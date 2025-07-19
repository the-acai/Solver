//! src/main.rs
use anyhow::{anyhow, bail, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use itertools::iproduct;
use meval::Expr;
use ordered_float::OrderedFloat;
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    fs::File,
    io::{BufWriter, Write},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

/// Batch size for progress‑bar updates
const PB_BATCH: u64 = 10_000;

/// Command‑line interface ---------------------------------------------------
#[derive(Parser, Debug)]
#[command(name = "Compute Optimizer")]
struct Config {
    /// Minimum / maximum for parameter a
    #[arg(long, default_value_t = -10.0)] min_a: f64,
    #[arg(long, default_value_t =  10.0)] max_a: f64,

    /// Minimum / maximum for parameter b
    #[arg(long, default_value_t = -10.0)] min_b: f64,
    #[arg(long, default_value_t =  10.0)] max_b: f64,

    /// Minimum / maximum for parameter c
    #[arg(long, default_value_t = -10.0)] min_c: f64,
    #[arg(long, default_value_t =  10.0)] max_c: f64,

    /// Grid resolution per axis (total ops = steps³)
    #[arg(long, default_value_t = 100)]
    steps: usize,

    /// Skip the “are you sure?” prompt for huge runs
    #[arg(long)]
    force: bool,

    /// Only keep the top N results (highest output). Omit for *all* rows.
    #[arg(long)]
    top: Option<usize>,

    /// Custom math expression, e.g. `--expr "sin(a*b+c)"`.
    /// Available variables: `a`, `b`, `c`
    #[arg(long)]
    expr: Option<String>,

    /// Output CSV file
    #[arg(long, default_value = "output.csv")]
    output: String,
}

/// NumPy‑style linspace (inclusive)
fn linspace(min: f64, max: f64, steps: usize) -> Vec<f64> {
    if steps <= 1 {
        vec![min]
    } else {
        let step = (max - min) / (steps - 1) as f64;
        (0..steps).map(|i| min + i as f64 * step).collect()
    }
}

/// Ask user whether to continue on extremely large runs
fn confirm_large_run(total: u64) -> Result<()> {
    eprintln!(
        "About to perform {total} computations – this could take a long time. Continue? [y/N] "
    );
    let mut ans = String::new();
    std::io::stdin().read_line(&mut ans)?;
    if !matches!(ans.trim(), "y" | "Y") {
        bail!("Aborted by user");
    }
    Ok(())
}

fn main() -> Result<()> {
    let cfg = Config::parse();
    let start = Instant::now();

    /* -------------------- set up compute function ------------------------ */
    let compute_fn: Arc<dyn Fn(f64, f64, f64) -> f64 + Send + Sync> = if let Some(expr_str) = &cfg.expr {
        let expr: Expr = expr_str
            .parse()
            .map_err(|e| anyhow!("Expression parse error: {e}"))?;
        let func = expr
            .bind3("a", "b", "c")
            .map_err(|e| anyhow!("Binding error: {e}"))?;
        Arc::new(move |a, b, c| func(a, b, c))
    } else {
        Arc::new(|a: f64, b: f64, c: f64| (a * b + c).sin().cos().sqrt().exp())
    };

    /* ------------------------- build ranges ----------------------------- */
    let a_vals = linspace(cfg.min_a, cfg.max_a, cfg.steps);
    let b_vals = linspace(cfg.min_b, cfg.max_b, cfg.steps);
    let c_vals = linspace(cfg.min_c, cfg.max_c, cfg.steps);

    let total = (a_vals.len() * b_vals.len() * c_vals.len()) as u64;
    if !cfg.force && total > 10_000_000 {
        confirm_large_run(total)?;
    }

    /* --------------------- progress bar --------------------------------- */
    let pb = Arc::new(
        ProgressBar::new(total).with_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] [{wide_bar}] {percent}% ({pos}/{len})",
            )
            .unwrap(),
        ),
    );
    let counter = Arc::new(AtomicU64::new(0));

    /* ----------------------- main workload ------------------------------ */
    if let Some(top_n) = cfg.top {
        /* -------- Top‑N mode (memory‑bounded) --------------------------- */
        type Item = Reverse<(OrderedFloat<f64>, f64, f64, f64)>; // min‑heap
        let heap = iproduct!(
            a_vals.iter().cloned(),
            b_vals.iter().cloned(),
            c_vals.iter().cloned()
        )
        .par_bridge()
        .fold(
            || BinaryHeap::<Item>::new(),
            |mut heap, (a, b, c)| {
                let res = (compute_fn)(a, b, c);

                // progress
                let prev = counter.fetch_add(1, Ordering::Relaxed) + 1;
                if prev % PB_BATCH == 0 {
                    pb.inc(PB_BATCH);
                }

                heap.push(Reverse((OrderedFloat(res), a, b, c)));
                if heap.len() > top_n {
                    heap.pop(); // discard lowest
                }
                heap
            },
        )
        .reduce(
            || BinaryHeap::<Item>::new(),
            |mut h1, h2| {
                for item in h2 {
                    h1.push(item);
                    if h1.len() > top_n {
                        h1.pop();
                    }
                }
                h1
            },
        );

        let mut results: Vec<_> = heap
            .into_sorted_vec() // ascending
            .into_iter()
            .map(|Reverse((score, a, b, c))| (a, b, c, score.into_inner()))
            .collect();
        results.reverse(); // descending

        // flush remaining progress ticks & finish bar
        let rem = counter.load(Ordering::Relaxed) % PB_BATCH;
        if rem > 0 {
            pb.inc(rem);
        }
        pb.finish_with_message("Computation complete");

        // write CSV
        let mut file = csv::Writer::from_writer(BufWriter::new(File::create(&cfg.output)?));
        file.write_record(&["a", "b", "c", "result"])?;
        for (a, b, c, r) in results {
            file.serialize((a, b, c, r))?;
        }
        file.flush()?;
    } else {
        /* -------- Stream‑all mode (bounded channel → single writer) ----- */
        use crossbeam_channel::bounded;

        let (tx, rx) = bounded::<(f64, f64, f64, f64)>(1_024);

        // Writer thread
        let output_path = cfg.output.clone();
        let writer_handle = std::thread::spawn(move || -> Result<()> {
            let mut wtr =
                csv::Writer::from_writer(BufWriter::new(File::create(output_path)?));
            wtr.write_record(&["a", "b", "c", "result"])?;
            for (a, b, c, r) in rx {
                wtr.serialize((a, b, c, r))?;
            }
            wtr.flush()?;
            Ok(())
        });

        // Compute threads
        iproduct!(
            a_vals.iter().cloned(),
            b_vals.iter().cloned(),
            c_vals.iter().cloned()
        )
        .par_bridge()
        .for_each(|(a, b, c)| {
            let res = (compute_fn)(a, b, c);
            tx.send((a, b, c, res)).expect("writer thread gone");

            let prev = counter.fetch_add(1, Ordering::Relaxed) + 1;
            if prev % PB_BATCH == 0 {
                pb.inc(PB_BATCH);
            }
        });

        drop(tx); // close channel
        writer_handle
            .join()
            .expect("writer thread panicked")?; // propagate error

        // flush remaining progress ticks & finish bar
        let rem = counter.load(Ordering::Relaxed) % PB_BATCH;
        if rem > 0 {
            pb.inc(rem);
        }
        pb.finish_with_message("Computation complete");
    }

    /* ------------------------- wrap‑up ---------------------------------- */
    println!(
        "Finished in {:.2?}. Results written to {}.",
        start.elapsed(),
        cfg.output
    );
    Ok(())
}