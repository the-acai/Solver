//! src/main.rs
use anyhow::{anyhow, bail, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use meval::{Context, Expr};
use ordered_float::OrderedFloat;
use rayon::prelude::*;
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
    #[arg(long, default_value_t = -10.0)]
    min_a: f64,
    #[arg(long, default_value_t = 10.0)]
    max_a: f64,

    /// Minimum / maximum for parameter b
    #[arg(long, default_value_t = -10.0)]
    min_b: f64,
    #[arg(long, default_value_t = 10.0)]
    max_b: f64,

    /// Minimum / maximum for parameter c
    #[arg(long, default_value_t = -10.0)]
    min_c: f64,
    #[arg(long, default_value_t = 10.0)]
    max_c: f64,

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

/// Compute function variants
#[derive(Clone)]
enum ComputeFn {
    Default(fn(f64, f64, f64) -> f64),
    Expr(Arc<Expr>),
}

impl ComputeFn {
    fn eval(&self, a: f64, b: f64, c: f64) -> f64 {
        match self {
            Self::Default(f) => f(a, b, c),
            Self::Expr(expr) => expr
                .eval_with_context(([("a", a), ("b", b), ("c", c)], Context::new()))
                .expect("eval failed"),
        }
    }
}

fn default_compute(a: f64, b: f64, c: f64) -> f64 {
    (a * b + c).sin().cos().sqrt().exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linspace() {
        let vals = linspace(0.0, 1.0, 5);
        assert_eq!(vals, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn test_compute_default() {
        let cf = ComputeFn::Default(default_compute as fn(f64, f64, f64) -> f64);
        let direct = default_compute(1.0, 2.0, 3.0);
        assert!((cf.eval(1.0, 2.0, 3.0) - direct).abs() < 1e-12);
    }

    #[test]
    fn test_compute_expr() {
        let expr: Expr = "a + b + c".parse().unwrap();
        let cf = ComputeFn::Expr(Arc::new(expr));
        assert_eq!(cf.eval(1.0, 2.0, 3.0), 6.0);
    }
}

/// NumPy‑style linspace (inclusive)
fn linspace(min: f64, max: f64, steps: usize) -> Vec<f64> {
    if steps <= 1 {
        vec![min]
    } else {
        let step = (max - min) / (steps - 1) as f64;
        let mut vals = Vec::with_capacity(steps);
        for i in 0..steps {
            vals.push(min + i as f64 * step);
        }
        vals
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
    let compute_fn = if let Some(expr_str) = &cfg.expr {
        let expr: Expr = expr_str
            .parse()
            .map_err(|e| anyhow!("Expression parse error: {e}"))?;
        Arc::new(ComputeFn::Expr(Arc::new(expr)))
    } else {
        Arc::new(ComputeFn::Default(
            default_compute as fn(f64, f64, f64) -> f64,
        ))
    };

    /* ------------------------- build ranges ----------------------------- */
    let a_vals = linspace(cfg.min_a, cfg.max_a, cfg.steps);
    let b_vals = linspace(cfg.min_b, cfg.max_b, cfg.steps);
    let c_vals = linspace(cfg.min_c, cfg.max_c, cfg.steps);

    let na = a_vals.len();
    let nb = b_vals.len();
    let nc = c_vals.len();
    let total = (na * nb * nc) as u64;
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
        type Item = Reverse<(
            OrderedFloat<f64>,
            OrderedFloat<f64>,
            OrderedFloat<f64>,
            OrderedFloat<f64>,
        )>; // min‑heap
        let heap = (0..na * nb * nc)
            .into_par_iter()
            .fold(
                || BinaryHeap::<Item>::new(),
                |mut heap, idx| {
                    let ia = idx % na;
                    let ib = (idx / na) % nb;
                    let ic = idx / (na * nb);
                    let a = a_vals[ia];
                    let b = b_vals[ib];
                    let c = c_vals[ic];
                    let res = compute_fn.eval(a, b, c);

                    // progress
                    let prev = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    if prev % PB_BATCH == 0 {
                        pb.inc(PB_BATCH);
                    }

                    heap.push(Reverse((
                        OrderedFloat(res),
                        OrderedFloat(a),
                        OrderedFloat(b),
                        OrderedFloat(c),
                    )));
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
            .map(|Reverse((score, a, b, c))| {
                (
                    a.into_inner(),
                    b.into_inner(),
                    c.into_inner(),
                    score.into_inner(),
                )
            })
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
            let mut wtr = csv::Writer::from_writer(BufWriter::new(File::create(output_path)?));
            wtr.write_record(&["a", "b", "c", "result"])?;
            for (a, b, c, r) in rx {
                wtr.serialize((a, b, c, r))?;
            }
            wtr.flush()?;
            Ok(())
        });

        // Compute threads
        (0..na * nb * nc).into_par_iter().for_each(|idx| {
            let ia = idx % na;
            let ib = (idx / na) % nb;
            let ic = idx / (na * nb);
            let a = a_vals[ia];
            let b = b_vals[ib];
            let c = c_vals[ic];
            let res = compute_fn.eval(a, b, c);
            tx.send((a, b, c, res)).expect("writer thread gone");

            let prev = counter.fetch_add(1, Ordering::Relaxed) + 1;
            if prev % PB_BATCH == 0 {
                pb.inc(PB_BATCH);
            }
        });

        drop(tx); // close channel
        writer_handle.join().expect("writer thread panicked")?; // propagate error

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
