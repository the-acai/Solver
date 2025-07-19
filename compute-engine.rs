use clap::Parser;
// For pretending we're making progress when in reality it's just a big damn loop
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs::File;
use std::io::Write;
// For safely sharing progress between a million threads clawing at each other
use std::sync::{Arc, Mutex};
// For timing the code
use std::time::Instant;

/// Grabs command-line args
#[derive(Parser, Debug)]
#[command(name = "Compute Optimizer")]
#[command(about = "A benchmark utility for computing combinations", long_about = None)]
struct Config {
    // defaults are honestly mostly to make sure I don't need to call the file with args when testing
    // Start and end for variable `a`, because one loop isn’t enough
    #[arg(long, default_value = "-10.0")]
    min_a: f64,
    #[arg(long, default_value = "10.0")]
    max_a: f64,

    // Same song, second verse: this time for `b`
    #[arg(long, default_value = "-10.0")]
    min_b: f64,
    #[arg(long, default_value = "10.0")]
    max_b: f64,

    // And again, because three dimensions are apparently necessary, would hate to see what 10 args looks like
    #[arg(long, default_value = "-10.0")]
    min_c: f64,
    #[arg(long, default_value = "10.0")]
    max_c: f64,

    // Number of steps for the grid. The bigger this is, the more the CPU hates me
    // Seriously, this can get out of hand fast: total_loops=steps^args, some sort of smarter search is probably preferred
    #[arg(long, default_value = "100")]
    steps: usize,

    // Output CSV file name, because they still rule the world
    #[arg(long, default_value = "output.csv")]
    output: String,
}

/// The actual computation
/// TODO: replace with something useful
fn compute_output(inputs: &[f64]) -> f64 {
    let a = inputs[0];
    let b = inputs[1];
    let c = inputs[2];

    // Expensive math, turns electric bill into a CSV of numbers
    (a * b + c).sin().cos().sqrt().exp()
}

/// Generates linearly spaced values like numpy's linspace
fn linspace(min: f64, max: f64, steps: usize) -> Vec<f64> {
    if steps <= 1 {
        return vec![min];
    }
    let step = (max - min) / (steps - 1) as f64;
    (0..steps).map(|i| min + i as f64 * step).collect()
}

fn main() {
    // Parse all those arguments
    let config = Config::parse();
    let start_time = Instant::now();

    // Make value ranges for a, b, and c
    let a_vals = linspace(config.min_a, config.max_a, config.steps);
    let b_vals = linspace(config.min_b, config.max_b, config.steps);
    let c_vals = linspace(config.min_c, config.max_c, config.steps);

    // This is how many combinations we’re about to throw a all the CPU threads at
    // TODO: warn and request a y/n  before running a few trilli ops
    let total = (a_vals.len() * b_vals.len() * c_vals.len()) as u64;

    // Fancy progress bar so you can feel better while your CPU catches fire
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] [{wide_bar}] {percent}% ({pos}/{len})")
            .unwrap(),
    );
    let pb = Arc::new(pb); // Share it like a hot potato
    let pb_inc_mutex = Arc::new(Mutex::new(0u64)); // Shared counter so threads don't fistfight and kill eachother

    // And now... the hell loop
    let results: Vec<(f64, f64, f64, f64)> = a_vals
        .par_iter() // Outer loop
        .flat_map_iter(|&a| {
            let b_vals = &b_vals;
            let c_vals_cloned = c_vals.clone(); // cloning
            let pb = Arc::clone(&pb);
            let pb_inc_mutex = Arc::clone(&pb_inc_mutex);

            b_vals.iter().flat_map(move |&b| {
                let c_vals_inner = c_vals_cloned.clone(); // Another clone, Kill me
                let pb = Arc::clone(&pb);
                let pb_inc_mutex = Arc::clone(&pb_inc_mutex);

                // Final loop: welcome to inner-loop hell
                c_vals_inner.into_iter().map(move |c| {
                    let input = vec![a, b, c];
                    let output = compute_output(&input); // Do the actual math

                    // Update the progress bar every 1000 ops, because otherwise it lags like hell, might make this larger or match CPU threads
                    let mut counter = pb_inc_mutex.lock().unwrap();
                    *counter += 1;
                    if *counter >= 1000 {
                        pb.inc(*counter);
                        *counter = 0;
                    }

                    (a, b, c, output) // Save this sweet combo, could be a possible solution but probably not
                })
            })
        })
        .collect();

    // Last gasp of progress reporting
    pb.inc(*pb_inc_mutex.lock().unwrap());
    pb.finish_with_message("Computation complete");

    // Save the results to a CSV, because they're still the king
    let mut file = File::create(&config.output).expect("Couldn't create output file. Fuck.");
    writeln!(file, "a,b,c,result").expect("Failed to write headers. Seriously?");
    for (a, b, c, result) in results {
        writeln!(file, "{},{},{},{}", a, b, c, result).expect("Couldn't write data. Goddammit.")
    }

    // Celebrate that it didn't crash and burn if it even makes it this far
    println!(
        "Finished in {:.2?}. Results written to {}.",
        start_time.elapsed(),
        config.output
    );
}
