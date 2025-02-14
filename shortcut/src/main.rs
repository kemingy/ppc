use std::env;
use std::time::Instant;

use rand::Rng;

fn step(graph: &[f32], res: &mut [f32], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut min_distance = f32::MAX;
            for k in 0..n {
                min_distance = min_distance.min(graph[i * n + k] + graph[k * n + j]);
            }
            res[i * n + j] = min_distance;
        }
    }
}

fn gen_random_graph(n: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    (0..n * n)
        .map(|i| if i % n == i / n { 0.0 } else { rng.random() })
        .collect()
}

fn gen_default_shortcut(n: usize) -> Vec<f32> {
    vec![0.0; n * n]
}

fn display(mat: &[f32], n: usize) {
    println!("Matrix:");
    for i in 0..n {
        for j in 0..n {
            print!("{:.2} ", mat[i * n + j]);
        }
        println!();
    }
    println!("-----------------");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut n = 5;
    if args.len() > 1 {
        n = args[1].parse().unwrap();
    }

    let graph = gen_random_graph(n);
    let mut res = gen_default_shortcut(n);
    let start = Instant::now();
    step(&graph, &mut res, n);
    println!("Time elapsed in step() is: {:?}", start.elapsed());

    if n < 16 {
        display(&graph, n);
        display(&res, n);
    }
}
