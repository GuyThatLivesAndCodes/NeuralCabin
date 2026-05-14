//! NeuralCabin — Web-based interface only.
//!
//! The main application is now web-based (React frontend).
//! This binary is kept for backward compatibility and headless testing.
//!
//! Usage:
//!   neuralcabin              Show help (UI moved to web frontend)
//!   neuralcabin --xor-demo   Run XOR training demo headlessly
//!   neuralcabin --help       Show this help

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--xor-demo" || a == "--demo") {
        match xor_demo() {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("xor demo failed: {e}");
                ExitCode::FAILURE
            }
        }
    } else {
        println!("{}", help_text());
        ExitCode::SUCCESS
    }
}

fn help_text() -> &'static str {
    "🧠 NeuralCabin — Pure Rust Neural Network Workbench\n\n\
     The interactive UI is now web-based!\n\n\
     Quick Start:\n\
     \n\
     1. Start the backend server:\n\
     \t  cargo run --package neuralcabin-backend --release\n\
     \n\
     2. Start the frontend (in another terminal):\n\
     \t  cd frontend && npm install && npm run dev\n\
     \n\
     3. Open your browser:\n\
     \t  http://localhost:5173\n\
     \n\
     Or use the convenience script:\n\
     \t  ./start-dev.sh        (Linux/macOS)\n\
     \t  start-dev.bat         (Windows)\n\
     \n\
     Alternative:\n\
     \t  neuralcabin --xor-demo   Run XOR training headlessly\n\
     \t  neuralcabin --help       Show this help\n\
     \n\
     For more info, see QUICKSTART.md and REFACTOR_README.md"
}

fn xor_demo() -> Result<(), Box<dyn std::error::Error>> {
    use neuralcabin_engine::nn::{LayerSpec, Model};
    use neuralcabin_engine::optimizer::{Optimizer, OptimizerKind};
    use neuralcabin_engine::tensor::Tensor;
    use neuralcabin_engine::{Activation, Loss};

    let specs = vec![
        LayerSpec::Linear { in_dim: 2, out_dim: 8 },
        LayerSpec::Activation(Activation::Tanh),
        LayerSpec::Linear { in_dim: 8, out_dim: 1 },
        LayerSpec::Activation(Activation::Sigmoid),
    ];
    let mut model = Model::from_specs(2, &specs, 42);
    let mut opt = Optimizer::new(
        OptimizerKind::Adam { lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
        &model.parameter_shapes(),
    );
    let x = Tensor::new(vec![4, 2], vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    let y = Tensor::new(vec![4, 1], vec![0.0, 1.0, 1.0, 0.0]);
    println!("Training XOR MLP...");
    let mut last = f32::INFINITY;
    for epoch in 1..=2000 {
        last = model.train_step(&mut opt, Loss::MeanSquaredError, &x, &y);
        if epoch % 200 == 0 {
            println!("  epoch {epoch:>5}  loss = {last:.6}");
        }
    }
    println!("Final loss = {last:.6}");
    let pred = model.predict(&x);
    println!("Predictions:");
    for (i, p) in pred.data.iter().enumerate() {
        let xi = &x.data[i * 2..(i + 1) * 2];
        let ti = y.data[i];
        println!("  XOR({}, {}) -> {p:.4}  (target {ti})", xi[0], xi[1]);
    }
    if last > 0.05 {
        return Err(format!("convergence check failed: final loss {last} > 0.05").into());
    }
    Ok(())
}
