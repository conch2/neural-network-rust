use std::{fs::OpenOptions, io::Write};

use neunet::*;

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .target(env_logger::Target::Stdout)
        .init();
    test_nn();
    // test_sync();
}

fn test_sync() {
    let mut nn = NeuralNetworkSync::new(vec![3, 4, 1]);
    // 生成随机权重和偏置
    nn.rand_weights();
    nn.rand_biases();

    let x = std::sync::Arc::new(vec![
        Matrix::with_data(1, 3, vec![0., 0., 1.]),
        Matrix::with_data(1, 3, vec![0., 1., 1.]),
        Matrix::with_data(1, 3, vec![1., 0., 1.]),
        Matrix::with_data(1, 3, vec![1., 1., 1.]),
    ]);
    let y = std::sync::Arc::new(vec![
        Matrix::with_data(1, 1, vec![0.]),
        Matrix::with_data(1, 1, vec![1.]),
        Matrix::with_data(1, 1, vec![1.]),
        Matrix::with_data(1, 1, vec![0.]),
    ]);
    let input = Matrix::with_data(1, 3, vec![1., 1., 0.]);

    // 获取时间间隔
    let start = std::time::Instant::now();
    let count = 10;
    let step = 1000;
    println!("\nStart training...\n");
    for i in 1..=count {
        let t = std::time::Instant::now();
        if let Err(e) = nn.practice(x.clone(), y.clone(), 0.1, step) {
            log::error!("{:?}", e);
        }

        let output = nn.infer(&input).unwrap();
        println!("output: {}", output[output.len() - 1]);
        println!("{}/{}\t{:?}", i, count, t.elapsed());
    }
    // 输出耗时
    println!("Time: {:?}", start.elapsed());

    test_for_sync(&mut nn);
}

fn test_for_sync(nn: &mut NeuralNetworkSync) {
    for i in vec![
        Matrix::with_cols(3, vec![vec![0., 0., 0.]]),
        Matrix::with_cols(3, vec![vec![0., 0., 1.]]),
        Matrix::with_cols(3, vec![vec![0., 1., 0.]]),
        Matrix::with_cols(3, vec![vec![0., 1., 1.]]),
        Matrix::with_cols(3, vec![vec![1., 0., 0.]]),
        Matrix::with_cols(3, vec![vec![1., 0., 1.]]),
        Matrix::with_cols(3, vec![vec![1., 1., 0.]]),
        Matrix::with_cols(3, vec![vec![1., 1., 1.]]),
    ] {
        let output = nn.infer(&i).unwrap();
        println!("{} -> {}", i, output[output.len() - 1]);
    }
}

fn test_nn() {
    let mut nn = NeuralNetwork::new(vec![3, 28 * 28, 128, 10, 2], Activation::Relu);
    // 生成随机权重和偏置
    nn.rand_weights(0.0..0.04);
    nn.rand_biases(-30.0..0.01);
    // println!("{}\n", nn.to_string());
    let input = Matrix::with_data(1, 3, vec![1., 1., 0.]);
    println!("input: \n{}\n", input);

    test(&mut nn);

    let x = vec![
        Matrix::with_data(1, 3, vec![0., 0., 1.]),
        Matrix::with_data(1, 3, vec![0., 1., 1.]),
        Matrix::with_data(1, 3, vec![1., 0., 1.]),
        Matrix::with_data(1, 3, vec![1., 1., 1.]),
    ];
    let y = vec![
        Matrix::with_data(1, 2, vec![0., 1.]),
        Matrix::with_data(1, 2, vec![1., 0.]),
        Matrix::with_data(1, 2, vec![1., 0.]),
        Matrix::with_data(1, 2, vec![0., 1.]),
    ];

    // 获取时间间隔
    let start = std::time::Instant::now();
    let count = 2;
    let step = 100;
    let mut learning_rate = 0.1;
    println!("\nStart training...\n");
    for i in 1..=count {
        let t = std::time::Instant::now();
        nn.practice(&x, &y, learning_rate, step).unwrap();

        let output = nn.infer(&input).unwrap();
        println!(
            "LR: {}\t output: {}",
            learning_rate,
            output[output.len() - 1]
        );
        println!("{}/{}\t{:?}", i, count, t.elapsed());

        learning_rate *= 0.95;
    }
    // 输出耗时
    println!("Time: {:?}", start.elapsed());
    // println!("{}", nn.to_string());

    test(&mut nn);

    // 保存神经网络
    OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("cache.json")
        .unwrap()
        .write_all(serde_json::to_string(&nn).unwrap().as_bytes())
        .unwrap();
}

fn test(nn: &mut NeuralNetwork) {
    for i in vec![
        Matrix::with_cols(3, vec![vec![0., 0., 0.]]),
        Matrix::with_cols(3, vec![vec![0., 0., 1.]]),
        Matrix::with_cols(3, vec![vec![0., 1., 0.]]),
        Matrix::with_cols(3, vec![vec![0., 1., 1.]]),
        Matrix::with_cols(3, vec![vec![1., 0., 0.]]),
        Matrix::with_cols(3, vec![vec![1., 0., 1.]]),
        Matrix::with_cols(3, vec![vec![1., 1., 0.]]),
        Matrix::with_cols(3, vec![vec![1., 1., 1.]]),
    ] {
        let output = nn.fp(&i).unwrap();
        println!("{} -> {}", i, output[output.len() - 1]);
    }
}
