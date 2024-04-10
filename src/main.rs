use std::{fs::OpenOptions, io::Write};

use crate::matrix::Matrix;

mod matrix;
mod neural_network;

fn main() {
    let mut nn = neural_network::NeuralNetwork::new(vec![3, 6, 1]);
    // 生成随机权重和偏置
    nn.rand_weights();
    nn.rand_biases();
    nn.set_learning_rate(0.05);
    println!("{}\n", nn.to_string());
    let input = matrix::Matrix::with_data(1, 3, vec![1., 1., 0.]);
    println!("input: \n{}", input);
    let output = nn.fp(&input).unwrap();
    println!("output: \n{}", output[output.len() - 1]);

    let x = vec![
        Matrix::with_data(1, 3, vec![0., 0., 1.]),
        Matrix::with_data(1, 3, vec![0., 1., 1.]),
        Matrix::with_data(1, 3, vec![1., 0., 1.]),
        Matrix::with_data(1, 3, vec![1., 1., 1.]),
    ];
    let y = vec![
        Matrix::with_data(1, 1, vec![0.]),
        Matrix::with_data(1, 1, vec![1.]),
        Matrix::with_data(1, 1, vec![1.]),
        Matrix::with_data(1, 1, vec![0.]),
    ];

    // 获取时间间隔
    let start = std::time::Instant::now();
    let count = 50;
    let step = 1000;
    for i in 1..=count {
        nn.practice(&x, &y, step).unwrap();
        if i % 10 == 0 {
            let output = nn.fp(&input).unwrap();
            println!("output: {}", output[output.len() - 1]);
            println!("{}/{}", i, count);
        }
    }
    // 输出耗时
    println!("Time: {:?}", start.elapsed());

    println!("{}\n", nn.to_string());

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
        println!("{}: {}", i, output[output.len() - 1]);
    }

    // 保存神经网络
    OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("nn.json")
        .unwrap()
        .write_all(serde_json::to_string(&nn).unwrap().as_bytes())
        .unwrap();
}
