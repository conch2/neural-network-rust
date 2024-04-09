use crate::matrix::Matrix;

mod matrix;
mod neural_network;

fn main() {
    let mut nn = neural_network::NeuralNetwork::new(vec![3, 1]);
    // 生成随机权重和偏置
    nn.rand_weights();
    nn.rand_biases();
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
        Matrix::with_data(1, 1, vec![0.]),
        Matrix::with_data(1, 1, vec![1.]),
        Matrix::with_data(1, 1, vec![1.]),
    ];

    let count = 5;
    let step = 1000;
    for i in 0..count {
        nn.practice(&x, &y, step).unwrap();
        if i % 10 == 0 {
            let output = nn.fp(&input).unwrap();
            println!("output: \n{}", output[output.len() - 1]);
            println!("{}/{}\n", i, count);
        }
    }

    println!("{}\n", nn.to_string());
    let input = matrix::Matrix::with_data(1, 3, vec![1., 0., 0.]);
    println!("input: \n{}", input);
    let output = nn.fp(&input).unwrap();
    println!("output: \n{}", output[output.len() - 1]);
}
