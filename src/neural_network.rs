use std::fmt::Display;

use serde::{Deserialize, Serialize};

use crate::matrix::{self, Matrix};

#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralNetwork {
    // 权重
    weights: Vec<Matrix<f32>>,
    // 偏置
    biases: Vec<Matrix<f32>>,
    // 激活函数
    activation: Activation,
}

/// 激活函数
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    Sigmoid,
    Relu,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<usize>, activation: Activation) -> Self {
        let mut weights = Vec::with_capacity(layers.len() - 1);
        for i in 0..layers.len() - 1 {
            weights.push(Matrix::new(layers[i], layers[i + 1]));
        }
        let mut biases = Vec::with_capacity(layers.len() - 1);
        for i in 0..layers.len() - 1 {
            biases.push(Matrix::new(1, layers[i + 1]));
        }

        Self {
            weights,
            biases,
            activation,
        }
    }

    pub fn rand_weights<R>(&mut self, range: R)
    where
        R: rand::distributions::uniform::SampleRange<f32> + Clone,
    {
        for i in 0..self.weights.len() {
            self.weights[i].rand(range.clone());
        }
    }

    pub fn rand_biases<R>(&mut self, range: R)
    where
        R: rand::distributions::uniform::SampleRange<f32> + Clone,
    {
        for i in 0..self.biases.len() {
            self.biases[i].rand(range.clone());
        }
    }

    pub fn set_activation(&mut self, activation: Activation) {
        self.activation = activation
    }

    pub fn infer(&self, input: &Matrix) -> Result<Vec<Matrix>, Error> {
        self.fp(input)
    }

    pub fn get_layers(&self) -> Vec<usize> {
        let mut layers = Vec::with_capacity(self.weights.len() + 1);
        layers.push(self.weights[0].row());
        for i in 0..self.weights.len() {
            layers.push(self.weights[i].col());
        }
        layers
    }

    // 开始训练
    // X: 输入
    // Y: 输出
    // count: 训练次数
    pub fn practice(
        &mut self,
        x: &Vec<Matrix>,
        y: &Vec<Matrix>,
        learning_rate: f32,
        count: usize,
    ) -> Result<(), Error> {
        for _ in 0..count {
            for i in 0..x.len() {
                let x = x.get(i).ok_or(Error::Empty)?;
                let y = y.get(i).ok_or(Error::Empty)?;
                // 1. Feedforward
                let ls = self.fp(x)?;
                // 2. Backpropagation
                let deltas = self.bp(y, &ls)?;
                // 3. Update weights and biases
                Self::update_weight(
                    learning_rate,
                    x,
                    self.weights.get_mut(0).ok_or(Error::Empty)?,
                    self.biases.get_mut(0).ok_or(Error::Empty)?,
                    deltas.get(0).ok_or(Error::Empty)?,
                )?;
                for i in 1..self.weights.len() {
                    Self::update_weight(
                        learning_rate,
                        ls.get(i - 1).ok_or(Error::Empty)?,
                        self.weights.get_mut(i).ok_or(Error::Empty)?,
                        self.biases.get_mut(i).ok_or(Error::Empty)?,
                        deltas.get(i).ok_or(Error::Empty)?,
                    )?;
                }
            }
        }
        Ok(())
    }

    pub fn update_weight(
        learning_rate: f32,
        x: &Matrix,
        weight: &mut Matrix,
        bias: &mut Matrix,
        delta: &Matrix,
    ) -> Result<(), Error> {
        let d = x.transpose()?.cross(delta)?;
        weight.add_to(&(d.mul_num(learning_rate)))?;
        bias.add_to(&delta.mul_num(learning_rate))?;
        Ok(())
    }

    pub fn bp(&self, y: &Matrix, ls: &Vec<Matrix>) -> Result<Vec<Matrix>, Error> {
        let mut deltas = Vec::with_capacity(self.weights.len());
        deltas.resize(deltas.capacity(), Matrix::default());
        {
            let ln = ls.get(ls.len() - 1).unwrap();
            // 最后一个权重的误差
            let ln_error = y.sub(ln)?;
            // 最后一个权重的斜率(激活函数偏导)
            let ln_slope = {
                match self.activation {
                    Activation::Sigmoid => ln.dot(&(1.0f32 - ln))?,
                    Activation::Relu => Self::relu_derivative(ln),
                }
            };
            // 最后一个权重的增量
            let ln_delta = ln_error.dot(&ln_slope)?;
            deltas[self.weights.len() - 1] = ln_delta;
        }
        if deltas.len() == 1 {
            return Ok(deltas);
        }
        let mut i = self.weights.len() - 2;
        loop {
            let l = ls.get(i).ok_or(Error::Empty)?;
            let w = self.weights.get(i + 1).ok_or(Error::Empty)?;
            let d = deltas.get(i + 1).ok_or(Error::Empty)?;
            let error = d.cross(&w.transpose()?)?;
            let slope = {
                match self.activation {
                    Activation::Sigmoid => l.dot(&(1.0f32 - l))?,
                    Activation::Relu => Self::relu_derivative(l),
                }
            };
            deltas[i] = error.dot(&slope)?;

            if i == 0 {
                break;
            }
            i -= 1;
        }
        Ok(deltas)
    }

    pub fn fp(&self, input: &Matrix) -> Result<Vec<Matrix>, Error> {
        if self.weights.len() == 0 {
            return Err(Error::Empty);
        }
        let mut ls = Vec::with_capacity(self.weights.len());
        ls.push(Self::fp_layer(
            input,
            &self.weights[0],
            &self.biases[0],
            self.activation,
        )?);
        for i in 1..self.weights.len() - 1 {
            ls.push(Self::fp_layer(
                &ls[i - 1],
                &self.weights[i],
                &self.biases[i],
                self.activation,
            )?);
        }
        match self.activation {
            Activation::Sigmoid => {
                let i = self.weights.len() - 1;
                ls.push(Self::fp_layer(
                    &ls[i - 1],
                    &self.weights[i],
                    &self.biases[i],
                    self.activation,
                )?);
            }
            Activation::Relu => {
                let i = self.weights.len() - 1;
                ls.push({
                    let mut z = ls[i - 1].cross(&self.weights[i])?;
                    z.add_to(&self.biases[i])?;
                    Self::softmax(z)
                });
            }
        }
        Ok(ls)
    }

    pub fn fp_layer(
        input: &Matrix,
        weight: &Matrix,
        bias: &Matrix,
        activation: Activation,
    ) -> Result<Matrix, Error> {
        let mut z = input.cross(weight)?;
        z.add_to(bias)?;
        match activation {
            Activation::Sigmoid => Ok(Self::sigmoid(z)),
            Activation::Relu => Ok(Self::relu(z)),
        }
    }

    pub fn sigmoid(z: Matrix) -> Matrix {
        return 1.0f32 / (1.0f32 + (-z).exp());
    }

    pub fn relu(mut z: Matrix) -> Matrix {
        let z_data = z.get_data_mut();
        for i in 0..z_data.len() {
            if z_data[i] < 0.0 {
                z_data[i] = 0.0
            }
        }
        z
    }

    pub fn relu_derivative(l: &Matrix) -> Matrix {
        let mut r = Vec::with_capacity(l.row() * l.col());
        unsafe { r.set_len(r.capacity()) }
        let l_data = l.get_data();
        for i in 0..l_data.len() {
            r[i] = if l_data[i] > 0.0 { 1.0 } else { 0.0 };
        }
        Matrix::with_data(l.row(), l.col(), r)
    }

    pub fn softmax(mut z: Matrix) -> Matrix {
        let z_data = z.get_data_mut();
        let mut sum = 0.0;
        for i in 0..z_data.len() {
            z_data[i] = z_data[i].exp();
            sum += z_data[i];
        }
        for i in 0..z_data.len() {
            z_data[i] /= sum;
        }
        z
    }
}

impl ToString for NeuralNetwork {
    fn to_string(&self) -> String {
        let mut s = String::new();
        for i in 0..self.weights.len() {
            s.push_str(&format!("Layer {}\n", i + 1));
            s.push_str(&self.weights[i].to_string());
            s.push_str(&format!("\nBiaese {}\n", i + 1));
            s.push_str(&self.biases[i].to_string());
            s.push_str("\n");
        }
        s.pop();
        s
    }
}

#[derive(Debug)]
pub enum Error {
    // 计算错误
    ComputeError,
    // 空
    Empty,
}

impl From<matrix::Error> for Error {
    fn from(_: matrix::Error) -> Self {
        Error::ComputeError
    }
}

impl PartialEq for Activation {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Activation::Sigmoid, Activation::Sigmoid) => true,
            (Activation::Relu, Activation::Relu) => true,
            _ => false,
        }
    }

    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}

impl Display for Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Activation::Sigmoid => "Sigmoid",
                Activation::Relu => "Relu",
            }
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_neural_network() {
        let mut nn = NeuralNetwork::new(vec![280, 16, 16, 10], Activation::Sigmoid);
        nn.rand_weights(-1.0..1.0);
        println!("{}", nn.to_string());
    }

    // 测试sigmoid函数
    #[test]
    fn test_sigmoid() {
        let z = Matrix::with_data(1, 3, vec![9.65, 0.12, 0.63]);
        let a = NeuralNetwork::sigmoid(z);
        println!("{}", a);
    }

    // 测试正向传播
    #[test]
    fn test_fp() {
        let mut nn = NeuralNetwork::new(vec![2, 3, 1], Activation::Sigmoid);
        nn.rand_weights(-1.0..1.0);
        nn.rand_biases(-1.0..1.0);
        println!("{}\n", nn.to_string());
        let input = Matrix::with_data(1, 2, vec![0.65, 0.12]);
        println!("input: \n{}", input);
        let output = nn.fp(&input).unwrap();
        println!("output: \n{:?}", output);
    }
}
