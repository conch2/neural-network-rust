use crate::matrix::{self, Matrix};

#[derive(Debug)]
pub struct NeuralNetwork {
    // 权重
    weights: Vec<Matrix<f32>>,
    // 偏置
    biases: Vec<Matrix<f32>>,
    // 学习率
    learning_rate: f32,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<usize>) -> Self {
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
            learning_rate: 0.1,
        }
    }

    pub fn rand_weights(&mut self) {
        for i in 0..self.weights.len() {
            self.weights[i].rand(-1.0f32..1.0f32);
        }
    }

    pub fn rand_biases(&mut self) {
        for i in 0..self.biases.len() {
            self.biases[i].rand(-1.0f32..1.0f32);
        }
    }

    // 开始训练
    // X: 输入
    // Y: 输出
    // count: 训练次数
    pub fn practice(&mut self, X: &Vec<Matrix>, Y: &Vec<Matrix>, count: usize) -> Result<(), Error> {
        for _ in 0..count {
            for i in 0..X.len() {
                let x = X.get(i).ok_or(Error::Empty)?;
                let y = Y.get(i).ok_or(Error::Empty)?;
                // 1. Feedforward
                let ls = self.fp(x)?;
                // 2. Backpropagation
                let deltas = self.bp(y, &ls)?;
                // 3. Update weights and biases
                Self::update_weight(
                    self.learning_rate,
                    x,
                    self.weights
                        .get_mut(0)
                        .map_or(Err(Error::Empty), |w| Ok(w))?,
                    self.biases
                        .get_mut(0)
                        .map_or(Err(Error::Empty), |bia| Ok(bia))?,
                    deltas.get(0).map_or(Err(Error::Empty), |d| Ok(d))?,
                )?;
                for i in 1..self.weights.len() {
                    Self::update_weight(
                        self.learning_rate,
                        ls.get(i - 1).map_or(Err(Error::Empty), |l| Ok(l))?,
                        self.weights
                            .get_mut(i)
                            .map_or(Err(Error::Empty), |w| Ok(w))?,
                        self.biases
                            .get_mut(i)
                            .map_or(Err(Error::Empty), |bia| Ok(bia))?,
                        deltas.get(i).map_or(Err(Error::Empty), |d| Ok(d))?,
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
        weight.sub_to(&(d * learning_rate))?;
        bias.sub_to(&delta.mul_num(learning_rate))?;
        Ok(())
    }

    pub fn bp(&self, y: &Matrix, ls: &Vec<Matrix>) -> Result<Vec<Matrix>, Error> {
        let mut deltas = Vec::with_capacity(self.weights.len());
        deltas.resize(deltas.capacity(), Matrix::default());
        {
            let ln = ls.get(ls.len() - 1).unwrap();
            // 最后一个权重的误差
            let ln_error = y.sub(ln)?;
            // 最后一个权重的斜率(激活函数求导)
            let ln_slope = ln.dot(&(1.0f32 - ln))?;
            // 最后一个权重的增量
            let ln_delta = ln_error.dot(&ln_slope)?;
            deltas[self.weights.len() - 1] = ln_delta;
        }
        if deltas.len() == 1 {
            return Ok(deltas);
        }
        let mut i = self.weights.len() - 2;
        loop {
            let l = ls.get(i).map_or(Err(Error::Empty), |l| Ok(l))?;
            let w = self
                .weights
                .get(i + 1)
                .map_or(Err(Error::Empty), |w| Ok(w))?;
            let d = deltas.get(i + 1).map_or(Err(Error::Empty), |d| Ok(d))?;
            let error = d.cross(&w.transpose()?)?;
            let slope = l.dot(&(1.0f32 - l))?;
            deltas[i] = error.dot(&slope)?;

            if i == 0 {
                break;
            } else {
                i -= 1;
            }
        }
        Ok(deltas)
    }

    pub fn fp(&mut self, input: &Matrix) -> Result<Vec<Matrix>, Error> {
        if self.weights.len() == 0 {
            return Err(Error::Empty);
        }
        let mut ls = Vec::with_capacity(self.weights.len());
        ls.push(Self::fp_layer(input, &self.weights[0], &self.biases[0])?);
        for i in 1..self.weights.len() {
            ls.push(Self::fp_layer(
                &ls[i - 1],
                &self.weights[i],
                &self.biases[i],
            )?)
        }
        Ok(ls)
    }

    pub fn fp_layer(input: &Matrix, weight: &Matrix, bias: &Matrix) -> Result<Matrix, Error> {
        let mut z = input.cross(weight)?;
        z = (z + bias)?;
        Ok(Self::sigmoid(&z))
    }

    pub fn sigmoid(z: &Matrix) -> Matrix {
        return 1.0f32 / (1.0f32 + z.exp());
    }
}

impl ToString for NeuralNetwork {
    fn to_string(&self) -> String {
        let mut s = String::new();
        for i in 0..self.weights.len() {
            s.push_str(&format!("Layer {}\n", i + 1));
            s.push_str(&self.weights[i].to_string());
            s.push_str(&format!("Biaese {}\n", i + 1));
            s.push_str(&self.biases[i].to_string());
        }
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

mod test {
    use super::*;

    #[test]
    fn test_neural_network() {
        let mut nn = NeuralNetwork::new(vec![280, 16, 16, 10]);
        nn.rand_weights();
        println!("{}", nn.to_string());
    }

    // 测试sigmoid函数
    #[test]
    fn test_sigmoid() {
        let z = Matrix::with_data(1, 3, vec![0.65, 0.12, 0.63]);
        let a = NeuralNetwork::sigmoid(&z);
        println!("{}", a);
    }

    // 测试正向传播
    #[test]
    fn test_fp() {
        let mut nn = NeuralNetwork::new(vec![2, 3, 1]);
        nn.rand_weights();
        nn.rand_biases();
        println!("{}\n", nn.to_string());
        let input = Matrix::with_data(1, 2, vec![0.65, 0.12]);
        println!("input: \n{}", input);
        let output = nn.fp(&input).unwrap();
        println!("output: \n{:?}", output);
    }
}
