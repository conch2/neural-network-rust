use std::sync::Arc;

use crate::matrix::Matrix;
use crate::neural_network::Error;

#[derive(Debug)]
pub struct NeuralNetworkSync {
    // 权重
    weights: Arc<Vec<Matrix<f32>>>,
    // 偏置
    biases: Arc<Vec<Matrix<f32>>>,
}

impl NeuralNetworkSync {
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
            weights: Arc::new(weights),
            biases: Arc::new(biases),
        }
    }

    // 开始训练
    // X: 输入
    // Y: 输出
    // count: 训练次数
    pub fn practice(
        &mut self,
        x: Arc<Vec<Matrix>>,
        y: Arc<Vec<Matrix>>,
        learning_rate: f32,
        count: usize,
    ) -> Result<(), Error> {
        if let Ok(rt) = tokio::runtime::Runtime::new() {
            rt.block_on(self.practice_sync(x, y, learning_rate, count))?;
        } else {
            return Err(Error::Empty);
        }
        Ok(())
    }

    async fn practice_sync(
        &mut self,
        x: Arc<Vec<Matrix>>,
        y: Arc<Vec<Matrix>>,
        learning_rate: f32,
        count: usize,
    ) -> Result<(), Error> {
        for _ in 0..count {
            let mut joins: tokio::task::JoinSet<Result<(Vec<Matrix>, Vec<Matrix>), Error>> =
                tokio::task::JoinSet::new();
            for i in 0..x.len() {
                let x_clone = x.clone();
                let y_clone = y.clone();
                let weights_clone = self.weights.clone();
                let biases_clone = self.biases.clone();
                joins.spawn(async move {
                    let x = x_clone.get(i).ok_or(Error::Empty)?;
                    let y = y_clone.get(i).ok_or(Error::Empty)?;
                    let ls = Self::fp(&weights_clone, &biases_clone, x)?;
                    let deltas = Self::bp(&weights_clone, y, &ls)?;
                    let mut up_weights = Vec::with_capacity(weights_clone.len());

                    up_weights.push(x.transpose()?.cross(&deltas[0])?);
                    for i in 1..weights_clone.len() {
                        up_weights.push(
                            ls.get(i - 1)
                                .ok_or(Error::Empty)?
                                .transpose()?
                                .cross(&deltas[i])?,
                        )
                    }
                    Ok((up_weights, deltas))
                });
            }

            let (mut up_weights, mut deltas) = {
                let mut w: Vec<Matrix> = Vec::with_capacity(self.weights.len());
                let mut d: Vec<Matrix> = Vec::with_capacity(self.weights.len());
                for sw in self.weights.iter() {
                    w.push(Matrix::new(sw.row(), sw.col()));
                    d.push(Matrix::new(1, sw.col()));
                }
                (w, d)
            };
            while let Some(res) = joins.join_next().await {
                match res {
                    Ok(Ok(v)) => {
                        let (w, d) = v;
                        if up_weights.len() != w.len() {
                            log::error!("Some Error");
                            continue;
                        }
                        for i in 0..w.len() {
                            up_weights[i].add_to(&w[i]).unwrap();
                            deltas[i].add_to(&d[i]).unwrap();
                        }
                    },
                    Ok(e) => {
                        log::error!("Error: {:?}", e);
                        continue;
                    }
                    Err(e) => {
                        log::error!("Error: {:?}", e);
                        continue;
                    }
                }
            }
            let weight = Arc::make_mut(&mut self.weights);
            let bias = Arc::make_mut(&mut self.biases);
            for i in 0..weight.len() {
                weight[i].add_to(&up_weights[i].mul_num(learning_rate))?;
                bias[i].add_to(&deltas[i].mul_num(learning_rate))?;
            }
        }
        Ok(())
    }

    pub fn infer(&self, input: &Matrix) -> Result<Vec<Matrix>, Error> {
        Self::fp(&self.weights, &self.biases, input)
    }

    fn fp(
        weights: &Arc<Vec<Matrix>>,
        biases: &Arc<Vec<Matrix>>,
        input: &Matrix,
    ) -> Result<Vec<Matrix>, Error> {
        if weights.len() == 0 {
            return Err(Error::Empty);
        }
        let mut ls = Vec::with_capacity(weights.len());
        ls.push(Self::fp_layer(input, &weights[0], &biases[0])?);
        for i in 1..weights.len() {
            ls.push(Self::fp_layer(&ls[i - 1], &weights[i], &biases[i])?)
        }
        Ok(ls)
    }

    fn fp_layer(input: &Matrix, weight: &Matrix, bias: &Matrix) -> Result<Matrix, Error> {
        let mut z = input.cross(weight)?;
        z = (z + bias)?;
        Ok(Self::sigmoid(z))
    }

    fn sigmoid(z: Matrix) -> Matrix {
        return 1.0f32 / (1.0f32 + (-z).exp());
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

    fn bp(weights: &Arc<Vec<Matrix>>, y: &Matrix, ls: &Vec<Matrix>) -> Result<Vec<Matrix>, Error> {
        let mut deltas = Vec::with_capacity(weights.len());
        deltas.resize(deltas.capacity(), Matrix::default());
        {
            let ln = ls.get(ls.len() - 1).unwrap();
            // 最后一个权重的误差
            let ln_error = y.sub(ln)?;
            // 最后一个权重的斜率(激活函数求导)
            let ln_slope = ln.dot(&(1.0f32 - ln))?;
            // 最后一个权重的增量
            let ln_delta = ln_error.dot(&ln_slope)?;
            deltas[weights.len() - 1] = ln_delta;
        }
        if deltas.len() == 1 {
            return Ok(deltas);
        }
        let mut i = weights.len() - 2;
        loop {
            let l = ls.get(i).ok_or(Error::Empty)?;
            let w = weights.get(i + 1).ok_or(Error::Empty)?;
            let d = deltas.get(i + 1).ok_or(Error::Empty)?;
            let error = d.cross(&w.transpose()?)?;
            let slope = l.dot(&(1.0f32 - l))?;
            deltas[i] = error.dot(&slope)?;

            if i == 0 {
                break;
            }
            i -= 1;
        }
        Ok(deltas)
    }

    pub fn rand_weights(&mut self) {
        let weights = Arc::make_mut(&mut self.weights);
        for i in 0..weights.len() {
            weights[i].rand(-1.0f32..1.0f32);
        }
    }

    pub fn rand_biases(&mut self) {
        let biases = Arc::make_mut(&mut self.biases);
        for i in 0..biases.len() {
            biases[i].rand(-1.0f32..1.0f32);
        }
    }
}
