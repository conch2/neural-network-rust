use rand::prelude::*;
use std::{
    fmt::{Debug, Display},
    ops::*,
    usize,
};

#[derive(Debug, Default, Clone)]
pub struct Matrix<T = f32> {
    rows: usize,  // 有多少行
    cols: usize,  // 有多少列
    data: Vec<T>, // 数据
}

impl<T> Matrix<T>
where
    T: Default + Clone,
{
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: {
                let mut data = Vec::with_capacity(rows * cols);
                data.resize(data.capacity(), T::default());
                data
            },
        }
    }

    pub fn with_data(rows: usize, cols: usize, data: Vec<T>) -> Self {
        Self { rows, cols, data }
    }

    // 随机初始化
    pub fn rand<R>(&mut self, range: R)
    where
        T: rand::distributions::uniform::SampleUniform,
        R: rand::distributions::uniform::SampleRange<T> + Clone,
    {
        for i in 0..self.data.len() {
            self.data[i] = rand::thread_rng().gen_range(range.clone());
        }
    }

    // 求矩阵每一个元素的exp
    pub fn exp(&self) -> Self
    where
        T: Exp,
    {
        let mut result = Self::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i].mexp();
        }
        result
    }

    // 矩阵转置
    pub fn transpose(&self) -> Result<Self, Error>
    where
        T: Copy,
    {
        let mut result = Self::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[(j, i)] = self[(i, j)];
            }
        }
        Ok(result)
    }

    // 数学上的叉乘
    pub fn cross(&self, rhs: &Self) -> Result<Self, Error>
    where
        T: Mul<Output = T> + Add<Output = T> + Default + Copy,
    {
        if self.cols != rhs.rows {
            return Err(Error::Uncomputable);
        }
        let mut result = Self::new(self.rows, rhs.cols);
        for i in 0..result.rows {
            for j in 0..result.cols {
                for k in 0..self.cols {
                    result[(i, j)] = result[(i, j)] + self[(i, k)] * rhs[(k, j)];
                }
            }
        }
        Ok(result)
    }

    // 点乘
    pub fn dot(&self, rhs: &Self) -> Result<Self, Error>
    where
        T: Mul<T, Output = T> + Copy,
    {
        if self.rows != rhs.rows && self.rows != 1 {
            return Err(Error::Uncomputable);
        }
        if self.cols != rhs.cols && self.cols != 1 {
            return Err(Error::Uncomputable);
        }
        let mut result = Self::new(rhs.rows, rhs.cols);
        if self.rows == 1 {
            for r in 0..result.rows {
                for c in 0..result.cols {
                    result[(r, c)] = self[(0, c)] * rhs[(r, c)];
                }
            }
            return Ok(result);
        } else if self.cols == 1 {
            for r in 0..result.rows {
                for c in 0..result.cols {
                    result[(r, c)] = self[(r, 0)] * rhs[(r, c)];
                }
            }
            return Ok(result);
        }
        for i in 0..result.data.len() {
            result.data[i] = self.data[i] * rhs.data[i];
        }
        Ok(result)
    }

    pub fn row(&self) -> usize {
        self.rows
    }
    pub fn col(&self) -> usize {
        self.cols
    }

    pub fn set_row(&mut self, row: usize) {
        self.rows = row;
        self.data.resize(self.rows * self.cols, T::default());
    }
    pub fn set_col(&mut self, col: usize) {
        self.cols = col;
        self.data.resize(self.rows * self.cols, T::default());
    }

    // 减去一个矩阵，获得一个新的矩阵
    pub fn sub(&self, rhs: &Self) -> Result<Self, Error>
    where
        T: Sub<Output = T> + Default + Copy,
    {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(Error::Uncomputable);
        }
        let mut result = Self::new(self.rows, self.cols);
        for i in 0..result.data.len() {
            result.data[i] = self.data[i] - rhs.data[i];
        }
        Ok(result)
    }

    pub fn sub_to(&mut self, rhs: &Self) -> Result<(), Error>
    where
        T: Sub<Output = T> + Default + Copy,
    {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(Error::Uncomputable);
        }
        for i in 0..self.data.len() {
            self.data[i] = self.data[i] - rhs.data[i];
        }
        Ok(())
    }

    pub fn mul_num(&self, num: T) -> Self
    where T: Mul<T, Output = T> + Copy,
    {
        let result = self.data.iter().map(|x| *x * num).collect();
        Self::with_data(self.rows, self.cols, result)
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[row * self.cols + col]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.data[row * self.cols + col]
    }
}

// 求自然数e的幂
pub trait Exp {
    fn mexp(&self) -> Self;
}

impl Exp for f32 {
    fn mexp(&self) -> Self {
        self.exp()
    }
}

impl Exp for f64 {
    fn mexp(&self) -> Self {
        self.exp()
    }
}

impl<T> Add<&Self> for Matrix<T>
where
    T: Add<Output = T> + Copy + Default + Clone,
{
    type Output = Result<Self, Error>;

    fn add(mut self, rhs: &Self) -> Self::Output {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(Error::Uncomputable);
        }

        for i in 0..self.data.len() {
            self.data[i] = self.data[i] + rhs.data[i];
        }
        Ok(self)
    }
}

impl<T> Add<Matrix<T>> for f32
where
    T: Add<f32, Output = T> + Copy + Default + Clone + AddAssign<f32>,
{
    type Output = Matrix<T>;

    fn add(self, mut rhs: Matrix<T>) -> Self::Output {
        for i in 0..rhs.data.len() {
            rhs.data[i] += self;
        }
        rhs
    }
}

impl<T> Sub<&Self> for Matrix<T>
where
    T: Sub<Output = T> + Copy + Default + Clone,
{
    type Output = Result<Self, Error>;

    fn sub(mut self, rhs: &Self) -> Self::Output {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(Error::Uncomputable);
        }

        for i in 0..self.data.len() {
            self.data[i] = self.data[i] - rhs.data[i];
        }
        Ok(self)
    }
}

impl Sub<&Matrix> for f32 {
    type Output = Matrix;

    fn sub(self, rhs: &Matrix) -> Self::Output {
        let mut result = rhs.clone();
        for i in 0..rhs.data.len() {
            result.data[i] = self - rhs.data[i];
        }
        result
    }
}

impl Div<Matrix> for f32 {
    type Output = Matrix;

    fn div(self, mut rhs: Matrix) -> Self::Output {
        for i in 0..rhs.data.len() {
            rhs.data[i] = self / rhs.data[i];
        }
        rhs
    }
}

impl<T, R> Mul<R> for Matrix<T>
where
    T: Clone + Mul<R, Output = T> + Copy + Default,
    R: Copy
{
    type Output = Self;

    fn mul(self, rhs: R) -> Self::Output {
        let mut result = self.clone();
        for i in 0..result.data.len() {
            result.data[i] = result.data[i] * rhs;
        }
        result
    }
}

impl<T> AsRef<Matrix<T>> for Matrix<T> {
    fn as_ref(&self) -> &Matrix<T> {
        self
    }
}

impl<T> AsMut<Matrix<T>> for Matrix<T> {
    fn as_mut(&mut self) -> &mut Matrix<T> {
        self
    }
}

impl<T> Display for Matrix<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        for i in 0..self.rows {
            let mut s1 = String::new();
            for j in 0..self.cols - 1 {
                let f = format!("{:8.5}", &self.data[i * self.cols + j]);
                s1.push_str(&f);
                s1.push_str(" ");
            }
            let f = format!("{:8.5}", &self.data[i * self.cols + self.cols - 1]);
            s1.push_str(&f);

            if self.rows == 1 {
                s.push_str("[");
                s.push_str(&s1);
                s.push_str("]\n");
            } else if i == 0 {
                s.push_str("╭");
                s.push_str(&s1);
                s.push_str("╮\n");
            } else if i == self.rows - 1 {
                s.push_str("╰");
                s.push_str(&s1);
                s.push_str("╯\n");
            } else {
                s.push_str("│");
                s.push_str(&s1);
                s.push_str("│\n");
            }
        }
        write!(f, "{}", s)
    }
}

mod test {
    use super::*;

    #[test]
    fn test_matrix() {
        let m = Matrix::with_data(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(m.data, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    // 测试矩阵相加
    #[test]
    fn test_matrix_add() {
        let m1 = Matrix::with_data(2, 2, vec![1, 2, 3, 4]);
        let m2 = Matrix::with_data(2, 2, vec![1, 2, 3, 4]);
        let m3 = m1 + &m2;
        assert_eq!(m3.map_or(Matrix::default(), |i| i).data, vec![2, 4, 6, 8]);
        let m4 = Matrix::with_data(2, 3, vec![1, 2, 3, 4, 5, 6])
            + &Matrix::with_data(2, 2, vec![1, 2, 3, 4]);
        assert!(m4.map_or(true, |_| false));
    }

    // 测试矩阵切片
    #[test]
    fn test_matrix_index() {
        let m = Matrix::with_data(2, 2, vec![1, 2, 3, 4]);
        assert_eq!(m[(0, 0)], 1);
        assert_eq!(m[(0, 1)], 2);
        assert_eq!(m[(1, 0)], 3);
        assert_eq!(m[(1, 1)], 4);
    }

    // 测试叉乘
    #[test]
    fn test_matrix_cross() {
        let m1 = Matrix::with_data(2, 3, vec![1, 2, 3, 4, 5, 6]);
        let m2 = Matrix::with_data(3, 2, vec![1, 2, 3, 4, 5, 6]);
        let m3 = m1.cross(&m2).unwrap();
        assert_eq!(m3.data, vec![22, 28, 49, 64]);
    }

    // 测试点乘
    #[test]
    fn test_matrix_dot() {
        let m1 = Matrix::with_data(2, 2, vec![1, 2, 3, 4]);
        let m2 = Matrix::with_data(2, 2, vec![1, 2, 3, 4]);
        let m3 = m1.dot(&m2).unwrap();
        assert_eq!(m3.data, vec![1, 4, 9, 16]);
        let m1 = Matrix::with_data(1, 3, vec![1, 2, 3]);
        let m2 = Matrix::with_data(2, 3, vec![1, 2, 3, 4, 5, 6]);
        let m3 = m1.dot(&m2).unwrap();
        assert_eq!(m3.data, vec![1, 4, 9, 4, 10, 18]);
        let m1 = Matrix::with_data(2, 1, vec![1, 2]);
        let m2 = Matrix::with_data(2, 3, vec![1, 2, 3, 4, 5, 6]);
        let m3 = m1.dot(&m2).unwrap();
        assert_eq!(m3.data, vec![1, 2, 3, 8, 10, 12]);
        let m1 = Matrix::with_data(2, 2, vec![1, 2, 3, 4]);
        let m2 = Matrix::with_data(2, 3, vec![1, 2, 3, 4, 5, 6]);
        assert!(m1.dot(&m2).map_or(true, |_| false));
        let m1 = Matrix::with_data(1, 2, vec![1, 2]);
        let m2 = Matrix::with_data(2, 3, vec![1, 2, 3, 4, 5, 6]);
        assert!(m1.dot(&m2).map_or(true, |_| false));
        let m1 = Matrix::with_data(3, 1, vec![1, 2, 3]);
        let m2 = Matrix::with_data(2, 3, vec![1, 2, 3, 4, 5, 6]);
        assert!(m1.dot(&m2).map_or(true, |_| false));
    }

    // 测试矩阵转置
    #[test]
    fn test_matrix_transpose() {
        let m = Matrix::with_data(2, 3, vec![1, 2, 3, 4, 5, 6]);
        let m1 = m.transpose().unwrap();
        assert_eq!(m1.data, vec![1, 4, 2, 5, 3, 6]);
    }
}

#[derive(Debug)]
pub enum Error {
    // 不可计算
    Uncomputable,
}
