mod matrix;
mod neural_network;
mod neural_network_sync;

pub use matrix::Matrix;
pub use neural_network::{NeuralNetwork, Activation, Error};
pub use neural_network_sync::NeuralNetworkSync;
