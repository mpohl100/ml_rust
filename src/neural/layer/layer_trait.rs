use crate::neural::mat::matrix::Matrix;

use dyn_clone::DynClone;
use std::error::Error;
// A trait representing a layer in a neural network.
/// Provides methods for the forward pass, backward pass, weight updates, and layer size information.
pub trait Layer: std::fmt::Debug + DynClone {
    /// Performs the forward pass of the layer, computing the output based on the input vector.
    ///
    /// # Arguments
    ///
    /// * `input` - A reference to a vector of `f64` values representing the input data.
    ///
    /// # Returns
    ///
    /// * A vector of `f64` values representing the output of the layer.
    fn forward(&mut self, input: &[f64]) -> Vec<f64>;

    /// Performs the forward pass of the layer for inputs doing batch caching.
    fn forward_batch(&mut self, input: &[f64]) -> Vec<f64>;

    /// Returns the input size of the layer.
    ///
    /// # Returns
    ///
    /// * A `usize` value representing the number of input neurons.
    fn input_size(&self) -> usize;

    /// Returns the output size of the layer.
    ///
    /// # Returns
    ///
    /// * A `usize` value representing the number of output neurons.
    fn output_size(&self) -> usize;

    /// Saves the layer to a file at the specified path.
    fn save(&self, path: &str) -> Result<(), Box<dyn Error>>;

    /// Reads the layer from a file at the specified path.
    fn read(&mut self, path: &str) -> Result<(), Box<dyn Error>>;

    /// Returns the weights of the layer.
    fn get_weights(&self) -> Matrix<f64>;

    /// Returns the biases of the layer.
    fn get_biases(&self) -> Vec<f64>;
}

dyn_clone::clone_trait_object!(Layer);

pub trait TrainableLayer: Layer {
    /// Performs the backward pass of the layer, computing the gradient based on the output gradient.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - A reference to a vector of `f64` values representing the gradient of the loss
    ///   with respect to the output of this layer.
    ///
    /// # Returns
    ///
    /// * A vector of `f64` values representing the gradient of the loss with respect to the input.
    fn backward(&mut self, grad_output: &[f64]) -> Vec<f64>;

    /// Performs the backward pass of the layer for inputs doing batch caching.
    fn backward_batch(&mut self, grad_output: &[f64]) -> Vec<f64>;

    /// Updates the weights of the layer based on the specified learning rate.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - A `f64` value representing the learning rate for weight updates.
    fn update_weights(&mut self, learning_rate: f64);

    /// Resizes the layer to the input dimensions.
    fn resize(&mut self, input_size: usize, output_size: usize);

    /// Assigns the weight of the input other layer
    fn assign_weights(&mut self, other: &dyn TrainableLayer);

    /// Adjusts the weights according to the Adam optimizer.
    fn adjust_adam(&mut self, t: usize, learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64);
}

dyn_clone::clone_trait_object!(TrainableLayer);
