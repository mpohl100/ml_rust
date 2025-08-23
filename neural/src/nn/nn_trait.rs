use crate::nn::shape::NeuralNetworkShape;
use crate::{nn::directory::Directory, utilities::util::WrappedUtils};
use std::sync::{Arc, Mutex};
use utils::safer::safe_lock;

pub trait NeuralNetwork: std::fmt::Debug {
    fn predict(
        &mut self,
        input: Vec<f64>,
    ) -> Vec<f64>;
    fn shape(&self) -> NeuralNetworkShape;
    /// Saves the neural network to the specified user model directory.
    ///
    /// # Errors
    /// Returns an error if saving the model fails due to IO issues or serialization errors.
    fn save(
        &mut self,
        user_model_directory: String,
    ) -> Result<(), Box<dyn std::error::Error>>;
    fn get_model_directory(&self) -> Directory;
    fn allocate(&mut self);
    fn deallocate(&mut self);
    fn set_internal(&mut self);
    fn duplicate(&self) -> WrappedNeuralNetwork;
    fn get_utils(&self) -> WrappedUtils;
}

#[derive(Debug, Clone)]
pub struct WrappedNeuralNetwork {
    nn: Arc<Mutex<Box<dyn NeuralNetwork + Send>>>,
}

impl WrappedNeuralNetwork {
    #[must_use]
    pub fn new(nn: Box<dyn NeuralNetwork + Send>) -> Self {
        Self { nn: Arc::new(Mutex::new(nn)) }
    }

    pub fn predict(
        &mut self,
        input: Vec<f64>,
    ) -> Vec<f64> {
        safe_lock(&self.nn).predict(input)
    }

    #[must_use]
    pub fn shape(&self) -> NeuralNetworkShape {
        safe_lock(&self.nn).shape()
    }

    /// Saves the neural network to the specified user model directory.
    ///
    /// # Errors
    /// Returns an error if saving the model fails due to IO issues or serialization errors.
    pub fn save(
        &mut self,
        user_model_directory: String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        safe_lock(&self.nn).save(user_model_directory)
    }

    pub fn allocate(&self) {
        safe_lock(&self.nn).allocate();
    }

    pub fn deallocate(&self) {
        safe_lock(&self.nn).deallocate();
    }

    pub fn set_internal(&mut self) {
        safe_lock(&self.nn).set_internal();
    }

    #[must_use]
    pub fn duplicate(&self) -> Self {
        safe_lock(&self.nn).duplicate()
    }

    #[must_use]
    pub fn get_utils(&self) -> WrappedUtils {
        safe_lock(&self.nn).get_utils()
    }
}

pub trait TrainableNeuralNetwork: NeuralNetwork {
    /// Trains the neural network using the given inputs, targets, learning rate, and number of epochs.
    /// Includes validation using a split of the data.
    #[allow(clippy::too_many_arguments)]
    fn train(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        learning_rate: f64,
        epochs: usize,
        tolerance: f64,
        use_adam: bool,
        validation_split: f64,
        sample_match_percentage: f64,
    ) -> f64;

    /// Trains the neural network doing batch back propagation.
    fn train_batch(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        learning_rate: f64,
        epochs: usize,
        tolerance: f64,
        batch_size: usize,
    );

    /// Returns the input size of the first layer in the network.
    fn input_size(&self) -> usize;

    /// Returns the output size of the last layer in the network.
    fn output_size(&self) -> usize;

    fn duplicate_trainable(&self) -> WrappedTrainableNeuralNetwork;
}

#[derive(Debug, Clone)]
pub struct WrappedTrainableNeuralNetwork {
    nn: Arc<Mutex<Box<dyn TrainableNeuralNetwork + Send>>>,
}

impl WrappedTrainableNeuralNetwork {
    #[must_use]
    pub fn new(nn: Box<dyn TrainableNeuralNetwork + Send>) -> Self {
        Self { nn: Arc::new(Mutex::new(nn)) }
    }

    pub fn predict(
        &mut self,
        input: Vec<f64>,
    ) -> Vec<f64> {
        safe_lock(&self.nn).predict(input)
    }

    #[must_use]
    pub fn shape(&self) -> NeuralNetworkShape {
        safe_lock(&self.nn).shape()
    }

    /// Saves the trainable neural network to the specified user model directory.
    ///
    /// # Errors
    /// Returns an error if saving the model fails due to IO issues or serialization errors.
    pub fn save(
        &mut self,
        user_model_directory: String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        safe_lock(&self.nn).save(user_model_directory)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn train(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        learning_rate: f64,
        epochs: usize,
        tolerance: f64,
        use_adam: bool,
        validation_split: f64,
        sample_match_percentage: f64,
    ) -> f64 {
        safe_lock(&self.nn).train(
            inputs,
            targets,
            learning_rate,
            epochs,
            tolerance,
            use_adam,
            validation_split,
            sample_match_percentage,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn train_batch(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        learning_rate: f64,
        epochs: usize,
        tolerance: f64,
        batch_size: usize,
    ) {
        safe_lock(&self.nn).train_batch(
            inputs,
            targets,
            learning_rate,
            epochs,
            tolerance,
            batch_size,
        );
    }

    #[must_use]
    pub fn input_size(&self) -> usize {
        safe_lock(&self.nn).input_size()
    }

    #[must_use]
    pub fn output_size(&self) -> usize {
        safe_lock(&self.nn).output_size()
    }

    #[must_use]
    pub fn get_model_directory(&self) -> Directory {
        safe_lock(&self.nn).get_model_directory()
    }

    pub fn allocate(&self) {
        safe_lock(&self.nn).allocate();
    }

    pub fn deallocate(&self) {
        safe_lock(&self.nn).deallocate();
    }

    pub fn set_internal(&mut self) {
        safe_lock(&self.nn).set_internal();
    }

    #[must_use]
    pub fn duplicate_trainable(&self) -> Self {
        safe_lock(&self.nn).duplicate_trainable()
    }

    #[must_use]
    pub fn get_utils(&self) -> WrappedUtils {
        safe_lock(&self.nn).get_utils()
    }
}
