use crate::activation::{
    activate::ActivationTrait, relu::ReLU, sigmoid::Sigmoid, softmax::Softmax, tanh::Tanh,
};
use crate::layer::dense_layer::DenseLayer;
use crate::layer::dense_layer::TrainableDenseLayer;
use crate::layer::layer_trait::WrappedLayer;
use crate::layer::layer_trait::WrappedTrainableLayer;
use crate::nn::nn_trait::{NeuralNetwork, TrainableNeuralNetwork};
use crate::nn::shape::{ActivationType, LayerType, NeuralNetworkShape};
use crate::utilities::util::WrappedUtils;
use alloc::allocatable::WrappedAllocatableTrait;

use indicatif::ProgressDrawTarget;
use indicatif::{ProgressBar, ProgressStyle};
use num_traits::NumCast;
use rand::prelude::SliceRandom;

use std::path::Path;

use std::boxed::Box;

use super::directory::Directory;
use super::nn_factory::{copy_dir_recursive, get_first_free_model_directory};
use super::nn_trait::{WrappedNeuralNetwork, WrappedTrainableNeuralNetwork};

/// A neural network.
#[derive(Debug)]
pub struct ClassicNeuralNetwork {
    layers: Vec<WrappedLayer>,
    activations: Vec<Box<dyn ActivationTrait + Send>>,
    shape: NeuralNetworkShape,
    model_directory: Directory,
    past_internal_directory: Vec<String>,
    utils: WrappedUtils,
}

impl ClassicNeuralNetwork {
    /// Creates a new `NeuralNetwork` from the given shape.
    ///
    /// # Panics
    ///
    /// This function will panic if the activation type is Softmax and its temperature is not set.
    #[must_use]
    pub fn new(
        shape: NeuralNetworkShape,
        internal_model_directory: String,
        utils: WrappedUtils,
    ) -> Self {
        let shape_clone = shape.clone();
        let mut network = Self {
            layers: Vec::new(),
            activations: Vec::new(),
            shape,
            model_directory: Directory::Internal(internal_model_directory),
            past_internal_directory: Vec::new(),
            utils,
        };

        // Initialize layers and activations based on the provided shape.
        for (i, layer_shape) in shape_clone.layers.iter().enumerate() {
            // Here you would instantiate the appropriate Layer and Activation objects.
            let dense_layer = DenseLayer::new(
                layer_shape.input_size(),
                layer_shape.output_size(),
                network.model_directory.clone(),
                i,
            );
            let layer = WrappedLayer::new(Box::new(dense_layer));
            let activation = match layer_shape.activation.activation_type() {
                ActivationType::ReLU => Box::new(ReLU::new()) as Box<dyn ActivationTrait + Send>,
                ActivationType::Sigmoid => Box::new(Sigmoid) as Box<dyn ActivationTrait + Send>,
                ActivationType::Tanh => Box::new(Tanh) as Box<dyn ActivationTrait + Send>,
                ActivationType::Softmax => {
                    Box::new(Softmax::new(layer_shape.activation.temperature().unwrap()))
                        as Box<dyn ActivationTrait + Send>
                },
            };

            network.add_activation_and_layer(activation, layer);
        }

        network
    }

    /// Creates a new `NeuralNetwork` from the given model directory.
    ///
    /// # Panics
    ///
    /// This function will panic if the model directory is valid but the shape cannot be unwrapped.
    #[allow(clippy::question_mark)]
    #[must_use]
    pub fn from_disk(
        model_directory: String,
        utils: WrappedUtils,
    ) -> Option<Self> {
        let shape = NeuralNetworkShape::from_disk(&model_directory);
        if shape.is_none() {
            return None;
        }
        let sh = shape.unwrap();
        let mut network = Self {
            layers: Vec::new(),
            activations: Vec::new(),
            shape: sh.clone(),
            model_directory: Directory::User(model_directory),
            past_internal_directory: Vec::new(),
            utils,
        };

        for i in 0..sh.layers.len() {
            let layer = match &sh.layers[i].layer_type() {
                LayerType::Dense { input_size, output_size } => {
                    let layer = DenseLayer::new(
                        *input_size,
                        *output_size,
                        network.model_directory.clone(),
                        i,
                    );
                    WrappedLayer::new(Box::new(layer))
                },
            };
            let activation = match sh.layers[i].activation.activation_type() {
                ActivationType::ReLU => Box::new(ReLU::new()) as Box<dyn ActivationTrait + Send>,
                ActivationType::Sigmoid => Box::new(Sigmoid) as Box<dyn ActivationTrait + Send>,
                ActivationType::Tanh => Box::new(Tanh) as Box<dyn ActivationTrait + Send>,
                ActivationType::Softmax => {
                    Box::new(Softmax::new(sh.layers[i].activation.temperature().unwrap()))
                        as Box<dyn ActivationTrait + Send>
                },
            };

            network.add_activation_and_layer(activation, layer);
        }

        Some(network)
    }

    /// Saves the neural network to disk with the internal logic.
    fn save_internal(
        &self,
        model_directory: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // remove the directory if it exists
        let backup_directory = format!("{model_directory}_backup");
        if std::fs::metadata(model_directory).is_ok() {
            // copy the directory to a backup
            copy_dir_recursive(Path::new(&model_directory), Path::new(&backup_directory))?;
            std::fs::create_dir_all(model_directory)?;
        } else {
            // create directory if it doesn't exist
            std::fs::create_dir_all(model_directory)?;
        }

        let shape = self.shape();
        shape.to_yaml(model_directory);
        self.save_layers(model_directory)?;

        // if backup directory exists, remove it
        if std::fs::metadata(&backup_directory).is_ok() {
            std::fs::remove_dir_all(&backup_directory)?;
        }

        Ok(())
    }

    /// Retrieves the first free model directory.
    fn get_first_free_model_directory(&self) -> String {
        get_first_free_model_directory(&self.model_directory)
    }

    /// Adds an activation and a layer to the neural network.
    fn add_activation_and_layer(
        &mut self,
        activation: Box<dyn ActivationTrait + Send>,
        layer: WrappedLayer,
    ) {
        self.activations.push(activation);
        self.layers.push(layer);
    }

    /// Saves the layers of the neural network to disk.
    fn save_layers(
        &self,
        model_directory: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // make a layers subdirectory
        std::fs::create_dir_all(format!("{model_directory}/layers"))?;
        for (i, layer) in self.layers.iter().enumerate() {
            layer.save(format!("{model_directory}/layers/layer_{i}.txt"))?;
        }
        Ok(())
    }

    /// Saves the layout of the neural network to disk.
    fn save_layout(&self) {
        let shape = self.shape();
        shape.to_yaml(&self.model_directory.path());
    }

    /// Performs a forward pass through the network with the given input.
    fn forward(
        &mut self,
        input: &[f64],
    ) -> Vec<f64> {
        let mut output = input.to_vec();
        for (layer, activation) in self.layers.iter_mut().zip(&mut self.activations) {
            layer.mark_for_use();
            self.utils.allocate(layer);
            output = layer.forward(&output, self.utils.clone());
            layer.free_from_use();
            // this operation should not change the dimension of output
            output = activation.forward(&output);
        }
        output
    }
}

impl NeuralNetwork for ClassicNeuralNetwork {
    /// Makes a prediction based on a single input by performing a forward pass.
    fn predict(
        &mut self,
        input: Vec<f64>,
    ) -> Vec<f64> {
        self.forward(input.as_slice())
    }

    fn shape(&self) -> NeuralNetworkShape {
        self.shape.clone()
    }

    fn save(
        &mut self,
        user_model_directory: String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Check if we're in test mode - if so, force save to Internal directory
        if self.utils.is_test_mode() {
            let workspace = self.utils.get_workspace();
            let internal_path = if workspace.is_empty() {
                user_model_directory
            } else {
                format!("{workspace}/{user_model_directory}")
            };
            // Don't add current directory to past_internal when in test mode to avoid cleanup
            // since we want to keep the saved model for testing
            self.model_directory = Directory::Internal(internal_path);
            let model_directory = self.model_directory.path();
            return self.save_internal(&model_directory);
        }

        if let Directory::Internal(_) = self.model_directory {
            self.past_internal_directory.push(self.model_directory.path());
        }
        self.model_directory = Directory::User(user_model_directory);
        let model_directory = self.model_directory.path();
        self.save_internal(&model_directory)
    }

    fn get_model_directory(&self) -> Directory {
        self.model_directory.clone()
    }

    /// Allocates the layers of the neural network.
    fn allocate(&mut self) {
        for layer in &self.layers {
            self.utils.allocate(layer);
        }
    }

    /// Deallocates the layers of the neural network.
    fn deallocate(&mut self) {
        for layer in &self.layers {
            self.utils.deallocate(layer);
        }
    }

    fn set_internal(&mut self) {
        // set the model directory to internal
        self.model_directory = Directory::Internal(self.model_directory.path());
    }

    fn duplicate(&self) -> WrappedNeuralNetwork {
        // create a sibling directory with the postfix _clone appendended to model_direcotory path
        let model_directory = self.get_first_free_model_directory();
        // Save the model to the new directory
        self.save_internal(&model_directory).unwrap();
        // Clone the neural network by cloning its layers and activations
        let mut new_layers = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            new_layers.push(layer.duplicate(model_directory.clone(), i));
            layer.cleanup();
        }
        WrappedNeuralNetwork::new(Box::new(Self {
            layers: new_layers,
            activations: self.activations.clone(),
            shape: self.shape.clone(),
            model_directory: Directory::Internal(model_directory),
            past_internal_directory: Vec::new(),
            utils: self.utils.clone(),
        }))
    }

    fn get_utils(&self) -> WrappedUtils {
        self.utils.clone()
    }
}

impl Drop for ClassicNeuralNetwork {
    fn drop(&mut self) {
        // Save the model to ensure that everything is on disk if it is a user_model_directory
        // ensure that the model_directory exists
        if let Directory::User(_) = &self.model_directory {
            if std::fs::metadata(self.model_directory.path()).is_err() {
                std::fs::create_dir_all(self.model_directory.path()).unwrap();
            }
            self.save_layout();
            self.deallocate();
        }
        // Interne Verzeichnisse immer entfernen, unabhängig vom Testmodus
        if let Directory::Internal(dir) = &self.model_directory {
            if std::fs::metadata(dir).is_ok() {
                std::fs::remove_dir_all(dir).unwrap();
            }
        }
        for dir in &self.past_internal_directory {
            if dir != &self.model_directory.path() && std::fs::metadata(dir).is_ok() {
                std::fs::remove_dir_all(dir).unwrap();
            }
        }
    }
}

/// A neural network.
#[derive(Debug)]
pub struct TrainableClassicNeuralNetwork {
    layers: Vec<WrappedTrainableLayer>,
    activations: Vec<Box<dyn ActivationTrait + Send>>,
    shape: NeuralNetworkShape,
    model_directory: Directory,
    past_internal_model_directory: Vec<String>,
    utils: WrappedUtils,
}

impl TrainableClassicNeuralNetwork {
    /// Creates a new `NeuralNetwork` from the given shape.
    ///
    /// # Panics
    ///
    /// This function will panic if the activation type is Softmax and its temperature is not set.
    #[must_use]
    pub fn new(
        shape: NeuralNetworkShape,
        model_directory: &Directory,
        utils: WrappedUtils,
    ) -> Self {
        let shape_clone = shape.clone();
        let mut network = Self {
            layers: Vec::new(),
            activations: Vec::new(),
            shape,
            model_directory: Directory::Internal(get_first_free_model_directory(model_directory)),
            past_internal_model_directory: Vec::new(),
            utils,
        };

        // Initialize layers and activations based on the provided shape.
        for (i, layer_shape) in shape_clone.layers.iter().enumerate() {
            // Here you would instantiate the appropriate Layer and Activation objects.
            let layer = WrappedTrainableLayer::new(Box::new(TrainableDenseLayer::new(
                layer_shape.input_size(),
                layer_shape.output_size(),
                network.model_directory.clone(),
                i,
            )));
            let activation = match layer_shape.activation.activation_type() {
                ActivationType::ReLU => Box::new(ReLU::new()) as Box<dyn ActivationTrait + Send>,
                ActivationType::Sigmoid => Box::new(Sigmoid) as Box<dyn ActivationTrait + Send>,
                ActivationType::Tanh => Box::new(Tanh) as Box<dyn ActivationTrait + Send>,
                ActivationType::Softmax => {
                    Box::new(Softmax::new(layer_shape.activation.temperature().unwrap()))
                        as Box<dyn ActivationTrait + Send>
                },
            };

            network.add_activation_and_trainable_layer(activation, layer);
        }

        network.save_layout();

        network
    }

    #[must_use]
    pub fn new_dir(
        model_directory: &Directory,
        utils: WrappedUtils,
    ) -> Self {
        let network = Self {
            layers: Vec::new(),
            activations: Vec::new(),
            shape: NeuralNetworkShape::default(),
            model_directory: Directory::Internal(get_first_free_model_directory(model_directory)),
            past_internal_model_directory: Vec::new(),
            utils,
        };

        network.save_layout();

        network
    }

    fn save_internal(
        &self,
        model_directory: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // remove the directory if it exists
        let backup_directory = format!("{model_directory}_backup");
        if std::fs::metadata(model_directory).is_ok() {
            // copy the directory to a backup not move
            copy_dir_recursive(Path::new(&model_directory), Path::new(&backup_directory))?;
            std::fs::create_dir_all(model_directory)?;
        } else {
            // create directory if it doesn't exist
            std::fs::create_dir_all(model_directory)?;
        }

        let shape = self.shape();
        shape.to_yaml(model_directory);
        self.save_layers(model_directory)?;

        // if backup directory exists, remove it
        if std::fs::metadata(&backup_directory).is_ok() {
            std::fs::remove_dir_all(&backup_directory)?;
        }

        Ok(())
    }

    /// Adjusts the weights of the neural network using the Adam optimizer.
    fn adjust_adam(
        &mut self,
        t: usize,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) {
        for layer in &mut self.layers {
            layer.adjust_adam(t, learning_rate, beta1, beta2, epsilon, self.utils.clone());
        }
    }

    /// Saves the layers of the neural network to disk.
    fn save_layers(
        &self,
        model_directory: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // make a layers subdirectory
        if std::fs::metadata(format!("{model_directory}/layers")).is_err() {
            std::fs::create_dir_all(format!("{model_directory}/layers"))?;
        }
        for (i, layer) in self.layers.iter().enumerate() {
            layer.save_weights(format!("{model_directory}/layers/layer_{i}.txt"))?;
        }
        Ok(())
    }

    /// Saves the layout of the neural network to disk.
    fn save_layout(&self) {
        let shape = self.shape();
        // ensure the directory exists
        if std::fs::metadata(self.model_directory.path()).is_err() {
            std::fs::create_dir_all(self.model_directory.path()).unwrap();
        }
        shape.to_yaml(&self.model_directory.path());
    }

    /// Adds an activation and a layer to the neural network.
    fn add_activation_and_trainable_layer(
        &mut self,
        activation: Box<dyn ActivationTrait + Send>,
        layer: WrappedTrainableLayer,
    ) {
        self.activations.push(activation);
        self.layers.push(layer);
    }

    /// Performs a forward pass through the network with the given input.
    fn forward(
        &mut self,
        input: &[f64],
    ) -> Vec<f64> {
        let mut output = input.to_vec();
        for (layer, activation) in self.layers.iter_mut().zip(&mut self.activations) {
            layer.mark_for_use();
            self.utils.allocate_trainable(layer);
            output = layer.forward(&output, self.utils.clone());
            layer.free_from_use();
            // this operation should not change the dimension of output
            output = activation.forward(&output);
        }
        output
    }

    /// Performs a forward pass through the network with the given input doing batch caching.
    fn forward_batch(
        &mut self,
        input: &[f64],
    ) -> Vec<f64> {
        let mut output = input.to_vec();
        for (layer, activation) in self.layers.iter_mut().zip(&mut self.activations) {
            output = layer.forward_batch(&output);
            output = activation.forward(&output);
        }
        output
    }

    /// Performs a backward pass through the network with the given output gradient.
    fn backward(
        &mut self,
        grad_output: Vec<f64>,
    ) {
        let mut grad = grad_output;
        for (layer, activation) in
            self.layers.iter_mut().rev().zip(self.activations.iter_mut().rev())
        {
            grad = activation.backward(&grad);
            layer.mark_for_use();
            self.utils.allocate_trainable(layer);
            grad = layer.backward(&grad, self.utils.clone());
            layer.free_from_use();
        }
    }

    /// Performs a backward pass through the network with the given output gradient doing batch caching.
    fn backward_batch(
        &mut self,
        grad_output: Vec<f64>,
    ) {
        let mut grad = grad_output;
        for (layer, activation) in
            self.layers.iter_mut().rev().zip(self.activations.iter_mut().rev())
        {
            grad = activation.backward(&grad);
            grad = layer.backward_batch(&grad);
        }
    }

    /// Creates a new `NeuralNetwork` from the given model directory.
    ///
    /// # Panics
    ///
    /// This function will panic if the model directory is invalid or if the YAML
    /// deserialization fails.
    #[allow(clippy::question_mark)]
    #[must_use]
    pub fn from_disk(
        model_directory: String,
        utils: WrappedUtils,
    ) -> Option<Self> {
        let shape = NeuralNetworkShape::from_disk(&model_directory);
        if shape.is_none() {
            return None;
        }
        let sh = shape.unwrap();
        let mut network = Self {
            layers: Vec::new(),
            activations: Vec::new(),
            shape: sh.clone(),
            model_directory: Directory::User(model_directory),
            past_internal_model_directory: Vec::new(),
            utils,
        };

        for i in 0..sh.layers.len() {
            let layer = match &sh.layers[i].layer_type() {
                LayerType::Dense { input_size, output_size } => {
                    let layer = TrainableDenseLayer::new(
                        *input_size,
                        *output_size,
                        network.model_directory.clone(),
                        i,
                    );
                    WrappedTrainableLayer::new(Box::new(layer))
                },
            };
            let activation = match sh.layers[i].activation.activation_type() {
                ActivationType::ReLU => Box::new(ReLU::new()) as Box<dyn ActivationTrait + Send>,
                ActivationType::Sigmoid => Box::new(Sigmoid) as Box<dyn ActivationTrait + Send>,
                ActivationType::Tanh => Box::new(Tanh) as Box<dyn ActivationTrait + Send>,
                ActivationType::Softmax => {
                    Box::new(Softmax::new(sh.layers[i].activation.temperature().unwrap()))
                        as Box<dyn ActivationTrait + Send>
                },
            };

            network.add_activation_and_trainable_layer(activation, layer);
        }

        Some(network)
    }

    /// Retrieves the first free model directory.
    fn get_first_free_model_directory(&self) -> String {
        get_first_free_model_directory(&self.model_directory)
    }

    fn transform(
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let repeat_n_times = 1000 / inputs.len();
        // if factor is greater than 1, then repeat the inputs and targets
        let (transformed_inputs, transformed_targets) = if repeat_n_times > 1 {
            let mut transformed_inputs = Vec::new();
            let mut transformed_targets = Vec::new();
            for (i, input) in inputs.iter().cycle().enumerate() {
                if i >= repeat_n_times * inputs.len() {
                    break;
                }
                // repeat the input n times
                transformed_inputs.push(input.clone());
            }
            for (i, target) in targets.iter().cycle().enumerate() {
                if i >= repeat_n_times * targets.len() {
                    break;
                }
                // repeat the target n times
                transformed_targets.push(target.clone());
            }
            (transformed_inputs, transformed_targets)
        } else {
            // if factor is 1, then just return the inputs and targets
            let transformed_inputs = inputs.to_vec();
            let transformed_targets = targets.to_vec();
            (transformed_inputs, transformed_targets)
        };

        // zip the transformed inputs and targets
        let mut zipped = transformed_inputs
            .iter()
            .zip(transformed_targets.iter())
            .map(|(input, target)| {
                let new_input = input.clone();
                let new_target = target.clone();
                (new_input, new_target)
            })
            .collect::<Vec<_>>();

        // shuffle the zipped inputs and targets
        let mut thread_rng = rand::thread_rng();
        zipped.shuffle(&mut thread_rng);

        // unzip the zipped inputs and targets
        let (transformed_inputs, transformed_targets): (Vec<_>, Vec<_>) =
            zipped.into_iter().unzip();
        (transformed_inputs, transformed_targets)
    }
}

impl NeuralNetwork for TrainableClassicNeuralNetwork {
    /// Makes a prediction based on a single input by performing a forward pass.
    fn predict(
        &mut self,
        input: Vec<f64>,
    ) -> Vec<f64> {
        self.forward(input.as_slice())
    }

    fn shape(&self) -> NeuralNetworkShape {
        self.shape.clone()
    }

    fn save(
        &mut self,
        user_model_directory: String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Check if we're in test mode - if so, force save to Internal directory
        if self.utils.is_test_mode() {
            let workspace = self.utils.get_workspace();
            let internal_path = if workspace.is_empty() {
                user_model_directory
            } else {
                format!("{workspace}/{user_model_directory}")
            };
            // Don't add current directory to past_internal when in test mode to avoid cleanup
            // since we want to keep the saved model for testing
            self.model_directory = Directory::Internal(internal_path);
            let model_directory = self.model_directory.path();
            return self.save_internal(&model_directory);
        }

        if self.model_directory.path() != user_model_directory {
            self.past_internal_model_directory.push(self.model_directory.path());
        }
        self.model_directory = Directory::User(user_model_directory);
        let model_directory = self.model_directory.path();
        self.save_internal(&model_directory)
    }

    fn get_model_directory(&self) -> Directory {
        self.model_directory.clone()
    }

    /// Allocates the layers of the neural network.
    fn allocate(&mut self) {
        for layer in &self.layers {
            self.utils.allocate_trainable(layer);
        }
    }

    /// Deallocates the layers of the neural network.
    fn deallocate(&mut self) {
        for layer in &self.layers {
            self.utils.deallocate_trainable(layer);
        }
    }

    fn set_internal(&mut self) {
        // set the model directory to internal
        self.model_directory = Directory::Internal(self.model_directory.path());
    }

    fn duplicate(&self) -> WrappedNeuralNetwork {
        unimplemented!()
    }

    fn get_utils(&self) -> WrappedUtils {
        self.utils.clone()
    }
}

impl TrainableNeuralNetwork for TrainableClassicNeuralNetwork {
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
    ) -> f64 {
        // in case one does not have enough samples, don't train and return zero accuracy
        let (transformed_inputs, transformed_targets) = Self::transform(inputs, targets);
        assert!(
            (0.0..=1.0).contains(&validation_split),
            "validation_split must be between 0 and 1"
        );

        let inputs_len: f64 =
            NumCast::from(inputs.len()).expect("Failed to convert inputs.len() to f64");
        let split_index: usize = NumCast::from((inputs_len * validation_split).round())
            .expect("Failed to convert split index to usize");
        let (train_inputs, validation_inputs) = transformed_inputs.split_at(split_index);
        let (train_targets, validation_targets) = transformed_targets.split_at(split_index);

        let mut accuracy = 0.0;

        let multi_progress = self.utils.get_multi_progress();

        for epoch in 0..epochs {
            // Initialize progress bar
            let pb = multi_progress.add(ProgressBar::new(train_inputs.len() as u64));
            pb.set_draw_target(ProgressDrawTarget::stdout());
            pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} | {msg}")
                .expect("Invalid template")
                .progress_chars("#>-"),);

            let mut loss = 0.0;
            let mut success_count = 0.0;

            train_inputs.iter().zip(train_targets).enumerate().for_each(|(j, (input, target))| {
                // Forward pass
                let output = self.forward(input.as_slice());

                // Calculate accuracy
                let correct_outputs = output
                    .iter()
                    .zip(target.iter())
                    .filter(|(&o, &t)| (o - t).abs() < tolerance)
                    .count();
                let correct_outputs_f64: f64 = NumCast::from(correct_outputs)
                    .expect("Failed to convert correct_outputs to f64");
                let target_len_f64: f64 =
                    NumCast::from(target.len()).expect("Failed to convert target.len() to f64");

                let match_percentage = correct_outputs_f64 / target_len_f64;
                if match_percentage >= sample_match_percentage {
                    success_count += 1.0;
                }
                // Calculate loss gradient
                let grad_output: Vec<f64> = output
                    .iter()
                    .zip(target)
                    .map(|(o, t)| {
                        let error = o - t;
                        loss += error * error;
                        2.0 * error
                    })
                    .collect();

                // Backward pass
                self.backward(grad_output);

                // Update weights
                if use_adam {
                    self.adjust_adam(j + 1, learning_rate, 0.9, 0.999, 1e-8);
                } else {
                    self.layers
                        .iter_mut()
                        .for_each(|layer| layer.update_weights(learning_rate, self.utils.clone()));
                }

                // Update the progress bar
                let train_inputs_len: f64 = NumCast::from(train_inputs.len())
                    .expect("Failed to convert train_inputs.len() to f64");
                let accuracy = success_count / train_inputs_len * 100.0;
                let loss_display = loss / train_inputs_len;
                pb.set_position((j + 1) as u64);
                pb.set_message(format!("Accuracy: {accuracy:.2} %, Loss: {loss_display:.4}"));
            });

            // Validation phase
            let mut validation_loss = 0.0;
            let mut validation_success_count = 0.0;

            validation_inputs.iter().zip(validation_targets).for_each(|(input, target)| {
                let output = self.forward(input.as_slice());
                let correct_outputs = output
                    .iter()
                    .zip(target.iter())
                    .filter(|(&o, &t)| (o - t).abs() < tolerance)
                    .count();
                let correct_outputs_f64: f64 = NumCast::from(correct_outputs)
                    .expect("Failed to convert correct_outputs to f64");
                let target_len_f64: f64 =
                    NumCast::from(target.len()).expect("Failed to convert target.len() to f64");
                let match_percentage = correct_outputs_f64 / target_len_f64;
                if match_percentage >= sample_match_percentage {
                    validation_success_count += 1.0;
                }

                validation_loss += output
                    .iter()
                    .zip(target)
                    .map(|(o, t)| {
                        let error = o - t;
                        error * error
                    })
                    .sum::<f64>();
            });

            let validation_inputs_len: f64 = NumCast::from(validation_inputs.len())
                .expect("Failed to convert validation_inputs.len() to f64");
            validation_loss /= validation_inputs_len;
            let validation_accuracy = validation_success_count / validation_inputs_len * 100.0;
            accuracy = validation_accuracy;
            // Finish the progress bar
            let train_inputs_len: f64 = NumCast::from(train_inputs.len())
                .expect("Failed to convert train_inputs.len() to f64");
            loss /= train_inputs_len;
            let accuracy = success_count / train_inputs_len * 100.0;
            let message = format!(
            "Epoch {epoch} finished | Train Acc: {accuracy:.2} %, Train Loss: {loss:.4} | Val Acc: {validation_accuracy:.2} %, Val Loss: {validation_loss:.4}");
            pb.finish_with_message(message);
            multi_progress.remove(&pb);
        }
        accuracy
    }

    /// Trains the neural network doing batch back propagation.
    fn train_batch(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        learning_rate: f64,
        epochs: usize,
        tolerance: f64,
        batch_size: usize,
    ) {
        for i in 0..epochs {
            println!("Epoch: {i}\r");
            let mut loss = 0.0;
            let input_chunks = inputs.chunks(batch_size);
            let target_chunks = targets.chunks(batch_size);
            let mut success_count = 0.0;
            for batch in input_chunks.zip(target_chunks) {
                let input_chunk_batch = batch.0;
                let target_chunk_batch = batch.1;
                for (input, target) in input_chunk_batch.iter().zip(target_chunk_batch) {
                    let output = self.forward_batch(input.as_slice());

                    // Check if the output matches the target
                    let mut nb_correct_outputs = 0;
                    for (o, t) in output.iter().zip(target.iter()) {
                        if (o - t).abs() < tolerance {
                            nb_correct_outputs += 1;
                        }
                    }
                    let nb_correct_outputs_f64: f64 =
                        NumCast::from(nb_correct_outputs).expect("Failed to convert to f64");
                    let target_len_f64: f64 =
                        NumCast::from(target.len()).expect("Failed to convert target.len() to f64");
                    success_count += nb_correct_outputs_f64 / target_len_f64;
                    let mut grad_output = Vec::new();
                    for j in 0..output.len() {
                        let error = output[j] - target[j];
                        grad_output.push(2.0 * error);
                        loss += error * error;
                    }
                    self.backward_batch(grad_output);
                }
                for layer in &mut self.layers {
                    layer.update_weights(learning_rate, self.utils.clone());
                }
            }
            let inputs_len: f64 =
                NumCast::from(inputs.len()).expect("Failed to convert inputs.len() to f64");
            let accuracy = success_count / inputs_len * 100.0;
            println!("Epoch {}: Loss {}, Accuracy {}%\r", i, loss / inputs_len, accuracy);
            if accuracy < 0.01 && i > 10 {
                break;
            }
        }
    }

    /// Returns the input size of the first layer in the network.
    fn input_size(&self) -> usize {
        self.shape.layers.first().map_or(0, super::shape::LayerShape::input_size)
    }

    /// Returns the output size of the last layer in the network.
    fn output_size(&self) -> usize {
        self.shape.layers.last().map_or(0, super::shape::LayerShape::output_size)
    }

    fn duplicate_trainable(&self) -> WrappedTrainableNeuralNetwork {
        // create a sibling directory with the postfix _clone appendended to model_direcotory path
        let model_directory = self.get_first_free_model_directory();
        // Save the model to the new directory
        self.save_internal(&model_directory).unwrap();
        // Clone the neural network by cloning its layers and activations
        let mut new_layers = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            new_layers.push(layer.duplicate(model_directory.clone(), i));
            layer.cleanup();
        }
        WrappedTrainableNeuralNetwork::new(Box::new(Self {
            layers: new_layers,
            activations: self.activations.clone(),
            shape: self.shape.clone(),
            model_directory: Directory::Internal(model_directory),
            past_internal_model_directory: Vec::new(),
            utils: self.utils.clone(),
        }))
    }
}

impl Drop for TrainableClassicNeuralNetwork {
    fn drop(&mut self) {
        // Save the model to ensure that everything is on disk if it is a user_model_directory
        // ensure that the model_directory exists
        if let Directory::User(_) = &self.model_directory {
            if std::fs::metadata(self.model_directory.path()).is_err() {
                std::fs::create_dir_all(self.model_directory.path()).unwrap();
            }
            self.save_layout();
            self.deallocate();
        }
        // Interne Verzeichnisse immer entfernen, unabhängig vom Testmodus
        if let Directory::Internal(dir) = &self.model_directory {
            if std::fs::metadata(dir).is_ok() {
                std::fs::remove_dir_all(dir).unwrap();
            }
        }
        for dir in &self.past_internal_model_directory {
            if dir != &self.model_directory.path() && std::fs::metadata(dir).is_ok() {
                std::fs::remove_dir_all(dir).unwrap();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        nn::shape::{ActivationData, ActivationType, LayerShape},
        utilities::util::Utils,
    };

    #[test]
    fn test_neural_network_train() {
        use std::path::Path;
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Helper to generate unique workspace names for tests
        static TEST_COUNTER: AtomicUsize = AtomicUsize::new(100); // Start at 100 to avoid conflicts

        let counter = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let workspace = format!("/tmp/test_workspace_unit_{counter}");
        let utils =
            WrappedUtils::new(Utils::new_with_test_mode(1_000_000_000, 4, workspace.clone()));

        let mut nn = TrainableClassicNeuralNetwork::new(
            NeuralNetworkShape {
                layers: vec![
                    LayerShape {
                        layer_type: LayerType::Dense { input_size: 3, output_size: 3 },
                        activation: ActivationData::new(ActivationType::Sigmoid),
                    },
                    LayerShape {
                        layer_type: LayerType::Dense { input_size: 3, output_size: 3 },
                        activation: ActivationData::new(ActivationType::ReLU),
                    },
                ],
            },
            &Directory::Internal("internal_model".to_string()),
            utils,
        );

        let input = vec![1.0, 1.0, 1.0];
        // put input 200 times in inputs
        let inputs = vec![input; 200];
        let target = vec![0.0, 0.0, 0.0];
        let targets = vec![target; 200];

        nn.train(&inputs, &targets, 0.01, 5, 0.1, true, 0.7, 1.0);

        let prediction = nn.predict(inputs[0].clone());
        // print targets[0]
        println!("{:?}", targets[0]);
        // print prediction
        println!("{prediction:?}");
        assert_eq!(prediction.len(), 3);
        // assert that the prediction is close to the target
        for (p, t) in prediction.iter().zip(&targets[0]) {
            assert!((p - t).abs() < 1e-4);
        }

        // Clean up workspace
        if !workspace.is_empty() && Path::new(&workspace).exists() {
            let _ = std::fs::remove_dir_all(&workspace);
        }
    }
}
