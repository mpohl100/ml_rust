use std::path::Path;

use super::nn_factory::copy_dir_recursive;
use super::nn_factory::neural_network_from_disk;
use super::nn_factory::new_trainable_neural_network;
use super::nn_factory::trainable_neural_network_from_disk;
use super::nn_factory::NeuralNetworkCreationArguments;
use super::nn_trait::WrappedNeuralNetwork;
use super::nn_trait::WrappedTrainableNeuralNetwork;
use super::shape::NeuralNetworkShape;

use crate::nn::directory::Directory;
use crate::nn::neuralnet::ClassicNeuralNetwork;
use crate::nn::neuralnet::TrainableClassicNeuralNetwork;
use crate::nn::nn_factory::get_first_free_model_directory;
use crate::nn::nn_trait::NeuralNetwork;
use crate::nn::nn_trait::TrainableNeuralNetwork;
use crate::utilities::util::WrappedUtils;

#[derive(Debug)]
pub struct EitherNeuralNetwork {
    pre_nn: WrappedNeuralNetwork,
    left_nn: Option<WrappedNeuralNetwork>,
    right_nn: Option<WrappedNeuralNetwork>,
    shape: NeuralNetworkShape,
    model_directory: Directory,
    past_internal_model_directories: Vec<String>,
    utils: WrappedUtils,
}

impl EitherNeuralNetwork {
    /// Creates a new `EitherNeuralNetwork` from the given model directory.
    ///
    /// # Panics
    ///
    /// This function will panic if the model directory is invalid or if the neural networks cannot be loaded.
    #[must_use]
    pub fn from_disk(
        model_directory: String,
        utils: WrappedUtils,
    ) -> WrappedNeuralNetwork {
        let pre_model_directory = append_dir(model_directory.clone(), "pre");
        let left_model_directory = append_dir(model_directory.clone(), "left");
        let right_model_directory = append_dir(model_directory.clone(), "right");
        if std::path::Path::new(&pre_model_directory).exists() {
            let pre_nn = neural_network_from_disk(pre_model_directory, utils.clone());

            let left_nn = if std::path::Path::new(&left_model_directory).exists() {
                Some(neural_network_from_disk(left_model_directory, utils.clone()))
            } else {
                None
            };

            let right_nn = if std::path::Path::new(&right_model_directory).exists() {
                Some(neural_network_from_disk(right_model_directory, utils.clone()))
            } else {
                None
            };

            let shape = pre_nn.shape();
            WrappedNeuralNetwork::new(Box::new(Self {
                pre_nn,
                left_nn,
                right_nn,
                shape,
                model_directory: Directory::User(model_directory),
                past_internal_model_directories: vec![],
                utils,
            }))
        } else {
            WrappedNeuralNetwork::new(Box::new(
                ClassicNeuralNetwork::from_disk(model_directory, utils).unwrap(),
            ))
        }
    }

    fn forward(
        &mut self,
        input: Vec<f64>,
    ) -> Vec<f64> {
        let pre_output = self.pre_nn.predict(input.clone());
        if self.left_nn.is_none() && self.right_nn.is_none() {
            return pre_output;
        }
        // if the last value in primary output is as close to zero as some tolerance, then we need to use the backup neural network
        if (pre_output[0] - 1.0).abs() < 0.2 {
            if self.left_nn.is_none() {
                return pre_output;
            }
            let mut left_nn = self.left_nn.as_ref().unwrap().clone();

            left_nn.predict(input)
        } else {
            if self.right_nn.is_none() {
                return pre_output;
            }
            let mut right_nn = self.right_nn.as_ref().unwrap().clone();
            right_nn.predict(input)
        }
    }
}

fn append_dir(
    model_directory: String,
    subdir: &str,
) -> String {
    let mut path = model_directory;
    path.push('/');
    path.push_str(subdir);
    path
}

impl NeuralNetwork for EitherNeuralNetwork {
    fn predict(
        &mut self,
        input: Vec<f64>,
    ) -> Vec<f64> {
        self.forward(input)
    }

    fn shape(&self) -> NeuralNetworkShape {
        if self.left_nn.is_some() {
            self.left_nn.as_ref().unwrap().shape()
        } else if self.right_nn.is_some() {
            self.right_nn.as_ref().unwrap().shape()
        } else {
            self.shape.clone()
        }
    }

    fn save(
        &mut self,
        user_model_directory: String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Directory::Internal(_) = self.model_directory {
            self.past_internal_model_directories.push(self.model_directory.path());
        }
        self.model_directory = Directory::User(user_model_directory.clone());
        let pre_user_model_directory = append_dir(user_model_directory.clone(), "pre");
        self.pre_nn.save(pre_user_model_directory)?;

        if let Some(ref mut left_nn) = self.left_nn {
            let left_user_model_directory = append_dir(user_model_directory.clone(), "left");
            left_nn.save(left_user_model_directory)?;
        }
        if let Some(ref mut right_nn) = self.right_nn {
            let right_user_model_directory = append_dir(user_model_directory, "right");
            right_nn.save(right_user_model_directory)?;
        }
        Ok(())
    }

    fn get_model_directory(&self) -> Directory {
        self.model_directory.clone()
    }

    fn allocate(&mut self) {
        self.pre_nn.allocate();
        if let Some(ref mut left_nn) = self.left_nn {
            left_nn.allocate();
        }
        if let Some(ref mut right_nn) = self.right_nn {
            right_nn.allocate();
        }
    }

    fn deallocate(&mut self) {
        self.pre_nn.deallocate();
        if let Some(ref mut left_nn) = self.left_nn {
            left_nn.deallocate();
        }
        if let Some(ref mut right_nn) = self.right_nn {
            right_nn.deallocate();
        }
    }

    fn set_internal(&mut self) {
        self.model_directory = Directory::Internal(self.model_directory.path());
        self.pre_nn.set_internal();
        if let Some(ref mut left_nn) = self.left_nn {
            left_nn.set_internal();
        }
        if let Some(ref mut right_nn) = self.right_nn {
            right_nn.set_internal();
        }
    }

    fn duplicate(&self) -> WrappedNeuralNetwork {
        let new_model_directory = get_first_free_model_directory(&self.model_directory);
        copy_dir_recursive(
            Path::new(&self.model_directory.path()),
            Path::new(&new_model_directory),
        )
        .expect("Failed to copy model directory for retry neural network");
        let mut cloned_retry_nn = neural_network_from_disk(new_model_directory, self.utils.clone());
        cloned_retry_nn.set_internal();
        cloned_retry_nn
    }

    fn get_utils(&self) -> WrappedUtils {
        self.utils.clone()
    }
}

impl Drop for EitherNeuralNetwork {
    fn drop(&mut self) {
        // Save the model to ensure that everything is on disk if it is a user_model_directory
        // ensure that the model_directory exists
        if let Directory::User(_) = &self.model_directory {
            if std::fs::metadata(self.model_directory.path()).is_err() {
                std::fs::create_dir_all(self.model_directory.path()).unwrap();
            }
            self.deallocate();
        }
        // Remove the internal model directory from disk
        if let Directory::Internal(dir) = &self.model_directory {
            if std::fs::metadata(dir).is_ok() {
                std::fs::remove_dir_all(dir).unwrap();
            }
        }
        // Remove all past internal model directories
        for dir in &self.past_internal_model_directories {
            if dir != &self.model_directory.path() && std::fs::metadata(dir).is_ok() {
                std::fs::remove_dir_all(dir).unwrap();
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainableEitherNeuralNetwork {
    pre_nn: WrappedTrainableNeuralNetwork,
    left_nn: Option<WrappedTrainableNeuralNetwork>,
    right_nn: Option<WrappedTrainableNeuralNetwork>,
    // The shape of the neural network that it should pretend to have to the outside world
    shape: NeuralNetworkShape,
    pre_shape: NeuralNetworkShape,
    max_levels: i32,
    model_directory: Directory,
    past_internal_model_directories: Vec<String>,
    utils: WrappedUtils,
}

impl TrainableEitherNeuralNetwork {
    #[must_use]
    pub fn new(
        shape: NeuralNetworkShape,
        pre_shape: NeuralNetworkShape,
        max_levels: i32,
        internal_model_directory: String,
        utils: WrappedUtils,
    ) -> Self {
        let pre_nn =
            WrappedTrainableNeuralNetwork::new(Box::new(TrainableClassicNeuralNetwork::new(
                shape.clone(),
                &Directory::Internal(append_dir(internal_model_directory.clone(), "pre")),
                utils.clone(),
            )));
        Self {
            pre_nn,
            left_nn: None,
            right_nn: None,
            shape,
            pre_shape,
            max_levels,
            model_directory: Directory::Internal(internal_model_directory),
            past_internal_model_directories: vec![],
            utils,
        }
    }

    /// Creates a new `EitherNeuralNetwork` from the given model directory.
    ///
    /// # Panics
    ///
    /// This function will panic if the model directory is invalid or if the neural networks cannot be loaded.
    #[must_use]
    pub fn from_disk(
        model_directory: String,
        utils: WrappedUtils,
    ) -> WrappedTrainableNeuralNetwork {
        let pre_model_directory = append_dir(model_directory.clone(), "pre");
        let primary_model_directory = append_dir(model_directory.clone(), "left");
        let backup_model_directory = append_dir(model_directory.clone(), "right");
        if std::path::Path::new(&pre_model_directory).exists() {
            let pre_nn = trainable_neural_network_from_disk(pre_model_directory, utils.clone());

            let left_nn = if std::path::Path::new(&primary_model_directory).exists() {
                Some(trainable_neural_network_from_disk(primary_model_directory, utils.clone()))
            } else {
                None
            };

            let right_nn = if std::path::Path::new(&backup_model_directory).exists() {
                Some(trainable_neural_network_from_disk(backup_model_directory, utils.clone()))
            } else {
                None
            };

            let shape = pre_nn.shape();
            let pre_shape = pre_nn.shape();
            WrappedTrainableNeuralNetwork::new(Box::new(Self {
                pre_nn,
                left_nn,
                right_nn,
                shape,
                pre_shape,
                max_levels: 2, // allow extension of maximum two levels when retraining
                model_directory: Directory::User(model_directory),
                past_internal_model_directories: vec![],
                utils,
            }))
        } else {
            WrappedTrainableNeuralNetwork::new(Box::new(
                TrainableClassicNeuralNetwork::from_disk(model_directory, utils).unwrap(),
            ))
        }
    }

    fn forward(
        &mut self,
        input: Vec<f64>,
    ) -> Vec<f64> {
        let pre_output = self.pre_nn.predict(input.clone());
        if self.left_nn.is_none() && self.right_nn.is_none() {
            return pre_output;
        }
        // if the last value in primary output is as close to zero as some tolerance, then we need to use the backup neural network
        if (pre_output[0] - 1.0).abs() < 0.2 {
            if self.left_nn.is_none() {
                return pre_output;
            }
            let mut left_nn = self.left_nn.as_ref().unwrap().clone();

            left_nn.predict(input)
        } else {
            if self.right_nn.is_none() {
                return pre_output;
            }
            let mut right_nn = self.right_nn.as_ref().unwrap().clone();
            right_nn.predict(input)
        }
    }

    const fn not_enough_samples(inputs: &[Vec<f64>]) -> bool {
        inputs.len() < 100
    }

    #[allow(clippy::too_many_arguments)]
    fn train_temp_network(
        &self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        learning_rate: f64,
        epochs: usize,
        tolerance: f64,
        use_adam: bool,
        validation_split: f64,
        sample_match_percentage: f64,
    ) -> (WrappedTrainableNeuralNetwork, f64) {
        let mut temp_nn = new_trainable_neural_network(NeuralNetworkCreationArguments::new(
            self.shape.clone(),
            None,
            None,
            append_dir(self.model_directory.path(), "temp"),
            self.utils.clone(),
        ));

        let acc = temp_nn.train(
            inputs,
            targets,
            learning_rate,
            epochs,
            tolerance,
            use_adam,
            validation_split,
            sample_match_percentage,
        );

        (temp_nn, acc)
    }

    const fn no_more_levels(&self) -> bool {
        self.max_levels <= 0
    }

    fn save_pre_network(
        &mut self,
        nn: &WrappedTrainableNeuralNetwork,
        dir: &str,
    ) {
        self.pre_nn = nn.clone();
        self.pre_nn
            .save(append_dir(self.model_directory.path(), dir))
            .expect("Failed to save pre neural network");
    }

    #[allow(clippy::type_complexity)]
    fn split_by_prediction(
        network: &mut WrappedTrainableNeuralNetwork,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        tolerance: f64,
        sample_match_percentage: f64,
    ) -> ((Vec<Vec<f64>>, Vec<Vec<f64>>), (Vec<Vec<f64>>, Vec<Vec<f64>>)) {
        let (left_inputs, left_targets): (Vec<_>, Vec<_>) = inputs
            .iter()
            .zip(targets.iter())
            .map(|(input, target)| {
                let prediction = network.predict(input.clone());
                (input, target, prediction)
            })
            .filter(|(_, target, prediction)| {
                let mut nb_correct = 0;
                for (o, t) in prediction.iter().zip(target.iter()) {
                    if (o - t).abs() < tolerance {
                        nb_correct += 1;
                    }
                }
                let match_percentage = nb_correct as f64 / target.len() as f64;
                match_percentage >= sample_match_percentage
            })
            .map(|(input, target, _)| (input.clone(), target.clone()))
            .unzip();

        let (right_inputs, right_targets): (Vec<_>, Vec<_>) = inputs
            .iter()
            .zip(targets.iter())
            .map(|(input, target)| {
                let prediction = network.predict(input.clone());
                (input, target, prediction)
            })
            .filter(|(_, target, prediction)| {
                let mut nb_correct = 0;
                for (o, t) in prediction.iter().zip(target.iter()) {
                    if (o - t).abs() < tolerance {
                        nb_correct += 1;
                    }
                }
                let match_percentage = nb_correct as f64 / target.len() as f64;
                match_percentage < sample_match_percentage
            })
            .map(|(input, target, _)| (input.clone(), target.clone()))
            .unzip();

        ((left_inputs, left_targets), (right_inputs, right_targets))
    }

    const fn too_few_mispredictions(right_inputs: &[Vec<f64>]) -> bool {
        right_inputs.len() < 100
    }

    fn prepare_pre_training_data(
        left_inputs: &[Vec<f64>],
        right_inputs: &[Vec<f64>],
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let (left_inputs_pre, left_targets_pre): (Vec<_>, Vec<_>) =
            left_inputs.iter().map(|input| (input.clone(), vec![1.0, 0.0])).collect();

        let (right_inputs_pre, right_targets_pre): (Vec<_>, Vec<_>) =
            right_inputs.iter().map(|input| (input.clone(), vec![0.0, 1.0])).collect();

        let pre_inputs = [left_inputs_pre, right_inputs_pre].concat();
        let pre_targets = [left_targets_pre, right_targets_pre].concat();

        (pre_inputs, pre_targets)
    }

    #[allow(clippy::too_many_arguments)]
    fn train_and_save_network(
        &self,
        shape: NeuralNetworkShape,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        dir_name: &str,
        learning_rate: f64,
        epochs: usize,
        tolerance: f64,
        use_adam: bool,
        validation_split: f64,
        sample_match_percentage: f64,
    ) -> (WrappedTrainableNeuralNetwork, f64) {
        let model_dir = append_dir(self.model_directory.path(), dir_name);
        let mut nn = new_trainable_neural_network(NeuralNetworkCreationArguments::new(
            shape,
            Some(self.max_levels - 1),
            Some(self.pre_shape.clone()),
            model_dir.clone(),
            self.utils.clone(),
        ));

        let acc =
            nn.train(inputs, targets, learning_rate, epochs, tolerance, use_adam, validation_split, sample_match_percentage);

        let error_message = format!("Failed to save {dir_name} neural network");
        nn.save(model_dir).expect(&error_message);

        (nn, acc)
    }
}

impl NeuralNetwork for TrainableEitherNeuralNetwork {
    fn predict(
        &mut self,
        input: Vec<f64>,
    ) -> Vec<f64> {
        self.forward(input)
    }

    fn shape(&self) -> NeuralNetworkShape {
        self.shape.clone()
    }

    fn save(
        &mut self,
        user_model_directory: String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Directory::Internal(_) = self.model_directory {
            self.past_internal_model_directories.push(self.model_directory.path());
        }
        self.model_directory = Directory::User(user_model_directory.clone());

        let pre_user_model_directory = append_dir(user_model_directory.clone(), "pre");
        self.pre_nn.save(pre_user_model_directory)?;
        if let Some(ref mut left_nn) = self.left_nn {
            let left_user_model_directory = append_dir(user_model_directory.clone(), "left");
            left_nn.save(left_user_model_directory)?;
        }
        if let Some(ref mut right_nn) = self.right_nn {
            let right_user_model_directory = append_dir(user_model_directory, "right");
            right_nn.save(right_user_model_directory)?;
        }

        Ok(())
    }

    fn get_model_directory(&self) -> Directory {
        self.model_directory.clone()
    }

    fn allocate(&mut self) {
        self.pre_nn.allocate();
        if let Some(ref mut left_nn) = self.left_nn {
            left_nn.allocate();
        }
        if let Some(ref mut right_nn) = self.right_nn {
            right_nn.allocate();
        }
    }

    fn deallocate(&mut self) {
        self.pre_nn.deallocate();
        if let Some(ref mut left_nn) = self.left_nn {
            left_nn.deallocate();
        }
        if let Some(ref mut right_nn) = self.right_nn {
            right_nn.deallocate();
        }
    }

    fn set_internal(&mut self) {
        self.model_directory = Directory::Internal(self.model_directory.path());
        self.pre_nn.set_internal();
        if let Some(ref mut left_nn) = self.left_nn {
            left_nn.set_internal();
        }
        if let Some(ref mut right_nn) = self.right_nn {
            right_nn.set_internal();
        }
    }

    fn duplicate(&self) -> WrappedNeuralNetwork {
        unimplemented!()
    }

    fn get_utils(&self) -> WrappedUtils {
        self.utils.clone()
    }
}

impl TrainableNeuralNetwork for TrainableEitherNeuralNetwork {
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
        if Self::not_enough_samples(inputs) {
            return 0.0;
        }

        let (mut temp_nn, temp_accuracy) = self.train_temp_network(
            inputs,
            targets,
            learning_rate,
            epochs,
            tolerance,
            use_adam,
            validation_split,
            sample_match_percentage,
        );

        if self.no_more_levels() {
            self.save_pre_network(&temp_nn, "pre");
            return temp_accuracy;
        }

        let ((left_inputs, left_targets), (right_inputs, right_targets)) =
            Self::split_by_prediction(&mut temp_nn, inputs, targets, tolerance, sample_match_percentage);

        if Self::too_few_mispredictions(&right_inputs) {
            self.save_pre_network(&temp_nn, "pre");
            return temp_accuracy;
        }

        let (pre_inputs, pre_targets) =
            Self::prepare_pre_training_data(&left_inputs, &right_inputs);

        let (pre_nn, _) = self.train_and_save_network(
            self.pre_shape.clone(),
            &pre_inputs,
            &pre_targets,
            "pre",
            learning_rate,
            epochs,
            tolerance,
            use_adam,
            validation_split,
            sample_match_percentage,
        );
        self.pre_nn = pre_nn;

        let (_, left_accuracy) = self.train_and_save_network(
            self.shape.clone(),
            &left_inputs,
            &left_targets,
            "left",
            learning_rate,
            epochs,
            tolerance,
            use_adam,
            validation_split,
            sample_match_percentage,
        );

        let (_, right_accuracy) = self.train_and_save_network(
            self.shape.clone(),
            &right_inputs,
            &right_targets,
            "right",
            learning_rate,
            epochs,
            tolerance,
            use_adam,
            validation_split,
            sample_match_percentage,
        );

        left_accuracy + right_accuracy
    }

    fn train_batch(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        learning_rate: f64,
        epochs: usize,
        tolerance: f64,
        batch_size: usize,
    ) {
        self.pre_nn.train_batch(inputs, targets, learning_rate, epochs, tolerance, batch_size);
    }

    fn input_size(&self) -> usize {
        self.shape.layers[0].input_size()
    }

    fn output_size(&self) -> usize {
        self.shape.layers[self.shape.layers.len() - 1].output_size()
    }

    fn duplicate_trainable(&self) -> WrappedTrainableNeuralNetwork {
        let new_model_directory = get_first_free_model_directory(&self.model_directory);
        copy_dir_recursive(
            Path::new(&self.model_directory.path()),
            Path::new(&new_model_directory),
        )
        .expect("Failed to copy model directory for trainable retry neural network");
        let mut cloned_retry_nn =
            trainable_neural_network_from_disk(new_model_directory, self.utils.clone());
        cloned_retry_nn.set_internal();
        cloned_retry_nn
    }
}

impl Drop for TrainableEitherNeuralNetwork {
    fn drop(&mut self) {
        // Save the model to ensure that everything is on disk if it is a user_model_directory
        // ensure that the model_directory exists
        if let Directory::User(_) = &self.model_directory {
            if std::fs::metadata(self.model_directory.path()).is_err() {
                std::fs::create_dir_all(self.model_directory.path()).unwrap();
            }
            self.deallocate();
        }
        // Remove the internal model directory from disk
        if let Directory::Internal(dir) = &self.model_directory {
            if std::fs::metadata(dir).is_ok() {
                std::fs::remove_dir_all(dir).unwrap();
            }
        }
        // Remove all past internal model directories
        for dir in &self.past_internal_model_directories {
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
        nn::shape::{ActivationData, ActivationType, LayerShape, LayerType},
        utilities::util::Utils,
    };

    #[test]
    fn test_either_neural_network_train() {
        let utils = WrappedUtils::new(Utils::new(1_000_000_000, 4));
        let mut nn = TrainableEitherNeuralNetwork::new(
            NeuralNetworkShape {
                layers: vec![
                    LayerShape {
                        layer_type: LayerType::Dense { input_size: 3, output_size: 3 },
                        activation: ActivationData::new(ActivationType::ReLU),
                    },
                    LayerShape {
                        layer_type: LayerType::Dense { input_size: 3, output_size: 3 },
                        activation: ActivationData::new(ActivationType::ReLU),
                    },
                ],
            },
            NeuralNetworkShape {
                layers: vec![LayerShape {
                    layer_type: LayerType::Dense { input_size: 3, output_size: 3 },
                    activation: ActivationData::new_softmax(1.0),
                }],
            },
            2,
            "internal_model".to_string(),
            utils,
        );

        let input = vec![1.0, 1.0, 1.0];
        // put input 200 times in inputs
        let inputs = vec![input; 500];
        let target = vec![0.0, 0.0, 0.0];
        let targets = vec![target; 500];

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
    }
}
