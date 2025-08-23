use std::path::Path;

use super::nn_factory::copy_dir_recursive;
use super::nn_factory::neural_network_from_disk;
use super::nn_factory::trainable_neural_network_from_disk;
use super::nn_trait::WrappedNeuralNetwork;
use super::nn_trait::WrappedTrainableNeuralNetwork;
use super::shape::NeuralNetworkShape;

use crate::nn::directory::Directory;
use crate::nn::neuralnet::ClassicNeuralNetwork;
use crate::nn::neuralnet::TrainableClassicNeuralNetwork;
use crate::nn::nn_factory::get_first_free_model_directory;
use crate::nn::nn_trait::NeuralNetwork;
use crate::nn::nn_trait::TrainableNeuralNetwork;
use crate::nn::shape::AnnotatedNeuralNetworkShape;
use crate::nn::shape::LayerShape;
use crate::nn::shape::LayerType;
use crate::utilities::util::WrappedUtils;

#[derive(Debug)]
pub struct RetryNeuralNetwork {
    primary_nn: WrappedNeuralNetwork,
    backup_nn: WrappedNeuralNetwork,
    // The shape of the neural network that it should pretend to have to the outside world
    shape: NeuralNetworkShape,
    model_directory: Directory,
    past_internal_model_directories: Vec<String>,
    utils: WrappedUtils,
}

impl RetryNeuralNetwork {
    /// Creates a new `RetryNeuralNetwork` with the given shape, levels, and model directory.
    ///
    /// # Panics
    ///
    /// This function will panic if the internal model directory is invalid or if the
    /// neural networks cannot be created.
    #[must_use]
    pub fn new(
        shape: NeuralNetworkShape,
        levels: i32,
        internal_model_directory: String,
        utils: WrappedUtils,
    ) -> Self {
        let actual_shape = add_internal_dimensions(&shape);
        let primary_nn = WrappedNeuralNetwork::new(Box::new(ClassicNeuralNetwork::new(
            actual_shape,
            append_dir(internal_model_directory.clone(), "primary"),
            utils.clone(),
        )));
        let backup_nn = match levels {
            1..=i32::MAX => WrappedNeuralNetwork::new(Box::new(Self::new(
                shape.clone(),
                levels - 1,
                append_dir(internal_model_directory.clone(), "backup"),
                utils.clone(),
            ))),
            0 => WrappedNeuralNetwork::new(Box::new(ClassicNeuralNetwork::new(
                shape.clone(),
                append_dir(internal_model_directory.clone(), "backup"),
                utils.clone(),
            ))),
            _ => panic!("Invalid level: {levels}"),
        };
        Self {
            primary_nn,
            backup_nn,
            shape,
            model_directory: Directory::Internal(internal_model_directory),
            past_internal_model_directories: vec![],
            utils,
        }
    }

    /// Creates a new `RetryNeuralNetwork` from the given model directory.
    ///
    /// # Panics
    ///
    /// This function will panic if the model directory is invalid or if the neural
    /// networks cannot be created.
    #[must_use]
    pub fn from_disk(
        model_directory: String,
        utils: WrappedUtils,
    ) -> WrappedNeuralNetwork {
        let primary_model_directory = append_dir(model_directory.clone(), "primary");
        let backup_model_directory = append_dir(model_directory.clone(), "backup");
        if std::path::Path::new(&primary_model_directory).exists() {
            let primary_nn = WrappedNeuralNetwork::new(Box::new(
                ClassicNeuralNetwork::from_disk(primary_model_directory, utils.clone()).unwrap(),
            ));
            let backup_nn = Self::from_disk(backup_model_directory, utils.clone());
            let shape = backup_nn.shape();
            WrappedNeuralNetwork::new(Box::new(Self {
                primary_nn,
                backup_nn,
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
        let primary_output = self.primary_nn.predict(input.clone());
        // if the last value in primary output is as close to zero as some tolerance, then we need to use the backup neural network
        if (primary_output[primary_output.len() - 1] - 1.0).abs() < 0.2 {
            self.backup_nn.predict(input)
        } else {
            // return the primary output despite the last internal value
            primary_output[0..primary_output.len() - 1].to_vec()
        }
    }
}

fn add_internal_dimensions(shape: &NeuralNetworkShape) -> NeuralNetworkShape {
    // Add internal dimensions to the shape
    let mut annotated_shape = AnnotatedNeuralNetworkShape::new(shape);
    let first_layer = shape.layers.first().unwrap();

    // Add internal dimensions to the first layer
    let internal_layer = first_layer.clone();
    let new_dense_layer_type = LayerShape {
        layer_type: LayerType::Dense {
            input_size: internal_layer.input_size(),
            output_size: internal_layer.output_size() + 1,
        },
        activation: internal_layer.activation,
    };
    annotated_shape.change_layer(0, new_dense_layer_type);

    // Add internal dimensions to the rest of the layers
    for (i, layer) in shape.layers.iter().skip(1).enumerate() {
        // Add internal dimensions to the layer
        let internal_layer = layer.clone();
        let new_dense_layer_type = LayerShape {
            layer_type: LayerType::Dense {
                input_size: internal_layer.input_size() + 1,
                output_size: internal_layer.output_size() + 1,
            },
            activation: internal_layer.activation.clone(),
        };
        annotated_shape.change_layer(i + 1, new_dense_layer_type);
    }
    annotated_shape.to_neural_network_shape()
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

impl NeuralNetwork for RetryNeuralNetwork {
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
        let primary_user_model_directory = append_dir(user_model_directory.clone(), "primary");
        self.primary_nn.save(primary_user_model_directory)?;
        let backup_user_model_directory = append_dir(user_model_directory, "backup");
        self.backup_nn.save(backup_user_model_directory)?;
        Ok(())
    }

    fn get_model_directory(&self) -> Directory {
        self.model_directory.clone()
    }

    fn allocate(&mut self) {
        self.primary_nn.allocate();
        self.backup_nn.allocate();
    }

    fn deallocate(&mut self) {
        self.primary_nn.deallocate();
        self.backup_nn.deallocate();
    }

    fn set_internal(&mut self) {
        self.model_directory = Directory::Internal(self.model_directory.path());
        self.primary_nn.set_internal();
        self.backup_nn.set_internal();
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

impl Drop for RetryNeuralNetwork {
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
pub struct TrainableRetryNeuralNetwork {
    primary_nn: WrappedTrainableNeuralNetwork,
    backup_nn: WrappedTrainableNeuralNetwork,
    // The shape of the neural network that it should pretend to have to the outside world
    shape: NeuralNetworkShape,
    model_directory: Directory,
    past_internal_model_directories: Vec<String>,
    utils: WrappedUtils,
}

impl TrainableRetryNeuralNetwork {
    /// Creates a new `TrainableRetryNeuralNetwork` with the given shape, levels, and model directory.
    ///
    /// # Panics
    ///
    /// This function will panic if the internal model directory is invalid or if the
    /// neural networks cannot be created.
    #[must_use]
    pub fn new(
        shape: NeuralNetworkShape,
        levels: i32,
        internal_model_directory: String,
        utils: WrappedUtils,
    ) -> Self {
        let actual_shape = add_internal_dimensions(&shape);
        let primary_nn =
            WrappedTrainableNeuralNetwork::new(Box::new(TrainableClassicNeuralNetwork::new(
                actual_shape,
                &Directory::Internal(append_dir(internal_model_directory.clone(), "primary")),
                utils.clone(),
            )));
        let backup_nn = match levels {
            1..=i32::MAX => WrappedTrainableNeuralNetwork::new(Box::new(Self::new(
                shape.clone(),
                levels - 1,
                append_dir(internal_model_directory.clone(), "backup"),
                utils.clone(),
            ))),
            0 => WrappedTrainableNeuralNetwork::new(Box::new(TrainableClassicNeuralNetwork::new(
                shape.clone(),
                &Directory::Internal(append_dir(internal_model_directory.clone(), "backup")),
                utils.clone(),
            ))),
            _ => panic!("Invalid level: {levels}"),
        };
        Self {
            primary_nn,
            backup_nn,
            shape,
            model_directory: Directory::Internal(internal_model_directory),
            past_internal_model_directories: vec![],
            utils,
        }
    }

    /// Creates a new `TrainableRetryNeuralNetwork` from the given model directory.
    ///
    /// # Panics
    ///
    /// This function will panic if the model directory is invalid or if the neural
    /// networks cannot be created.
    #[must_use]
    pub fn from_disk(
        model_directory: String,
        utils: WrappedUtils,
    ) -> WrappedTrainableNeuralNetwork {
        let primary_model_directory = append_dir(model_directory.clone(), "primary");
        let backup_model_directory = append_dir(model_directory.clone(), "backup");
        if std::path::Path::new(&primary_model_directory).exists() {
            let primary_nn = WrappedTrainableNeuralNetwork::new(Box::new(
                TrainableClassicNeuralNetwork::from_disk(primary_model_directory, utils.clone())
                    .unwrap(),
            ));
            let backup_nn = Self::from_disk(backup_model_directory, utils.clone());
            let shape = backup_nn.shape();
            WrappedTrainableNeuralNetwork::new(Box::new(Self {
                primary_nn,
                backup_nn,
                shape,
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
        let primary_output = self.primary_nn.predict(input.clone());
        // if the last value in primary output is as close to zero as some tolerance, then we need to use the backup neural network
        if primary_output[primary_output.len() - 1].abs() < 0.05 {
            self.backup_nn.predict(input)
        } else {
            // return the primary output despite the last internal value
            primary_output[0..primary_output.len() - 1].to_vec()
        }
    }
}

impl NeuralNetwork for TrainableRetryNeuralNetwork {
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
        let primary_user_model_directory = append_dir(user_model_directory.clone(), "primary");
        self.primary_nn.save(primary_user_model_directory)?;
        let backup_user_model_directory = append_dir(user_model_directory, "backup");
        self.backup_nn.save(backup_user_model_directory)?;
        Ok(())
    }

    fn get_model_directory(&self) -> Directory {
        self.model_directory.clone()
    }

    fn allocate(&mut self) {
        self.primary_nn.allocate();
        self.backup_nn.allocate();
    }

    fn deallocate(&mut self) {
        self.primary_nn.deallocate();
        self.backup_nn.deallocate();
    }

    fn set_internal(&mut self) {
        self.model_directory = Directory::Internal(self.model_directory.path());
        self.primary_nn.set_internal();
        self.backup_nn.set_internal();
    }

    fn duplicate(&self) -> WrappedNeuralNetwork {
        unimplemented!()
    }

    fn get_utils(&self) -> WrappedUtils {
        self.utils.clone()
    }
}

impl TrainableNeuralNetwork for TrainableRetryNeuralNetwork {
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
        if inputs.len() < 100 {
            return 0.0;
        }
        let mut temp_neural_network = TrainableClassicNeuralNetwork::new(
            self.shape.clone(),
            &Directory::Internal(append_dir(self.model_directory.path(), "temp_primary")),
            self.utils.clone(),
        );
        let _ = temp_neural_network.train(
            inputs,
            targets,
            learning_rate,
            epochs,
            tolerance,
            use_adam,
            validation_split,
            sample_match_percentage,
        );

        let (primary_inputs, primary_targets): (Vec<Vec<f64>>, Vec<Vec<f64>>) = inputs
            .iter()
            .zip(targets.iter())
            .map(|(input, target)| {
                let prediction = temp_neural_network.predict(input.clone());
                (input, target, prediction)
            })
            .map(|(input, target, prediction)| {
                // Check if the output matches the target
                let mut nb_correct_outputs = 0;
                for (o, t) in prediction.iter().zip(target.iter()) {
                    if (o - t).abs() < tolerance {
                        nb_correct_outputs += 1;
                    }
                }
                let mut t = target.clone();
                let match_percentage = nb_correct_outputs as f64 / target.len() as f64;
                if match_percentage >= sample_match_percentage {
                    t.push(0.0);
                } else {
                    t.push(1.0);
                }
                (input.clone(), t)
            })
            .unzip();

        // train the primary neural network with the modified outputs
        let primary_accuracy = self.primary_nn.train(
            &primary_inputs,
            &primary_targets,
            learning_rate,
            epochs,
            tolerance,
            use_adam,
            validation_split,
            sample_match_percentage,
        );

        let (backup_inputs, backup_targets): (Vec<Vec<f64>>, Vec<Vec<f64>>) = primary_inputs
            .iter()
            .zip(primary_targets.iter())
            .map(|(input, target)| {
                let prediction = self.primary_nn.predict(input.clone());
                (input, target, prediction)
            })
            .filter(|(_, target, prediction)| {
                // Check if the output matches the target
                let mut nb_correct_outputs = 0;
                for (o, t) in prediction.iter().zip(target.iter()) {
                    if (o - t).abs() < tolerance {
                        nb_correct_outputs += 1;
                    }
                }

                let match_percentage = nb_correct_outputs as f64 / target.len() as f64;
                match_percentage < sample_match_percentage
            })
            .map(|(input, target, _)| {
                let mut t = target.clone();
                t.remove(t.len() - 1);
                (input.clone(), t)
            })
            .unzip();

        let backup_accuracy = self.backup_nn.train(
            &backup_inputs,
            &backup_targets,
            learning_rate,
            epochs,
            tolerance,
            use_adam,
            validation_split,
            sample_match_percentage,
        );

        primary_accuracy + backup_accuracy
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
        self.primary_nn.train_batch(inputs, targets, learning_rate, epochs, tolerance, batch_size);
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

impl Drop for TrainableRetryNeuralNetwork {
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
        nn::shape::{ActivationData, ActivationType, LayerShape},
        utilities::util::Utils,
    };

    #[test]
    fn test_retry_neural_network_train() {
        let utils = WrappedUtils::new(Utils::new(1_000_000_000, 4));
        let mut nn = TrainableRetryNeuralNetwork::new(
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
            1,
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
