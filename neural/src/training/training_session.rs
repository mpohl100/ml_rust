use super::data_importer::DataImporter;
use super::training_params::TrainingParams;
use crate::nn::directory::Directory;
use crate::nn::nn_factory::new_trainable_neural_network;
use crate::nn::nn_factory::trainable_neural_network_from_disk;
use crate::nn::nn_factory::NeuralNetworkCreationArguments;
use crate::nn::nn_trait::WrappedTrainableNeuralNetwork;
use crate::utilities::util::WrappedUtils;

use num_traits::NumCast;

use std::error::Error;

pub struct TrainingSession {
    params: TrainingParams,
    neural_network: WrappedTrainableNeuralNetwork,
    data_importer: Box<dyn DataImporter>,
}

impl TrainingSession {
    /// Creates a new `TrainingSession` with the given parameters, data importer, model directory, and utilities.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided training parameters are invalid.
    pub fn new(
        params: TrainingParams,
        data_importer: Box<dyn DataImporter>,
        model_directory: &Directory,
        utils: WrappedUtils,
    ) -> Result<Self, Box<dyn Error>> {
        validate_params(&params)?;
        let shape = params.shape().clone();
        let pre_shape = params.pre_shape();
        let levels = params.levels();
        Ok(Self {
            params,
            neural_network: new_trainable_neural_network(NeuralNetworkCreationArguments::new(
                shape,
                levels,
                pre_shape,
                model_directory.path(),
                utils,
            )),
            data_importer,
        })
    }

    /// Creates a new `TrainingSession` from an existing neural network, training parameters, and data importer.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided training parameters are invalid.
    pub fn from_network(
        nn: WrappedTrainableNeuralNetwork,
        params: TrainingParams,
        data_importer: Box<dyn DataImporter>,
    ) -> Result<Self, Box<dyn Error>> {
        let mut changed_params = params;
        changed_params.set_shape(nn.shape());
        validate_params(&changed_params)?;
        Ok(Self { params: changed_params, neural_network: nn, data_importer })
    }

    /// Loads a model from disk and creates a training session.
    ///
    /// # Errors
    ///
    /// Returns an error if the model directory does not exist or if loading the neural network fails.
    pub fn from_disk(
        model_directory: String,
        params: TrainingParams,
        data_importer: Box<dyn DataImporter>,
        utils: WrappedUtils,
    ) -> Result<Self, Box<dyn Error>> {
        // if the directory does not esist, return an error
        if std::fs::metadata(model_directory.clone()).is_err() {
            return Err("Model directory does not exist".into());
        }
        let nn = trainable_neural_network_from_disk(model_directory, utils);
        Ok(Self { params, neural_network: nn, data_importer })
    }

    /// Train method
    ///
    /// # Errors
    ///
    /// This function will return an error if the training parameters are not valid.
    ///
    /// # Panics
    ///
    /// This function will panic if the neural network is not properly initialized.
    pub fn train(&mut self) -> Result<f64, Box<dyn Error>> {
        // Prepare the data
        let data = self.data_importer.get_data();
        let inputs = data.data;
        let targets = data.labels;

        let input_size = inputs[0].len();
        println!("Inputs: {} x {}", inputs.len(), inputs[0].len());
        println!("Targets: {} x {}", targets.len(), targets[0].len());

        // Prepare and validate the neural network
        let nn = &mut self.neural_network;
        if nn.input_size() != input_size {
            return Err("Input size mismatch with neural network".into());
        }
        if nn.output_size() != targets[0].len() {
            return Err("Output size mismatch with neural network".into());
        }

        println!("Training neural network with shape: {:?}", nn.shape());
        // Train the neural network
        nn.train(
            &inputs,
            &targets,
            self.params.learning_rate(),
            self.params.epochs(),
            self.params.tolerance(),
            self.params.use_adam(),
            self.params.validation_split(),
            self.params.sample_match_percentage(),
        );

        // Validation phase
        let mut success_count = 0.0;
        let inputs_len_f64: f64 =
            NumCast::from(inputs.len()).expect("Failed to convert inputs.len() to f64");
        let num_training_samples: usize =
            NumCast::from((inputs_len_f64 * self.params.validation_split()).round())
                .expect("Failed to convert result to usize");
        let num_verification_samples = inputs.len() - num_training_samples;
        for i in 0..num_verification_samples {
            let sample_idx = num_training_samples + i;
            if sample_idx >= inputs.len() {
                return Err("Not enough verification samples".into());
            }

            let output = nn.predict(inputs[sample_idx].clone());
            let target = &targets[sample_idx];

            // Check if the output matches the target
            let mut nb_correct_outputs = 0;
            for (o, t) in output.iter().zip(target.iter()) {
                if (o - t).abs() < self.params.tolerance() {
                    nb_correct_outputs += 1;
                }
            }
            let num_correct_outputs_f64: f64 =
                NumCast::from(nb_correct_outputs).expect("Failed to convert to f64");
            let target_len_f64: f64 =
                NumCast::from(target.len()).expect("Failed to convert target.len() to f64");
            let match_percentage = num_correct_outputs_f64 / target_len_f64;
            if match_percentage >= self.params.sample_match_percentage() {
                success_count += 1.0;
            }
        }
        // Return the accuracy as the fraction of successful predictions
        let num_verification_samples_f64: f64 =
            NumCast::from(num_verification_samples).expect("Failed to convert to f64");
        let accuracy: f64 = NumCast::from(success_count / num_verification_samples_f64)
            .expect("Failed to convert to f64");
        Ok(accuracy)
    }

    /// Save the model to disk
    ///
    /// # Errors
    ///
    /// Returns an error if saving the neural network to disk fails.
    pub fn save_model(
        &mut self,
        model_directory: String,
    ) -> Result<(), Box<dyn Error>> {
        self.neural_network.save(model_directory)
    }

    /// get the resulting neural network
    #[must_use]
    pub fn get_nn(&self) -> WrappedTrainableNeuralNetwork {
        self.neural_network.clone()
    }
}

fn validate_params(params: &TrainingParams) -> Result<(), Box<dyn Error>> {
    if !(params.validation_split() >= 0.0 && params.validation_split() <= 1.0) {
        return Err("Number of training to verification ratio must be between 0.0 and 1.0".into());
    }
    if params.learning_rate() <= 0.0 {
        return Err("Learning rate must be positive".into());
    }
    if params.learning_rate() >= 1.0 {
        return Err("Learning rate must be less than 1".into());
    }
    if params.epochs() == 0 {
        return Err("Number of epochs must be positive".into());
    }
    if !params.shape().is_valid() {
        // put the shape in the error message
        return Err(format!("Invalid neural network shape: {:?}", params.shape()).into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::shape::NeuralNetworkShape;
    use crate::nn::shape::{ActivationData, ActivationType, LayerShape, LayerType};
    use crate::training::data_importer::{DataImporter, SessionData};
    use crate::utilities::util::Utils;

    // Mock DataImporter implementation for testing
    #[derive(Clone)]
    struct MockDataImporter {
        shape: NeuralNetworkShape,
    }

    impl MockDataImporter {
        const fn new(shape: NeuralNetworkShape) -> Self {
            Self { shape }
        }
    }

    impl DataImporter for MockDataImporter {
        fn get_data(&self) -> SessionData {
            let num_samples = 1000;
            let input_size = self.shape.layers[0].input_size(); // Input dimension (e.g., 28x28 image flattened)
            let num_classes = self.shape.layers[self.shape.layers.len() - 1].output_size(); // Number of output classes (e.g., for digit classification)

            // Initialize inputs and targets with zeros
            let data = vec![vec![0.0; input_size]; num_samples];
            let labels = vec![vec![0.0; num_classes]; num_samples];

            SessionData { data, labels }
        }
    }

    #[test]
    fn test_simple_neural_net() {
        // Define the neural network shape
        let nn_shape = NeuralNetworkShape {
            layers: vec![
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 128, output_size: 128 },
                    activation: ActivationData::new(ActivationType::ReLU),
                },
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 128, output_size: 64 },
                    activation: ActivationData::new(ActivationType::ReLU),
                },
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 64, output_size: 64 },
                    activation: ActivationData::new_softmax(2.0),
                },
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 64, output_size: 10 },
                    activation: ActivationData::new(ActivationType::Sigmoid),
                },
            ],
        };

        // Define training parameters
        let training_params =
            TrainingParams::new(nn_shape.clone(), None, None, 0.7, 0.01, 10, 0.1, 32, true, 1.0);

        // Create a training session using the mock data importer
        let data_importer = MockDataImporter::new(nn_shape);

        let utils = WrappedUtils::new(Utils::new(1_000_000_000, 4));

        let mut training_session = TrainingSession::new(
            training_params,
            Box::new(data_importer),
            &Directory::Internal("test_session_model".to_string()),
            utils,
        )
        .expect("Failed to create TrainingSession");

        // Train the neural network and check the success rate
        let success_rate = training_session.train().expect("Training failed");
        assert!(success_rate >= 0.9, "Expected success rate >= 0.9, got {success_rate}");
    }
}
