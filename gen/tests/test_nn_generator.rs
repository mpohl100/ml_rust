use evol::evolution::{EvolutionOptions, LogLevel};
use gen::neuralnet_gen::NeuralNetworkGenerator;
use neural::nn::shape::NeuralNetworkShape;
use neural::nn::shape::{ActivationData, ActivationType, LayerShape, LayerType};
use neural::training::data_importer::{DataImporter, SessionData};
use neural::training::training_params::TrainingParams;
use neural::utilities::util::{Utils, WrappedUtils};

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

// Helper to generate unique workspace names for tests
static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn create_test_utils() -> WrappedUtils {
    let counter = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let workspace = format!("/tmp/test_workspace_gen_{counter}");
    WrappedUtils::new(Utils::new_with_test_mode(1000000000, 4, workspace))
}

fn cleanup_workspace(utils: &WrappedUtils) {
    let workspace = utils.get_workspace();
    if !workspace.is_empty() && Path::new(&workspace).exists() {
        let _ = std::fs::remove_dir_all(&workspace);
    }
}

// Mock DataImporter implementation for testing
#[derive(Clone)]
struct MockDataImporter {
    shape: NeuralNetworkShape,
}

impl MockDataImporter {
    fn new(shape: NeuralNetworkShape) -> Self {
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
fn test_neural_network_generator() {
    let model_directory = "tests/test_model_generation".to_string();
    let utils = create_test_utils();

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
                layer_type: LayerType::Dense { input_size: 64, output_size: 10 },
                activation: ActivationData::new(ActivationType::Sigmoid),
            },
        ],
    };

    let training_params =
        TrainingParams::new(nn_shape.clone(), None, None, 0.7, 0.01, 2, 0.1, 32, true, 1.0);

    let data_importer = MockDataImporter::new(nn_shape.clone());

    let evolution_params = EvolutionOptions::new(2, LogLevel::Verbose, 3, 4);

    let mut nn_generator = NeuralNetworkGenerator::new(
        training_params,
        evolution_params,
        Box::new(data_importer),
        model_directory.clone(),
        4,
        utils.clone(),
    );
    nn_generator.generate();
    nn_generator.save();

    // Clean up workspace
    cleanup_workspace(&utils);
}
