use neural::nn::directory::Directory;
use neural::nn::neuralnet::ClassicNeuralNetwork;
use neural::nn::nn_trait::NeuralNetwork;
use neural::nn::shape::NeuralNetworkShape;
use neural::nn::shape::{ActivationData, ActivationType, LayerShape, LayerType};
use neural::training::data_importer::{DataImporter, SessionData};
use neural::training::training_params::TrainingParams;
use neural::training::training_session::TrainingSession;
use neural::utilities::util::{Utils, WrappedUtils};

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

// Helper to generate unique workspace names for tests
static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn create_test_utils() -> WrappedUtils {
    let counter = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let workspace = format!("/tmp/test_workspace_{counter}");
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

fn train_model(
    model_directory: String,
    internal_model_directory: String,
    utils: WrappedUtils,
) -> TrainingSession {
    println!("train_model called with model_directory: {model_directory}");
    println!("train_model utils test_mode: {}", utils.is_test_mode());
    println!("train_model utils workspace: {}", utils.get_workspace());

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
        TrainingParams::new(nn_shape.clone(), None, None, 0.7, 0.01, 3, 0.1, 32, true, 1.0);

    // Create a training session using the mock data importer
    let data_importer = MockDataImporter::new(nn_shape);

    let training_session = TrainingSession::from_disk(
        model_directory.clone(),
        training_params.clone(),
        Box::new(data_importer.clone()),
        utils.clone(),
    );
    match training_session {
        Ok(mut training_session) => {
            // Train the neural network and check the success rate
            let success_rate = training_session.train().expect("Training failed");
            // print the success rate
            println!("Success rate: {success_rate}");
            training_session.save_model(model_directory.clone()).expect("Failed to save model");
            return training_session;
        },
        Err(e) => {
            println!("Failed to load model: {e}");
        },
    }

    let mut training_session = TrainingSession::new(
        training_params,
        Box::new(data_importer),
        &Directory::Internal(internal_model_directory),
        utils.clone(),
    )
    .expect("Failed to create TrainingSession");

    // Train the neural network and check the success rate
    let success_rate = training_session.train().expect("Training failed");
    // print the success rate
    println!("Success rate: {success_rate}");
    training_session.save_model(model_directory.clone()).expect("Failed to save model");
    training_session
}

#[test]
fn new_model_is_persisted() {
    // Arrange
    let model_directory = "tests/test_model_persistence_1".to_string();
    let utils = create_test_utils();

    // Act
    let _training_session = train_model(
        model_directory.clone(),
        "tests/test_model_persistence_1_internal".to_string(),
        utils.clone(),
    );

    // Assert
    // Check if the model directory exists (should be in workspace)
    let workspace = utils.get_workspace();
    let expected_path = format!("{workspace}/{model_directory}");
    assert!(std::path::Path::new(&expected_path).exists());

    // Clean up workspace
    cleanup_workspace(&utils);
}

#[test]
fn already_trained_model_is_loaded() {
    // Arrange
    let model_directory = "tests/test_model_persistence_2".to_string();
    let utils = create_test_utils();

    let _training_session_1 = train_model(
        model_directory.clone(),
        "tests/test_model_persistence_2_internal".to_string(),
        utils.clone(),
    );

    // Act
    let _training_session_2 = train_model(
        model_directory.clone(),
        "tests/test_model_persistence_3_internal".to_string(),
        utils.clone(),
    );

    // Assert
    // Check if the model directory exists (should be in workspace)
    let workspace = utils.get_workspace();
    let expected_path = format!("{workspace}/{model_directory}");
    assert!(std::path::Path::new(&expected_path).exists());
    // Check that backup directory is removed (should be in workspace)
    let backup_path = format!("{workspace}/{model_directory}_backup");
    assert!(!std::path::Path::new(&backup_path).exists());

    // Clean up workspace
    cleanup_workspace(&utils);
}

#[test]
fn trained_model_is_convertible_to_ordinary_model_and_back() {
    let model_directory = "tests/test_model_persistence_3".to_string();
    let new_model_directory = "tests/test_model_persistence_3_new".to_string();
    let utils = create_test_utils();

    // Arrange
    let _training_session = train_model(
        model_directory.clone(),
        "tests/test_model_persistence_3_internal".to_string(),
        utils.clone(),
    );

    let workspace = utils.get_workspace();
    let expected_model_path = format!("{workspace}/{model_directory}");

    let mut ordinary_model =
        ClassicNeuralNetwork::from_disk(expected_model_path, utils.clone()).unwrap();
    ordinary_model.allocate();
    // Act
    ordinary_model.save(new_model_directory.clone()).expect("Failed to save model");

    let expected_new_model_path = format!("{workspace}/{new_model_directory}");
    let mut trainable_ordinary_model =
        ClassicNeuralNetwork::from_disk(expected_new_model_path, utils.clone()).unwrap();

    trainable_ordinary_model.allocate();

    trainable_ordinary_model
        .save(new_model_directory.clone())
        .expect("Failed to save trainable model");

    // Assert
    // Check if the new model directory exists (should be in workspace)
    let workspace = utils.get_workspace();
    let expected_new_model_path = format!("{workspace}/{new_model_directory}");
    assert!(std::path::Path::new(&expected_new_model_path).exists());
    // Check that the backup directories are removed (should be in workspace)
    let model_backup_path = format!("{workspace}/{model_directory}_backup");
    assert!(!std::path::Path::new(&model_backup_path).exists());
    let new_model_backup_path = format!("{workspace}/{new_model_directory}_backup");
    assert!(!std::path::Path::new(&new_model_backup_path).exists());

    // Clean up workspace
    cleanup_workspace(&utils);
}
