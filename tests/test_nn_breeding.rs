use learn::evol::rng::RandomNumberGenerator;
use learn::neural::nn::neuralnet::TrainableNeuralNetwork;
use learn::neural::nn::shape::NeuralNetworkShape;
use learn::neural::nn::shape::{ActivationData, ActivationType, LayerShape, LayerType};

use learn::gen::pheno::nn_pheno::NeuralNetworkPhenotype;
use learn::gen::strategy::nn_strategy::NeuralNetworkStrategy;

use learn::evol::evolution::EvolutionOptions;
use learn::evol::evolution::LogLevel;
use learn::evol::strategy::BreedStrategy;

#[test]
fn test_neural_network_breeding() {
    // Define the neural network shape
    let nn_shape = NeuralNetworkShape {
        layers: vec![
            LayerShape {
                layer_type: LayerType::Dense {
                    input_size: 128,
                    output_size: 128,
                },
                activation: ActivationData::new(ActivationType::ReLU),
            },
            LayerShape {
                layer_type: LayerType::Dense {
                    input_size: 128,
                    output_size: 64,
                },
                activation: ActivationData::new(ActivationType::ReLU),
            },
            LayerShape {
                layer_type: LayerType::Dense {
                    input_size: 64,
                    output_size: 10,
                },
                activation: ActivationData::new(ActivationType::Sigmoid),
            },
        ],
    };

    // Create a neural network phenotype
    let nn = TrainableNeuralNetwork::new(nn_shape);
    let nn_phenotype = NeuralNetworkPhenotype::new(nn);
    let mut parents = vec![nn_phenotype];

    let evol_opts = EvolutionOptions::new(100, LogLevel::None, 4, 10);

    let mut rng = RandomNumberGenerator::new();

    let model_directory = "model".to_owned();

    let nn_strategy = NeuralNetworkStrategy::new(model_directory.clone());

    for _ in 0..20 {
        let children = nn_strategy
            .breed(&parents, &evol_opts, &mut rng)
            .expect("Breed failed");
        assert_eq!(children.len(), 10);
        assert!(children
            .iter()
            .all(|child| child.get_nn().shape().is_valid()));
        parents.clear();
        for child in children.iter().take(4) {
            parents.push(child.clone());
        }
    }

    // Remove model directory
    std::fs::remove_dir_all(model_directory).expect("Failed to remove model directory");
}
