use crate::evol::evolution::EvolutionOptions;
use crate::evol::rng::RandomNumberGenerator;
use crate::evol::strategy::AdjustStrategy;
use crate::evol::strategy::BreedStrategy;
use crate::gen::pheno::nn_pheno::NeuralNetworkPhenotype;

use std::fmt::Error;

#[derive(Debug, Clone)]
pub struct NeuralNetworkStrategy {
    model_directory: String,
}

impl NeuralNetworkStrategy {
    pub fn new(model_directory: String) -> Self {
        Self { model_directory }
    }
}

impl BreedStrategy<NeuralNetworkPhenotype> for NeuralNetworkStrategy {
    fn breed(
        &self,
        parents: &[NeuralNetworkPhenotype],
        evol_options: &EvolutionOptions,
        rng: &mut RandomNumberGenerator,
    ) -> Result<Vec<NeuralNetworkPhenotype>, Error> {
        let adjust_strategy = AdjustStrategy::default();
        let mut nn = parents[0].get_nn();
        println!(
            "Saving model to: {} with shape: {:?}",
            self.model_directory,
            nn.shape()
        );
        let _ = nn.save(self.model_directory.clone());
        adjust_strategy.breed(parents, evol_options, rng)
    }
}
