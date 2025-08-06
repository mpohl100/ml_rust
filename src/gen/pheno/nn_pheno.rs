use super::rng_wrapper::RngWrapper;
use super::{
    nn_mutater::fetch_activation_data, nn_mutater::NeuralNetworkMutater, rng_wrapper::RealRng,
};
use crate::evol::rng::RandomNumberGenerator;
use crate::evol::strategy::Adjust;
use crate::neural::nn::nn_factory::{new_trainable_neural_network, NeuralNetworkCreationArguments};
use crate::neural::nn::shape::NeuralNetworkShape;
use crate::{evol::phenotype::Phenotype, neural::nn::nn_trait::WrappedTrainableNeuralNetwork};

#[derive(Debug)]
pub struct NeuralNetworkPhenotype {
    nn: WrappedTrainableNeuralNetwork,
    left_half_shape: Option<NeuralNetworkShape>,
    right_half_shape: Option<NeuralNetworkShape>,
    nb_mutates: usize,
}

impl Clone for NeuralNetworkPhenotype {
    fn clone(&self) -> Self {
        Self {
            nn: self.get_nn().clone(),
            left_half_shape: self.left_half_shape.clone(),
            right_half_shape: self.right_half_shape.clone(),
            nb_mutates: self.nb_mutates,
        }
    }
}

impl NeuralNetworkPhenotype {
    pub fn new(nn: WrappedTrainableNeuralNetwork) -> Self {
        Self {
            nn: nn.duplicate_trainable(),
            left_half_shape: None,
            right_half_shape: None,
            nb_mutates: 0,
        }
    }

    pub fn get_nn(&self) -> WrappedTrainableNeuralNetwork {
        self.nn.clone()
    }

    pub fn set_nn(
        &mut self,
        nn: WrappedTrainableNeuralNetwork,
    ) {
        self.nn = nn.clone();
    }

    fn set_left_half_shape(
        &mut self,
        shape: NeuralNetworkShape,
    ) {
        self.left_half_shape = Some(shape);
    }

    fn set_right_half_shape(
        &mut self,
        shape: NeuralNetworkShape,
    ) {
        self.right_half_shape = Some(shape);
    }

    fn reset_half_shapes(&mut self) {
        self.left_half_shape = None;
        self.right_half_shape = None;
    }

    pub fn allocate(&mut self) {
        self.nn.allocate();
    }

    fn mutate_classic_nn(
        &mut self,
        rng_wrapper: &mut RealRng,
    ) {
        let left_half_shape = self.left_half_shape.clone();
        let right_half_shape = self.right_half_shape.clone();
        let previous_shape = if left_half_shape.is_some() && right_half_shape.is_some() {
            let left_shape = left_half_shape.unwrap();
            let right_shape = right_half_shape.unwrap();
            left_shape.merge(right_shape, fetch_activation_data(rng_wrapper))
        } else {
            self.get_nn().shape().clone()
        };

        let mut mutater = NeuralNetworkMutater::new(rng_wrapper);

        let mut mutated_shape = mutater.mutate_shape(previous_shape.clone());
        let mut i = 0;
        while mutated_shape.to_neural_network_shape() == previous_shape {
            mutated_shape = mutater.mutate_shape(previous_shape.clone());
            i += 1;
            if i > 10 {
                break;
            }
        }
        let nn = new_trainable_neural_network(NeuralNetworkCreationArguments::new(
            self.get_nn().shape().clone(),
            None,
            None,
            self.nn.get_model_directory().path(),
            self.nn.get_utils(),
        ));
        self.set_nn(nn);
        self.reset_half_shapes();
    }

    fn mutate_levels(
        &mut self,
        rng_wrapper: &mut RealRng,
    ) {
        let left_half_shape = self.left_half_shape.clone();
        let right_half_shape = self.right_half_shape.clone();
        let previous_shape = if left_half_shape.is_some() && right_half_shape.is_some() {
            let left_shape = left_half_shape.unwrap();
            let right_shape = right_half_shape.unwrap();
            left_shape.merge(right_shape, fetch_activation_data(rng_wrapper))
        } else {
            self.get_nn().shape().clone()
        };

        let random_number = rng_wrapper.fetch_uniform(1.0, 5.0, 1);
        // round do to integer
        let random_number = random_number[0].round() as i32;
        let nn = new_trainable_neural_network(NeuralNetworkCreationArguments::new(
            previous_shape.clone(),
            Some(random_number),
            None,
            self.nn.get_model_directory().path(),
            self.nn.get_utils(),
        ));
        self.set_nn(nn);
        self.reset_half_shapes();
    }
}

impl Phenotype for NeuralNetworkPhenotype {
    fn crossover(
        &mut self,
        other: &Self,
    ) {
        let left_original_nn = self.get_nn();
        let right_original_nn = other.get_nn();
        let left_index_begin = 0;
        let mut left_index_end = left_original_nn.shape().num_layers() / 2;
        if left_index_end == 0 {
            left_index_end = 1;
        }
        let right_index_begin = right_original_nn.shape().num_layers() / 2;
        let mut right_index_end = right_original_nn.shape().num_layers();
        if right_index_end == right_index_begin {
            right_index_end += 1;
        }
        let left_half_shape =
            left_original_nn.shape().clone().cut_out(left_index_begin, left_index_end);
        let right_half_shape =
            right_original_nn.shape().clone().cut_out(right_index_begin, right_index_end);
        self.set_left_half_shape(left_half_shape);
        self.set_right_half_shape(right_half_shape);
    }

    fn mutate(
        &mut self,
        rng: &mut RandomNumberGenerator,
    ) {
        let mut rng_wrapper = RealRng::new(rng);
        // fetch a random number between 0 and 1
        let random_number = rng_wrapper.fetch_uniform(0.0, 10.0, 1);
        if random_number[0] < 5.0 {
            self.mutate_classic_nn(&mut rng_wrapper);
        } else {
            self.mutate_levels(&mut rng_wrapper);
        }
    }
}

impl Adjust<NeuralNetworkPhenotype> for NeuralNetworkPhenotype {
    fn incr_number_mutates(&mut self) -> usize {
        self.nb_mutates += 1;
        self.nb_mutates
    }

    fn decr_number_mutates(&mut self) -> usize {
        if self.nb_mutates > 0 {
            self.nb_mutates -= 1;
        }
        self.nb_mutates
    }

    fn get_number_mutates(&self) -> usize {
        self.nb_mutates
    }
}
