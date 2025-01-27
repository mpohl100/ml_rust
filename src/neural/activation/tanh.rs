use crate::neural::nn::shape::ActivationData;
use crate::neural::nn::shape::ActivationType;

use super::activate::ActivationTrait;

/// Tanh activation function.
#[derive(Debug, Clone)]
pub struct Tanh;

impl Tanh {
    /// Creates a new Tanh instance.
    pub fn new() -> Self {
        Self
    }

    fn tanh_vec(&self, input: &[f64]) -> Vec<f64> {
        input.iter().map(|&x| x.tanh()).collect()
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl ActivationTrait for Tanh {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        self.tanh_vec(input)
    }

    fn backward(&mut self, grad_output: &[f64]) -> Vec<f64> {
        grad_output
            .iter()
            .zip(self.tanh_vec(grad_output).iter())
            .map(|(&grad, &output)| grad * (1.0 - output.powi(2)))
            .collect()
    }

    fn get_activation_data(&self) -> ActivationData {
        ActivationData::new(ActivationType::Tanh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tanh() {
        let mut tanh = Tanh;
        let input = vec![0.0];
        let output = tanh.forward(&input);
        // print output
        println!("{:?}", output);
        assert!((output[0] - 0.0).abs() < 1e-7);

        let grad_output = vec![1.0];
        let grad_input = tanh.backward(&grad_output);
        // print grad_input
        println!("{:?}", grad_input);
        assert!((grad_input[0] - 0.4199743).abs() < 1e-7);
    }
}
