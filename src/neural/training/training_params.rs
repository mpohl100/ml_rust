use crate::neural::nn::shape::NeuralNetworkShape;

#[derive(Clone)]
pub struct TrainingParams {
    shape: NeuralNetworkShape,
    validation_split: f64,
    learning_rate: f64,
    epochs: usize,
    tolerance: f64,
    batch_size: usize,
    use_adam: bool,
}

impl TrainingParams {
    pub fn new(
        shape: NeuralNetworkShape,
        validation_split: f64,
        learning_rate: f64,
        epochs: usize,
        tolerance: f64,
        batch_size: usize,
        use_adam: bool,
    ) -> Self {
        Self {
            shape,
            validation_split,
            learning_rate,
            epochs,
            tolerance,
            batch_size,
            use_adam,
        }
    }

    pub fn shape(&self) -> &NeuralNetworkShape {
        &self.shape
    }

    pub fn validation_split(&self) -> f64 {
        self.validation_split
    }

    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    pub fn epochs(&self) -> usize {
        self.epochs
    }

    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn use_adam(&self) -> bool {
        self.use_adam
    }

    pub fn set_shape(&mut self, shape: NeuralNetworkShape) {
        self.shape = shape;
    }
}
