use crate::nn::shape::NeuralNetworkShape;

#[derive(Clone)]
pub struct TrainingParams {
    shape: NeuralNetworkShape,
    levels: Option<i32>,
    pre_shape: Option<NeuralNetworkShape>,
    validation_split: f64,
    learning_rate: f64,
    epochs: usize,
    tolerance: f64,
    batch_size: usize,
    use_adam: bool,
    sample_match_percentage: f64,
}

impl TrainingParams {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub const fn new(
        shape: NeuralNetworkShape,
        levels: Option<i32>,
        pre_shape: Option<NeuralNetworkShape>,
        validation_split: f64,
        learning_rate: f64,
        epochs: usize,
        tolerance: f64,
        batch_size: usize,
        use_adam: bool,
        sample_match_percentage: f64,
    ) -> Self {
        Self {
            shape,
            levels,
            pre_shape,
            validation_split,
            learning_rate,
            epochs,
            tolerance,
            batch_size,
            use_adam,
            sample_match_percentage,
        }
    }

    #[must_use]
    pub const fn shape(&self) -> &NeuralNetworkShape {
        &self.shape
    }

    #[must_use]
    pub const fn levels(&self) -> Option<i32> {
        self.levels
    }

    #[must_use]
    pub fn pre_shape(&self) -> Option<NeuralNetworkShape> {
        self.pre_shape.clone()
    }

    #[must_use]
    pub const fn validation_split(&self) -> f64 {
        self.validation_split
    }

    #[must_use]
    pub const fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    #[must_use]
    pub const fn epochs(&self) -> usize {
        self.epochs
    }

    #[must_use]
    pub const fn tolerance(&self) -> f64 {
        self.tolerance
    }

    #[must_use]
    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }

    #[must_use]
    pub const fn use_adam(&self) -> bool {
        self.use_adam
    }

    #[must_use]
    pub const fn sample_match_percentage(&self) -> f64 {
        self.sample_match_percentage
    }

    pub fn set_shape(
        &mut self,
        shape: NeuralNetworkShape,
    ) {
        self.shape = shape;
    }
}
