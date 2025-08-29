use matrix::{mat::WrappedMatrix, sum_mat::SumMatrix};
use regret::provider::Provider;
use regret::provider::ProviderType;
use regret::provider::WrappedChildrenProvider;
use regret::provider::WrappedExpectedValueProvider;
use regret::provider::WrappedProvider;
use regret::provider::{ChildrenProvider, ExpectedValueProvider};
use regret::regret_node::RegretNode;
use regret::regret_node::WrappedRegret;
use regret::user_data::DecisionTrait;
use regret::user_data::WrappedDecision;

use num_traits::cast::NumCast;

pub trait StateTrait: Default + Clone + Eq + std::fmt::Debug {
    fn get_data_as_string(&self) -> String;
}

#[derive(Debug, Clone)]
struct MarkovUserData<State: StateTrait + 'static> {
    state: State,
    probability: f64,
}

impl<State: StateTrait> Default for MarkovUserData<State> {
    fn default() -> Self {
        Self { state: State::default(), probability: 0.0 }
    }
}

impl<State: StateTrait> MarkovUserData<State> {
    const fn new(state: State) -> Self {
        Self { state, probability: 0.0 }
    }

    fn get_state(&self) -> State {
        self.state.clone()
    }
}

impl<State: StateTrait> DecisionTrait for MarkovUserData<State> {
    fn get_probability(&self) -> f64 {
        self.probability
    }

    fn set_probability(
        &mut self,
        probability: f64,
    ) {
        self.probability = probability;
    }

    fn get_data_as_string(&self) -> String {
        format!("State: {:?}, Probability: {}", self.state.get_data_as_string(), self.probability)
    }
}

#[derive(Debug, Clone)]
struct MarkovChildrenProvider<State: StateTrait + 'static> {
    all_states: Vec<State>,
    expected_value_calc_func: fn(&State) -> f64,
    _marker: std::marker::PhantomData<State>,
}

impl<State: StateTrait> MarkovChildrenProvider<State> {
    fn new(
        all_states: Vec<State>,
        expected_value_calc_func: fn(&State) -> f64,
    ) -> Self {
        Self { all_states, expected_value_calc_func, _marker: std::marker::PhantomData }
    }
}

impl<State: StateTrait + 'static> ChildrenProvider<MarkovUserData<State>>
    for MarkovChildrenProvider<State>
{
    fn get_children(
        &self,
        parents_data: Vec<WrappedDecision<MarkovUserData<State>>>,
    ) -> Vec<WrappedRegret<MarkovUserData<State>>> {
        let all_states_len_f64: f64 = NumCast::from(self.all_states.len()).unwrap();
        let probability = 1.0 / all_states_len_f64;
        let min_probability = 1e-4;
        match parents_data.len().cmp(&2) {
            std::cmp::Ordering::Less => self
                .all_states
                .iter()
                .map(|state| {
                    let data = WrappedDecision::new(MarkovUserData::new(state.clone()));
                    let new_children_provider =
                        Box::new(Self::new(self.all_states.clone(), self.expected_value_calc_func));
                    let provider = Provider::new(
                        ProviderType::Children(WrappedChildrenProvider::new(new_children_provider)),
                        Some(data),
                    );
                    let node = RegretNode::new(
                        probability,
                        min_probability,
                        parents_data.clone(),
                        WrappedProvider::new(provider),
                        None,
                    );
                    WrappedRegret::new(node)
                })
                .collect(),
            std::cmp::Ordering::Equal => self
                .all_states
                .iter()
                .map(|state| {
                    let data = WrappedDecision::new(MarkovUserData::new(state.clone()));
                    let new_expected_value_provider =
                        Box::new(MarkovExpectedValueProvider::new(self.expected_value_calc_func));
                    let provider = Provider::new(
                        ProviderType::ExpectedValue(WrappedExpectedValueProvider::new(
                            new_expected_value_provider,
                        )),
                        Some(data),
                    );
                    let node = RegretNode::new(
                        probability,
                        min_probability,
                        parents_data.clone(),
                        WrappedProvider::new(provider),
                        None,
                    );
                    WrappedRegret::new(node)
                })
                .collect(),
            std::cmp::Ordering::Greater => {
                // Handle the case where there are multiple parents
                vec![]
            },
        }
    }
}

#[derive(Debug, Clone)]
struct MarkovExpectedValueProvider<State: StateTrait + 'static> {
    expected_value_calc_func: fn(&State) -> f64,
}

impl<State: StateTrait + 'static> MarkovExpectedValueProvider<State> {
    fn new(expected_value_calc_func: fn(&State) -> f64) -> Self {
        Self { expected_value_calc_func }
    }
}

impl<State: StateTrait> ExpectedValueProvider<MarkovUserData<State>>
    for MarkovExpectedValueProvider<State>
{
    fn get_expected_value(
        &self,
        parents_data: Vec<WrappedDecision<MarkovUserData<State>>>,
    ) -> f64 {
        let state_last = parents_data
            .last()
            .expect("Expected at least one parent data")
            .get_decision_data()
            .get_state();
        let state_prev = parents_data[parents_data.len() - 2].get_decision_data().get_state();
        let last_expected_value = (self.expected_value_calc_func)(&state_last);
        let prev_expected_value = (self.expected_value_calc_func)(&state_prev);
        last_expected_value - prev_expected_value
    }
}

#[derive(Clone)]
pub struct MarkovSolver<State: StateTrait + 'static> {
    transition_matrix: SumMatrix,
    states: Vec<State>,
    expected_value_calc_func: fn(&State) -> f64,
}

impl<State: StateTrait> MarkovSolver<State> {
    pub fn new(
        states: Vec<State>,
        expected_value_calc_func: fn(&State) -> f64,
    ) -> Self {
        let mat = WrappedMatrix::new(states.len(), states.len());
        Self { transition_matrix: SumMatrix::new(mat), states, expected_value_calc_func }
    }

    /// Solves the Markov process for the given number of iterations.
    ///
    /// # Panics
    ///
    /// This function will panic if the conversion from probability to `i64` fails,
    /// or if a state is not found in the list of states.
    pub fn solve(
        &mut self,
        iterations: usize,
    ) {
        let mut node = RegretNode::new(
            1.0,
            0.01,
            vec![],
            WrappedProvider::new(Provider::new(
                ProviderType::Children(WrappedChildrenProvider::new(Box::new(
                    MarkovChildrenProvider::new(self.states.clone(), self.expected_value_calc_func),
                ))),
                None,
            )),
            Some(1.0),
        );

        node.solve(iterations);

        println!("{}", node.get_data_as_string(0));

        // add average probabilities * 100000 to transformation matrix
        for first_child in &node.get_children() {
            for second_child in &first_child.get_children() {
                let probability: i64 =
                    NumCast::from(second_child.get_average_probability() * 100_000.0).unwrap();
                let state_index = self
                    .states
                    .iter()
                    .position(|s| {
                        s == &first_child.get_user_data().unwrap().get_decision_data().get_state()
                    })
                    .unwrap();
                let next_state_index = self
                    .states
                    .iter()
                    .position(|s| {
                        s == &second_child.get_user_data().unwrap().get_decision_data().get_state()
                    })
                    .unwrap();
                self.transition_matrix.set_val_unchecked(
                    state_index,
                    next_state_index,
                    probability,
                );
            }
        }
    }

    #[allow(dead_code)]
    const fn get_transition_matrix(&self) -> &SumMatrix {
        &self.transition_matrix
    }

    /// Get the transition probability from one state to another.
    ///
    /// # Panics
    ///
    /// This function will panic if the `from` or `to` probability can not be calculated
    pub fn get_transition_probability(
        &self,
        from: &State,
        to: &State,
    ) -> f64 {
        let from_index = self.states.iter().position(|s| s == from).unwrap();
        let to_index = self.states.iter().position(|s| s == to).unwrap();
        self.transition_matrix.get_ratio(from_index, to_index).unwrap_or(0.0)
    }
}

// add a tests module
#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    use rand_distr::{Distribution, Normal};

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    struct TestState {
        value: i32,
    }

    impl TestState {
        fn new(value: i32) -> Self {
            Self { value }
        }

        fn get_value(&self) -> i32 {
            self.value
        }
    }

    impl StateTrait for TestState {
        fn get_data_as_string(&self) -> String {
            self.value.to_string()
        }
    }

    #[test]
    fn test_get_transition_probability_goes_to_zero_state() {
        let state_1 = TestState::new(1);
        let state_2 = TestState::new(2);
        let states = vec![state_1, state_2];

        let expected_value_calc_func = |state: &TestState| match state.get_value() {
            1 => 1.0,
            2 => -1.0,
            _ => 0.0,
        };

        let mut solver = MarkovSolver::new(states, expected_value_calc_func);

        solver.solve(1000);

        let transition_mat = solver.get_transition_matrix();
        println!("Transition Matrix 0 to 0: {:?}", transition_mat.get_ratio(0, 0));
        println!("Transition Matrix 0 to 1: {:?}", transition_mat.get_ratio(0, 1));
        println!("Transition Matrix 1 to 0: {:?}", transition_mat.get_ratio(1, 0));
        println!("Transition Matrix 1 to 1: {:?}", transition_mat.get_ratio(1, 1));

        // assert the ratios with an epsilon of 1e-4
        let epsilon = 1e-4;
        assert!((transition_mat.get_ratio(0, 0).unwrap() - 1.0).abs() < epsilon);
        assert!((transition_mat.get_ratio(0, 1).unwrap() - 0.0).abs() < epsilon);
        assert!((transition_mat.get_ratio(1, 0).unwrap() - 1.0).abs() < epsilon);
        assert!((transition_mat.get_ratio(1, 1).unwrap() - 0.0).abs() < epsilon);
    }

    #[test]
    fn test_get_transition_probability_goes_to_both_states() {
        let state_1 = TestState::new(1);
        let state_2 = TestState::new(2);
        let states = vec![state_1, state_2];

        let expected_value_calc_func = |_state: &TestState| 1.0;

        let mut solver = MarkovSolver::new(states, expected_value_calc_func);

        solver.solve(1000);

        let transition_mat = solver.get_transition_matrix();

        // assert the ratios with an epsilon of 1e-4
        let epsilon = 1e-4;
        assert!((transition_mat.get_ratio(0, 0).unwrap() - 0.5).abs() < epsilon);
        assert!((transition_mat.get_ratio(0, 1).unwrap() - 0.5).abs() < epsilon);
        assert!((transition_mat.get_ratio(1, 0).unwrap() - 0.5).abs() < epsilon);
        assert!((transition_mat.get_ratio(1, 1).unwrap() - 0.5).abs() < epsilon);
    }

    #[test]
    fn test_get_transition_probability_goes_to_one_state() {
        let state_1 = TestState::new(1);
        let state_2 = TestState::new(2);
        let states = vec![state_1, state_2];

        let expected_value_calc_func = |state: &TestState| match state.get_value() {
            1 => -1.0,
            2 => 1.0,
            _ => 0.0,
        };

        let mut solver = MarkovSolver::new(states, expected_value_calc_func);

        solver.solve(1000);

        let transition_mat = solver.get_transition_matrix();

        // assert the ratios with an epsilon of 1e-4
        let epsilon = 1e-4;
        assert!((transition_mat.get_ratio(0, 0).unwrap() - 0.0).abs() < epsilon);
        assert!((transition_mat.get_ratio(0, 1).unwrap() - 1.0).abs() < epsilon);
        assert!((transition_mat.get_ratio(1, 0).unwrap() - 0.0).abs() < epsilon);
        assert!((transition_mat.get_ratio(1, 1).unwrap() - 1.0).abs() < epsilon);
    }

    #[test]
    fn test_get_transition_probability_approximates() {
        let state_1 = TestState::new(1);
        let state_2 = TestState::new(2);
        let states = vec![state_1, state_2];

        // in case of 1 return a random number normally distributed around average 1.0 with std dev 2.0
        // in case of 2 return a random number normally distributed around average 0.5 with std dev 4.0
        // Create a random number generator
        let expected_value_calc_func = |state: &TestState| {
            let mut rng = thread_rng();
            match state.get_value() {
                1 => {
                    // Create a normal distribution with mean = 1.0 and std dev = 2.0
                    let normal = Normal::new(1.0, 2.0).unwrap();
                    normal.sample(&mut rng)
                },
                2 => {
                    // Create a normal distribution with mean = 0.5 and std dev = 4.0
                    let normal = Normal::new(0.0, 4.0).unwrap();
                    normal.sample(&mut rng)
                },
                _ => 0.0,
            }
        };

        let mut solver = MarkovSolver::new(states, expected_value_calc_func);

        solver.solve(1000);

        let transition_mat = solver.get_transition_matrix();
        println!("Transition Matrix 0 to 0: {:?}", transition_mat.get_ratio(0, 0));
        println!("Transition Matrix 0 to 1: {:?}", transition_mat.get_ratio(0, 1));
        println!("Transition Matrix 1 to 0: {:?}", transition_mat.get_ratio(1, 0));
        println!("Transition Matrix 1 to 1: {:?}", transition_mat.get_ratio(1, 1));

        // assert the ratios with an epsilon of 1e-1
        let epsilon = 1e-1;
        assert!((transition_mat.get_ratio(0, 0).unwrap() - 0.536).abs() < epsilon);
        assert!((transition_mat.get_ratio(0, 1).unwrap() - 0.464).abs() < epsilon);
        assert!((transition_mat.get_ratio(1, 0).unwrap() - 0.556).abs() < epsilon);
        assert!((transition_mat.get_ratio(1, 1).unwrap() - 0.444).abs() < epsilon);
    }

    #[test]
    fn test_get_transition_probability_approximates_2() {
        let state_1 = TestState::new(1);
        let state_2 = TestState::new(2);
        let states = vec![state_1, state_2];

        // in case of 1 return a random number normally distributed around average 1.0 with std dev 2.0
        // in case of 2 return a random number normally distributed around average 0.5 with std dev 4.0
        // Create a random number generator
        let expected_value_calc_func = |state: &TestState| {
            let mut rng = thread_rng();
            match state.get_value() {
                1 => {
                    // Create a normal distribution with mean = 1.0 and std dev = 2.0
                    let normal = Normal::new(1.0, 2.0).unwrap();
                    normal.sample(&mut rng)
                },
                2 => {
                    // Create a normal distribution with mean = 0.0 and std dev = 2.0
                    let normal = Normal::new(0.0, 2.0).unwrap();
                    normal.sample(&mut rng)
                },
                _ => 0.0,
            }
        };

        let mut solver = MarkovSolver::new(states, expected_value_calc_func);

        solver.solve(1000);

        let transition_mat = solver.get_transition_matrix();
        println!("Transition Matrix 0 to 0: {:?}", transition_mat.get_ratio(0, 0));
        println!("Transition Matrix 0 to 1: {:?}", transition_mat.get_ratio(0, 1));
        println!("Transition Matrix 1 to 0: {:?}", transition_mat.get_ratio(1, 0));
        println!("Transition Matrix 1 to 1: {:?}", transition_mat.get_ratio(1, 1));

        // assert the ratios with an epsilon of 1e-1
        let epsilon = 1e-1;
        assert!((transition_mat.get_ratio(0, 0).unwrap() - 0.6).abs() < epsilon);
        assert!((transition_mat.get_ratio(0, 1).unwrap() - 0.4).abs() < epsilon);
        assert!((transition_mat.get_ratio(1, 0).unwrap() - 0.6).abs() < epsilon);
        assert!((transition_mat.get_ratio(1, 1).unwrap() - 0.4).abs() < epsilon);
    }
}
