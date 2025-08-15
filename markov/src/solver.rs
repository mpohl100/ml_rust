use matrix::{mat::WrappedMatrix, sum_mat::SumMatrix};
use regret::solver_node::Provider;
use regret::solver_node::ProviderType;
use regret::solver_node::WrappedChildrenProvider;
use regret::solver_node::WrappedExpectedValueProvider;
use regret::solver_node::WrappedProvider;
use regret::solver_node::WrappedRegret;
use regret::solver_node::{ChildrenProvider, ExpectedValueProvider, RegretNode, UserDataTrait};

use num_traits::cast::NumCast;

pub trait StateTrait: Default + Clone + Eq {
    fn get_data_as_string(&self) -> String;
}

#[derive(Clone)]
struct MarkovUserData<State: StateTrait + 'static> {
    state: State,
    all_states: Vec<State>,
    expected_value_calc_func: fn(&State) -> f64,
    probability: f64,
}

impl<State: StateTrait> Default for MarkovUserData<State> {
    fn default() -> Self {
        Self {
            state: State::default(),
            all_states: Vec::new(),
            expected_value_calc_func: |_| 0.0,
            probability: 0.0,
        }
    }
}

impl<State: StateTrait> MarkovUserData<State> {
    fn new(
        state: State,
        all_states: Vec<State>,
        expected_value_calc_func: fn(&State) -> f64,
    ) -> Self {
        Self { state, all_states, expected_value_calc_func, probability: 0.0 }
    }

    fn get_state(&self) -> State {
        self.state.clone()
    }

    fn get_expected_value(&self) -> f64 {
        (self.expected_value_calc_func)(&self.state)
    }

    fn get_all_states(&self) -> Vec<State> {
        self.all_states.clone()
    }
}

impl<State: StateTrait> UserDataTrait for MarkovUserData<State> {
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

#[derive(Clone)]
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
        parents_data: Vec<MarkovUserData<State>>,
    ) -> Vec<WrappedRegret<MarkovUserData<State>>> {
        let all_states_len_f64: f64 = NumCast::from(self.all_states.len()).unwrap();
        let probability = 1.0 / all_states_len_f64;
        let min_probability = 1e-4;
        match parents_data.len().cmp(&1) {
            std::cmp::Ordering::Less => self
                .all_states
                .iter()
                .map(|state| {
                    let data = MarkovUserData::new(
                        state.clone(),
                        self.all_states.clone(),
                        self.expected_value_calc_func,
                    );
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
                    let data = MarkovUserData::new(
                        state.clone(),
                        self.all_states.clone(),
                        self.expected_value_calc_func,
                    );
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
        parents_data: Vec<MarkovUserData<State>>,
    ) -> f64 {
        let state = parents_data.last().expect("Expected at least one parent data").get_state();
        (self.expected_value_calc_func)(&state)
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

    pub fn solve(&self) {
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

        node.solve(1000);

        // TODO add average probabilities * 100000 to transformation matrix
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
