use std::sync::{Arc, Mutex};

use utils::safer::safe_lock;

pub trait UserDataTrait: Default + Clone {
    // Define the methods that UserData should implement
    fn get_probability(&self) -> f64;
    fn set_probability(
        &mut self,
        probability: f64,
    );
    fn get_data_as_string(&self) -> String;
}

pub trait ChildrenProvider<UserData: UserDataTrait> {
    fn get_children(
        &self,
        parents_data: Vec<UserData>,
    ) -> Vec<WrappedRegret<UserData>>;
}

#[derive(Clone)]
pub struct WrappedChildrenProvider<UserData: UserDataTrait> {
    provider: Arc<Mutex<Box<dyn ChildrenProvider<UserData>>>>,
}

impl<UserData: UserDataTrait> WrappedChildrenProvider<UserData> {
    #[must_use]
    pub fn new(provider: Box<dyn ChildrenProvider<UserData>>) -> Self {
        Self { provider: Arc::new(Mutex::new(provider)) }
    }

    #[must_use]
    pub fn get_children(
        &self,
        parents_data: Vec<UserData>,
    ) -> Vec<WrappedRegret<UserData>> {
        safe_lock(&self.provider).get_children(parents_data)
    }
}

pub trait ExpectedValueProvider<UserData: UserDataTrait> {
    fn get_expected_value(
        &self,
        parents_data: Vec<UserData>,
    ) -> f64;
}

#[derive(Clone)]
pub struct WrappedExpectedValueProvider<UserData: UserDataTrait> {
    provider: Arc<Mutex<Box<dyn ExpectedValueProvider<UserData>>>>,
}

impl<UserData: UserDataTrait> WrappedExpectedValueProvider<UserData> {
    #[must_use]
    pub fn new(provider: Box<dyn ExpectedValueProvider<UserData>>) -> Self {
        Self { provider: Arc::new(Mutex::new(provider)) }
    }

    #[must_use]
    pub fn get_expected_value(
        &self,
        parents_data: Vec<UserData>,
    ) -> f64 {
        safe_lock(&self.provider).get_expected_value(parents_data)
    }
}

#[derive(Clone)]
pub enum ProviderType<UserData: UserDataTrait> {
    Children(WrappedChildrenProvider<UserData>),
    ExpectedValue(WrappedExpectedValueProvider<UserData>),
}

#[derive(Clone)]
pub struct Provider<UserData: UserDataTrait> {
    pub provider_type: ProviderType<UserData>,
    pub user_data: Option<UserData>,
}

impl<UserData: UserDataTrait> Provider<UserData> {
    pub fn new(
        provider_type: ProviderType<UserData>,
        user_data: Option<UserData>,
    ) -> Self {
        Self { provider_type, user_data }
    }
}

#[derive(Clone)]
pub struct WrappedProvider<UserData: UserDataTrait> {
    provider: Arc<Mutex<Provider<UserData>>>,
}

impl<UserData: UserDataTrait> WrappedProvider<UserData> {
    #[must_use]
    pub fn new(provider: Provider<UserData>) -> Self {
        Self { provider: Arc::new(Mutex::new(provider)) }
    }

    pub fn get_provider_type(&self) -> ProviderType<UserData> {
        safe_lock(&self.provider).provider_type.clone()
    }

    pub fn get_user_data(&self) -> Option<UserData> {
        safe_lock(&self.provider).user_data.clone()
    }
}

#[derive(Clone)]
pub struct RegretNode<UserData: UserDataTrait> {
    probability: f64,
    min_probability: f64,
    current_expected_value: f64,
    parents_data: Vec<UserData>,
    provider: WrappedProvider<UserData>,
    children: Vec<WrappedRegret<UserData>>,
    regret: f64,
    sum_probabilities: f64,
    num_probabilities: f64,
    average_probability: f64,
    sum_expected_values: f64,
    num_expected_values: f64,
    average_expected_value: f64,
    fixed_probability: Option<f64>,
}

impl<UserData: UserDataTrait> RegretNode<UserData> {
    pub fn new(
        probability: f64,
        min_probability: f64,
        parents_data: Vec<UserData>,
        provider: WrappedProvider<UserData>,
        fixed_probability: Option<f64>,
    ) -> Self {
        provider.get_user_data().as_mut().map_or_else(
            || {
                // If no user data is provided, we set the probability to 0.0
                UserData::set_probability(
                    &mut UserData::default(),
                    fixed_probability.unwrap_or(1.0),
                );
            },
            |data| data.set_probability(probability),
        );
        Self {
            probability,
            min_probability,
            current_expected_value: 0.0,
            parents_data,
            children: Vec::new(),
            provider: provider.clone(),
            regret: 0.0,
            sum_probabilities: 0.0,
            num_probabilities: 0.0,
            average_probability: 0.0,
            sum_expected_values: 0.0,
            num_expected_values: 0.0,
            average_expected_value: 0.0,
            fixed_probability,
        }
    }

    #[allow(clippy::format_push_string)]
    pub fn get_data_as_string(
        &self,
        indentation: usize,
    ) -> String {
        // first put average probability and average expected value
        let mut result = format!(
            "{}Average Probability: {}\n",
            " ".repeat(indentation),
            self.average_probability
        );
        result.push_str(&format!(
            "{}Average Expected Value: {}\n",
            " ".repeat(indentation),
            self.average_expected_value
        ));
        // then push the provider user_data.get_data_as_string
        if let Some(data) = self.provider.get_user_data().as_ref() {
            // put indentation and newline
            result.push_str(&format!("{}{}\n", " ".repeat(indentation), data.get_data_as_string()));
        }
        // then put all children data
        // put "children" as caption
        result.push_str(&format!("{}Children:\n", " ".repeat(indentation)));
        for child in &self.children {
            result.push_str(&format!("{}\n", child.get_data_as_string(indentation + 2)));
        }
        result
    }

    pub fn solve(
        &mut self,
        num_iterations: usize,
    ) {
        self.populate_children();
        for _ in 0..num_iterations {
            // Implement the regret minimization algorithm here
            self.calculate_expected_value();
            self.calculate_regrets(self.current_expected_value);
            let sum_regrets = self.get_sum_regrets();
            self.calculate_probabilities(sum_regrets, self.children.len());
            self.calculate_normalized_probabilities(1.0);
            self.update_average_values();
        }
    }

    fn update_average_values(&mut self) {
        // Update the average values based on the current probabilities and expected values
        // check if probability is less than epsilon or nan and the nset to zero
        if self.probability < f64::EPSILON || self.probability.is_nan() {
            self.probability = 0.0;
        }
        let prob = self.get_probability();
        self.sum_probabilities += prob;
        self.num_probabilities += 1.0;
        self.average_probability = self.sum_probabilities / self.num_probabilities;

        if self.current_expected_value.is_nan() || self.current_expected_value < f64::EPSILON {
            self.current_expected_value = 0.0;
        }
        self.sum_expected_values += self.current_expected_value;
        self.num_expected_values += 1.0;
        self.average_expected_value = self.sum_expected_values / self.num_expected_values;

        // update average values of children
        for child in &mut self.children {
            child.update_average_values();
        }
    }

    fn calculate_normalized_probabilities(
        &mut self,
        total_probability: f64,
    ) {
        if total_probability > 0.0 {
            self.probability /= total_probability;
        } else {
            self.probability = 0.0;
        }
        if let Some(data) = self.provider.get_user_data().as_mut() {
            data.set_probability(self.probability);
        }
        let total_probability_sum: f64 =
            self.children.iter().map(WrappedRegret::get_probability).sum();
        for child in &mut self.children {
            child.calculate_normalized_probabilities(total_probability_sum);
        }
    }

    fn calculate_probabilities(
        &mut self,
        sum_regrets: f64,
        total_siblings: usize,
    ) {
        if self.fixed_probability.is_none() {
            if self.regret < 0.0 {
                self.add_probability(0.0);
            } else if sum_regrets <= 0.0 {
                if total_siblings == 0 {
                    self.add_probability(1.0);
                } else {
                    self.add_probability(
                        1.0 / f64::from(usize::try_into(total_siblings).unwrap_or(1)),
                    );
                }
            } else {
                let probability = self.regret / sum_regrets;
                self.add_probability(probability);
            }
        } else {
            // If it's a fixed node, we do not calculate probabilities
            self.add_probability(self.fixed_probability.unwrap_or(0.0));
        }

        let new_sum_regrets = self.get_sum_regrets();
        let total_siblings = self.children.len();
        self.children.iter_mut().for_each(|child| {
            child.calculate_probabilities(new_sum_regrets, total_siblings);
        });
    }

    fn add_probability(
        &mut self,
        probability: f64,
    ) {
        self.probability += probability;
    }

    pub fn calculate_regrets(
        &mut self,
        outer_expected_value: f64,
    ) {
        // Calculate regrets based on the expected value
        self.regret = outer_expected_value - self.current_expected_value;
        self.probability = 0.0; // Reset probability for next iteration
        self.children.iter_mut().for_each(|child| {
            child.calculate_regrets(self.current_expected_value);
        });
    }

    fn calculate_expected_value(&mut self) -> f64 {
        // Calculate the expected value based on the current probabilities
        if self.get_total_probability() < self.min_probability {
            self.current_expected_value = 0.0;
            return 0.0;
        }

        self.current_expected_value = match self.provider.get_provider_type() {
            ProviderType::ExpectedValue(ref provider) => {
                let mut cloned_parents_data = self.parents_data.clone();
                cloned_parents_data.extend(
                    self.provider
                        .get_user_data()
                        .as_ref()
                        .map_or_else(|| Vec::new(), |data| vec![data.clone()]),
                );
                provider.get_expected_value(cloned_parents_data)
            },
            ProviderType::Children(ref _provider) => self
                .children
                .iter()
                .map(|child| child.get_expected_value() * child.get_probability())
                .sum::<f64>(),
        };
        self.current_expected_value
    }

    fn get_sum_regrets(&self) -> f64 {
        self.children.iter().map(|child| child.node.lock().unwrap().regret).sum()
    }

    fn get_total_probability(&self) -> f64 {
        if self.parents_data.is_empty() {
            return self.probability; // If no parents, return the node's own probability
        }
        let parent_probability =
            self.parents_data.iter().fold(1.0, |acc, data| acc * data.get_probability());
        parent_probability * self.probability
    }

    #[allow(dead_code)]
    fn get_children(&self) -> Vec<WrappedRegret<UserData>> {
        self.children.clone()
    }

    fn populate_children(&mut self) {
        if !self.children.is_empty() {
            return; // No children to populate
        }
        match self.provider.get_provider_type() {
            ProviderType::Children(ref provider) => {
                let mut cloned_parents_data = self.parents_data.clone();
                if let Some(ref user_data) = self.provider.get_user_data() {
                    cloned_parents_data.push(user_data.clone());
                } else {
                    // If no user data is provided, we just use the parents data
                }
                self.children = provider.get_children(cloned_parents_data);
                // populate children of children
                self.children.iter_mut().for_each(|child| {
                    child.populate_children();
                });
            },
            ProviderType::ExpectedValue(_) => {},
        }
    }

    const fn get_expected_value(&self) -> f64 {
        self.current_expected_value
    }

    pub const fn get_probability(&self) -> f64 {
        self.probability
    }

    pub const fn get_average_probability(&self) -> f64 {
        self.average_probability
    }

    pub const fn get_average_expected_value(&self) -> f64 {
        self.average_expected_value
    }
}

#[derive(Clone)]
pub struct WrappedRegret<UserData: UserDataTrait> {
    node: Arc<Mutex<RegretNode<UserData>>>,
}

impl<UserData: UserDataTrait> WrappedRegret<UserData> {
    pub fn new(node: RegretNode<UserData>) -> Self {
        Self { node: Arc::new(Mutex::new(node)) }
    }

    #[must_use]
    pub fn get_total_probability(&self) -> f64 {
        safe_lock(&self.node).get_total_probability()
    }

    #[must_use]
    pub fn get_expected_value(&self) -> f64 {
        safe_lock(&self.node).get_expected_value()
    }

    pub fn calculate_regrets(
        &self,
        outer_expected_value: f64,
    ) {
        safe_lock(&self.node).calculate_regrets(outer_expected_value);
    }

    pub fn calculate_probabilities(
        &self,
        sum_regrets: f64,
        total_siblings: usize,
    ) {
        safe_lock(&self.node).calculate_probabilities(sum_regrets, total_siblings);
    }

    #[must_use]
    pub fn get_probability(&self) -> f64 {
        safe_lock(&self.node).get_probability()
    }

    pub fn calculate_normalized_probabilities(
        &self,
        total_probability: f64,
    ) {
        safe_lock(&self.node).calculate_normalized_probabilities(total_probability);
    }

    #[must_use]
    pub fn get_average_probability(&self) -> f64 {
        safe_lock(&self.node).get_average_probability()
    }

    #[must_use]
    pub fn get_average_expected_value(&self) -> f64 {
        safe_lock(&self.node).get_average_expected_value()
    }

    pub fn populate_children(&mut self) {
        safe_lock(&self.node).populate_children();
    }

    pub fn update_average_values(&mut self) {
        safe_lock(&self.node).update_average_values();
    }

    #[must_use]
    pub fn get_data_as_string(
        &self,
        indentation: usize,
    ) -> String {
        safe_lock(&self.node).get_data_as_string(indentation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default, Debug, Clone)]
    enum Choice {
        #[default]
        Rock,
        Paper,
        Scissors,
    }

    #[derive(Default, Clone)]
    struct RoshamboData {
        choice: Choice,
        probability: f64,
    }

    impl UserDataTrait for RoshamboData {
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
            format!("Choice: {:?}", self.choice.clone())
        }
    }

    struct RoshamboChildrenProvider {}

    impl RoshamboChildrenProvider {
        pub const fn new() -> Self {
            Self {}
        }
    }

    impl ChildrenProvider<RoshamboData> for RoshamboChildrenProvider {
        fn get_children(
            &self,
            parents_data: Vec<RoshamboData>,
        ) -> Vec<WrappedRegret<RoshamboData>> {
            let probabilities = [0.4, 0.4, 0.2];
            match parents_data.len().cmp(&1) {
                std::cmp::Ordering::Less => {
                    let mut children = Vec::new();
                    for (i, choice) in
                        [Choice::Rock, Choice::Paper, Choice::Scissors].iter().enumerate()
                    {
                        let data =
                            RoshamboData { choice: choice.clone(), probability: probabilities[i] };
                        let node = RegretNode::new(
                            probabilities[i],
                            0.01,
                            parents_data.clone(),
                            WrappedProvider::new(Provider::new(
                                ProviderType::Children(WrappedChildrenProvider::new(Box::new(
                                    Self::new(),
                                ))),
                                Some(data.clone()),
                            )),
                            None,
                        );
                        children.push(WrappedRegret::new(node));
                    }
                    children
                },
                std::cmp::Ordering::Equal => {
                    let mut children = Vec::new();
                    for (i, choice) in
                        [Choice::Rock, Choice::Paper, Choice::Scissors].iter().enumerate()
                    {
                        let data =
                            RoshamboData { choice: choice.clone(), probability: probabilities[i] };
                        let provider = Provider::new(
                            ProviderType::ExpectedValue(WrappedExpectedValueProvider::new(
                                Box::new(RoshamboExpectedValueProvider::new()),
                            )),
                            Some(data.clone()),
                        );
                        let node = RegretNode::new(
                            probabilities[i],
                            0.01,
                            parents_data.clone(),
                            WrappedProvider::new(provider),
                            None,
                        );
                        children.push(WrappedRegret::new(node));
                    }
                    children
                },
                std::cmp::Ordering::Greater => Vec::new(),
            }
        }
    }

    struct RoshamboExpectedValueProvider {}

    impl RoshamboExpectedValueProvider {
        pub const fn new() -> Self {
            Self {}
        }
    }

    impl ExpectedValueProvider<RoshamboData> for RoshamboExpectedValueProvider {
        fn get_expected_value(
            &self,
            parents_data: Vec<RoshamboData>,
        ) -> f64 {
            assert!(
                parents_data.len() >= 2,
                "Expected at least two parents data for expected value calculation"
            );
            let player_1_choice = &parents_data[parents_data.len() - 2].choice;
            let player_2_choice = &parents_data[parents_data.len() - 1].choice;
            match player_1_choice {
                Choice::Rock => match player_2_choice {
                    Choice::Rock => 0.0,     // Tie
                    Choice::Paper => -1.0,   // Paper beats Rock
                    Choice::Scissors => 1.0, // Rock beats Scissors
                },
                Choice::Paper => match player_2_choice {
                    Choice::Rock => 1.0,      // Paper beats Rock
                    Choice::Paper => 0.0,     // Tie
                    Choice::Scissors => -1.0, // Scissors beats Paper
                },
                Choice::Scissors => match player_2_choice {
                    Choice::Rock => -1.0,    // Rock beats Scissors
                    Choice::Paper => 1.0,    // Paper beats Rock
                    Choice::Scissors => 0.0, // Tie
                },
            }
        }
    }

    #[test]
    fn test_roshambo_children_provider() {
        let provider = RoshamboChildrenProvider::new();
        let parents_data = vec![];
        let children = provider.get_children(parents_data);
        assert_eq!(children.len(), 3); // Should have three children for Rock, Paper, Scissors
    }

    #[test]
    fn test_roshambo_expected_value_provider() {
        let provider = RoshamboExpectedValueProvider::new();
        let parents_data = vec![
            RoshamboData { choice: Choice::Rock, probability: 0.333 },
            RoshamboData { choice: Choice::Paper, probability: 0.333 },
        ];
        let expected_value = provider.get_expected_value(parents_data);
        assert!((expected_value + 1.0).abs() < f64::EPSILON, "Paper beats Rock");
    }

    #[test]
    fn test_roshambo_regret_minimization() {
        let mut node = RegretNode::new(
            1.0,
            0.01,
            vec![],
            WrappedProvider::new(Provider::new(
                ProviderType::Children(WrappedChildrenProvider::new(Box::new(
                    RoshamboChildrenProvider {},
                ))),
                None,
            )),
            Some(1.0),
        );

        node.solve(1000);
        // print node as string
        println!("{}", node.get_data_as_string(0));
        let children = node.get_children();
        assert_eq!(children.len(), 3); // Should have three children for Rock, Paper, Scissors
        for child in children {
            let expected_value = child.get_average_expected_value();
            let probability = child.get_average_probability();
            // assert expected value is close to zero
            assert!((expected_value - 0.0).abs() < 0.01, "Expected value should be close to zero");
            // assert probability is close to 1/3
            assert!((probability - 1.0 / 3.0).abs() < 0.01, "Probability should be close to 1/3");
        }
    }
}
