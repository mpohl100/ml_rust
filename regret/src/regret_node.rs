//! # Regret Node Module
//!
//! This module implements the core regret minimization tree node and its thread-safe wrapper.
//! Nodes maintain probabilities, expected values, regrets, and children, and support iterative
//! regret minimization algorithms.

use crate::provider::{ProviderType, WrappedProvider};
use crate::user_data::{DecisionTrait, WrappedDecision};
use std::sync::{Arc, Mutex};
use utils::safer::safe_lock;

/// Represents a node in the regret minimization tree.
#[derive(Debug, Clone)]
pub struct RegretNode<Decision: DecisionTrait> {
    /// Probability of this node.
    probability: f64,
    /// Minimum allowed probability.
    _min_probability: f64,
    /// Current expected value of this node.
    current_expected_value: f64,
    /// Data from parent nodes.
    parents_data: Vec<WrappedDecision<Decision>>,
    /// Provider for children or expected value.
    provider: WrappedProvider<Decision>,
    /// Child nodes.
    children: Vec<WrappedRegret<Decision>>,
    /// Regret value for this node.
    regret: f64,
    /// Sum of probabilities over iterations.
    sum_probabilities: f64,
    /// Number of probability updates.
    num_probabilities: f64,
    /// Average probability over iterations.
    average_probability: f64,
    /// Sum of expected values over iterations.
    sum_expected_values: f64,
    /// Number of expected value updates.
    num_expected_values: f64,
    /// Average expected value over iterations.
    average_expected_value: f64,
    /// Fixed probability for leaf nodes (if any).
    fixed_probability: Option<f64>,
}

impl<Decision: DecisionTrait> RegretNode<Decision> {
    /// Creates a new regret node.
    #[must_use]
    pub fn new(
        probability: f64,
        min_probability: f64,
        parents_data: Vec<WrappedDecision<Decision>>,
        provider: WrappedProvider<Decision>,
        fixed_probability: Option<f64>,
    ) -> Self {
        provider.get_user_data().as_mut().map_or_else(
            || {
                // If no user data is provided, we set the probability to 0.0
                Decision::set_probability(
                    &mut Decision::default(),
                    fixed_probability.unwrap_or(1.0),
                );
            },
            |data| data.set_probability(probability),
        );
        Self {
            probability,
            _min_probability: min_probability,
            current_expected_value: 0.0,
            parents_data,
            children: Vec::new(),
            provider,
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

    /// Returns a string representation of the node and its children.
    #[allow(clippy::format_push_string)]
    #[must_use]
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
        // put the current provider type
        result.push_str(&format!(
            "{}Provider Type: {:?}\n",
            " ".repeat(indentation),
            self.provider.get_provider_type()
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

    /// Solves the node using regret minimization for the given number of iterations.
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

    /// Updates the average probability and expected value statistics.
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

    /// Normalizes the probability of this node and its children.
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

    /// Calculates probabilities for this node and its children based on regrets.
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

    /// Adds the given probability to this node.
    fn add_probability(
        &mut self,
        probability: f64,
    ) {
        self.probability += probability;
    }

    /// Calculates regrets for this node and its children.
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

    /// Calculates the expected value for this node.
    fn calculate_expected_value(&mut self) -> f64 {
        // Calculate the expected value based on the current probabilities
        // TODO fix this later
        // if self.get_total_probability() < self.min_probability {
        //     self.current_expected_value = 0.0;
        //     return 0.0;
        // }

        self.current_expected_value = match self.provider.get_provider_type() {
            ProviderType::ExpectedValue(ref provider) => {
                let mut cloned_parents_data = self.parents_data.clone();
                cloned_parents_data.extend(
                    self.provider
                        .get_user_data()
                        .as_ref()
                        .map_or_else(Vec::new, |data| vec![data.clone()]),
                );
                provider.get_expected_value(cloned_parents_data)
            },
            ProviderType::Children(ref _provider) => self
                .children
                .iter()
                .map(|child| child.calculate_expected_value() * child.get_probability())
                .sum::<f64>(),
            ProviderType::None => 0.0, // Terminal node has zero expected value
        };
        self.current_expected_value
    }

    /// Returns the sum of regrets of all children.
    fn get_sum_regrets(&self) -> f64 {
        self.children.iter().map(|child| child.node.lock().unwrap().regret).sum()
    }

    /// Returns the total probability including parent probabilities.
    fn get_total_probability(&self) -> f64 {
        if self.parents_data.is_empty() {
            return self.probability; // If no parents, return the node's own probability
        }
        let parent_probability =
            self.parents_data.iter().fold(1.0, |acc, data| acc * data.get_probability());
        parent_probability * self.probability
    }

    /// Returns the children of this node.
    #[must_use]
    pub fn get_children(&self) -> Vec<WrappedRegret<Decision>> {
        self.children.clone()
    }

    /// Returns the user data associated with this node.
    #[must_use]
    pub fn get_user_data(&self) -> Option<WrappedDecision<Decision>> {
        self.provider.get_user_data()
    }

    /// Populates the children of this node using the provider.
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
            ProviderType::ExpectedValue(_) | ProviderType::None => {
                // Expected value and terminal nodes have no children to populate
            },
        }
    }

    /// Returns the current expected value.
    const fn get_expected_value(&self) -> f64 {
        self.current_expected_value
    }

    /// Returns the current probability.
    #[must_use]
    pub const fn get_probability(&self) -> f64 {
        self.probability
    }

    /// Returns the average probability.
    #[must_use]
    pub const fn get_average_probability(&self) -> f64 {
        self.average_probability
    }

    /// Returns the average expected value.
    #[must_use]
    pub const fn get_average_expected_value(&self) -> f64 {
        self.average_expected_value
    }
}

/// Thread-safe wrapper for a `RegretNode`.
#[derive(Debug, Clone)]
pub struct WrappedRegret<Decision: DecisionTrait> {
    /// The underlying regret node, wrapped in Arc<Mutex>.
    node: Arc<Mutex<RegretNode<Decision>>>,
}

impl<Decision: DecisionTrait> WrappedRegret<Decision> {
    /// Creates a new wrapped regret node.
    #[must_use]
    pub fn new(node: RegretNode<Decision>) -> Self {
        Self { node: Arc::new(Mutex::new(node)) }
    }

    /// Returns the total probability including parent probabilities.
    #[must_use]
    pub fn get_total_probability(&self) -> f64 {
        safe_lock(&self.node).get_total_probability()
    }

    /// Returns the expected value of the node.
    #[must_use]
    pub fn get_expected_value(&self) -> f64 {
        safe_lock(&self.node).get_expected_value()
    }

    /// Calculates and returns the expected value of the node.
    #[must_use]
    pub fn calculate_expected_value(&self) -> f64 {
        safe_lock(&self.node).calculate_expected_value()
    }

    /// Returns the user data associated with the node.
    #[must_use]
    pub fn get_user_data(&self) -> Option<WrappedDecision<Decision>> {
        safe_lock(&self.node).get_user_data()
    }

    /// Returns the children of the node.
    #[must_use]
    pub fn get_children(&self) -> Vec<Self> {
        safe_lock(&self.node).get_children()
    }

    /// Calculates regrets for the node.
    pub fn calculate_regrets(
        &self,
        outer_expected_value: f64,
    ) {
        safe_lock(&self.node).calculate_regrets(outer_expected_value);
    }

    /// Calculates probabilities for the node.
    pub fn calculate_probabilities(
        &self,
        sum_regrets: f64,
        total_siblings: usize,
    ) {
        safe_lock(&self.node).calculate_probabilities(sum_regrets, total_siblings);
    }

    /// Returns the probability of the node.
    #[must_use]
    pub fn get_probability(&self) -> f64 {
        safe_lock(&self.node).get_probability()
    }

    /// Normalizes the probability of the node.
    pub fn calculate_normalized_probabilities(
        &self,
        total_probability: f64,
    ) {
        safe_lock(&self.node).calculate_normalized_probabilities(total_probability);
    }

    /// Returns the average probability of the node.
    #[must_use]
    pub fn get_average_probability(&self) -> f64 {
        safe_lock(&self.node).get_average_probability()
    }

    /// Returns the average expected value of the node.
    #[must_use]
    pub fn get_average_expected_value(&self) -> f64 {
        safe_lock(&self.node).get_average_expected_value()
    }

    /// Populates the children of the node.
    pub fn populate_children(&mut self) {
        safe_lock(&self.node).populate_children();
    }

    /// Updates the average values of the node.
    pub fn update_average_values(&mut self) {
        safe_lock(&self.node).update_average_values();
    }

    /// Returns a string representation of the node and its children.
    #[must_use]
    pub fn get_data_as_string(
        &self,
        indentation: usize,
    ) -> String {
        safe_lock(&self.node).get_data_as_string(indentation)
    }
}
