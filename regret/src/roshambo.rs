//! # Rock-Paper-Scissors Example Module
//!
//! This module provides example types and providers for the Rock-Paper-Scissors game,
//! demonstrating how to use custom user data, children providers, and expected value providers
//! with the regret minimization framework.

use crate::provider::{
    ChildrenProvider, ExpectedValueProvider, Provider, ProviderType, WrappedChildrenProvider,
    WrappedExpectedValueProvider, WrappedProvider,
};
use crate::regret_node::{RegretNode, WrappedRegret};
use crate::user_data::{DecisionTrait, WrappedDecision};
use serde::{Deserialize, Serialize};

/// Enum representing choices in Rock-Paper-Scissors.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// Enum representing choices in Rock-Paper-Scissors.
pub enum Choice {
    /// The Rock move.
    #[default]
    Rock,
    /// The Paper move.
    Paper,
    /// The Scissors move.
    Scissors,
}

/// User data for Rock-Paper-Scissors.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct RoshamboData {
    /// The chosen move.
    pub choice: Choice,
    /// Probability of this choice.
    pub probability: f64,
}

impl DecisionTrait for RoshamboData {
    /// Returns the probability of this choice.
    fn get_probability(&self) -> f64 {
        self.probability
    }
    /// Sets the probability of this choice.
    fn set_probability(
        &mut self,
        probability: f64,
    ) {
        self.probability = probability;
    }
    /// Returns a string representation of the choice.
    fn get_data_as_string(&self) -> String {
        format!("Choice: {:?}", self.choice.clone())
    }
}

/// Children provider for Rock-Paper-Scissors.
#[derive(Default, Debug, Clone)]
pub struct RoshamboChildrenProvider {}

impl RoshamboChildrenProvider {
    /// Creates a new children provider for Rock-Paper-Scissors.
    #[must_use]
    pub const fn new() -> Self {
        Self {}
    }
}

impl ChildrenProvider<RoshamboData> for RoshamboChildrenProvider {
    /// Returns the children nodes for Rock-Paper-Scissors.
    fn get_children(
        &self,
        parents_data: Vec<WrappedDecision<RoshamboData>>,
    ) -> Vec<WrappedRegret<RoshamboData>> {
        let probabilities = [0.4, 0.4, 0.2];
        match parents_data.len().cmp(&1) {
            std::cmp::Ordering::Less => {
                let mut children = Vec::new();
                for (i, choice) in
                    [Choice::Rock, Choice::Paper, Choice::Scissors].iter().enumerate()
                {
                    let data = WrappedDecision::new(RoshamboData {
                        choice: choice.clone(),
                        probability: probabilities[i],
                    });
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
                    let data = WrappedDecision::new(RoshamboData {
                        choice: choice.clone(),
                        probability: probabilities[i],
                    });
                    let provider = Provider::new(
                        ProviderType::ExpectedValue(WrappedExpectedValueProvider::new(Box::new(
                            RoshamboExpectedValueProvider::new(),
                        ))),
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

/// Expected value provider for Rock-Paper-Scissors.
#[derive(Default, Debug, Clone)]
pub struct RoshamboExpectedValueProvider {}

impl RoshamboExpectedValueProvider {
    /// Creates a new expected value provider for Rock-Paper-Scissors.
    #[must_use]
    pub const fn new() -> Self {
        Self {}
    }
}

impl ExpectedValueProvider<RoshamboData> for RoshamboExpectedValueProvider {
    /// Returns the expected value for Rock-Paper-Scissors.
    fn get_expected_value(
        &self,
        parents_data: Vec<WrappedDecision<RoshamboData>>,
    ) -> f64 {
        assert!(
            parents_data.len() >= 2,
            "Expected at least two parents data for expected value calculation"
        );
        let player_1_choice = &parents_data[parents_data.len() - 2].get_decision_data().choice;
        let player_2_choice = &parents_data[parents_data.len() - 1].get_decision_data().choice;
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
                Choice::Paper => 1.0,    // Scissors beats Paper
                Choice::Scissors => 0.0, // Tie
            },
        }
    }
}
