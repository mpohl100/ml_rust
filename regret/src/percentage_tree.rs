//! # `PercentageNode` Module
//!
//! This module implements a percentage-based decision node that can make random decisions
//! based on probability distributions. It uses wrapped decisions and providers to manage
//! decision-making processes.

use crate::provider::{ProviderType, WrappedProvider};
use crate::user_data::{DecisionTrait, WrappedDecision};
use evol::rng::RandomNumberGenerator;
use rand::Rng;
use std::sync::{Arc, Mutex};
use utils::safer::safe_lock;

/// A percentage-based decision node that makes random decisions based on probability distributions.
#[derive(Debug, Clone)]
pub struct PercentageNode<Decision: DecisionTrait> {
    /// Vector of parent wrapped decisions that led to this node.
    parent_decisions: Vec<WrappedDecision<Decision>>,
    /// The provider type which is either `ChildrenProvider` or None for terminal nodes.
    provider: WrappedProvider<Decision, WrappedPercentageNode<Decision>>,
    /// Child nodes for tree structure.
    children: Vec<WrappedPercentageNode<Decision>>,
}

/// A thread-safe wrapper around `PercentageNode`.
#[derive(Debug, Clone)]
pub struct WrappedPercentageNode<Decision: DecisionTrait> {
    /// The underlying percentage node, wrapped in Arc<Mutex>.
    node: Arc<Mutex<PercentageNode<Decision>>>,
}

impl<Decision: DecisionTrait> PercentageNode<Decision> {
    /// Creates a new `PercentageNode` with the given parent decisions and provider.
    ///
    /// # Arguments
    ///
    /// * `parent_decisions` - Vector of wrapped decisions that led to this node
    /// * `provider` - The provider which can be `ChildrenProvider` or None
    ///
    /// # Examples
    ///
    /// ```
    /// use regret::percentage_tree::PercentageNode;
    /// use regret::provider::{Provider, ProviderType, WrappedProvider};
    /// use regret::user_data::{DecisionTrait, WrappedDecision};
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Debug, Default, Clone, Serialize, Deserialize)]
    /// struct MyDecision {
    ///     value: i32,
    ///     probability: f64,
    /// }
    ///
    /// impl DecisionTrait for MyDecision {
    ///     fn get_probability(&self) -> f64 { self.probability }
    ///     fn set_probability(&mut self, prob: f64) { self.probability = prob; }
    ///     fn get_data_as_string(&self) -> String { format!("Decision: {}", self.value) }
    /// }
    ///
    /// let parent_decisions: Vec<WrappedDecision<MyDecision>> = vec![];
    /// let provider = WrappedProvider::new(Provider::new(ProviderType::None, None));
    /// let node = PercentageNode::new(parent_decisions, provider);
    /// ```
    #[must_use]
    pub const fn new(
        parent_decisions: Vec<WrappedDecision<Decision>>,
        provider: WrappedProvider<Decision, WrappedPercentageNode<Decision>>,
    ) -> Self {
        Self { parent_decisions, provider, children: Vec::new() }
    }

    /// Creates children for this node based on the provider.
    ///
    /// This method populates the children vector by creating child nodes
    /// from the provider's children.
    pub fn create_children(&mut self) {
        match self.provider.get_provider_type() {
            ProviderType::Children(ref children_provider) => {
                // Get children from the provider
                let child_nodes = children_provider.get_children(self.parent_decisions.clone());

                // Assign the children directly since they're already WrappedPercentageNode<Decision>
                self.children = child_nodes;
            },
            ProviderType::ExpectedValue(_) | ProviderType::None => {
                // Terminal nodes have no children
                self.children.clear();
            },
        }
    }

    /// Makes a random decision by selecting a random child and appending its decision.
    ///
    /// This method selects a random child node and recursively gets its random decision,
    /// appending the result to the parent decisions.
    ///
    /// # Arguments
    ///
    /// * `rng` - A mutable reference to a `RandomNumberGenerator`
    ///
    /// # Returns
    ///
    /// A vector containing the parent decisions plus the randomly selected child decision.
    /// If this is a terminal node (no children), returns just the parent decisions.
    ///
    /// # Examples
    ///
    /// ```
    /// use regret::percentage_tree::{PercentageNode, WrappedPercentageNode};
    /// use regret::provider::{Provider, ProviderType, WrappedProvider};
    /// use regret::user_data::{WrappedDecision, DecisionTrait};
    /// use evol::rng::RandomNumberGenerator;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Debug, Default, Clone, Serialize, Deserialize)]
    /// struct MyDecision {
    ///     value: i32,
    ///     probability: f64,
    /// }
    ///
    /// impl DecisionTrait for MyDecision {
    ///     fn get_probability(&self) -> f64 { self.probability }
    ///     fn set_probability(&mut self, prob: f64) { self.probability = prob; }
    ///     fn get_data_as_string(&self) -> String { format!("Decision: {}", self.value) }
    /// }
    ///
    /// // Create a terminal percentage node
    /// let parent_decisions: Vec<WrappedDecision<MyDecision>> = vec![];
    /// let provider = WrappedProvider::new(Provider::new(ProviderType::None, None));
    /// let node = PercentageNode::new(parent_decisions, provider);
    /// let mut rng = RandomNumberGenerator::new();
    /// let result = node.random_decision(&mut rng);
    /// ```
    #[must_use]
    pub fn random_decision(
        &self,
        rng: &mut RandomNumberGenerator,
    ) -> Vec<WrappedDecision<Decision>> {
        let mut result = self.parent_decisions.clone();

        if self.children.is_empty() {
            // Terminal node - return parent decisions
            return result;
        }

        // Select a random child
        let child_index = rng.rng.gen_range(0..self.children.len());
        let selected_child = &self.children[child_index];

        // Get random decision from the selected child
        let child_decisions = selected_child.random_decision(rng);

        // Append child decisions to result
        result.extend(child_decisions.into_iter().skip(self.parent_decisions.len()));

        result
    }

    /// Returns a reference to the parent decisions.
    #[must_use]
    pub fn get_parent_decisions(&self) -> &[WrappedDecision<Decision>] {
        &self.parent_decisions
    }

    /// Returns a reference to the provider.
    #[must_use]
    pub const fn get_provider(
        &self
    ) -> &WrappedProvider<Decision, WrappedPercentageNode<Decision>> {
        &self.provider
    }

    /// Returns a reference to the children.
    #[must_use]
    pub fn get_children(&self) -> &[WrappedPercentageNode<Decision>] {
        &self.children
    }

    /// Checks if this is a terminal node (has no children).
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.children.is_empty()
    }
}

impl<Decision: DecisionTrait> WrappedPercentageNode<Decision> {
    /// Creates a new wrapped percentage node.
    #[must_use]
    pub fn new(node: PercentageNode<Decision>) -> Self {
        Self { node: Arc::new(Mutex::new(node)) }
    }

    /// Creates children for this node by delegating to the inner node.
    pub fn create_children(&self) {
        safe_lock(&self.node).create_children();
    }

    /// Makes a random decision by delegating to the inner node.
    #[must_use]
    pub fn random_decision(
        &self,
        rng: &mut RandomNumberGenerator,
    ) -> Vec<WrappedDecision<Decision>> {
        safe_lock(&self.node).random_decision(rng)
    }

    /// Returns a clone of the parent decisions.
    #[must_use]
    pub fn get_parent_decisions(&self) -> Vec<WrappedDecision<Decision>> {
        safe_lock(&self.node).get_parent_decisions().to_vec()
    }

    /// Returns the provider.
    #[must_use]
    pub fn get_provider(&self) -> WrappedProvider<Decision, Self> {
        safe_lock(&self.node).get_provider().clone()
    }

    /// Returns a clone of the children.
    #[must_use]
    pub fn get_children(&self) -> Vec<Self> {
        safe_lock(&self.node).get_children().to_vec()
    }

    /// Checks if this is a terminal node.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        safe_lock(&self.node).is_terminal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{Provider, ProviderType};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Default, Clone, Serialize, Deserialize)]
    struct TestDecision {
        value: i32,
        probability: f64,
    }

    impl DecisionTrait for TestDecision {
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
            format!("TestDecision(value: {}, prob: {})", self.value, self.probability)
        }
    }

    #[derive(Debug, Clone)]
    struct TestChildrenProvider;

    impl crate::provider::ChildrenProvider<TestDecision, WrappedPercentageNode<TestDecision>>
        for TestChildrenProvider
    {
        fn get_children(
            &self,
            parents_data: Vec<WrappedDecision<TestDecision>>,
        ) -> Vec<WrappedPercentageNode<TestDecision>> {
            // Create three children with different probabilities
            let children_data = [
                (1, 0.5), // 50% probability
                (2, 0.3), // 30% probability
                (3, 0.2), // 20% probability
            ];

            children_data
                .iter()
                .map(|(value, prob)| {
                    let decision =
                        WrappedDecision::new(TestDecision { value: *value, probability: *prob });
                    let mut child_parent_decisions = parents_data.clone();
                    child_parent_decisions.push(decision);

                    // Create child node with terminal provider
                    let child_provider = WrappedProvider::new(crate::provider::Provider::new(
                        ProviderType::None,
                        None,
                    ));

                    let child_node = PercentageNode::new(child_parent_decisions, child_provider);
                    WrappedPercentageNode::new(child_node)
                })
                .collect()
        }
    }

    #[test]
    fn test_percentage_node_creation() {
        let parent_decisions =
            vec![WrappedDecision::new(TestDecision { value: 1, probability: 0.5 })];
        let provider = WrappedProvider::new(Provider::new(ProviderType::None, None));
        let node = PercentageNode::new(parent_decisions.clone(), provider);

        assert_eq!(node.get_parent_decisions().len(), 1);
        assert!(node.is_terminal());
        assert_eq!(node.get_children().len(), 0);
    }

    #[test]
    fn test_wrapped_percentage_node_creation() {
        let parent_decisions =
            vec![WrappedDecision::new(TestDecision { value: 1, probability: 0.5 })];
        let provider = WrappedProvider::new(Provider::new(ProviderType::None, None));
        let node = PercentageNode::new(parent_decisions.clone(), provider);
        let wrapped_node = WrappedPercentageNode::new(node);

        assert_eq!(wrapped_node.get_parent_decisions().len(), 1);
        assert!(wrapped_node.is_terminal());
        assert_eq!(wrapped_node.get_children().len(), 0);
    }

    #[test]
    fn test_create_children() {
        let parent_decisions = vec![];
        let children_provider = Box::new(TestChildrenProvider);
        let provider = WrappedProvider::new(Provider::new(
            ProviderType::Children(crate::provider::WrappedChildrenProvider::new(
                children_provider,
            )),
            None,
        ));
        let mut node = PercentageNode::new(parent_decisions, provider);

        // Initially no children
        assert_eq!(node.get_children().len(), 0);

        // Create children
        node.create_children();

        // Should now have 3 children
        assert_eq!(node.get_children().len(), 3);
        assert!(!node.is_terminal());
    }

    #[test]
    fn test_random_decision_with_children() {
        let parent_decisions = vec![];
        let children_provider = Box::new(TestChildrenProvider);
        let provider = WrappedProvider::new(Provider::new(
            ProviderType::Children(crate::provider::WrappedChildrenProvider::new(
                children_provider,
            )),
            None,
        ));
        let mut node = PercentageNode::new(parent_decisions, provider);

        // Create children first
        node.create_children();

        let mut rng = RandomNumberGenerator::new();

        // Test multiple random decisions to ensure they're within expected range
        let mut results = std::collections::HashMap::new();
        for _ in 0..300 {
            let result = node.random_decision(&mut rng);
            assert_eq!(result.len(), 1); // Should have one decision from the children

            let value = result[0].get_decision_data().value;
            *results.entry(value).or_insert(0) += 1;
        }

        // Verify that all three possible values were selected
        assert!(results.contains_key(&1));
        assert!(results.contains_key(&2));
        assert!(results.contains_key(&3));
    }

    #[test]
    fn test_random_decision_with_terminal_node() {
        let parent_decisions =
            vec![WrappedDecision::new(TestDecision { value: 1, probability: 0.5 })];
        let provider = WrappedProvider::new(Provider::new(ProviderType::None, None));
        let node = PercentageNode::new(parent_decisions.clone(), provider);
        let mut rng = RandomNumberGenerator::new();

        let result = node.random_decision(&mut rng);

        // Terminal node should return the same parent decisions
        assert_eq!(result.len(), parent_decisions.len());
        assert_eq!(
            result[0].get_decision_data().value,
            parent_decisions[0].get_decision_data().value
        );
    }

    #[test]
    fn test_wrapped_node_random_decision() {
        let parent_decisions = vec![];
        let children_provider = Box::new(TestChildrenProvider);
        let provider = WrappedProvider::new(Provider::new(
            ProviderType::Children(crate::provider::WrappedChildrenProvider::new(
                children_provider,
            )),
            None,
        ));
        let node = PercentageNode::new(parent_decisions, provider);
        let wrapped_node = WrappedPercentageNode::new(node);

        // Create children
        wrapped_node.create_children();

        let mut rng = RandomNumberGenerator::new();

        // Test multiple random decisions
        for _ in 0..10 {
            let result = wrapped_node.random_decision(&mut rng);
            assert_eq!(result.len(), 1); // Should have one decision from the children

            let value = result[0].get_decision_data().value;
            assert!([1, 2, 3].contains(&value));
        }
    }

    #[test]
    fn test_node_properties() {
        let parent_decisions =
            vec![WrappedDecision::new(TestDecision { value: 42, probability: 0.75 })];
        let provider = WrappedProvider::new(Provider::new(ProviderType::None, None));
        let node = PercentageNode::new(parent_decisions, provider);

        // Test basic properties
        assert_eq!(node.get_parent_decisions().len(), 1);
        assert_eq!(node.get_parent_decisions()[0].get_decision_data().value, 42);
        assert_eq!(node.get_parent_decisions()[0].get_decision_data().probability, 0.75);
        assert!(node.is_terminal());
        assert_eq!(node.get_children().len(), 0);
    }
}

/// Backward compatibility alias for `PercentageNode`.
///
/// This alias ensures that existing code using `PercentageTree` continues to work
/// without modification after the renaming to `PercentageNode`.
pub type PercentageTree<Decision> = PercentageNode<Decision>;
