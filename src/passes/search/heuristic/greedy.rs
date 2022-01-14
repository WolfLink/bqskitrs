use crate::{
    circuit::Circuit,
    gates::{Size, Unitary},
    minimizers::{CostFn, CostFunction},
    utils::matrix_distance_squared,
};
use ndarray::ArrayView2;
use num_complex::Complex64;

use super::Heuristic;

#[derive(Clone, Copy)]
pub struct GreedyHeuristic();

impl GreedyHeuristic {
    pub fn new() -> Self {
        Self {}
    }
}

impl Heuristic for GreedyHeuristic {
    fn get_value(&self, _circuit: &Circuit, cost_fn: impl CostFn) -> f64 {
        cost_fn.get_cost(&[])
    }
}
