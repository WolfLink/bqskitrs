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
pub struct BreadthHeuristic();

impl BreadthHeuristic {
    pub fn new() -> Self {
        Self {}
    }
}

impl Heuristic for BreadthHeuristic {
    fn get_value(&self, circuit: &Circuit, _cost_fn: impl CostFn) -> f64 {
        circuit
            .ops
            .iter()
            .filter(|op| op.gate.num_qudits() > 1)
            .count() as f64
    }
}
