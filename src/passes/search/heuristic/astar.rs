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
pub struct AStarHeuristic {
    heuristic_factor: f64,
    cost_factor: f64,
}

impl Default for AStarHeuristic {
    fn default() -> Self {
        Self {
            heuristic_factor: 10.0,
            cost_factor: 1.0,
        }
    }
}

impl AStarHeuristic {
    pub fn new(heuristic_factor: f64, cost_factor: f64) -> Self {
        Self {
            heuristic_factor,
            cost_factor,
        }
    }
}

impl Heuristic for AStarHeuristic {
    fn get_value(&self, circuit: &Circuit, cost_fn: impl CostFn) -> f64 {
        let mut cost = 0.;
        cost += circuit
            .ops
            .iter()
            .filter(|op| op.gate.num_qudits() > 1)
            .count() as f64;

        let heuristic = cost_fn.get_cost(&[]);
        self.heuristic_factor * heuristic + self.cost_factor * cost
    }
}
