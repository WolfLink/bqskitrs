mod astar;
mod breadth;
mod greedy;

pub use astar::AStarHeuristic;
pub use breadth::BreadthHeuristic;
pub use greedy::GreedyHeuristic;

use crate::circuit::Circuit;
use crate::minimizers::CostFn;
use enum_dispatch::enum_dispatch;
use ndarray::ArrayView2;
use num_complex::Complex64;

use crate::{minimizers::CostFunction, utils::matrix_distance_squared};

#[enum_dispatch]
pub trait Heuristic {
    fn get_value(&self, circuit: &Circuit, cost_fn: impl CostFn) -> f64;
}

#[enum_dispatch(Heuristic)]
#[derive(Clone, Copy)]
pub enum HeuristicFunction {
    AStar(AStarHeuristic),
    Greedy(GreedyHeuristic),
    Breadth(BreadthHeuristic),
}
