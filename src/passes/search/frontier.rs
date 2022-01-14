use std::{cmp::Ordering, collections::BinaryHeap};

use num_complex::Complex64;

use ndarray::{Array2, ArrayView2};

use crate::{circuit::Circuit, minimizers::HilbertSchmidtResidualFn};

use super::{Heuristic, HeuristicFunction};

pub struct FrontierElement<T>
where
    T: PartialEq,
{
    pub cost: f64,
    pub id: usize,
    pub circuit: Circuit,
    pub extra: Option<T>,
}

impl<T> FrontierElement<T>
where
    T: PartialEq,
{
    pub fn new(cost: f64, id: usize, circuit: Circuit, extra: Option<T>) -> Self {
        Self {
            cost,
            id,
            circuit,
            extra,
        }
    }
}

impl<T> PartialEq for FrontierElement<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for FrontierElement<T> where T: PartialEq {}

impl<T> Ord for FrontierElement<T>
where
    T: PartialEq,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        Ordering::reverse(
            self.cost
                .partial_cmp(&other.cost)
                .unwrap_or(self.id.cmp(&other.id)),
        )
    }
}

impl<T> PartialOrd for FrontierElement<T>
where
    T: PartialEq,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct Frontier<'a, T>
where
    T: PartialEq,
{
    target: ArrayView2<'a, Complex64>,
    heuristic_func: HeuristicFunction,
    frontier: BinaryHeap<FrontierElement<T>>,
    count: usize,
}

impl<'a, T> Frontier<'a, T>
where
    T: PartialEq,
{
    pub fn new(target: ArrayView2<'a, Complex64>, heuristic_func: HeuristicFunction) -> Self {
        Self {
            target,
            heuristic_func,
            frontier: BinaryHeap::new(),
            count: 0,
        }
    }

    fn next_count(&mut self) -> usize {
        let count = self.count;
        self.count += 1;
        count
    }

    pub fn add(&mut self, circuit: Circuit, extra: Option<T>) {
        let cost_fn = HilbertSchmidtResidualFn::new(circuit.clone(), self.target.to_owned());
        let heuristic_val = self.heuristic_func.get_value(&circuit, cost_fn);
        let count = self.next_count();
        let elem = FrontierElement::new(heuristic_val, count, circuit, extra);
        self.frontier.push(elem);
    }

    pub fn pop(&mut self) -> Option<(Circuit, Option<T>)> {
        match self.frontier.pop() {
            Some(elem) => Some((elem.circuit, elem.extra)),
            None => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.frontier.is_empty()
    }

    pub fn clear(&mut self) {
        self.frontier.clear();
    }
}
