use ndarray::Array1;
use ndarray::ArrayView2;
use num_complex::Complex64;
use rayon::prelude::*;

use super::SynthesisPass;
use crate::circuit::Circuit;
use crate::circuit::Instantiators;
use crate::gates::Unitary;
use crate::instantiators::*;
use crate::minimizers::*;
use crate::passes::search::*;
use crate::utils::linregress;

struct LEAPData {
    best_dist: f64,
    best_circ: Circuit,
    best_layer: usize,
    best_dists: Vec<f64>,
    best_layers: Vec<usize>,
    last_prefix_layer: usize,
}

impl LEAPData {
    fn new(
        best_dist: f64,
        best_circ: Circuit,
        best_layer: usize,
        best_dists: Vec<f64>,
        best_layers: Vec<usize>,
        last_prefix_layer: usize,
    ) -> Self {
        Self {
            best_dist,
            best_circ,
            best_layer,
            best_dists,
            best_layers,
            last_prefix_layer,
        }
    }
}

pub struct LeapSynthesisPass {
    heuristic_function: HeuristicFunction,
    layer_gen: LayerGenerators,
    success_threshold: f64,
    max_layer: Option<usize>,
    min_prefix_size: usize,
}

impl LeapSynthesisPass {
    pub fn new(
        heuristic_function: HeuristicFunction,
        layer_gen: LayerGenerators,
        success_threshold: f64,
        max_layer: Option<usize>,
        min_prefix_size: usize,
    ) -> Self {
        Self {
            heuristic_function,
            layer_gen,
            success_threshold,
            max_layer,
            min_prefix_size,
        }
    }

    fn evaluate_node(
        &self,
        circ: &Circuit,
        utry: ArrayView2<Complex64>,
        frontier: &mut Frontier<usize>,
        layer: usize,
        leap_data: &mut LEAPData,
    ) -> bool {
        let cost = HilbertSchmidtResidualFn::new(circ.clone(), utry.to_owned());
        let dist = cost.get_cost(&[]);
        if dist < self.success_threshold {
            return true;
        }

        if self.check_new_best(layer + 1, dist, leap_data.best_layer, leap_data.best_dist) {
            leap_data.best_dist = dist;
            leap_data.best_circ = circ.clone();
            leap_data.best_layer = layer + 1;
            if self.check_leap_condition(layer + 1, leap_data) {
                leap_data.last_prefix_layer = layer + 1;
                frontier.clear();
                // TODO: Window markers
                // TODO: check if this is important?
                //if self.max_layer.is_none() || layer + 1 < self.max_layer {
                //    frontier.add(circuit, layer + 1)
                //}
            }
        }

        if self.max_layer.is_none() || layer + 1 < self.max_layer.unwrap() {
            frontier.add(circ.clone(), Some(layer + 1));
        }
        false
    }

    fn check_new_best(&self, layer: usize, dist: f64, best_layer: usize, best_dist: f64) -> bool {
        let better_layer =
            dist < best_dist && best_dist >= self.success_threshold || layer <= best_layer;
        let better_dist_and_layer = dist < self.success_threshold && layer < best_layer;
        better_layer || better_dist_and_layer
    }

    fn check_leap_condition(&self, new_layer: usize, leap_data: &mut LEAPData) -> bool {
        if leap_data.best_layers.len() < 2 {
            leap_data.best_layers.push(new_layer);
            leap_data.best_dists.push(leap_data.best_dist);
            return false;
        }
        let (m, y_int) = linregress(
            Array1::from_iter(leap_data.best_layers.iter().map(|&i| i as f64)),
            Array1::from_iter(leap_data.best_dists.iter().cloned()),
        )
        .unwrap();
        let predicted_best = m * new_layer as f64 + y_int;
        leap_data.best_layers.push(new_layer);
        leap_data.best_dists.push(leap_data.best_dist);
        if predicted_best.is_nan() {
            return false;
        }

        let delta = predicted_best - leap_data.best_dist;

        let layers_added = new_layer - leap_data.last_prefix_layer;
        delta < 0. && layers_added > self.min_prefix_size
    }
}

fn calc_cost(circuit: &Circuit, target: ArrayView2<Complex64>) -> f64 {
    let costfn = HilbertSchmidtCostFn::new(circuit.clone(), target.to_owned());
    costfn.get_cost(&[])
}

impl SynthesisPass for LeapSynthesisPass {
    fn synthesize(&self, utry: ArrayView2<Complex64>) -> Circuit {
        let mut frontier = Frontier::new(utry.view(), self.heuristic_function.clone());
        // TODO: Window markers

        let mut initial_layer = self.layer_gen.initial_layer(utry.view());
        let x0 = vec![0.; initial_layer.num_params()];
        initial_layer.instantiate(utry.to_owned(), &x0, Instantiators::Ceres);
        frontier.add(initial_layer.clone(), Some(0usize));
        let init_cost = calc_cost(&initial_layer, utry);
        let mut leap_data = LEAPData::new(init_cost, initial_layer, 0, vec![init_cost], vec![0], 0);

        while !frontier.is_empty() {
            let (top_circ, layer) = frontier.pop().unwrap();
            let mut successors = self.layer_gen.successors(top_circ);
            let result = successors.par_iter_mut().find_any(|&circuit| {
                let x0 = vec![0.; circuit.num_params()];
                circuit.instantiate(utry.to_owned(), &x0, Instantiators::Ceres);
                self.evaluate_node(
                    &circuit,
                    utry,
                    &mut frontier,
                    layer.unwrap(),
                    &mut leap_data,
                )
            });
            if let Some(circuit) = result {
                return circuit.clone();
            }
        }

        leap_data.best_circ
    }
}
