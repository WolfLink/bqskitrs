use ndarray::{ArrayView2, Array2};
use num_complex::Complex64;

use crate::{
    circuit::Circuit,
    gates::{Gate, Size, Unitary},
    operation::Operation,
    utils::log,
};

use super::LayerGenerator;

pub struct SimpleLayerGenerator {
    two_qudit_gate: Gate,
    single_qudit_gate_1: Gate,
    single_qudit_gate_2: Gate,
    initial_layer_gate: Gate,
    constant_gates: Vec<Array2<Complex64>>,
    coupling: Vec<(usize, usize)>,
}

impl SimpleLayerGenerator {
    pub fn new(
        two_qudit_gate: Gate,
        single_qudit_gate_1: Gate,
        single_qudit_gate_2: Gate,
        initial_layer_gate: Gate,
        constant_gates: Vec<Array2<Complex64>>,
        coupling: Vec<(usize, usize)>,
    ) -> Self {
        assert_eq!(two_qudit_gate.num_qudits(), 2);
        assert_eq!(single_qudit_gate_1.num_qudits(), 1);
        assert_eq!(single_qudit_gate_2.num_qudits(), 1);
        Self {
            two_qudit_gate,
            single_qudit_gate_1,
            single_qudit_gate_2,
            initial_layer_gate,
            constant_gates,
            coupling,
        }
    }
}

impl LayerGenerator for SimpleLayerGenerator {
    fn initial_layer(&self, target: ArrayView2<Complex64>) -> Circuit {
        let dim = target.shape()[0];
        let (qudits, radix) = if dim.is_power_of_two() {
            (log::<2>(dim) as usize, 2)
        } else if 3usize.pow(log::<3>(dim)) == dim {
            (log::<3>(dim) as usize, 3)
        } else {
            panic!("Unknown array shape {}", dim)
        };
        let mut initial_ops = Vec::new();
        for i in 0..qudits {
            initial_ops.push(Operation::new(
                self.initial_layer_gate.clone(),
                vec![i],
                vec![0.; self.initial_layer_gate.num_params()],
            ));
        }
        Circuit::new(
            qudits,
            vec![radix; qudits],
            initial_ops,
            self.constant_gates.clone(),
        )
    }

    fn successors(&self, circuit: Circuit) -> Vec<Circuit> {
        let mut successors = Vec::new();
        for edge in &self.coupling {
            let mut circ = circuit.clone();
            circ.append_gate(self.two_qudit_gate.clone(), vec![edge.0, edge.1], None);
            circ.append_gate(self.single_qudit_gate_1.clone(), vec![edge.0], None);
            circ.append_gate(self.single_qudit_gate_2.clone(), vec![edge.1], None);
            successors.push(circ);
        }
        successors
    }
}
