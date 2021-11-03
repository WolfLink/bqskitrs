use bqskitrs::circuit::Circuit;
use bqskitrs::{gates::*, r};
use bqskitrs::minimizers::*;
use bqskitrs::instantiators::*;
use bqskitrs::operation::Operation;

use ndarray::{Array2, array};
use num_complex::Complex64;

extern "C" {
    fn srand(seed: u32);
}

fn make_circuit(positions: Vec<usize>, num_qubits: usize, u3: bool) -> Circuit {
    let mut ops = vec![];
    let mut constant_gates = vec![];
    // CNOT
    let v: Vec<Complex64> = vec![
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
    ]
    .iter()
    .map(|i| Complex64::new(*i, 0.0))
    .collect();
    constant_gates.push(Array2::from_shape_vec((4, 4), v).unwrap());
    // Fill in the first row
    for qubit in 0..num_qubits {
        if u3 {
            ops.push(Operation::new(
                U3Gate::new().into(),
                vec![qubit],
                vec![0.0; 3],
            ));
        } else {
            ops.push(Operation::new(
                VariableUnitaryGate::new(1, vec![2]).into(),
                vec![qubit],
                vec![0.0; 8],
            ));
        }
    }

    for position in positions {
        ops.push(Operation::new(
            ConstantGate::new(0, 2).into(),
            vec![position, position + 1],
            vec![],
        ));
        ops.push(Operation::new(
            RXGate::new().into(),
            vec![position],
            vec![0.0],
        ));
        ops.push(Operation::new(
            RZGate::new().into(),
            vec![position],
            vec![0.0],
        ));
        ops.push(Operation::new(
            RXGate::new().into(),
            vec![position],
            vec![0.0],
        ));
        ops.push(Operation::new(
            RZGate::new().into(),
            vec![position],
            vec![0.0],
        ));
        if u3 {
            ops.push(Operation::new(
                U3Gate::new().into(),
                vec![position + 1],
                vec![0.0; 3],
            ));
        } else {
            ops.push(Operation::new(
                VariableUnitaryGate::new(1, vec![2]).into(),
                vec![position + 1],
                vec![0.0; 8],
            ));
        }
    }
    Circuit::new(num_qubits, vec![2; num_qubits], ops, constant_gates)
}

fn optimize_ceres(minimizer: &CeresJacSolver, cost: HilbertSchmidtResidualFn, x0: Vec<f64>) -> Vec<f64> {
    minimizer.minimize(ResidualFunction::HilbertSchmidt(cost), x0.clone())
}

fn optimize_qfactor(
    instantiator: &QFactorInstantiator,
    circ: Circuit,
    target: Array2<Complex64>,
    x0: &[f64],
) {
    let _x = instantiator.instantiate(circ, target, x0);
}

/* fn target_from_npy(npy: Vec<u8>, qubits: usize) -> Array2<Complex64> {
    let file = NpyFile::new(&npy[..]).unwrap();
    let size = 2usize.pow(qubits as u32);
    Array2::from_shape_vec((size, size), file.into_vec().unwrap()).unwrap()
}
 */
fn main() {
    // Set random seed for reproducability
    unsafe { srand(21211411) }

    // Positions of CNOTs in each circuit
    let qft3 = vec![1, 0, 1, 0, 1, 0, 1];
    let qft5 = vec![0, 3, 0, 2, 2, 1, 2, 0, 1, 2, 3, 0, 0, 1, 2, 3, 1, 2, 3, 0, 1, 1, 2, 1, 2, 0, 2, 3, 0, 0, 1];
    let mul = vec![2, 1, 0, 1, 2, 3, 1, 2, 1, 2, 0, 0, 1, 1];
    let fredkin = vec![0, 1, 1, 0, 1, 1, 0, 1];
    let qaoa = vec![0, 2, 3, 1, 3, 0, 3, 2, 2, 2, 3, 2, 1, 0, 1, 1, 2, 1, 2, 3, 0, 3, 0, 1, 2, 1, 0];
    let hhl = vec![1, 0, 1, 0];
    let peres = vec![1, 0, 1, 0, 0, 1, 0];
    let grover3 = vec![1, 0, 1, 0, 0, 1, 0];
    let tfim_5_100 = vec![1, 3, 0, 2, 3, 1, 2, 2, 3, 1, 3, 0, 2, 3, 1, 2, 3, 0, 0, 1];
    let toffoli = vec![0, 1, 0, 1, 0, 0, 1, 0];
    let tfim_6_1 = vec![0, 2, 3, 4, 0, 2, 4, 1, 3, 1];
    let adder = vec![1, 2, 1, 0, 1, 0, 1, 0, 1, 2, 1, 2, 0, 1];
    let qft4 = vec![0, 2, 1, 2, 0, 2, 1, 2, 0, 1, 0, 1, 2, 1, 2, 0];
    let or = vec![1, 0, 1, 0, 0, 1, 0, 0];
    let tfim_4_95 = vec![1, 2, 0, 2, 1, 1, 2, 0, 0, 1, 2, 2];
    let vqe = vec![0, 2, 0, 0, 1, 0, 0, 1, 1, 1, 2, 0, 0, 1, 2, 0, 2, 1, 2, 0, 2, 1, 1, 0];
    let hlf = vec![1, 3, 0, 3, 0, 1, 1, 2, 2, 3, 1, 2, 0, 2, 1];

    /*let qft3_bytes = include_bytes!("qft3.npy");
    let qft5_bytes = include_bytes!("qft5.npy");
    let mul_bytes = include_bytes!("mul.npy");*/
    let fredkin_bytes = include_bytes!("fredkin.npy");
    /*let qaoa_bytes = include_bytes!("qaoa.npy");
    let hhl_bytes = include_bytes!("hhl.npy");
    let peres_bytes = include_bytes!("peres.npy");
    let grover3_bytes = include_bytes!("grover3.npy");
    let tfim_5_100_bytes = include_bytes!("tfim-5-100.npy");
    let toffoli_bytes = include_bytes!("toffoli.npy");
    let tfim_6_1_bytes = include_bytes!("tfim-6-1.npy");
    let adder_bytes = include_bytes!("adder.npy");
    let qft4_bytes = include_bytes!("qft4.npy");
    let or_bytes = include_bytes!("or.npy");
    let tfim_4_95_bytes = include_bytes!("tfim-4-95.npy");
    let vqe_bytes = include_bytes!("vqe.npy");
    let hlf_bytes = include_bytes!("hlf.npy");*/

    // Setup numerical optimizers
    let instantiator = QFactorInstantiator::new(None, None, None, None, None, None, None);
    let minimizer = CeresJacSolver::new(1, 1e-6, 1e-10, false);

    //let mut group = c.benchmark_group("ceres");
    let one = r!(1.0);
    let zero = r!(0.0);
    let target: Array2<Complex64> = array![
        [one, zero, zero, zero, zero, zero, zero, zero],
        [zero, one, zero, zero, zero, zero, zero, zero],
        [zero, zero, one, zero, zero, zero, zero, zero],
        [zero, zero, zero, one, zero, zero, zero, zero],
        [zero, zero, zero, zero, one, zero, zero, zero],
        [zero, zero, zero, zero, zero, zero, one, zero],
        [zero, zero, zero, zero, zero, one, zero, zero],
        [zero, zero, zero, zero, zero, zero, zero, one],
    ];
    let circ = make_circuit(fredkin, 3, true);
    let x0 = vec![0.0; circ.num_params()];
    let cost = HilbertSchmidtResidualFn::new(circ, target);
    for _ in 0..10 {
        let cc = cost.clone();
        let x = optimize_ceres(&minimizer, cc, x0.clone());
        assert!(x.iter().sum::<f64>() != 0.);
    }
}
