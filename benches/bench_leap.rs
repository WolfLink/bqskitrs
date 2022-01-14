use bqskitrs::gates::{ConstantGate, U3Gate};
use bqskitrs::passes::{AStarHeuristic, HeuristicFunction};
use bqskitrs::passes::{LayerGenerators, SimpleLayerGenerator};
use bqskitrs::passes::{LeapSynthesisPass, SynthesisPass};
use bqskitrs::r;
use bqskitrs::utils::qft;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use ndarray::arr2;
use num_complex::Complex64;

fn bench_leap(c: &mut Criterion) {
    let qft4 = qft(8);
    let heuristic_func = HeuristicFunction::AStar(AStarHeuristic::default());
    // Linear topology
    let coupling = vec![(0, 1), (1, 2)];
    let cnot = arr2(&[
        [r!(1.), r!(0.), r!(0.), r!(0.)],
        [r!(0.), r!(1.), r!(0.), r!(0.)],
        [r!(0.), r!(0.), r!(0.), r!(1.)],
        [r!(0.), r!(0.), r!(1.), r!(0.)]
    ]);
    let const_gates = vec![cnot];
    let layer_gen = LayerGenerators::Simple(SimpleLayerGenerator::new(
        ConstantGate::new(0, 2).into(),
        U3Gate::new().into(),
        U3Gate::new().into(),
        U3Gate::new().into(),
        const_gates,
        coupling,
    ));
    let leap = LeapSynthesisPass::new(heuristic_func, layer_gen, 1e-10, None, 5);

    let mut group = c.benchmark_group("leap");
    group.sample_size(10);
    group.bench_function("LEAP qft4", |b| b.iter(|| leap.synthesize(qft4.view())));
    group.finish();
}

criterion_group!(leap, bench_leap);
criterion_main!(leap);
