use crate::circuit::Circuit;
use enum_dispatch::enum_dispatch;
use ndarray::ArrayView2;
use num_complex::Complex64;

mod simple;

pub use simple::SimpleLayerGenerator;

#[enum_dispatch(LayerGenerator)]
pub enum LayerGenerators {
    Simple(SimpleLayerGenerator),
}

#[enum_dispatch]
pub trait LayerGenerator {
    fn initial_layer(&self, target: ArrayView2<Complex64>) -> Circuit;
    fn successors(&self, circuit: Circuit) -> Vec<Circuit>;
}
