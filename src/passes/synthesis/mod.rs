use ndarray::ArrayView2;
use num_complex::Complex64;

use crate::circuit::Circuit;
use crate::gates::Unitary;

use super::Pass;

mod leap;
pub use leap::LeapSynthesisPass;

pub trait SynthesisPass {
    fn synthesize(&self, utry: ArrayView2<Complex64>) -> Circuit;
}

impl<T> Pass for T
where
    T: SynthesisPass,
{
    fn run(&self, circuit: &mut Circuit) {
        let utry = circuit.get_utry(&[], &circuit.constant_gates);
        std::mem::swap(
            circuit,
            &mut <Self as SynthesisPass>::synthesize(&self, utry.view()),
        );
    }
}
