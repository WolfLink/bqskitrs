use crate::circuit::Circuit;

mod search;
mod synthesis;

pub use search::*;
pub use synthesis::*;

pub trait Pass {
    fn run(&self, circuit: &mut Circuit);
}
