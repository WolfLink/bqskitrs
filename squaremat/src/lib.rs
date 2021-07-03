use ndarray::Array2;
use num_complex::Complex64;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
extern crate cblas;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
#[cfg(any(feature = "openblas-static", feature = "openblas-system"))]
extern crate openblas_src;

#[macro_export]
macro_rules! c {
    ($re:expr, $im:expr) => {
        Complex64::new($re, $im)
    };
}

#[macro_export]
macro_rules! r {
    ($re:expr) => {
        Complex64::new($re, 0.0)
    };
}

#[macro_export]
macro_rules! i {
    ($im:expr) => {
        Complex64::new(0.0, $im)
    };
}

mod conj;
mod kron;
mod multiply;
mod split_complex;
mod swap_rows;
mod trace;

pub use conj::Conj;
pub use kron::Kronecker;
pub use multiply::Multiply;
pub use split_complex::SplitComplex;
pub use swap_rows::SwapRows;
pub use trace::Trace;
