use ndarray::{s, stack, Array1, Array2, Array3, ArrayView2, ArrayView3, ArrayView4, Axis, Ix2};
use ndarray_einsum_beta::einsum;
use ndarray_stats::CorrelationExt;
use num_complex::Complex64;
use squaremat::*;

use crate::{i, r};

use std::f64::consts::{E, PI};

use itertools::Itertools;

pub fn log<const T: usize>(x: usize) -> u32 {
    let mut e = 1;
    while T.pow(e) < x {
        e += 1
    }
    e
}

pub fn trace(arr: ArrayView4<Complex64>) -> Array2<Complex64> {
    let mut out = Array2::<Complex64>::zeros((arr.shape()[2], arr.shape()[3]));
    for i in 0..arr.shape()[2] {
        for j in 0..arr.shape()[3] {
            out[(i, j)] = arr
                .slice(s![.., .., i, ..])
                .slice(s![.., .., j])
                .into_dimensionality::<Ix2>()
                .unwrap()
                .into_owned()
                .diag()
                .sum();
        }
    }
    out
}

pub fn argsort(v: Vec<usize>) -> Vec<usize> {
    v.iter()
        .enumerate()
        .sorted_by(|(_idx_a, a), (_idx_b, b)| a.cmp(b))
        .map(|(idx, _a)| idx)
        .collect()
}

/// Calculate the linear regression of x and y, returning the slope and intercept
pub fn linregress(x: Array1<f64>, y: Array1<f64>) -> Result<(f64, f64), ()> {
    let xmean = x.iter().sum::<f64>() / x.len() as f64;
    let ymean = y.iter().sum::<f64>() / x.len() as f64;
    let m = stack![Axis(1), x, y];
    let mcov = m.cov(0.).unwrap();
    let cov = Array1::from_iter(mcov.iter());
    let ssxm = *cov[0];
    let ssxym = *cov[1];
    if ssxm != 0. {
        let slope = ssxym as f64 / ssxm as f64;
        let intercept = ymean - slope * xmean;
        Ok((slope, intercept))
    } else {
        Err(())
    }
}

pub fn matrix_distance_squared(a: ArrayView2<Complex64>, b: ArrayView2<Complex64>) -> f64 {
    // 1 - np.abs(np.trace(np.dot(A,B.H))) / A.shape[0]
    // converted to
    // 1 - np.abs(np.sum(np.multiply(A,np.conj(B)))) / A.shape[0]
    let prod = einsum("ij,ij->", &[&a, &b.conj()]).unwrap();
    let norm = prod.sum().norm();
    1f64 - norm / a.shape()[0] as f64
}

pub fn matrix_distance_squared_jac(
    u: ArrayView2<Complex64>,
    m: ArrayView2<Complex64>,
    j: ArrayView3<Complex64>,
) -> (f64, Vec<f64>) {
    let size = u.shape()[0];
    let s = u.multiply(&m.conj().view()).sum();
    let dsq = 1f64 - s.norm() / size as f64;
    if s == r!(0.0) {
        return (dsq, vec![std::f64::INFINITY; j.len()]);
    }
    let jus: Vec<Complex64> = j
        .outer_iter()
        .map(|ji| einsum("ij,ij->", &[&u, &ji.conj()]).unwrap().sum())
        .collect();
    let jacs = jus
        .iter()
        .map(|jusi| -(jusi.re * s.re + jusi.im * s.im) * size as f64 / s.norm())
        .collect();
    (dsq, jacs)
}

/// Calculates the residuals
pub fn matrix_residuals(
    a_matrix: &Array2<Complex64>,
    b_matrix: &Array2<Complex64>,
    identity: &Array2<f64>,
) -> Vec<f64> {
    let calculated_mat = b_matrix.matmul(a_matrix.conj().t());
    let (re, im) = calculated_mat.split_complex();
    let resid = re - identity;
    resid.iter().chain(im.iter()).copied().collect()
}

pub fn matrix_residuals_jac(
    u: &Array2<Complex64>,
    _m: &Array2<Complex64>,
    jacs: &Array3<Complex64>,
) -> Array2<f64> {
    let u_conj = u.conj();
    let size = u.shape()[0];
    let mut out = Array2::zeros((jacs.shape()[0], size * size * 2));
    for (jac, mut row) in jacs.outer_iter().zip(out.rows_mut()) {
        let m = jac.matmul(u_conj.t());
        let (re, im) = m.split_complex();
        let data = Array1::from_vec(re.iter().chain(im.iter()).copied().collect());
        row.assign(&data);
    }
    out.reversed_axes()
}

pub fn qft(n: usize) -> Array2<Complex64> {
    let root = r!(E).powc(i!(2f64) * PI / n as f64);
    Array2::from_shape_fn((n, n), |(x, y)| root.powf((x * y) as f64)) / (n as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::linregress;
    use ndarray::Array1;
    use proptest::prelude::*;

    fn gen_x_y() -> impl Strategy<Value = (Array1<f64>, Array1<f64>)> {
        (2..1000).prop_flat_map(|size| {
            (0f64..1000f64).prop_flat_map(move |upper| {
                (
                    Just(Array1::linspace(0f64, 1000f64, size as usize)),
                    Just(Array1::linspace(-1000f64, upper, size as usize)),
                )
            })
        })
    }

    proptest! {
        #[test]
        fn linregress_test((x, y) in gen_x_y()) {
            let (slope, intercept) = linregress(x.clone(), y.clone()).unwrap();
            assert!(slope * x[x.len() / 2] + intercept - y[y.len() / 2] < 1e-10);
        }
    }
}
