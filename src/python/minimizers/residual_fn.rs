use crate::minimizers::{
    CostFn, DifferentiableResidualFn, HilbertSchmidtResidualFn, ResidualFn, ResidualFunction,
};
use ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::{prelude::*, types::PyTuple};

struct PyResidualFn {
    cost_fn: PyObject,
}

impl PyResidualFn {
    pub fn new(cost_fn: PyObject) -> Self {
        PyResidualFn { cost_fn }
    }
}

impl CostFn for PyResidualFn {
    fn get_cost(&self, params: &[f64]) -> f64 {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let parameters = PyArray1::from_slice(py, params);
        let args = PyTuple::new(py, &[parameters]);
        match self.cost_fn.call_method1(py, "get_cost", args) {
            Ok(val) => val
                .extract::<f64>(py)
                .expect("Return type of get_cost was not a float."),
            Err(..) => panic!("Failed to call 'get_cost' on passed ResidualFunction."), // TODO: make a Python exception?
        }
    }
}

impl ResidualFn for PyResidualFn {
    fn num_residuals(&self) -> usize {
        let gil = Python::acquire_gil();
        let py = gil.python();
        match self.cost_fn.call_method0(py, "num_residuals") {
            Ok(val) => val
                .extract::<usize>(py)
                .expect("Return of num_residuals was not an integer."),
            Err(..) => panic!("Failed to call num_residuals on passed residual function."),
        }
    }

    fn get_residuals(&self, params: &[f64]) -> Vec<f64> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let parameters = PyArray1::from_slice(py, params);
        let args = PyTuple::new(py, &[parameters]);
        match self.cost_fn.call_method1(py, "get_cost", args) {
            Ok(val) => val
                .extract::<Vec<f64>>(py)
                .expect("Return type of get_cost was not a sequence of floats."),
            Err(..) => panic!("Failed to call 'get_cost' on passed ResidualFunction."), // TODO: make a Python exception?
        }
    }
}

impl DifferentiableResidualFn for PyResidualFn {
    fn get_grad(&self, params: &[f64]) -> Array2<f64> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let parameters = PyArray1::from_slice(py, params);
        let args = PyTuple::new(py, &[parameters]);
        let pyarray = match self.cost_fn.call_method1(py, "get_grad", args) {
            Ok(val) => val
                .extract::<Py<PyArray2<f64>>>(py)
                .expect("Return type of get_grad was not a matrix of floats."),
            Err(..) => panic!("Failed to call 'get_grad' on passed ResidualFunction."), // TODO: make a Python exception?
        };
        pyarray.as_ref(py).to_owned_array()
    }

    fn get_residuals_and_grad(&self, params: &[f64]) -> (Vec<f64>, Array2<f64>) {
        (self.get_residuals(params), self.get_grad(params))
    }
}

#[pyclass(name = "HilbertSchmidtResidualFunction", subclass, module = "bqskitrs")]
pub struct PyHilberSchmidtResidualFn {
    cost_fn: HilbertSchmidtResidualFn,
}

fn is_cost_fn_obj<'a>(obj: &'a PyAny) -> PyResult<bool> {
    if obj.hasattr("get_cost")? {
        let get_cost = obj.getattr("get_cost")?;
        if get_cost.is_callable() {
            if obj.hasattr("get_grad")? {
                if obj.getattr("get_grad")?.is_callable() {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

impl<'source> FromPyObject<'source> for ResidualFunction {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        match ob.extract::<Py<PyHilberSchmidtResidualFn>>() {
            Ok(fun) => Ok(ResidualFunction::HilbertSchmidt(
                fun.try_borrow(py)?.cost_fn.clone(),
            )),
            Err(..) => {
                if is_cost_fn_obj(ob)? {
                    let fun = PyResidualFn::new(ob.into());
                    Ok(ResidualFunction::Dynamic(Box::new(fun)))
                } else {
                    panic!("Failed to extract ResidualFn from obj."); // TODO: throw a Python error here.
                }
            }
        }
    }
}
