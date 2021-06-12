use crate::{
    circuit::Circuit,
    minimizers::{CostFn, CostFunction, DifferentiableCostFn, HilbertSchmidtCostFn},
};
use num_complex::Complex64;
use numpy::{PyArray1, PyArray2};
use pyo3::{prelude::*, types::PyTuple};
use squaremat::SquareMatrix;

struct PyCostFn {
    cost_fn: PyObject,
}

impl PyCostFn {
    pub fn new(cost_fn: PyObject) -> Self {
        PyCostFn { cost_fn }
    }
}

impl CostFn for PyCostFn {
    fn get_cost(&self, params: &[f64]) -> f64 {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let parameters = PyArray1::from_slice(py, params);
        let args = PyTuple::new(py, &[parameters]);
        match self.cost_fn.call_method1(py, "get_cost", args) {
            Ok(val) => val
                .extract::<f64>(py)
                .expect("Return type of get_cost was not a float."),
            Err(..) => panic!("Failed to call 'get_cost' on passed CostFunction."), // TODO: make a Python exception?
        }
    }
}

impl DifferentiableCostFn for PyCostFn {
    fn get_grad(&self, params: &[f64]) -> Vec<f64> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let parameters = PyArray1::from_slice(py, params);
        let args = PyTuple::new(py, &[parameters]);
        match self.cost_fn.call_method1(py, "get_grad", args) {
            Ok(val) => val
                .extract::<Vec<f64>>(py)
                .expect("Return type of get_grad was not a list of floats."),
            Err(..) => panic!("Failed to call 'get_grad' on passed CostFunction."), // TODO: make a Python exception?
        }
    }
}

#[pyclass(name = "HilbertSchmidtCostFunction", subclass, module = "bqskitrs")]
pub struct PyHilberSchmidtCostFn {
    cost_fn: HilbertSchmidtCostFn,
}

#[pymethods]
impl PyHilberSchmidtCostFn {
    #[new]
    pub fn new(circ: Circuit, target_matrix: &PyArray2<Complex64>) -> Self {
        let target = SquareMatrix::from_ndarray(target_matrix.to_owned_array());
        PyHilberSchmidtCostFn {
            cost_fn: HilbertSchmidtCostFn::new(circ, target),
        }
    }

    pub fn get_cost(&self, _py: Python, params: Vec<f64>) -> f64 {
        self.cost_fn.get_cost(&params)
    }

    pub fn get_grad(&self, _py: Python, params: Vec<f64>) -> Vec<f64> {
        self.cost_fn.get_grad(&params)
    }

    pub fn get_cost_and_grad(&self, _py: Python, params: Vec<f64>) -> (f64, Vec<f64>) {
        self.cost_fn.get_cost_and_grad(&params)
    }
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

impl<'source> FromPyObject<'source> for CostFunction {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        match ob.extract::<Py<PyHilberSchmidtCostFn>>() {
            Ok(fun) => Ok(CostFunction::HilbertSchmidt(
                fun.try_borrow(py)?.cost_fn.clone(),
            )),
            Err(..) => {
                if is_cost_fn_obj(ob)? {
                    let fun = PyCostFn::new(ob.into());
                    Ok(CostFunction::Dynamic(Box::new(fun)))
                } else {
                    panic!("Failed to extract CostFn from obj."); // TODO: throw a Python error here.
                }
            }
        }
    }
}
