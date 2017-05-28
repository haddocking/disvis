#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include "numpy/arrayobject.h"

#define SQUARE(x) ((x) * (x))
#define MIN(a, b) ((a) > (b) ? (b) : (a))
#define MAX(a, b) ((a) > (b) ? (a) : (b))


static int modulo(npy_intp x, npy_intp N)
{
  npy_intp ret = x % N;
  if (ret < 0)
    ret += N;
  return ret;
}


static void dilate_points(
    PyArrayObject *pypoints, PyArrayObject *pyradii, double value, 
    PyArrayObject *pyout)
{
  double *points = (double *) PyArray_DATA(pypoints);
  double *radii = (double *) PyArray_DATA(pyradii);
  double *out = (double *) PyArray_DATA(pyout);

  npy_intp *points_shape = PyArray_DIMS(pypoints);
  npy_intp *out_shape = PyArray_DIMS(pyout);
  npy_intp out_slice = out_shape[2] * out_shape[1];
  npy_intp out_size = out_shape[0] * out_slice;

  for (npy_intp n = 0; n < points_shape[1]; ++n) {
    double center_x = points[n];
    double center_y = points[n + points_shape[1]];
    double center_z = points[n + 2 * points_shape[1]];
    double radius_max = radii[n];
    double radius2_max = SQUARE(radius_max);
    int zmin = (int) ceil(center_z - radius_max);
    int ymin = (int) ceil(center_y - radius_max);
    int xmin = (int) ceil(center_x - radius_max);
    int zmax = (int) floor(center_z + radius_max);
    int ymax = (int) floor(center_y + radius_max);
    int xmax = (int) floor(center_x + radius_max);
    for (npy_intp z = zmin; z <= zmax; ++z) {
      npy_intp ind_z = modulo(z * out_slice, out_size);
      double radius2_z = SQUARE(z - center_z);
      for (npy_intp y = ymin; y <= ymax; ++y) {
        npy_intp ind_zy = modulo(y * out_shape[2], out_slice) + ind_z;
        double radius2_zy = radius2_z + SQUARE(y - center_y);
        for (npy_intp x = xmin; x <= xmax; ++x) {
          double radius2 = radius2_zy + SQUARE(x - center_x);
          if (radius2 <= radius2_max) {
            npy_intp ind_zyx = ind_zy + modulo(x, out_shape[2]);
            out[ind_zyx] = value;
          }
        }
      }
    }
  }
}


static PyObject *py_dilate_points(PyObject *dummy, PyObject *args)
{
  PyObject *pypoints_arg=NULL, *pyradii_arg=NULL, *pyout_arg=NULL;
  PyArrayObject *pypoints=NULL, *pyradii=NULL, *pyout=NULL;
  double value;
  if (!PyArg_ParseTuple(args, "OOdO", &pypoints_arg, &pyradii_arg, 
                        &value, &pyout_arg)) 
    return NULL;

  pypoints = (PyArrayObject *) 
      PyArray_FROM_OTF(pypoints_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
  if (pypoints == NULL)
    goto fail;
  pyradii = (PyArrayObject *) 
      PyArray_FROM_OTF(pyradii_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
  if (pyradii == NULL)
    goto fail;
  pyout = (PyArrayObject *) 
      PyArray_FROM_OTF(pyout_arg, NPY_FLOAT64, NPY_ARRAY_INOUT_ARRAY);
  if (pyout == NULL)
    goto fail;

  dilate_points(pypoints, pyradii, value, pyout);

  Py_DECREF(pypoints);
  Py_DECREF(pyradii);
  Py_DECREF(pyout);
  Py_INCREF(Py_None);
  return Py_None;
fail:
  // Clean up objects
  Py_XDECREF(pypoints);
  Py_XDECREF(pyradii);
  PyArray_XDECREF_ERR(pyout);
  return NULL;
}


static void fill_restraint_space(
    PyArrayObject *py_rsel, PyArrayObject *py_lsel, 
    double min_dis, double max_dis, int value,
    PyArrayObject *py_out
    )
{
  double *rsel = (double *) PyArray_DATA(py_rsel);
  double *lsel = (double *) PyArray_DATA(py_lsel);
  int *out = (int *) PyArray_DATA(py_out);

  npy_intp *rsel_shape = PyArray_DIMS(py_rsel);
  npy_intp *lsel_shape = PyArray_DIMS(py_lsel);
  npy_intp *out_shape = PyArray_DIMS(py_out);
  npy_intp out_slice = out_shape[2] * out_shape[1];
  double max_dis2 = SQUARE(max_dis);
  double min_dis2 = SQUARE(min_dis);

  for (npy_intp nr = 0; nr < rsel_shape[0]; ++nr) {
    for (npy_intp nl = 0; nl < lsel_shape[0]; ++nl) {
      double center_x = rsel[3 * nr] - lsel[3 * nl];
      double center_y = rsel[3 * nr + 1] - lsel[3 * nl + 1];
      double center_z = rsel[3 * nr + 2] - lsel[3 * nl + 2];

      int zmin = MAX((int) ceil(center_z - max_dis), 0);
      int ymin = MAX((int) ceil(center_y - max_dis), 0);
      int xmin = MAX((int) ceil(center_x - max_dis), 0);
      int zmax = MIN((int) floor(center_z + max_dis), out_shape[0] - 1);
      int ymax = MIN((int) floor(center_y + max_dis), out_shape[1] - 1);
      int xmax = MIN((int) floor(center_x + max_dis), out_shape[2] - 1);

      for (npy_intp z = zmin; z <= zmax; z++) {
        double dist2_z = SQUARE(z - center_z);
        npy_intp ind_z = z * out_slice;
        for (npy_intp y = ymin; y <= ymax; y++) {
          double dist2_zy = SQUARE(y - center_y) + dist2_z;
          npy_intp ind_zy = y * out_shape[2] + ind_z;
          for (npy_intp x = xmin; x <= xmax; x++) {
            double dist2_zyx = SQUARE(x - center_x) + dist2_zy;
            if ((dist2_zyx <= max_dis2) && (dist2_zyx >= min_dis2))
              out[ind_zy + x] |= value;
          }
        }
      }
    }
  }
}


static PyObject *py_fill_restraint_space(PyObject *dummy, PyObject *args)
{
   // Parse arguments
  PyObject 
      *arg1=NULL, *arg2=NULL, *arg6=NULL;
  PyArrayObject 
      *py_rsel=NULL, *py_lsel=NULL, *py_out=NULL;
  double 
      min_dis, max_dis;
  int value;

  if (!PyArg_ParseTuple(args, "OOddiO", &arg1, &arg2, &min_dis, 
                        &max_dis, &value, &arg6))
    return NULL;

  py_rsel = (PyArrayObject *) 
      PyArray_FROM_OTF(arg1, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
  if (py_rsel == NULL)
    goto fail;

  py_lsel = (PyArrayObject *) 
      PyArray_FROM_OTF(arg2, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
  if (py_lsel == NULL)
    goto fail;

  py_out = (PyArrayObject *) 
      PyArray_FROM_OTF(arg6, NPY_INT32, NPY_ARRAY_INOUT_ARRAY);
  if (py_out == NULL)
    goto fail;

  fill_restraint_space(py_rsel, py_lsel, min_dis, max_dis, value, py_out);

  // Clean up objects
  Py_DECREF(py_rsel);
  Py_DECREF(py_lsel);
  Py_DECREF(py_out);
  Py_INCREF(Py_None);
  return Py_None;

fail:
  // Clean up objects
  Py_XDECREF(py_rsel);
  Py_XDECREF(py_lsel);
  PyArray_XDECREF_ERR(py_out);
  return NULL;
}


static void fill_restraint_space_add(
    PyArrayObject *py_rsel, PyArrayObject *py_lsel, 
    double min_dis, double max_dis,
    PyArrayObject *py_out
    )
{
  double *rsel = (double *) PyArray_DATA(py_rsel);
  double *lsel = (double *) PyArray_DATA(py_lsel);
  int *out = (int *) PyArray_DATA(py_out);

  npy_intp *rsel_shape = PyArray_DIMS(py_rsel);
  npy_intp *lsel_shape = PyArray_DIMS(py_lsel);
  npy_intp *out_shape = PyArray_DIMS(py_out);
  npy_intp out_slice = out_shape[2] * out_shape[1];
  double max_dis2 = SQUARE(max_dis);
  double min_dis2 = SQUARE(min_dis);

  for (npy_intp nr = 0; nr < rsel_shape[0]; ++nr) {
    for (npy_intp nl = 0; nl < lsel_shape[0]; ++nl) {
      double center_x = rsel[3 * nr] - lsel[3 * nl];
      double center_y = rsel[3 * nr + 1] - lsel[3 * nl + 1];
      double center_z = rsel[3 * nr + 2] - lsel[3 * nl + 2];

      int zmin = MAX((int) ceil(center_z - max_dis), 0);
      int ymin = MAX((int) ceil(center_y - max_dis), 0);
      int xmin = MAX((int) ceil(center_x - max_dis), 0);
      int zmax = MIN((int) floor(center_z + max_dis), out_shape[0] - 1);
      int ymax = MIN((int) floor(center_y + max_dis), out_shape[1] - 1);
      int xmax = MIN((int) floor(center_x + max_dis), out_shape[2] - 1);

      for (npy_intp z = zmin; z <= zmax; z++) {
        double dist2_z = SQUARE(z - center_z);
        npy_intp ind_z = z * out_slice;
        for (npy_intp y = ymin; y <= ymax; y++) {
          double dist2_zy = SQUARE(y - center_y) + dist2_z;
          npy_intp ind_zy = y * out_shape[2] + ind_z;
          for (npy_intp x = xmin; x <= xmax; x++) {
            double dist2_zyx = SQUARE(x - center_x) + dist2_zy;
            if ((dist2_zyx <= max_dis2) && (dist2_zyx >= min_dis2))
              out[ind_zy + x] += 1;
          }
        }
      }
    }
  }
}


static PyObject *py_fill_restraint_space_add(PyObject *dummy, PyObject *args)
{
   // Parse arguments
  PyObject 
      *arg1=NULL, *arg2=NULL, *arg6=NULL;
  PyArrayObject 
      *py_rsel=NULL, *py_lsel=NULL, *py_out=NULL;
  double 
      min_dis, max_dis;

  if (!PyArg_ParseTuple(args, "OOddO", &arg1, &arg2, &min_dis, 
                        &max_dis, &arg6))
    return NULL;

  py_rsel = (PyArrayObject *) 
      PyArray_FROM_OTF(arg1, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
  if (py_rsel == NULL)
    goto fail;

  py_lsel = (PyArrayObject *) 
      PyArray_FROM_OTF(arg2, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
  if (py_lsel == NULL)
    goto fail;

  py_out = (PyArrayObject *) 
      PyArray_FROM_OTF(arg6, NPY_INT32, NPY_ARRAY_INOUT_ARRAY);
  if (py_out == NULL)
    goto fail;

  fill_restraint_space_add(py_rsel, py_lsel, min_dis, max_dis, py_out);

  // Clean up objects
  Py_DECREF(py_rsel);
  Py_DECREF(py_lsel);
  Py_DECREF(py_out);
  Py_INCREF(Py_None);
  return Py_None;

fail:
  // Clean up objects
  Py_XDECREF(py_rsel);
  Py_XDECREF(py_lsel);
  PyArray_XDECREF_ERR(py_out);
  return NULL;
}


static PyMethodDef mymethods[] = {
  {"dilate_points", py_dilate_points, METH_VARARGS, ""},
  {"fill_restraint_space", py_fill_restraint_space, METH_VARARGS, ""},
  {"fill_restraint_space_add", py_fill_restraint_space_add, METH_VARARGS, ""},
  {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC
init_extensions(void)
{
  (void) Py_InitModule("_extensions", mymethods);
  import_array();
};

