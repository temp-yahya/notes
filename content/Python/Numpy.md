## Arrays
Numpy provides an N-dimensional array called the ndarray, which describes a collection of items with the same data type and size. The number of dimensions in an array is defined by its `shape` attribute, and the type of items in the array is specified by its `dtype` attribute.

A numpy array consists of two parts: a contiguous data buffer and metadata. The metadata contains information such as:
- **shape**: a tuple specifying the size of each dimension of the array.
- **dtype**: the data type of the items in the array.
- **offset**: the offset from the beginning of the memory block containing the data.
- **strides**: the separation between elements in each dimension, specified in bytes.

An item extracted from an array (such as by indexing) is represented by an array scalar object whose type is one of the 24 fundamental types built in numpy (such as `numpy.int64`).

Arrays scalars have the same attributes and methods as ndarrays. This smooths out rough edges that result when mixing scalar and array operations.

| Attribute          | Description                                                        |
| ------------------ | ------------------------------------------------------------------ |
| `ndarray.shape`    | Tuple of array dimensions.                                         |
| `ndarray.strides`  | Tuple of bytes to step in each dimension when traversing an array. |
| `ndarray.ndim`     | Number of array dimensions.                                        |
| `ndarray.size`     | Number of elements in the array.                                   |
| `ndarray.itemsize` | Length of one array element in bytes.                              |
| `ndarray.nbytes`   | Total bytes consumed by the elements in the array.                 |
| `ndarray.base`     | Base object if array is a view of another object.                  |
| `ndarray.dtype`    | Data type of the array elements.                                   |
| `ndarray.T`        | View of the transposed array.                                      |
| `ndarray.real`     | The real part of an array.                                         |
| `ndarray.imag`     | The imaginary part of an array.                                    |
| `ndarray.flat`     | An iterator over the flattened array.                              |

## Fundamentals

### Indexing


### Broadcasting


### Copies and views
A numpy array can be accessed differently by simply changing certain metadata without changing the data buffer. This creates a new way of looking at the same data and these new arrays are called views. A view can be forced through the `ndarray.view()` method.

```python
x = np.array([1,2,3,4])
y = x.view(dtype=np.int16, type=np.matrix)
```

When a new array is created by duplicating the data buffer as well as the metadata, it is called a copy. Changes made to the copy do not reflect on the original array. A copy can be forced through the `ndarray.copy()` method.

```python
x = np.array([1,2,3,4])
y = x.copy()
```

The `base` attribute of an array can be used to tell if an array is a view or a copy:
- View: the `base` attribute returns the original array.
- Copy: the `base` attribute returns `None`.

Basic indexing and many numpy methods return views instead of copies. As a general rule of thumb, if an operation can return a view, it will do so by default.

## Routines
### Array creation
Numpy provide several methods for creating arrays, either from existing data, from shape or value information, or from random generators.

| Method                      | Description                                                               |
| --------------------------- | ------------------------------------------------------------------------- |
| `array(object)`             | Creates an array from a Python list or tuple.                             |
| `arange(start,stop,step)`   | Creates an array with evenly spaced values within the given interval.     |
| `linspace(start,stop,num)`  | Creates an array with a fixed number of values within the given interval. |
| `eye(rows,cols)`            | Creates a 2-D array with ones on the diagonal and zeros elsewhere.        |
| `zeros(shape)`              | Creates an array filled with zeros.                                       |
| `empty(shape)`              | Creates an array without initializing its values.                         |
| `random.rand(d0,d1,...,dn)` | Creates an array filled with random values from the interval $[0,1)$.     |

### Array manipulation
Numpy provides several methods for manipulating arrays. These manipulations include changing the array shape, transpose operations, changing number of dimensions, stacking arrays, and splitting arrays.

| Method                  | Description                                                   |
| ----------------------- | ------------------------------------------------------------- |
| `reshape(shape)`        | Returns an array with the same data but with a new shape.     |
| `ravel()`               | Returns the array collapsed into one dimension.               |
| `transpose()`           | Returns the array with axes transposed.                       |
| `squeeze()`             | Removes axes of length one from the array.                    |
| `expand_dims(a,axis)`   | Expands the shape of an array by inserting new axes.          |
| `stack(arrays,axis)`    | Joins a sequence of arrays along a new axis.                  |
| `split(a,indices,axis)` | Splits an array into multiple subarrays along the given axis. |
