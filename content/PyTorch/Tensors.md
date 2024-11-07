Tensors are a specialized data structure similar to NumPy's ndarray, except that tensors can be GPU accelerated and are optimized for automatic differentiation. Tensors are used to encode the inputs and outputs of a model, as well as the model's parameters.

In fact, tensors and Numpy arrays can often share the same underlying memory, eliminating the need to copy data.

## Tensor Creation
A tensor can be created in various ways: directly from data (the data type is automatically inferred), from a NumPy array, with random or constant values, or from another tensor (the new tensor retains the properties such as dtype of the argument tensor, unless explicitly overridden).

```python
# Directly from data
data = [[1, 2], [3, 4]]
x1 = torch.tensor(data)

# From NumPy array
array = np.array(data)
x2 = torch.from_numpy(array)

# With random or constant values
x3 = torch.empty((2, 3))
x4 = torch.zeros((2, 3))
x5 = torch.ones((2, 3))
x6 = torch.rand((2, 3))

# From another tensor
x7 = torch.ones_like(x1)
```

## Tensor Attributes
A tensor has some key attributes that define its properties:
- `dtype`: the data type of the underlying data
- `shape`: the number of elements along each dimension
- `device`: the device where the tensor is stored

```python
tensor = torch.tensor([[1, 2], [3, 4]])

tensor.dtype  # torch.int64
tensor.shape  # torch.Size([2, 2])
tensor.device # device(type='cpu')
```

## Tensor Operations
Tensors support a wide range of operations, including arithmetic, linear algebra, matrix manipulation, and random sampling. Each of these operations are supported on both CPU and GPU, and therefore can be accelerated by moving the tensor to the GPU.

```python

```

## Random
To  assure reproducibility with random tensors, you can set the seed of the random number generator.

```python
rng = torch.Generator().manual_seed(42)
x = torch.randn((3, 2), generator=rng)
```

