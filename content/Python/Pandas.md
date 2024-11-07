Pandas is an open-source data manipulation and analysis library for Python, primarily used for working with structured data. It provides a wide range of data operations, including data cleaning, merging, reshaping, filtering, and aggregation, making it a core tool in data science.

Pandas provides two types of data structures as classes:
- `Series`: a one-dimensional labeled array with homogeneous data.
- `DataFrame`: a two-dimensional labeled array with columns of potentially different types.

Customarily, `pandas` is imported as follows:

```python
import pandas as pd
```

## Object Creation
A `Series` can be created by passing a list, a dictionary, an ndarray, or a scalar value. You can optionally pass an `index`. If `index` is not passed, a default index will be created.

```python
# A Series with one missing element
s1 = pd.Series([1, 3, 5, np.nan, 6, 8])

# A Series of 5 random numbers with custom index and a name
s2 = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"], name="A")

# A Series with 3 elements and custom index derived from dictionary keys
s3 = pd.Series({"a": 0.0, "b": 1.0, "c": 2.0})

# A Series with repeated 42 elements and custom index
s4 = pd.Series(42, index=[1, 2, 3, 4, 5])
```

A `DataFrame` can be created by passing a dictionary where the keys are mapped to column names and their values are mapped to column values. The values of the dictionary keys should be 1-D array-like structures such as ndarrays, lists, and Series.

A `DataFrame` can also be created by passing a 2-D ndarray, along with an optional `index` (row labels) and `columns` (column labels).

```python
# Creates a DataFrame from dictionary of Series
df1 = pd.DataFrame({
	"A": pd.Series([1, 2, 3], index=["a", "b", "c"]),
	"B": pd.Series([4, 5, 6], index=["a", "b", "c"])
})

# Creates a DataFrame from dictionary of lists with custom index
df2 = pd.Series({
	"A": [1, 2, 3],
	"B": [4, 5, 6]
}, index=["a", "b", "c"])

# Creates a DataFrame from 2-D ndarray with custom index and column labels
df3 = pd.DataFrame(np.random.randn(6, 4),
				  index=pd.date_range("20130101", periods=6),
				  columns=["A", "B", "C", "D"])

# Creates a DataFrame from dictionary with columns of different types
df4 = pd.DataFrame({
	"A": 1.0,
	"B": pd.Timestamp("20130102"),
	"C": pd.Series(1, index=list(range(4)), dtype="float32"),
	"D": np.array([3] * 4, dtype="int32"),
	"E": pd.Categorical(["test", "train", "test", "train"]),
	"F": "foo",
})
```

## Viewing Data
To view a sample of the data, you can use the `head()` and `tail()` methods which returns the first few and last few rows respectively. The default number of elements in five, but a custom number can be passed as an argument.

```python
df.head() # Displays the first 5 rows of the data
df.head(3) # Displays the first 3 rows of the data

df.tail() # Displays the last 5 rows of the data
df.tail(3) # Displays the last 3 rows of the data
```

You can get the column labels through `DataFrame.columns` and index through `DataFrame.index`. You can get a NumPy representation of the underlying data without the index or column labels through `DataFrame.to_numpy()`.

```python
df.index # Returns the index of the DataFrame 
df.columns # Returns the column labels of the DataFrame
df.to_numpy() # Returns a NumPy array for the underlying data
```

You can view a quick statistic summary of the data through `DataFrame.describe()`.

```python
df.describe() # Displays a statistical summary of the data in each column
```

## Selection
In Pandas, data can be accessed using these methods: indexing with the `[]` operator, selection by label with the `loc[]` and `at[]` methods, selection by position with the `iloc[]` and `iat[]` methods, and boolean indexing.

Pandas support indexing notation similar to Python and NumPy:
- Passing a single label (such as `df["A"]`) selects that column and yields a Series.
- Passing a slice (such as `df[1:2]`) selects the corresponding rows.

Although Pandas support indexing notation, it is recommended to use the optimized data access methods:

| Method   | Description                                                      |
| -------- | ---------------------------------------------------------------- |
| `at[]`   | Access a single value for a row/column pair by label.            |
| `iat[]`  | Access a single value for a row/column pair by integer position. |
| `loc[]`  | Access a group of rows and columns by label(s).                  |
| `iloc[]` | Access a group of rows and columns by integer position(s).       |

The `loc[]` method accepts one or more arguments for each of the axis. Each argument can be:
- a single label, such as `loc["I1"]`,
- a list of labels, such as `loc[["I1", "I2"]]`, or
- a slice with labels, such as `loc["I1":"I2"]`, where both start and stop are included.

The `iloc[]` method is used when you want to use integer positions rather than labels. It should be noted that the slice notation in `iloc[]` method does not include the stop value. The `at[]` and `iat[]` methods are used when you want to get or set a single value in the DataFrame or Series.

If you want to filter a DataFrame using some condition, such as selecting all rows where the value of a column is greater than zero, you can use boolean indexing.

```python
df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
				  index=["I0", "I1", "I2"], columns=["A", "B", "C"])
#      A   B   C
# I0   1   2   3
# I1   4   5   6
# I2   7   8   9

df["A"] # Returns a Series for the column "A"
df.A # Equivalent to df["A"]
df[0:1] # Selects the first row
df[:] # Selects all rows

df.at["I1", "B"] # 5
df.at["I1"] # TypeError
df.loc["I1"].at["B"] # 5

df.iat[1, 1] # 5
df.loc["I1"].iloc[1] # 5

df.loc["I1"] # Returns the row with index "I1"
df.loc["B"] # KeyError
df.loc["I1:I2"] # Selects the rows I1 and I2
df.loc[["I1", "I2"]] # Selects the rows I1 and I2
df.loc["I1":"I2", "B"] # Selects the rows I1 and I2 with column B
df.loc[:, "B"] # Selects all rows with column B

df.iloc[1] # Selects the row at index 1
df.iloc[0:2] # Selects the rows at index 0 and 1
df.iloc[0:2, 1] # Selects the rows at index 0 and 1 with second column
df.iloc[:, 1] # Selects all rows with second column
df.iloc[[0, 1], [1, 2]] # Selects first two rows with second and third column

df[df["A"] > 3] # Selects the second and third row
df[df > 5] # All values less than 5 are changed to NaN while others are kept
df[df["B"].isin([2, 8])] # Selects the first and third row
```

## Missing Data
## Operations
## Sorting
There are two types of sorting in pandas: `sort_index()` (sorting by index) and `sort_values()` (sorting by values). These methods are available for both series and dataframes. You can also get the transposed version of a dataframe using the `df.T` attribute.

```python
df.sort_index() # Sorts by index
df.sort_index(axis=1) # Sorts by column labels
df.sort_index(ascending=False) # Sorts by index in descending order
df.sort_index(key=lambda idx: idx) # Sorts by index using custom function

df.sort_values(by="B") # Sorts by values of the column "B"
df.sort_values(by="B", ascending=False) # Sorts in descending order
df.sort_values(by=["A", "B"]) # Sorts by values of "A" but for tie, sorts by "B"
df.sort_values(by="B", key=lambda col: col) # Sorts by "B" using custom function

df.T # Transposed version where index become column labels and vice versa
```

## Merging


## Grouping
## Reshaping
## Time Series
## Categorical
## Plotting
## Importing Data
