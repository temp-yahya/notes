Python provides a variety of built-in data structures such as lists, tuples, sets, and dictionaries. Besides these, there are some more advanced data structures contained in the `collections` module. There are also many modules which provide some useful functions for working with these data structures. 

## Unpacking Items
If you want to unpack an $N$-item sequence into a collection of $N$ variables, you can simply use the assignment operator. The only requirement is that the number of variables match the number of items in the sequence.

Unpacking works with any iterable, not just tuples, so this includes lists, strings, files, iterators, and generators. If you want to discard certain values, you can use a throwaway variable for it.

```python
x, y = (4, 5)
a, b, c, d, e = "Hello"
_, share, price, _ = ["ACME", 50, 91.1, (2012, 12, 21)]

print(x)     # 4
print(a)     # "H"
print(share) # 50
```

If you want to unpack a sequence of arbitrary length, you can use the star expression (`*`). The star expression will consume the remaining items into a list.

```python
name, email, *phone_numbers = ("Dave", "dave@example.com", "773-555-1212", "847-555-1212")
*trailing, current = [10, 8, 7, 1, 9, 5, 10, 3]

print(trailing) # [10, 8, 7, 1, 9, 5, 10]
print(current)  # 3
```

## Named Tuples
If you use tuples and want to decouple your code from the position of elements, you can use `namedtuple()` from the `collections` module. It provides an interface similar to normal classes, while supporting standard tuple operations and being very memory efficient.

```python
from collections import namedtuple

Subscriber = namedtuple("Subscriber", ["name", "email"])
sub = Subscriber("Mike", "mike@example.com")

name, email = sub.name, sub.email
# name = Mike
# email = mike@example.com
```

## $N$-Largest Items
If you want to find the $N$-largest or $N$-smallest values in a container efficiently, you can use the following techniques, depending on the comparison between $N$ and the size of the container:
- If $N$ is $1$, it is faster to use `min()` and `max()` functions.
- If $N$ is small compared to `size`, it is faster to convert the container to a heap (which is sorted) then extract the smallest or largest values from its end.
- If $N$ is close to `size`, it is faster to sort the container and then use the slicing operation, as in `sorted(items)[:N]`.

The `nlargest()` and `nsmallest()` functions from the `heapq` module can be used to efficiently find the $N$-largest or $N$-smallest items in a container, by using these optimizations under the hood.

```python
import heapq

nums = [1, 8, 2, 23, 7, -4, 18, 23, 42, 37, 2]
print(heapq.nsmallest(3, nums)) # [-4, 1, 2]
print(heapq.nlargest(3, nums))  # [42, 37, 23]

portfolio = [
	{"name": "IBM", "shares": 100, "price": 91.1},
	{"name": "AAPL", "shares": 50, "price": 543.22},
	{"name": "FB", "shares": 200, "price": 21.09},
]

# With complicated containers, you can provide the key parameter
cheap = heapq.nsmallest(3, portfolio, key=lambda s: s["price"])
expensive = heapq.nlargest(3, portfolio, key=lambda s: s["price"])
```

## Last $N$-Items
If you want to keep a limited history of the last $N$-items seen during iteration, you can use `deque` from the `collections` module, which allows to push and pop items at either end in constant time.

Queues can be bounded by specifying a maximum size. When the maximum size is specified, the queue automatically pops items from its left size when new items are added from the right.

```python
from collections import deque

def search(lines, pattern, history=5):
	prevlines = deque(maxlen=history)
	for lines in line:
		if pattern in line:
			yield line, prevlines
		prevlines.append(line)

# Whenever this function is called
# "line" will contain the current line, and
# "prevlines" will contain the history of the last few lines
```



## Custom Sorting
If you want to sort objects without native comparison support, you can create provide a `lambda` expression to the `key` argument of the `sorted()` function.

```python
class User:
	def __init__(self, uid):
		self.uid = uid

	def __repr__(self):
		return f"User({self.uid})"

users = [User(23), User(3), User(99)]
sorted(users, key=lambda u: u.uid) # [User(3), User(23), User(99)]
```

## Default Dictionary
If you want to create a dictionary that has a container as the value for each of its keys. You can use `defaultdict` from the `collections` module. The `defaultdict` automatically initializes each key value with the specified sequence type. This has two implications:
- The user simply appends items, instead of initializing an empty container manually.
- When accessing a nonexistent key, it provides an empty container, instead of throwing a `KeyError`.

```python
from collections import defaultdict

pairs = (
	("Name", "Jessi"),
	("Age", 37),
)

d = defaultdict(list)
for key, value in pairs:
	d[key].append(value)

# The append() method is provided by the list class
# For sets, the approach would be to use d[key].add(value)

print(d) # defaultdict(<class 'list'>, {'Name': ['Jessi'], 'Age': [37]})
```

## Ordered Dictionary
If you want to control the order of items when iterating or serializing, you can use `OrderedDict` from the `collections` module. It exactly preserves the insertion order of data when iterating.

```python
from collections import OrderedDict

d = OrderedDict()
d["foo"] = 1
d["bar"] = 2
d["spam"] = 3
d["grok"] = 4

for key in d:
	print(key, d[key])

# foo 1
# bar 2
# spam 3
# grok 4
```

## Reducing Dictionaries
Data reduction operations like `min()`, `max()`, or `sorted()` on a dictionary only process the keys of the dictionary, not the values. After performing these operations, you will have to perform an extra lookup step to retrieve the corresponding values.

An efficient solution is to use `zip(d.values(), d.keys())` with these operations. This inverts the dictionary items, so that the operation is performed on the values instead of the keys.

```python
d = {"ACME": 45.23, "AAPL": 612.78, "IBM": 205.55, "FB": 10.75}

min(zip(d.values(), d.keys())) # (10.75, "FB")
max(zip(d.values(), d.keys())) # (612.78, "AAPL")

sorted(zip(d.values(), d.keys()))
# [(10.75, "FB"), (45.23, "ACME"), (205.55, "IBM"), (612.78, "AAPL")]
```

## Comparing Dictionaries
If you have two dictionaries and you want to find out whether they anything in common, such as same keys, you can perform common set operations using the `keys()` or `items()` methods. This is possible because the dictionary keys are guaranteed to be unique.

```python
a = {"x": 1, "y": 2, "z": 3}
b = {"w": 10, "x": 11, "y": 2}

# Find keys in common
a.keys() & b.keys() # {"x", "y"}

# Find keys in "a" that are not in "b"
a.keys() - b.keys() # {"z"}

# Find (key, value) pairs in common
a.items() & b.items() # {("y", 2)}
```

## Sorting Dictionaries
If you want to sort multiple dictionaries by a common key, you can use `itemgetter` from the `operator` module. The `itemgetter()` takes the lookup indices as arguments that can be fed to the object's `__getitem__()` method. After `f = itemgetter(2)`, the call `f(r)` returns `r[2]`.

```python
from operator import itemgetter

rows = [
	{"fname": "Brian", "lname": "Jones",   "uid": 1003},
	{"fname": "David", "lname": "Beazley", "uid": 1002},
	{"fname": "John",  "lname": "Cleese",  "uid": 1001},
	{"fname": "Big",   "lname": "Jones",   "uid": 1004},
]

rows_by_uid = sorted(rows, key=itemgetter("uid"))
print(rows_by_uid)

# [{"fname": "John", "lname": "Cleese",  "uid": 1001},
# {"fname": "David", "lname": "Beazley", "uid": 1002},
# {"fname": "Brian", "lname": "Jones",   "uid": 1003},
# {"fname": "Big",   "lname": "Jones",   "uid": 1004}]
```

## Dictionary Comprehension
If you want to create a subset of a dictionary using some criteria, you can use dictionary comprehension.

```python
prices = {"ACME": 45.23, "AAPL": 612.78, "IBM": 205.55, "FB": 10.75}
p = {k: v for k, v in prices.items() if v > 200}
# {"AAPL": 612.78, "IBM": 205.55}
```

## Removing Duplicates
To remove duplicates from a sequence, you can simply convert it to a set using `set()`. However, this approach does not preserve any order. To make sure that the order is maintained, you can use the following technique:

```python
def dedupe(items):
	seen = set()
	for item in items:
		if item not in seen:
			yield item
			seen.add(item)

c = [1, 5, 2, 1, 9, 1, 5, 10]
list(dedupe(c)) # [1, 5, 2, 9, 10]
```

## Named Slice
To improve code readability and maintainability, it is useful to name slices. This is possible because slices can be constructed using the `slice()` method.

```python
raw_txt = "Item: Biscuits ... Price: $42.00"
ITEM = slice(6, 14)
PRICE = slice(27, 32)

raw_txt[ITEM]  # Biscuits
raw_txt[PRICE] # 42.00
```

## Counting Items
If you want to count the number of occurrences of an item in a sequence, you can use `Counter` from the `collections` module.  It creates a dictionary that stores the items as dictionary keys and their counts as dictionary values. It also provides a very useful `most_common()` method.

```python
from collections import Counter

words = ["look", "into", "my", "eyes", "look", "into",
		 "my", "eyes", "my", "eyes", "you're", "under"]

c = Counter(words)
# Counter({"my": 3, "eyes": 3, "look": 2, "into": 2, "you're": 1, "under": 1})

c["eyes"] # 3
c.most_common(3) # [("my", 3), ("eyes", 3), ("look", 2)]

moreWords = ["the", "eyes", "the", "eyes", "the", "eyes", "not", "around"]
c.update(moreWords)
c["eyes"] # 6
```

## Grouping Records
If you have a sequence of dictionaries and you want to iterate over the data in groups based on the value of a particular field, you can first sort the data based on that field, and then use `groupby()` from the `itertools` module. The `groupby()` function works by examining consecutive items and finding sequential runs of identical values.

```python
from operator import itemgetter
from itertools import groupby

rows = [
	{"address": "5412 N CLARK", "date": "07/01/2012"},
	{"address": "5800 E 58TH", "date": "07/02/2012"},
	{"address": "2122 N CLARK", "date": "07/03/2012"},
	{"address": "5645 N RAVENSWOOD", "date": "07/02/2012"},
	{"address": "1060 W ADDISON", "date": "07/02/2012"},
	{"address": "4801 N BROADWAY", "date": "07/01/2012"},
]

rows.sort(key=itemgetter("date"))
for date, items in groupby(rows, itemgetter("date")):
	print(date)
	for it in items:
		print(f"\t{it}")

# 07/01/2012
#	{"address": "5412 N CLARK", "date": "07/01/2012"}
#	{"address": "4801 N BROADWAY", "date": "07/01/2012"}
# 07/02/2012
#	{"address": "5800 E 58TH", "date": "07/02/2012"}
#	{"address": "5645 N RAVENSWOOD", "date": "07/02/2012"}
#	{"address": "1060 W ADDISON", "date": "07/02/2012"}
# 07/03/2012
#	{"address": "2122 N CLARK", "date": "07/03/2012"}
```

## Filtering Sequences
If you want to filter the data inside a sequence using some criteria, you can use list comprehension or generator expressions.

```python
l = [1, 4, -5, 10, -7, 2, 3, -1]
[n for n in l if n > 0] # [1, 4, 10, 2, 3]

# Generator expressions can be used to save memory
list(n for n in l if n > 0) # [1, 4, 10, 2, 3]
```

However, if the criteria can not be easily expressed in a list comprehension or generate expression, you can use the `filter()` function.

```python
seq = ["1", "2", "-3", "-", "4", "N/A", "5"]

def is_int(val):
	try:
		_ = int(val)
		return True
	except ValueError:
		return False

list(filter(is_int, seq)) # ["1", "2", "-3", "4", "5"]
```

Another approach to filtering is to use `compress()` from the `itertools` module along with a Boolean selector sequence. It produces all of the items in the iterable where the corresponding element in the Boolean selector is `True`.

```python
from itertools import compress

seq = ["CLARK", "RAVENSWOOD", "ADDISON", "BROADWAY", "GRANVILLE"]
counts = [0, 10, 1, 6, 10]

selector = [n > 5 for n in counts]
list(compress(seq, selector)) # ["RAVENSWOOD", "BROADWAY", "GRANVILLE"]
```

## Transformation and Reduction
If you want to combine a data reduction and a transformation, a very elegant solution is to use a generator-expression argument to the reduction function. This is much more efficient then first creating a filtered list. You can also omit the parenthesis around the generator expression.

```python
nums = [1, 2, 3, 4, 5]
# Note that you do not need repeated parenthesis
s = sum(x * x for x in nums) # 55
```

## Chaining Mappings
Sometimes you have multiple dictionaries and you want to logically combine them into a single mapping, such that if a key is not found in one dictionary, the lookup should continue to the next dictionary and so on. For this purpose, you can use `ChainMap` from the `collections` module.

A `ChainMap` takes multiple mappings and makes them logically appear as one. If there are duplicate keys, the values from the first mapping get used. However, operations that mutate the mappings only affect the first mapping listed.

```python
from collections import ChainMap

a = {"x": 1, "z": 2}
b = {"y": 3, "z": 4}

c = ChainMap(a, b)

c.keys() # KeysView(ChainMap({"x": 1, "z": 2}, {"y": 3, "z": 4}))
xyz = c["x"], c["y"], c["z"] # (1, 3, 2)

c["x"] = 10
c["y"] = 20

a["x"] # 10: The mutation affected the underlying mapping
b["y"] # 3:  The mutation did not affect the underlying mapping
c["y"] # 20: The mutation only affected the chained mapping

del c["x"] # Deletes "x" from "a"
del c["z"] # Deletes "z" from "a"
```
