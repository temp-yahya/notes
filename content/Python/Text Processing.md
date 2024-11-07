The string class provides several useful methods for manipulating text such as slicing, searching, and substituting. However, more complicated operations might require the use of regular expressions.

## Splitting Strings
The `split()` method of string objects is meant only for very simple cases, and does not allow for multiple delimiters or account for possible whitespace around the delimiters. In such cases, you can use `split()` method from the `re` module.

```python
import re

line = "asdf fjdk; afed, fjek,asdf, foo"
re.split(r"[;,\s]\s*", line) # ["asdf", "fjdk", "afed", "fjek", "asdf", "foo"]
```

## Prefix Matching
If you want to check the start or end of a string for specific text patterns, such as filename extensions, you can the `startswith()` or `endswith()` method from the string class. If you need to check against multiple choices, you can provide a tuple of possibilities.

```python
url = "https://www.example.com"
filename = "foo.txt"
filenames = ["Makefile", "foo.c", "bar.py", "spam.c", "spam.h"]

url.startswith("https:")  # True
filename.endswith(".txt") # True

[name for name in filenames if name.endswith((".c", ".h"))]
# ["foo.c", "spam.c", "spam.h"]
any(name.endswith(".py") for name in filenames) # True
```

## Pattern Matching
If you are trying to a simple literal, you can use basic string operations or the `find()` method of the string class. For more complicated matching, use regular expressions. If you are going to perform a lot of matches, it is better to precompile the regular expression pattern.

```python
import re

text = "yeah, but no, but yeah, but no, but yeah"
date = "11/27/2012"
note = "Today is 11/27/2012. PyCon starts at 3/13/2013."

# Exact match
text == "yeah" # False

# Location of first occurrence
text.find("no") # 10

pattern = re.compile(r"\d+/\d+/\d+")
pattern.match(date)   # <re.Match object; span=(0, 10), match="11/27/2012">
pattern.findall(note) # ["11/27/2012", "3/13/2013"]
```

## Wildcard Matching
If you want to use wildcard matching as is available in Unix shells, you can the `fnmatch()` function from the `fnmatch` module. It matches strings uses the same case-sensitivity rules as the underlying system. If you want to force case-sensitive matching, `fnmatch` also provides the `fnmatchcase()` function.

```python
from fnmatch import fnmatch, fnmatchcase

fnmatch("foo.txt", "*.txt")       # True
fnmatch("foo.txt", "?oo.txt")     # True
fnmatch("dat45.csv", "dat[0-9]*") # True
fnmatchcase("foo.txt", "*.TXT")   # False
```

## Replacing Text
If you want to search and replace text, you can use the `replace()` method of the string class. For more complicated substitutions, you can use the `sub()` method of the `re` module. If you want more control over the substitution, you can pass a callback function to the `sub()` method.

```python
import re

def callback_fn(m):
	return "{} {} {}".format(m.group(3), m.group(1), m.group(2))

text = "yeah, but no, but yeah, but no, but yeah"
date = "11/27/2012"
note = "Today is 11/27/2012. PyCon starts 3/13/2013."

pattern = re.compile(r"(\d+)/(\d+)/(\d+)")

text.replace("yeah", "yep")    # "yep, but no, but yep, but no, but yep"
pattern.sub(r"\3-\1-\2", date) # "2012-11-27"
pattern.sub(callback_fn, note) # "Today is 2012 11 27. PyCon starts 2013 3 13."
```

If you want to perform case-insensitive operations, you can pass `re.IGNORECASE` flag to these operations. If you want to match the case of the matched text, you can use a support function.

```python
import re

text = "UPPER PYTHON. lower python. Mixed Python."

def support_fn(word):
	def replace(m):
		text = m.group()
		if text.isupper():
			return word.upper()
		elif text.islower():
			return word.lower()
		elif text[0].isupper():
			return word.capitalize()
		else:
			return word
	return replace

re.sub("python", "snake", text, flags=re.IGNORECASE)
# UPPER snake. lower snake. Mixed snake.
re.sub("python", support_fn("snake"), text, flags=re.IGNORECASE)
# "UPPER SNAKE. lower snake. Mixed Snake."
```

