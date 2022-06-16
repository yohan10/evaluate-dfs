# Evaluate DataFrames

**evaluate_dfs** provides features for evaluating a series or dataframe against another series or dataframe.


## Overview

`evaluate_dfs` contains many features to help evaluate series or dataframe against another series or dataframe:

- `dfs_evaluator`: Classes for evaluating two dataframes against each other.
- `series_evaluator`: Classes for evaluating a series against a dataframe.
- `match`: Functions for ranking and matching the highest ranking item in an iterable according to a hierarchy. 

Here's a tour of some of the main features of `evaluate_dfs` with examples:

### Series Evaluators

The main features of the `series_evaluator` module are the `SeriesEvaluator` class and its subclasses. 

You can store your **evaluation mapping** into these classes, composed of:
- **evaluation field name** (keys), the field name to assign the results from an evaluation function to an **evaluation**.
- **evaluation function** (values), a function that compares two series.

Example of an evaluation mapping:
```python
def evaluate_fruits(series1, series2): return series1.fruits == series2.fruits
def evaluate_vegies(series1, series2): return series1.vegies == series2.vegies
smoothie_evaluations = {
    'is_fruits_same': evaluate_fruits,
    'is_vegies_same': evaluate_vegies
}
```

An **evaluation** is a series returned or produced by methods from a `SeriesEvaluator` (or subclass) instance. The evaluation contains the return values from comparing  two series with all the **evaluation functions** from an **evaluation mapping**,
with **evaluation field names** as its indexes. By default, and evaluation stores the indexes of both series compared.

Example of an evaluation:
```
index_x              0
index_y             42
is_fruits_same    True
is_vegies_same    True
dtype: object
```

Additionally, the classes offer the following features:
- Pre-filtering the dataframe before evaluation.
- The field_names attribute, which dynamically stores the **evaluation field names** and/or the indexes field names to use for each evaluation.
- Options to specify the indexes field names or ommit recording the indexes for the evaluations.

Examples of using the `SeriesEvaluator` class:
```python
import pandas as pd
from evaluate_dfs.series_evaluator import SeriesEvaluator

apple = pd.Series(['apple'], index=['fruits'], name=0)
fruit_basket = pd.DataFrame({'fruits': ['apple', 'peach', 'orange']})

def is_fruits_equal(series1: pd.Series, series2: pd.Series) -> bool:
    if series1.fruits == series2.fruits:
        return True
    return False 

series_evaluator = SeriesEvaluator(
    mapping={'fruits_equal': is_fruits_equal},
)

evaluations = series_evaluator.evaluate_against_df(series=apple, df=fruit_basket)
pd.DataFrame(evaluations)
```

       index_x  index_y  fruits_equal
    0        0        0          True
    1        0        1         False
    2        0        2         False

With a filter function:
```python
def filter_out_orange(series: pd.Series, df: pd.DataFrame):
    return df[df['fruits'] != 'orange']

series_evaluator = SeriesEvaluator(
    mapping={'fruits_equal': is_fruits_equal},
    filter_df=filter_out_orange
)
evaluations = series_evaluator.evaluate_against_df(series=apple, df=fruit_basket)
pd.DataFrame(evaluations)
```
       index_x  index_y  fruits_equal
    0        0        0          True
    1        0        1         False

### DataFrames Evaluators

The main features of the `dfs_evaluator` module are the `DataFramesEvaluator` class and its subclasses. They provide methods for evaluating every row (series) from one dataframe against every row, if a filter function not provided for the series evaluator instance.

A dataframes evaluator instance is composed of an `SeriesEvaluator` (or subclass) instance found in the `series_evaluator` module.

Additionally, `dfs_evaluator` also provides the `ParallelEvaluator` that works just like `SeriesEvaluator`, but processes the evaluations in parallel.


A quick example of using the `DataFramesEvaluator` class:
```python
import pandas as pd
from evaluate_dfs.series_evaluator import SeriesEvaluator
from evaluate_dfs.dfs_evaluator import DataFramesEvaluator

fruit_basket1 = pd.DataFrame({'fruits': ['apple', 'pineapple']})
fruit_basket2 = pd.DataFrame({'fruits': ['apple', 'peach']})

def is_fruits_equal(series1: pd.Series, series2: pd.Series) -> bool:
    if series1.fruits == series2.fruits:
        return True
    return False 

series_evaluator = SeriesEvaluator(
    mapping={'fruits_equal': is_fruits_equal},
)
dfs_evaluator = DataFramesEvaluator(series_evaluator=series_evaluator)
dfs_evaluator.evaluate(df1=fruit_basket1, df2=fruit_basket2)
```
       index_x  index_y  fruits_equal
    0        0        0          True
    1        0        1         False
    2        1        0         False
    3        1        1         False