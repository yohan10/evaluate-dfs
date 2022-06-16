from contextlib import redirect_stdout
import doctest
from functools import partial
import io
from numpy.testing import assert_equal
import pandas as pd
import unittest
from evaluate_dfs import dfs_evaluator
from evaluate_dfs.dfs_evaluator import (
    DataFramesEvaluator,
    ParallelEvaluator
)
from evaluate_dfs.series_evaluator import (
    BaseSeriesEvaluator,
    SeriesEvaluator
)


def _test_evaluate_returns_dataframe(
        self: unittest.TestCase,
        evaluator_cls: type,
) -> None:
    """
    Tests if the evaluator returns a dataframe.
    Use for TestCases involving DataFramesEvaluator or its subclasses.

    Parameters
    ----------
    self : unittest.TestCase
        The TestCase instance.
    evaluator_cls : type
        DataFramesEvaluator or its subclasses.
    series_evaluator : BaseSeriesEvaluator
        BaseSeriesEvaluator or subclass instance.
    """

    series_evaluator = SeriesEvaluator(
        mapping=eval_funcs.copy(),
    )
    df_evaluator = evaluator_cls(
        series_evaluator
    )
    evals = df_evaluator.evaluate(smoothie_combos1, smoothie_combos2)
    desired_data = {
        'index_x': ['smoothie1', 'smoothie1', 'smoothie2', 'smoothie2'],
        'index_y': ['smoothie3', 'smoothie4', 'smoothie3', 'smoothie4'],
        'fruits_eval': [True, False, False, True],
        'vegetables_eval': [True, False, False, True]
    }
    desired_df = pd.DataFrame(desired_data)
    self.assertEqual(list(evals.columns), list(desired_df.columns))
    assert_equal(evals.values, desired_df.values)

    # setting indexes to None returns df without index
    df_evaluator.series_evaluator.indexes = None
    evals = df_evaluator.evaluate(smoothie_combos1, smoothie_combos2)
    desired_df = pd.DataFrame({
        'fruits_eval': [True, False, False, True],
        'vegetables_eval': [True, False, False, True]
    })
    self.assertEqual(list(evals.columns), list(desired_df.columns))
    assert_equal(evals.values, desired_df.values)


def _test_evaluate_filter_and_fillvalue(
        self: unittest.TestCase,
        evaluator_cls: type
) -> None:
    """
    Tests the filtering and filling of empty evaluations from  the
    evaluator.evaluate method.

    Use for TestCases involving DataFramesEvaluator or its subclasses.

    Parameters
    ----------
    self : unittest.TestCase
        The TestCase instance.
    evaluator_cls : type
        DataFramesEvaluator or its subclasses.
    """

    series_evaluator = SeriesEvaluator(
        mapping=eval_funcs.copy(),
        filter_df=filter_fruits
    )

    df_evaluator = evaluator_cls(
        series_evaluator=series_evaluator
    )

    evals = df_evaluator.evaluate(smoothie_combos1, smoothie_combos2)

    desired_df = pd.DataFrame({
        'index_x': ['smoothie1', 'smoothie2'],
        'index_y': ['smoothie3', 'smoothie4'],
        'fruits_eval': [True, True],
        'vegetables_eval': [True, True]
    })
    self.assertEqual(list(evals.columns), list(desired_df.columns))
    assert_equal(evals.values, desired_df.values)

    # setting series_evaluator indexes to None returns df without index
    df_evaluator.series_evaluator.indexes = None
    desired_df = pd.DataFrame({
        'fruits_eval': [True, True],
        'vegetables_eval': [True, True]
    })
    evals = df_evaluator.evaluate(smoothie_combos1, smoothie_combos2)
    self.assertEqual(list(evals.columns), list(desired_df.columns))
    assert_equal(evals.values, desired_df.values)

    # If evaluator has indexes, empty dataframes returned from filter_df
    # should result in the evaluation being series made of:
    #   -> index from the series
    #   -> n-1 fillvalues, n being the number of fields in field_names.
    #      The -1 excludes the first field in field_names, the index of
    #      the series.

    series_evaluator = SeriesEvaluator(
        mapping=eval_funcs.copy(),
        filter_df=filter_out_all,
        fillvalue='FILL'
    )
    df_evaluator = evaluator_cls(
        series_evaluator=series_evaluator
    )

    desired_df = pd.DataFrame({
        'index_x': ['smoothie1', 'smoothie2'],
        'index_y': ['FILL', 'FILL'],
        'fruits_eval': ['FILL', 'FILL'],
        'vegetables_eval': ['FILL', 'FILL']
    })
    evals = df_evaluator.evaluate(smoothie_combos1, smoothie_combos2)
    self.assertEqual(list(evals.columns), list(desired_df.columns))
    assert_equal(evals.values, desired_df.values)

    # If evaluator has indexes, empty dataframes returned from filter_df
    # should result in the evaluation being series made of n
    # fillvalues, n being the number of fields in field_names.
    df_evaluator.series_evaluator.indexes = None
    desired_df = pd.DataFrame({
        'fruits_eval': ['FILL', 'FILL'],
        'vegetables_eval': ['FILL', 'FILL']
    })
    evals = df_evaluator.evaluate(smoothie_combos1, smoothie_combos2)
    self.assertEqual(list(evals.columns), list(desired_df.columns))
    assert_equal(evals.values, desired_df.values)


def is_fruits_equal(x: pd.Series, y: pd.Series) -> bool:
    if x.fruits == y.fruits:
        return True
    return False


def is_vegetable_equal(x: pd.Series, y: pd.Series) -> bool:
    if x.vegetables == y.vegetables:
        return True
    return False


def filter_fruits(x: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    return df[x.fruits == df.fruits]


def filter_out_all(x: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame()


eval_funcs = {
    'fruits_eval': is_fruits_equal,
    'vegetables_eval': is_vegetable_equal
}

smoothie_combos1 = pd.DataFrame({
    'fruits': ['apple', 'oranges'],
    'vegetables': ['cabbage', 'eggplant']
}, index=['smoothie1', 'smoothie2'])

smoothie_combos2 = pd.DataFrame({
    'fruits': ['apple', 'oranges'],
    'vegetables': ['cabbage', 'eggplant']
}, index=['smoothie3', 'smoothie4'])


class TestDataFramesEvaluator(unittest.TestCase):
    def test_evaluate_returns_dataframe(self):

        _test_evaluate_returns_dataframe(
            self,
            evaluator_cls=DataFramesEvaluator
        )

    def test_evaluate_filter_and_fillvalue(self):
        _test_evaluate_filter_and_fillvalue(
            self,
            evaluator_cls=DataFramesEvaluator
        )


class TestParallelEvaluator(unittest.TestCase):
    def setUp(self):
        self.silent_parallel_evaluator = partial(
            ParallelEvaluator,
            verbose=0
        )

    def test_evaluate_returns_dataframe(self):
        _test_evaluate_returns_dataframe(
            self,
            evaluator_cls=self.silent_parallel_evaluator
        )

    def test_evaluate_filter_and_fillvalue(self):
        _test_evaluate_filter_and_fillvalue(
            self,
            evaluator_cls=self.silent_parallel_evaluator
        )

    def test_pandarallel_initializes_every_call(self):
        """
        Pandarallel integration test. Simply tests the verbosity to see if
        Pandarallel is initialized on every call to ParallelEvaluator.evaluate.
        """
        def foo(x, y): pass

        series_evaluator = SeriesEvaluator(
            mapping={'eval': foo}
        )
        evaluator = ParallelEvaluator(
            series_evaluator=series_evaluator,
            nb_workers=1,
            verbose=2  # Display all logs
        )
        f = io.StringIO()
        with redirect_stdout(f):
            evaluator.evaluate(smoothie_combos1, smoothie_combos2)
        self.assertIn('1 worker', f.getvalue())

        evaluator.verbose = 0  # Don't display any logs
        f = io.StringIO()
        with redirect_stdout(f):
            values = evaluator.evaluate(smoothie_combos1, smoothie_combos2)
        self.assertEqual('', f.getvalue())  # terminal output should be empty


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(dfs_evaluator))
    return tests


if __name__ == '__main__':
    unittest.main()
