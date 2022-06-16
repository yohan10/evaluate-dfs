import doctest
import numpy as np
import pandas as pd
import unittest
from evaluate_dfs import series_evaluator
from evaluate_dfs.series_evaluator import SeriesEvaluator, MatchEvaluator


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

smoothie = pd.Series(
    ['apple', 'cabbage'],
    index=['fruits', 'vegetables'],
    name='smoothie1'
)

smoothie_combos = pd.DataFrame({
    'fruits': ['apple', 'oranges'],
    'vegetables': ['cabbage', 'eggplant']
}, index=['smoothie2', 'smoothie3'])


class TestSeriesEvaluator(unittest.TestCase):
    def test_properties(self):
        evaluator = SeriesEvaluator(
            mapping=eval_funcs.copy(),
            indexes=('df1_index', 'df2_index')
        )
        self.assertEqual(('df1_index', 'df2_index'), evaluator.indexes)
        self.assertEqual(
            evaluator.field_names,
            ['df1_index', 'df2_index', 'fruits_eval', 'vegetables_eval']
        )

        # Changes to mapping is reflected in .field_names
        new_mapping = {'eval1': lambda x, y: None, 'eval2': lambda x, y: None}
        evaluator.mapping = new_mapping
        self.assertEqual(
            evaluator.field_names,
            ['df1_index', 'df2_index', 'eval1', 'eval2']
        )
        evaluator.mapping = {}
        self.assertEqual(
            evaluator.field_names,
            ['df1_index', 'df2_index']
        )

        # Changes to indexes is reflected in .field_names
        evaluator.indexes = ('x', 'y')
        self.assertEqual(
            evaluator.field_names,
            ['x', 'y']
        )
        evaluator.indexes = None
        self.assertEqual(
            evaluator.field_names,
            []
        )

    def test_indexes_setter(self):
        # Can't set indexes with iterables with len > 2 on initialization
        self.assertRaises(
            ValueError,
            SeriesEvaluator,
            mapping=eval_funcs.copy(),
            indexes=('index_x', 'index_y', 'index_z')
        )

        # Can't set indexes with iterables with len > 2 on an object
        def set_invalid_indexes(evaluator):
            evaluator.indexes = ('index_x', 'index_y', 'index_z')
        evaluator = SeriesEvaluator(
            mapping=eval_funcs.copy(),
        )
        self.assertRaises(
            ValueError,
            set_invalid_indexes,
            evaluator=evaluator
        )

        # Setting indexes as None results indexes being an empty tuple
        evaluator.indexes = None
        self.assertEqual(evaluator.indexes, tuple())

    def test_evaluate_series(self):
        evaluator = SeriesEvaluator(
            mapping=eval_funcs.copy(),
        )

        smoothie1 = pd.Series(
            ['apple', 'spinach'],
            index=['fruits', 'vegetables'],
            name='smoothie1'
        )
        smoothie2 = pd.Series(
            ['apple', 'cabbage'],
            index=['fruits', 'vegetables'],
            name='smoothie2'
        )
        results = evaluator.evaluate_series(smoothie1, smoothie2)
        self.assertEqual(
            results.to_list(),
            ['smoothie1', 'smoothie2', True, False]  # indexes + eval results
        )

        # setting indexes to None makes evaluate_series ignore returning indexes
        evaluator.indexes = None
        results = evaluator.evaluate_series(smoothie1, smoothie2)
        self.assertEqual(
            results.to_list(),
            [True, False]  # indexes + eval results
        )

    def test_evaluate_against_df(self):
        evaluator = SeriesEvaluator(
            mapping=eval_funcs.copy()
        )

        results = evaluator.evaluate_against_df(smoothie, smoothie_combos.copy())
        desired_results = [
            ['smoothie1', 'smoothie2', True, True],
            ['smoothie1', 'smoothie3', False, False]
        ]
        self.assertEqual(
            list(r.to_list() for r in results),
            desired_results
        )

        # setting indexes to None makes evaluate_against_df ignore series indexes
        evaluator.indexes = None
        results = evaluator.evaluate_against_df(smoothie, smoothie_combos.copy())
        desired_results = [
            [True, True],
            [False, False]
        ]
        self.assertEqual(
            list(r.to_list() for r in results),
            desired_results
        )

    def test_eval_against_df_filter_and_fillvalue(self):
        evaluator = SeriesEvaluator(
            mapping=eval_funcs.copy(),
            filter_df=filter_fruits
        )

        # filter_df filters out rows with fruits that is not apple from df2
        results = evaluator.evaluate_against_df(
            smoothie,
            smoothie_combos.copy()
        )

        self.assertEqual(
            list(r.to_list() for r in results),
            [['smoothie1', 'smoothie2', True, True]]
        )

        # setting indexes to None makes eval_against_df ignore series indexes
        evaluator.indexes = None
        results = evaluator.evaluate_against_df(
            smoothie.copy(),
            smoothie_combos.copy()
        )
        self.assertEqual(
            list(r.to_list() for r in results),
            [[True, True]]
        )

        # If evaluator has indexes, empty df from filter should return a
        # iterator of a single item (the evaluation) made of:
        #   -> index from the series
        #   -> n-1 fillvalues, n being the number of fields in field_names.
        #      The -1 excludes the first field in field_names, the index of
        #      the series.

        evaluator = SeriesEvaluator(
            mapping=eval_funcs.copy(),
            filter_df=filter_out_all,
            fillvalue='FILL'
        )

        results = evaluator.evaluate_against_df(
            smoothie.copy(),
            smoothie_combos.copy()
        )

        self.assertEqual(
            list(r.to_list() for r in results),
            [['smoothie1', 'FILL', 'FILL', 'FILL']]
        )

        # If evaluator does not have indexes, empty df from filter should
        # return a iterator of a single item (the evaluation) made of n
        # fillvalues, n being the number of fields in field_names.
        evaluator.indexes = None
        results = evaluator.evaluate_against_df(
            smoothie.copy(),
            smoothie_combos.copy()
        )

        self.assertEqual(
            list(r.to_list() for r in results),
            [['FILL', 'FILL']]
        )


class TestMatchEvaluator(unittest.TestCase):
    def setUp(self):
        def is_match(evaluation: pd.Series):
            is_fruits_match = evaluation.fruits_eval
            is_vegies_match = evaluation.vegetables_eval
            if is_fruits_match and is_vegies_match:
                if not (pd.isna(is_fruits_match) or pd.isna(is_vegies_match)):
                    return True
            return False

        self.match_hierarchy = {
            'matched': is_match,
            'not matched': lambda x: not is_match(x)
        }

    def test_match_yields_only_one_series(self):
        evaluator = MatchEvaluator(
            mapping=eval_funcs.copy(),
            hierarchy=self.match_hierarchy
        )

        results = evaluator.evaluate_against_df(smoothie, smoothie_combos.copy())
        results = list(results)
        self.assertEqual(len(results), 1)
        desired_data = {
            'index_x': 'smoothie1',
            'index_y': 'smoothie2',
            'fruits_eval': True,
            'vegetables_eval': True,
            'match': 'matched'
        }
        desired_series = pd.Series(desired_data)
        self.assertEqual(
            results[0].index.to_list(),
            desired_series.index.to_list()
        )
        self.assertEqual(
            results[0].values.tolist(),
            desired_series.values.tolist()
        )

    def test_match_col_can_be_specified(self):
        evaluator = MatchEvaluator(
            mapping=eval_funcs.copy(),
            hierarchy=self.match_hierarchy,
            match_col='is_smoothie_match',
        )

        results = evaluator.evaluate_against_df(
            smoothie,
            smoothie_combos.copy()
        )
        results = list(results)[0]
        self.assertIn(
            'is_smoothie_match',
            results.index
        )

    def test_match_status_not_recorded_if_match_col_is_none(self):
        evaluator = MatchEvaluator(
            mapping=eval_funcs.copy(),
            hierarchy=self.match_hierarchy,
            match_col=None
        )

        results = evaluator.evaluate_against_df(
            smoothie,
            smoothie_combos.copy()
        )

        # Should still match the closest (yield only 1 series)
        results = list(results)
        self.assertEqual(len(results), 1)
        desired_data = {
            'index_x': 'smoothie1',
            'index_y': 'smoothie2',
            'fruits_eval': True,
            'vegetables_eval': True,
        }
        desired_series = pd.Series(desired_data)
        self.assertEqual(
            results[0].index.to_list(),
            desired_series.index.to_list()
        )
        self.assertEqual(
            results[0].values.tolist(),
            desired_series.values.tolist()
        )

    def test_evaluator_attempts_to_match_despite_empty_result_from_filter_df(self):
        evaluator = MatchEvaluator(
            mapping=eval_funcs.copy(),
            hierarchy=self.match_hierarchy,
            filter_df=filter_out_all,
        )

        results = evaluator.evaluate_against_df(
            smoothie,
            smoothie_combos.copy()
        )
        results = list(results)[0]
        desired_data = {
            'index_x': 'smoothie1',
            'index_y': np.nan,
            'fruits_eval': np.nan,
            'vegetables_eval': np.nan,
            'match': 'not matched'
        }
        desired_series = pd.Series(desired_data)
        self.assertEqual(
            results.index.to_list(),
            desired_series.index.to_list()
        )
        self.assertEqual(
            results.values.tolist(),
            desired_series.values.tolist()
        )

    def test_match_status_dispute_yields_leftmost(self):
        smoothie_combos = pd.DataFrame({
            'fruits': ['apple', 'apple', 'apple'],
            'vegetables': ['cabbage', 'cabbage', 'cabbage']
        }, index=['smoothie2', 'smoothie3', 'smoothie4'])
        evaluator = MatchEvaluator(
            mapping=eval_funcs,
            hierarchy=self.match_hierarchy,
        )
        results = evaluator.evaluate_against_df(
            smoothie,
            smoothie_combos.copy(),
        )
        self.assertEqual(next(results).index_y, 'smoothie2')

    def test_empty_hierarchy_yields_leftmost_eval_with_default_value(self):
        evaluator = MatchEvaluator(
            mapping=eval_funcs.copy(),
            hierarchy={},
            default='unknown match type'
        )

        results = evaluator.evaluate_against_df(
            smoothie,
            smoothie_combos.copy(),
        )
        results = list(results)[0]
        desired_data = {
            'index_x': 'smoothie1',
            'index_y': 'smoothie2',
            'fruits_eval': True,
            'vegetables_eval': True,
            'match': 'unknown match type'
        }
        desired_series = pd.Series(desired_data)
        self.assertEqual(
            results.index.to_list(),
            desired_series.index.to_list()
        )
        self.assertEqual(
            results.values.tolist(),
            desired_series.values.tolist()
        )

    def test_no_evals_with_status_yields_leftmost_eval_with_default_value(self):
        evaluator = MatchEvaluator(
            mapping=eval_funcs.copy(),
            hierarchy={'is_match': lambda x: False},
            default='unknown match type'
        )

        results = evaluator.evaluate_against_df(
            smoothie,
            smoothie_combos.copy(),
        )
        results = list(results)[0]
        desired_data = {
            'index_x': 'smoothie1',
            'index_y': 'smoothie2',
            'fruits_eval': True,
            'vegetables_eval': True,
            'match': 'unknown match type'
        }
        desired_series = pd.Series(desired_data)
        self.assertEqual(
            results.index.to_list(),
            desired_series.index.to_list()
        )
        self.assertEqual(
            results.values.tolist(),
            desired_series.values.tolist()
        )


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(series_evaluator))
    return tests


if __name__ == '__main__':
    unittest.main()
