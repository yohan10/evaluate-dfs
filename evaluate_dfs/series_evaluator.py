"""
This module provides evaluator classes that compares a series against another
series or dataframe.

These classes store an evaluation mapping to evaluate the two. An evaluation
mapping is composed of:
    -> evaluation field name (keys):

       The field name to assign the results from an evaluation function to an
       evaluation.

    -> evaluation function (values)

       A function that compares two series.

An evaluation is a series returned or produced by methods from an evaluator.
It contains return values from comparing  two series with all the evaluation
functions from an evaluation mapping, with evaluation field names as its
indexes.

Classes Offered
---------------
- BaseSeriesEvaluator
     The abstract series evaluator class
- SeriesEvaluator
     An evaluator to compare a series against another series or dataframe.
- MatchEvaluator
     An evaluator to compare a series against another series or dataframe and
     return the closest matching series according to a given hierarchy.
"""
from abc import ABC, abstractmethod
from itertools import chain
from typing import (
    Any,
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Tuple,
    Union
)
import numpy as np
import pandas as pd
from .match import match_highest

EvaluationMapping = Mapping[Hashable, Callable[[pd.Series, pd.Series], Any]]
FilterFunc = Callable[[pd.Series, pd.DataFrame], pd.DataFrame]
HierarchyMapping = Mapping[Hashable, Callable[[pd.Series], bool]]
Indexes = Tuple[Hashable, Hashable]


class BaseSeriesEvaluator(ABC):
    def evaluate_series(self, series1: pd.Series, series2: pd.Series) -> Any:
        """Compares two series."""

    @abstractmethod
    def evaluate_against_df(
            self,
            series: pd.Series,
            df: pd.DataFrame
    ) -> Union[Iterable, Iterator]:
        """Compares a series against all series from df."""


class SeriesEvaluator(BaseSeriesEvaluator):
    """
    An evaluator to compare a series against another series or dataframe.

    Attributes
    ----------
    mapping: Mapping
        A mapping of evaluation operation name (keys) and the evaluation
        operation. The operation should take in two series as the first
        two positional argument (ex: def foo(series1, series2): ...).
    indexes: tuple of str, default ('index_x', 'index_y')
        The evaluation will return the indexes from each evaluated
        series from the first dataframe and the second dataframe under the
        field names from the first and second names from the tuple
        respectively. If None, indexes will be ignored from the evaluation.
    filter_df: Callable, optional
        A callable that filters the dataframe to be compared against the
        series for evaluation. The callable will be passed in a series and
        a dataframe as the first two positional argument. It should return
        a dataframe.
    fillvalue: Any, default np.nan
        Used to create a tuple with n fillvalues representing an empty
        evaluation only when filter_df returns an empty dataframe. n being
        the number of evaluation operations and (optionally) the indexes
        from each dataframe.
    field_names: list
        Field names composed for both indexes names and the evaluation fields
        from the evaluations mapping.

    Methods
    -------
    get_empty_evaluation(series)
        Returns an empty evaluation with indexes of the field_names attribute,
        each index associated with a fillvalue (attribute) value.
    evaluate_series(series1, series2)
        Evaluates two series according to the evaluation mapping.
    evaluate_against_dataframe(series, df)
        Produces evaluations from every comparison between series against every
        row of a dataframe.
    """
    def __init__(
            self,
            mapping: EvaluationMapping,
            indexes: Union[Indexes, None] = ('index_x', 'index_y'),
            filter_df: FilterFunc = None,
            fillvalue: Any = np.nan
    ) -> None:
        self.mapping = mapping
        self.indexes = indexes
        self.filter_df = filter_df
        self.fillvalue = fillvalue

    @property
    def mapping(self) -> dict:
        """ Returns evaluation mapping."""
        return self._mapping

    @mapping.setter
    def mapping(self, value: EvaluationMapping) -> None:
        """
        Setter for the evaluation mapping attribute.

        value: Mapping
            A mapping of evaluation operation name (keys) and the evaluation
            operation. The operation should take in two series as the first
            two positional argument (ex: def foo(series1, series2): ...).
        """
        self._mapping = dict(value)

    @property
    def indexes(self) -> Union[Tuple, None]:
        """ Returns the indexes attribute."""
        return self._indexes

    @indexes.setter
    def indexes(self, value: Union[Indexes, None]) -> None:
        """
        Sets the indexes attribute.

        Parameters
        ----------
        value: tuple of str or None
            The evaluation will return the indexes from each evaluated
            series from the first dataframe and the second dataframe under the
            field names from the first and second names from the tuple
            respectively. If None, indexes will be ignored from the evaluation.

        Raises
        ------
        ValueError
            When the length of the given indexes is not 2.
        """
        if value is None:
            self._indexes = tuple()
        else:
            if len(value) != 2:
                msg = "indexes must be an iterable of only 2 items."
                raise ValueError(msg)
            else:
                self._indexes = (value[0], value[1])

    @property
    def field_names(self) -> list:
        """
        Field names (index) of each evaluation.
        """
        if self._indexes:
            return list(chain(self._indexes, self.mapping))
        return list(self.mapping)

    def get_empty_evaluation(self, series: pd.Series) -> pd.Series:
        """
        Returns an empty evaluation with indexes of the field_names attribute,
        each index associated with a fillvalue (attribute) value.

        If the indexes attribute is not None, then associate the passed in
        series name with the compared-with-index*.

        *indexes attr composed of: (compared-with-index, compared-to-index).

        Examples
        --------

        If indexes exist, indexes from the series passed in is assigned to
        the evaluation.

        >>> def foo(x, y): pass
        >>> evaluator = SeriesEvaluator(
        ...     mapping={'eval_fields': foo},
        ...     indexes=('index_left', 'index_right')
        ... )
        >>> series = pd.Series(index=['name'], name='John', dtype='object')
        >>> evaluator.get_empty_evaluation(series)
        index_left     John
        index_right     NaN
        eval_fields     NaN
        dtype: object

        indexes=None results in no indexes being set for the empty evaluation.

        >>> evaluator = SeriesEvaluator(
        ...     mapping={'eval_fields': foo},
        ...     indexes=None
        ... )
        >>> evaluator.get_empty_evaluation(series)
        eval_fields   NaN
        dtype: float64
        """
        field_names = self.field_names
        offset = 1 if self._indexes else 0
        empty_range = range(len(field_names) - offset)
        if self._indexes:
            evaluation = series.name, *(self.fillvalue for i in empty_range)
        else:
            evaluation = tuple(self.fillvalue for i in empty_range)
        return pd.Series(evaluation, index=field_names)

    def evaluate_series(
            self,
            series1: pd.Series,
            series2: pd.Series
    ) -> pd.Series:
        """
        Evaluates two series according to the evaluation mapping.

        Parameters
        ----------
        series1: pd.Series
        series2: pd.Series

        Returns
        -------
        pd.Series
            Results from each evaluation operation (evaluation mapping values)
            called and passed with series1 and series2 as the first two
            positional argument.

        Examples
        --------
        Evaluating two series with an evaluation mapping

        >>> import pandas as pd
        >>> from evaluate_dfs.series_evaluator import SeriesEvaluator
        >>> smoothie1 = pd.Series(
        ...     data=['apple', 'celery'],
        ...     index=['fruit', 'vegie'],
        ...     name='smoothie1'
        ... )
        >>> smoothie2 = pd.Series(
        ...     data=['apple', 'spinach'],
        ...     index=['fruit', 'vegie'],
        ...     name='smoothie2'
        ... )
        >>> def is_field_equal(
        ...     series1: pd.Series,
        ...     series2: pd.Series,
        ...     field: Hashable
        ... ) -> bool:
        ...     return series1[field] == series2[field]
        ...
        >>> eval_mapping = {
        ...     'fruits_equal': lambda s1, s2: is_field_equal(s1, s2, 'fruit'),
        ...     'vegies_equal': lambda s1, s2: is_field_equal(s1, s2, 'vegie')
        ... }

        >>> series_evaluator = SeriesEvaluator(mapping=eval_mapping)
        >>> series_evaluator.evaluate_series(smoothie1, smoothie2)
        index_x         smoothie1
        index_y         smoothie2
        fruits_equal         True
        vegies_equal        False
        dtype: object

        With custom indexes names

        >>> series_evaluator = SeriesEvaluator(
        ...     mapping=eval_mapping,
        ...     indexes=('index_left', 'index_right')
        ... )
        >>> series_evaluator.evaluate_series(smoothie1, smoothie2)
        index_left      smoothie1
        index_right     smoothie2
        fruits_equal         True
        vegies_equal        False
        dtype: object

        Ommitting indexes

        >>> series_evaluator = SeriesEvaluator(
        ...     mapping=eval_mapping,
        ...     indexes=None
        ... )
        >>> series_evaluator.evaluate_series(smoothie1, smoothie2)
        fruits_equal     True
        vegies_equal    False
        dtype: bool
        """
        evaluation = (fn(series1, series2) for fields, fn
                      in self.mapping.items())
        if self.indexes:
            evaluation = chain([series1.name, series2.name], evaluation)
            eval_indexes = list(chain(self.indexes, self.mapping))
        else:
            eval_indexes = list(self.mapping)
        return pd.Series(evaluation, index=eval_indexes)

    def evaluate_against_df(
            self,
            series: pd.Series,
            df: pd.DataFrame
    ) -> Iterator[pd.Series]:
        """
        A generator that yields an evaluation of every series against every row
        of a dataframe.

        Parameters
        ----------
        series: pd.Series
            A series to compared against every row of df.
        df: pd.DataFrame
            A dataframes whose rows will be compared to series.

        Yields
        -------
        pd.Series
            Results from each evaluation operation (from the evaluation
            mapping) called and passed with series1 and a row from df as the
            first two positional argument.

        Examples
        --------

        Evaluating a series and a dataframe an evaluation mapping

        >>> import pandas as pd
        >>> from evaluate_dfs.series_evaluator import SeriesEvaluator
        >>> apple = pd.Series(['apple'], index=['fruits'], name=0)
        >>> fruit_basket = pd.DataFrame({
        ...     'fruits': ['apple', 'peach', 'orange']
        ... })
        >>> def is_fruits_equal(
        ...     series1: pd.Series,
        ...     series2: pd.Series
        ... ) -> bool:
        ...     return series1.fruits == series2.fruits
        ...
        >>> series_evaluator = SeriesEvaluator(
        ...     mapping={'fruits_equal': is_fruits_equal},
        ... )
        >>> evaluations = series_evaluator.evaluate_against_df(
        ...     series=apple, df=fruit_basket
        ... )
        >>> pd.DataFrame(evaluations)
           index_x  index_y  fruits_equal
        0        0        0          True
        1        0        1         False
        2        0        2         False

        With custom indexes names

        >>> series_evaluator = SeriesEvaluator(
        ...     mapping={'fruits_equal': is_fruits_equal},
        ...     indexes=('index_left', 'index_right')
        ... )
        >>> evaluations = series_evaluator.evaluate_against_df(
        ...     series=apple, df=fruit_basket
        ... )
        >>> pd.DataFrame(evaluations)
           index_left  index_right  fruits_equal
        0           0            0          True
        1           0            1         False
        2           0            2         False

        Ommitting indexes

        >>> series_evaluator = SeriesEvaluator(
        ...     mapping={'fruits_equal': is_fruits_equal},
        ...     indexes=None
        ... )
        >>> evaluations = series_evaluator.evaluate_against_df(
        ...     series=apple, df=fruit_basket
        ... )
        >>> pd.DataFrame(evaluations)
           fruits_equal
        0          True
        1         False
        2         False

        With Filtering

        >>> def filter_out_orange(series: pd.Series, df: pd.DataFrame):
        ...     return df[df['fruits'] != 'orange']
        >>> series_evaluator = SeriesEvaluator(
        ...     mapping={'fruits_equal': is_fruits_equal},
        ...     filter_df=filter_out_orange
        ... )
        >>> evaluations = series_evaluator.evaluate_against_df(
        ...     series=apple,
        ...     df=fruit_basket
        ... )
        >>> pd.DataFrame(evaluations)
           index_x  index_y  fruits_equal
        0        0        0          True
        1        0        1         False
        """
        if self.filter_df:
            df = self.filter_df(series, df)
        if df.empty:
            empty_evaluation = self.get_empty_evaluation(series)
            yield empty_evaluation
        else:
            for idx, s in df.iterrows():
                yield self.evaluate_series(series, s)


class MatchEvaluator(SeriesEvaluator):
    """
    An evaluator to compare a series against another series or dataframe and
    return the closest matching series according to a given hierarchy.

    Attributes
    ----------
    mapping : Mapping
        A mapping of evaluation operation name (keys) and the evaluation
        operation. The operation should take in two series as the first
        two positional argument (ex: def foo(series1, series2): ...).
    hierarchy : mapping of match type and callable
        Every evaluation will be passed into each of the callables (values)
        in this mapping. Once a callable returns truthy, the paired match
        type (keys) is associated to the evaluation.

        The evaluation with the highest match type is yielded. The first
        key-value pair is considered the highest rank, and the last pair is
        the second to lowest rank. The 'default' value is the lowest rank.
        Given two evaluations, both with the highest match type, the item
        with the lower position in the dataframe will be yielded.
    indexes : tuple of str, default ('index_x', 'index_y')
        The evaluation will return the indexes from each evaluated
        series from the first dataframe and the second dataframe under the
        field names from the first and second names from the tuple
        respectively. If None, indexes will be ignored from the evaluation.
    filter_df: Callable, optional
        A callable that filters the dataframe to be compared against the
        series for evaluation. The callable will be passed in a series and
        a dataframe as the first two positional argument. It should return
        a dataframe.
    fillvalue: Any, default np.nan
        Used to create a tuple with n fillvalues representing an empty
        evaluation only when filter_df returns an empty dataframe. n being
        the number of evaluation operations and (optionally) the indexes
        from each dataframe.
    default : any, optional
        The default case value when the evaluation returns falsy for all
        the match functions in the hierarchy. The default value is
        considered the lowest rank.
    match_col : hashable or None, default "match"
        The field name of the match status.
    field_names: list
        Field names composed for both indexes names and the evaluation fields
        from the evaluations mapping.

    Methods
    -------
    evaluate_against_dataframe(series, df)
        Produces evaluations from every comparison between series against every
        row of a dataframe.
    """
    def __init__(
            self,
            mapping: EvaluationMapping,
            hierarchy: HierarchyMapping,
            indexes: Union[Indexes, None] = ('index_x', 'index_y'),
            filter_df: FilterFunc = None,
            fillvalue: Any = np.nan,
            default: Any = None,
            match_col: Union[Hashable, None] = 'match',
    ) -> None:
        super().__init__(
            mapping=mapping,
            indexes=indexes,
            filter_df=filter_df,
            fillvalue=fillvalue
        )
        self.hierarchy = hierarchy
        self.match_col = match_col
        self.default = default

    @property
    def field_names(self) -> list:
        """
        Field names (index) of each evaluation.
        """
        field_names = super().field_names
        if self.match_col is None:
            return super().field_names
        else:
            return list((*field_names, self.match_col))

    def evaluate_against_df(
            self,
            series: pd.Series,
            df: pd.DataFrame
    ) -> Iterator[pd.Series]:
        """
        A generator that yields an evaluation of every series against every row
        of a dataframe.

        Parameters
        ----------
        series: pd.Series
            A series to compared against every row of df.
        df: pd.DataFrame
            A dataframes whose rows will be compared to series.

        Returns
        -------
        pd.Series
            Results from each evaluation operation (from the evaluation
            mapping) called and passed with series1 and a row from df as the
            first two positional argument.

        Examples
        --------
        >>> import pandas as pd
        >>> from evaluate_dfs.series_evaluator import MatchEvaluator
        >>> apple = pd.Series(['apple'], index=['fruits'], name=0)
        >>> fruit_basket = pd.DataFrame({
        ...     'fruits': ['apple', 'peach', 'orange']
        ... })
        >>> def is_fruits_equal(
        ...     series1: pd.Series,
        ...     series2: pd.Series
        ... ) -> bool:
        ...     return series1.fruits == series2.fruits
        ...
        >>> def is_match(evaluation: pd.Series) -> bool:
        ...     return evaluation.fruits_equal
        ...
        >>> series_evaluator = MatchEvaluator(
        ...     mapping={'fruits_equal': is_fruits_equal},
        ...     hierarchy={'matched': is_match}
        ... )
        >>> evaluations = series_evaluator.evaluate_against_df(
        ...     series=apple,
        ...     df=fruit_basket
        ... )
        >>> pd.DataFrame(evaluations)
           index_x  index_y  fruits_equal    match
        0        0        0          True  matched

        Given two evaluations, both with the highest match type relative to all
        the evaluations, the item with the lower position in the dataframe will
        be yielded.

        >>> apple = pd.Series(['apple'], index=['fruits'], name=0)
        >>> fruit_basket = pd.DataFrame(
        ...     {'fruits': ['apple', 'apple']},
        ...     index=['first', 'second']
        ... )
        >>> series_evaluator = MatchEvaluator(
        ...     mapping={'fruits_equal': is_fruits_equal},
        ...     hierarchy={'matched': is_match}
        ... )
        >>> evaluations = series_evaluator.evaluate_against_df(
        ...     series=apple,
        ...     df=fruit_basket
        ... )
        >>> pd.DataFrame(evaluations)
           index_x index_y  fruits_equal    match
        0        0   first          True  matched

        Given no evaluations are assigned a match type the first row in the
        dataframe is assigned the match type value of the default argument and
        is yielded:

        >>> apple = pd.Series(['apple'], index=['fruits'], name=0)
        >>> fruit_basket = pd.DataFrame(
        ...     {'fruits': ['pear', 'pear']},
        ...     index=['first', 'second']
        ... )
        >>> series_evaluator = MatchEvaluator(
        ...     mapping={'fruits_equal': is_fruits_equal},
        ...     hierarchy={'matched': is_match},
        ...     default='NOT MATCHED'
        ... )
        >>> evaluations = series_evaluator.evaluate_against_df(
        ...     series=apple,
        ...     df=fruit_basket
        ... )
        >>> pd.DataFrame(evaluations)
           index_x index_y  fruits_equal        match
        0        0   first         False  NOT MATCHED

        Change the match field name by specifying the match_col argument, or
        ommit by passing/assigning it in as None

        >>> apple = pd.Series(['apple'], index=['fruits'], name=0)
        >>> fruit_basket = pd.DataFrame({'fruits': ['apple', 'pear']})
        >>> series_evaluator = MatchEvaluator(
        ...     mapping={'fruits_equal': is_fruits_equal},
        ...     hierarchy={'matched': is_match},
        ...     match_col="eval status"
        ... )
        >>> evaluations = series_evaluator.evaluate_against_df(
        ...     series=apple,
        ...     df=fruit_basket
        ... )
        >>> pd.DataFrame(evaluations)
           index_x  index_y  fruits_equal eval status
        0        0        0          True     matched
        >>> series_evaluator.match_col = None
        >>> evaluations = series_evaluator.evaluate_against_df(
        ...     series=apple,
        ...     df=fruit_basket,
        ... )
        >>> pd.DataFrame(evaluations)
           index_x  index_y  fruits_equal
        0        0        0          True
        """
        evaluations = super().evaluate_against_df(series, df)
        evaluation, match_status = match_highest(
            evaluations,
            hierarchy=self.hierarchy,
            default=self.default
        )
        if self.match_col is not None:
            evaluation[self.match_col] = match_status
        yield evaluation
