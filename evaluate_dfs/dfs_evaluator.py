"""
This module provides evaluator classes that evaluates every row from one
dataframe against rows in another.

A dataframes evaluator instance is composed of an SeriesEvaluator (or subclass)
instance found in the series_evaluator module.

Additionally, this module also provides the ParallelEvaluator that works just
like SeriesEvaluator, but processes the evaluations in parallel.

Classes Offered
---------------
- BaseDataFramesEvaluator
     Abstract evaluator class
- DataFramesEvaluator
     An evaluator that evaluates every row from one dataframe against
     rows in another.
- ParallelEvaluator
     Just like DataFramesEvaluator except evaluates the dataframes in parallel.
"""
from abc import abstractmethod, ABC
from itertools import chain
from multiprocessing import get_context
from pandarallel import pandarallel
import pandas as pd
from .series_evaluator import SeriesEvaluator


class BaseDataFramesEvaluator(ABC):
    def __init__(self, series_evaluator: SeriesEvaluator) -> None:
        """
        Parameters
        ----------
        series_evaluator : BaseSeriesEvaluator
            A instance of a subclassed BaseSeriesEvaluator.
        """
        self.series_evaluator = series_evaluator

    @abstractmethod
    def evaluate(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Evaluates every row from df1 against rows in df2."""


class DataFramesEvaluator(BaseDataFramesEvaluator):
    """
    An evaluator that evaluates every row from df1 against rows in df2.

    Attributes
    ----------
    series_evaluator : BaseSeriesEvaluator
        A instance of a subclassed BaseSeriesEvaluator.

    Methods
    -------
    evalaute(df1, df2)
        Evaluates every row from df1 against rows in df2.
    Examples
    --------
    >>> import pandas as pd
    >>> from evaluate_dfs.series_evaluator import SeriesEvaluator
    >>> from evaluate_dfs.dfs_evaluator import DataFramesEvaluator
    >>> fruit_basket1 = pd.DataFrame({'fruits': ['apple', 'pineapple']})
    >>> fruit_basket2 = pd.DataFrame({'fruits': ['apple', 'peach']})
    >>> def is_fruits_equal(series1: pd.Series, series2: pd.Series) -> bool:
    ...     return series1.fruits == series2.fruits
    ...
    >>> series_evaluator = SeriesEvaluator(
    ... mapping={'fruits_equal': is_fruits_equal},
    ... )
    >>> dfs_evaluator = DataFramesEvaluator(series_evaluator=series_evaluator)
    >>> dfs_evaluator.evaluate(df1=fruit_basket1, df2=fruit_basket2)
       index_x  index_y  fruits_equal
    0        0        0          True
    1        0        1         False
    2        1        0         False
    3        1        1         False
    """
    def __init__(self, series_evaluator: SeriesEvaluator) -> None:
        super().__init__(series_evaluator=series_evaluator)

    def evaluate(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates every row from df1 against rows in df2.
        """
        eval_against_df = self.series_evaluator.evaluate_against_df
        evaluations = [eval_against_df(df1.loc[idx], df2) for idx in df1.index]
        evaluations = chain(*evaluations)
        return pd.DataFrame(evaluations)


class ParallelEvaluator(DataFramesEvaluator):
    """
    An evaluator that evaluates every row from df1 against rows in df2,
    processed in parallel.

    Attributes
    ----------
    The documentation for the following parameters is directly from the
    pandarallel.pandarallel.__init__(v.1.5.5) docs: nb_workers,
    progress_bar, verbose, and use_memory_fs.

    series_evaluator : BaseSeriesEvaluator
        A instance of a subclassed BaseSeriesEvaluator.
    nb_workers: int, optional
        Number of workers used for parallelisation
        If not set, all available CPUs will be used.
    progress_bar: bool, default False
        Display progress bars if set to `True`
    verbose: int, optional
        The verbosity level
        0 - Don't display any logs
        1 - Display only warning logs
        2 - Display all logs
    use_memory_fs: bool, optional
        If set to None and if memory file system is available, Pandaralllel
        will use it to transfer data between the main process and workers.
        If memory file system is not available, Pandarallel will default on
        multiprocessing data transfer (pipe).

        If set to True, Pandarallel will use memory file system to transfer
        data between the main process and workers and will raise a
        SystemError if memory file system is not available.

        If set to False, Pandarallel will use multiprocessing data transfer
        (pipe) to transfer data between the main process and workers.

        Using memory file system reduces data transfer time between the
        main process and workers, especially for big data.

        Memory file system is considered as available only if the
        directory `/dev/shm` exists and if the user has read and write
        permission on it.

        Basically memory file system is only available on some Linux
        distributions (including Ubuntu).

    Methods
    -------
    evaluate(df1, df2)
        Behaves just like DataFramesEvaluator.evaluate, evaluating every row
        from df1 against rows in df2, but processed in parallel.

    Examples
    --------
    >>> import pandas as pd
    >>> from evaluate_dfs.series_evaluator import SeriesEvaluator
    >>> from evaluate_dfs.dfs_evaluator import ParallelEvaluator
    >>> fruit_basket1 = pd.DataFrame({'fruits': ['apple', 'pineapple']})
    >>> fruit_basket2 = pd.DataFrame({'fruits': ['apple', 'peach']})
    >>> def is_fruits_equal(series1: pd.Series, series2: pd.Series) -> bool:
    ...     return series1.fruits == series2.fruits
    ...
    >>> series_evaluator = SeriesEvaluator(
    ... mapping={'fruits_equal': is_fruits_equal},
    ... )
    >>> dfs_evaluator = ParallelEvaluator(series_evaluator, verbose=0)
    >>> dfs_evaluator.evaluate(df1=fruit_basket1, df2=fruit_basket2)
       index_x  index_y  fruits_equal
    0        0        0          True
    1        0        1         False
    2        1        0         False
    3        1        1         False
    """
    def __init__(
            self,
            series_evaluator: SeriesEvaluator,
            nb_workers: int = None,
            progress_bar: bool = False,
            verbose: int = 2,
            use_memory_fs: bool = None
    ) -> None:
        super().__init__(series_evaluator=series_evaluator)
        if nb_workers:
            self.nb_workers = nb_workers
        else:
            context = get_context("fork")
            self.nb_workers = context.cpu_count()
        self.progress_bar = progress_bar
        self.verbose = verbose
        self.use_memory_fs = use_memory_fs

    def evaluate(
            self,
            df1: pd.DataFrame,
            df2: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Behaves just like DataFramesEvaluator.evaluate, evaluating every row
        from df1 against rows in df2, but processed in parallel.
        """
        pandarallel.initialize(
            nb_workers=self.nb_workers,
            progress_bar=self.progress_bar,
            verbose=self.verbose,
            use_memory_fs=self.use_memory_fs
        )
        eval_against_df = self.series_evaluator.evaluate_against_df
        # eval_fn returns tuple since generators/iterators can't be pickled.
        eval_fn = lambda x: tuple(eval_against_df(x.copy(), df2))
        evaluations = df1.parallel_apply(eval_fn, axis=1).to_list()
        evaluations = chain(*evaluations)
        return pd.DataFrame(evaluations)
