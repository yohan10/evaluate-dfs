from typing import Any, Callable, Hashable, Iterable, Mapping, Tuple

MatchType = Hashable


def get_match_type(
        obj: Any,
        hierarchy: Mapping[Hashable, Callable],
        default: Any = None
) -> MatchType:
    """
    Returns the evaluation's match type.

    Parameters
    ----------
    obj : any
    hierarchy : mapping of match type and callable
        Every item in obj will be passed into each of the callables (values)
        in this mapping. Once a callable returns truthy, the paired match type
        (keys) is associated to the item.

        The item with the highest match type is returned. The first
        key-value pair is considered the highest rank, and the last
        pair is the second to lowest rank. The 'default' value is the lowest
        rank. Given two items, both with the highest match type, the item with
        the lower position in obj will be returned.
    default : any
        The default case value when the evaluation returns falsy for all
        the match functions in the hierarchy. The default value is considered
        the lowest rank.

    Returns
    -------
    Hashable
        The match type (keys) associated with the first callable (values)
        that returns truthy with obj passed in. If no callables return truthy,
        return default.

    Examples
    --------
    >>> food_tastiness = {
    ...     'beyond belief': lambda x: x in ["Moms Cooking"],
    ...     'good': lambda x: x in ['Pho', 'Ice Cream'],
    ...     'ok': lambda x: x in ['Celery', 'Graham Crackers', 'Rice'],
    ...     'bad': lambda x: x in ['Casu Marzu', 'Jello Salad', 'Durian']
    ... }
    >>> get_match_type('Moms Cooking', food_tastiness)
    'beyond belief'
    >>> get_match_type('Rubber Tire', food_tastiness)
    >>> get_match_type('Rubber Tire', food_tastiness, default='not a food')
    'not a food'
    """
    for category, fn in hierarchy.items():
        if fn(obj):
            return category
    return default


def get_match_rank(
        hierarchy: Mapping[Hashable, Callable],
        match_type: MatchType,
        reverse: bool = False
) -> int:
    """
    Returns the index of hierarchy where match_type exists.

    Parameters
    ----------
    hierarchy : mapping of match type and callable
        The callable first parameter should accommodate an evaluation and it
        should return a truthy or falsy value. By default, the first key-
        value pair is considered the highest rank and the last key-value pair,
        the second to lowest rank. Match types that can't be found in the
        hierarchy are considered the lowest rank.
    match_type : any
        The match type to determine its rank.
    reverse : bool
        Hierarchy ranking is reversed. the last key-value pair is considered
        the highest rank and the first key-value pair, the second to lowest
        rank. Match types that can't be found in the hierarchy are
        considered the lowest rank.

    Returns
    -------
    int
        Rank of the match type. 0 being the highest rank.

    Examples
    --------
    >>> food_tastiness = {
    ...     'beyond belief': lambda x: x in ["Moms Cooking"],
    ...     'good': lambda x: x in ['Pho', 'Ice Cream'],
    ...     'ok': lambda x: x in ['Celery', 'Graham Crackers', 'Rice'],
    ...     'bad': lambda x: x in ['Casu Marzu', 'Jello Salad', 'Durian']
    ... }
    >>> get_match_rank(food_tastiness, 'beyond belief')
    0
    >>> get_match_rank(food_tastiness, 'beyond belief', reverse=True)
    3

    Match types not in the hierarchy is given an index outside (+1) of the
    hierarchy.

    >>> get_match_rank(food_tastiness, 'burger')
    4
    """
    order = list(hierarchy.keys())
    if reverse:
        order = order[::-1]
    try:
        return order.index(match_type)
    except ValueError:
        return len(order)


def match_highest(
        iterable: Iterable,
        hierarchy: Mapping[Hashable, Callable],
        default: Any = None,
        reverse: bool = False,
) -> Tuple[Any, MatchType]:
    """
    Returns evaluation with the highest match type in the hierarchy.

    Parameters
    ----------
    iterable : iterable
    hierarchy : mapping of match type and callable
        The callable first parameter should accommodate an evaluation and it
        should return a truthy or falsy value. By default, the first key-
        value pair is considered the highest rank and the last key-value pair,
        the second to lowest rank. The default value is always the lowest rank.
    default : any, default None
        The default case value when the evaluation returns falsy for all
        the match functions in the hierarchy. The default value is considered
        the lowest rank.
    reverse : bool, default False
        Hierarchy ranking is reversed. the last key-value pair is considered
        the highest rank and the first key-value pair, the second to lowest
        rank. The default value is always the lowest rank.

    Returns
    -------
    Tuple
        Tuple of the item with the highest match type in the hierarchy and
        its match type.

    Examples
    --------
    >>> food_tastiness = {
    ...     'beyond belief': lambda x: x in ["Moms Cooking"],
    ...     'good': lambda x: x in ['Pho', 'Ice Cream'],
    ...     'ok': lambda x: x in ['Celery', 'Graham Crackers', 'Rice'],
    ...     'bad': lambda x: x in ['Casu Marzu', 'Jello Salad', 'Durian']
    ... }
    >>> foods = ['Pho', 'Rice', 'Jello Salad', 'Moms Cooking']

    Hierarchy is ordered from highest rank to lowest rank by default

    >>> tastiest_food = match_highest(foods, food_tastiness)
    >>> tastiest_food
    ('Moms Cooking', 'beyond belief')

    reverse=True reverses the default order of the hierarchy.

    >>> nastiest_food = match_highest(foods, food_tastiness, reverse=True)
    >>> nastiest_food
    ('Jello Salad', 'bad')

    If no match type can be assigned using the hierarchy, set the match type
    to 'default' keyword argument. Default value for 'default' is None.

    >>> tires = ['Medium Tire', 'Big Tire', 'Small Tire']
    >>> match_highest(tires, food_tastiness)
    ('Medium Tire', None)
    >>> match_highest(tires, food_tastiness, default="not a food")
    ('Medium Tire', 'not a food')
    """
    matched = [(i, get_match_type(i, hierarchy, default)) for i in iterable]
    key = lambda x: get_match_rank(hierarchy, match_type=x[1], reverse=reverse)
    closest = min(matched, key=key)
    return closest
