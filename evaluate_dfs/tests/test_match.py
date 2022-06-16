import unittest
import doctest
from evaluate_dfs import match


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(match))
    return tests


if __name__ == '__main__':
    unittest.main()
