"""Unit tests for multiple_dispatch decorator"""

import unittest
from unittest.mock import Mock

from multiple_dispatch import multiple_dispatch

DOC = """Dispatch function.
Each signature & its docstring is appended below.

----
func()

This is a docstring for func with no arguments.

----
func(arg)

This is a docstring for func with one argument.

----
func(one: int, two: str)

This is a docstring for func with an int and a string as arguments."""


class TestMultipleDispatch(unittest.TestCase):
    """Test suite for the multiple_dispatch decorator"""

    def test_multiple_dispatch_on_functions(self):
        """Test multiple_dispatch on functions"""
        mock_no_args = Mock()
        mock_one_arg = Mock()
        mock_two_typed_args = Mock()

        @multiple_dispatch
        def func():
            mock_no_args()

        @multiple_dispatch
        def func(arg):
            mock_one_arg(arg)

        @multiple_dispatch
        def func(one: int, two: str):
            mock_two_typed_args(one, two)

        func()
        func(1)
        func(1, "2")
        mock_no_args.assert_called_once_with()
        mock_one_arg.assert_called_once_with(1)
        mock_two_typed_args.assert_called_once_with(1, "2")

        with self.assertRaisesRegex(
            TypeError, r"func\(\) dispatch function has no matching signatures"
        ):
            func("2", 1)

    def test_multiple_dispatch_on_methods(self):
        """Test multiple_dispatch on methods"""
        mock_no_args = Mock()
        mock_one_arg = Mock()
        mock_two_typed_args = Mock()

        class Class:
            @multiple_dispatch
            def method(self):
                mock_no_args(self)

            @multiple_dispatch
            def method(self, arg):
                mock_one_arg(self, arg)

            @multiple_dispatch
            def method(self, one: int, two: str):
                mock_two_typed_args(self, one, two)

        obj = Class()
        obj.method()
        obj.method(1)
        obj.method(1, "2")
        mock_no_args.assert_called_once_with(
            obj,
        )
        mock_one_arg.assert_called_once_with(obj, 1)
        mock_two_typed_args.assert_called_once_with(obj, 1, "2")

        with self.assertRaisesRegex(
            TypeError, r"method\(\) dispatch function has no matching signatures"
        ):
            obj.method("2", 1)

    def test_multiple_dispatch_namespaces(self):
        """Test multiple_dispatch handles namespaces appropriately"""
        mock_method = Mock()
        mock_func = Mock()

        class Class:
            @multiple_dispatch
            def combust(self):
                mock_method(self)

        @multiple_dispatch
        def combust(arg):
            mock_func(arg)

        obj = Class()
        obj.combust()
        combust(obj)
        mock_method.assert_called_once_with(obj)
        mock_func.assert_called_once_with(obj)

    def test_multiple_dispatch_doc(self):
        """Test multiple_dispatch docstring formatting"""

        @multiple_dispatch
        def func():
            """
            This is a docstring for func with no arguments.
            """

        @multiple_dispatch
        def func(arg):
            """
            This is a docstring for func with one argument.
            """

        @multiple_dispatch
        def func(one: int, two: str):
            """
            This is a docstring for func with an int and a string as arguments.
            """

        self.assertEqual(func.__doc__, DOC)


if __name__ == "__main__":
    unittest.main()
