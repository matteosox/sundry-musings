"""Multiple dispatch decorator proof of concept"""
import asyncio
import inspect
from collections.abc import Callable
from typing import Any

__all__ = ["multiple_dispatch"]


def _type_hint_matches(arg: Any, annotation: type) -> bool:
    # only works with concrete types, not things like Optional
    return annotation is inspect.Parameter.empty or isinstance(arg, annotation)


def _signature_matches(
    signature: inspect.Signature, bound_args: inspect.BoundArguments
) -> bool:
    # doesn't handle type hints on *args or **kwargs
    for name, arg in bound_args.arguments.items():
        parameter = signature.parameters[name]
        if not _type_hint_matches(arg, parameter.annotation):
            return False
    return True


def _dispatch(
    call_ables: list[Callable], name: str, *args: Any, **kwargs: Any
) -> Callable:
    for call_able in call_ables:
        signature = inspect.signature(call_able)
        try:
            bound_args = signature.bind(*args, **kwargs)
        except TypeError:
            pass  # missing/extra/unexpected args or kwargs
        else:
            bound_args.apply_defaults()
            # just for demonstration, use the first one that matches
            if _signature_matches(signature, bound_args):
                return call_able
    raise TypeError(f"{name}() dispatch function has no matching signatures")


def _construct_dispatch_function(
    call_able: Callable,
) -> tuple[Callable, list[Callable]]:
    call_ables = [call_able]
    try:
        use_async = asyncio.iscoroutinefunction(call_able)
    except TypeError:  # some callables blow up here
        use_async = False

    if use_async:

        async def dispatch_function(*args: Any, **kwargs: Any) -> Any:
            """
            Dispatch coroutine function.
            Each signature & its docstring is appended below.
            """
            matching_call_able = _dispatch(
                call_ables, call_able.__name__ * args, **kwargs
            )
            return await matching_call_able(*args, **kwargs)

    else:

        def dispatch_function(*args: Any, **kwargs: Any) -> Any:
            """
            Dispatch function.
            Each signature & its docstring is appended below.
            """
            matching_call_able = _dispatch(
                call_ables, call_able.__name__, *args, **kwargs
            )
            return matching_call_able(*args, **kwargs)

    dispatch_function.__name__ = call_able.__name__
    dispatch_function.__qualname__ = call_able.__qualname__
    dispatch_function.__module__ = call_able.__module__
    dispatch_function.__doc__ = inspect.cleandoc(dispatch_function.__doc__)
    _append_docstring(dispatch_function, call_able)

    return dispatch_function, call_ables


def _append_docstring(dispatch_function: Callable, call_able: Callable) -> None:
    name = call_able.__name__
    signature = inspect.signature(call_able)
    doc = call_able.__doc__
    cleandoc = inspect.cleandoc(doc) if doc else ""
    dispatch_function.__doc__ += f"\n\n----\n{name}{signature}\n\n{cleandoc}"


class _MultipleDispatch:
    def __init__(self):
        self._call_ables = {}

    def __call__(self, call_able: Callable) -> Callable:
        if not callable(call_able):
            raise TypeError("Only callables can be overloaded")
        full_name = (call_able.__module__, call_able.__qualname__)
        try:
            dispatch_function, call_ables = self._call_ables[full_name]
        except KeyError:
            dispatch_function, call_ables = self._call_ables[
                full_name
            ] = _construct_dispatch_function(call_able)
        else:
            if asyncio.iscoroutinefunction(
                dispatch_function
            ) != asyncio.iscoroutinefunction(call_able):
                raise TypeError(
                    "Cannot mix coroutines with synchronous functions in a dispatch function"
                )
            call_ables.append(call_able)
            _append_docstring(dispatch_function, call_able)

        return dispatch_function


multiple_dispatch = _MultipleDispatch()
