# Multiple Dispatch

In James Murphy's mCoding [video on metaclasses](https://www.youtube.com/watch?v=yWzMiaqnpkI), he created a proof-of-concept `overload` Python decorator. You can find the code [here](https://github.com/mCodingLLC/VideosSampleCode/blob/master/videos/077_metaclasses_in_python/overloading.py).

In general, it's pretty hard to come up with uses for metaclasses in Python that wouldn't be better served with a simpler, less magical implementation. But this struck me as one such example, truly requiring differentiation of the `__prepare__` step of class creation. Neat!

That's what I thought, until I started playing around with the code a bit, and realized a more general `overload` behavior could be achieved with a simpler, more functional approach. `multiple_dispatch.py` is a simple implementation of that approach, stealing James' code for the actual dispatch logic (the function and associated code for what he had called `best_match`). As mentioned in his video, this is not really a useful approach in practice, because of the performance issues with the `best_match` function. To get decent performance, you'd need to write it in Cython, and also probably fix some of the weaknesses like not supporting generic types, or dealing with [PEP 563](https://www.python.org/dev/peps/pep-0563/)'s postponed evaluation of annotations.

I still don't think you'd get the speed benefits of Julia's approach to multiple dispatch of many statically typed functions even with that though, since those seem to come from the underlying functions that are dispatched to being just-in-time compiled thanks to each being strongly typed.

I decided to rename the decorator to `multiple_dispatch` for two reasons:
1) Avoid the confusion of colliding with the [`typing.overload`](https://docs.python.org/3/library/typing.html#typing.overload) function.
2) Make more obvious the analogy to [`functools.singledispatch`](https://docs.python.org/3/library/functools.html#functools.singledispatch), which is already in the standard library (I wonder how fast that is...).

That said, I took a more purely functional approach than `functools.singledispatch`, which creates a callable object with convenience methods like `dispatch`, `register`, and `registry`. Sure, convenient, but not compatible when you _actually need a function_, which is rare but non-zero. A utility function ought to work everywhere I'd say. In my case, that includes asynchronous coroutines support (though you can't mix overloading synchronous and asynchronous functions).

Anyways, I also included a few simple unit tests in `test_multiple_dispatch.py` just to prove out functionality, including some nice little docstring formatting. It would be nice to also handle type annotation, but sadly the existing `typing.overload` function is just a directive to the type checker, not a thing that mutates the annotation (I was bummed out to learn this), so any solution would probably require coordination with mypy.

Works in Python 3.9.
