"""Minimal @tool decorator stub with .invoke support used in tests."""

from typing import Any, Callable


def tool(func: Callable) -> Any:
    class ToolWrapper:
        __name__ = getattr(func, "__name__", "tool")

        def __init__(self, fn: Callable):
            self._fn = fn

        def __call__(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

        def invoke(self, arg=None, /, **kwargs):
            if isinstance(arg, dict) and not kwargs:
                return self._fn(**arg)
            if arg is None:
                return self._fn(**kwargs)
            return self._fn(arg, **kwargs)

    return ToolWrapper(func)

