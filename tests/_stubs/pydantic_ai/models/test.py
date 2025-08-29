"""Testing model stubs used by example tests."""

from typing import Any, Callable, Optional


class TestModel:
    def __init__(self, custom_output_text: Optional[str] = None, call_tools: Any = None):
        self.custom_output_text = custom_output_text
        self.call_tools = call_tools or []


class FunctionModel:
    def __init__(self, function: Callable[[list, dict], str]):
        self.function = function

