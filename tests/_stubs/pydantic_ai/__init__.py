"""Very small stubs for pydantic_ai to support example tests."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Generic, TypeVar


T = TypeVar("T")


@dataclass
class RunContext(Generic[T]):
    deps: T


class _Result:
    def __init__(self, data: Any):
        self.data = data


class Agent:
    def __init__(self, model: Any, deps_type: Any, result_type: Any, system_prompt: Optional[str] = None):
        self.model = model
        self.deps_type = deps_type
        self.result_type = result_type
        self.system_prompt = system_prompt
        self._tools: Dict[str, Any] = {}

    def tool(self, fn):
        self._tools[fn.__name__] = fn
        return fn

    class _Override:
        def __init__(self, agent: "Agent", model: Any):
            self.agent = agent
            self.new_model = model
            self.prev_model = None

        def __enter__(self):
            self.prev_model = self.agent.model
            self.agent.model = self.new_model
            return self.agent

        def __exit__(self, exc_type, exc, tb):
            self.agent.model = self.prev_model
            return False

    def override(self, model: Any) -> "Agent._Override":
        return Agent._Override(self, model)

    def _default_response(self, prompt: str, tool_names: List[str]) -> Dict[str, Any]:
        msg = f"Processed: {prompt}"
        if tool_names:
            msg += f" | tools: {', '.join(tool_names)}"
        return {"message": msg, "confidence": 0.8, "actions": tool_names}

    def _invoke_tools_sync(self, deps: Any, call_tools: Any) -> List[str]:
        if call_tools == "all":
            names = list(self._tools.keys())
        elif isinstance(call_tools, list):
            names = [n for n in call_tools if n in self._tools]
        else:
            names = []
        ctx = RunContext(deps)
        for name in names:
            fn = self._tools[name]
            if asyncio.iscoroutinefunction(fn):
                try:
                    asyncio.run(fn(ctx, "test"))
                except RuntimeError:
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(fn(ctx, "test"))
            else:
                fn(ctx, "/test", data=None)
        return names

    async def _invoke_tools_async(self, deps: Any, call_tools: Any) -> List[str]:
        if call_tools == "all":
            names = list(self._tools.keys())
        elif isinstance(call_tools, list):
            names = [n for n in call_tools if n in self._tools]
        else:
            names = []
        ctx = RunContext(deps)
        for name in names:
            fn = self._tools[name]
            if asyncio.iscoroutinefunction(fn):
                await fn(ctx, "test")
            else:
                fn(ctx, "/test", data=None)
        return names

    def run_sync(self, prompt: str, deps: Any) -> _Result:
        if hasattr(self.model, "function") and callable(self.model.function):
            tools_view = {name: fn for name, fn in self._tools.items()}
            text = self.model.function([type("Msg", (), {"content": prompt})()], tools_view)
            try:
                data = json.loads(text)
            except Exception:
                data = self._default_response(prompt, [])
        else:
            call_tools = getattr(self.model, "call_tools", [])
            used_tools = self._invoke_tools_sync(deps, call_tools)
            custom_text = getattr(self.model, "custom_output_text", None)
            if custom_text:
                try:
                    data = json.loads(custom_text)
                except Exception:
                    data = self._default_response(prompt, used_tools)
            else:
                data = self._default_response(prompt, used_tools)
        result_obj = self.result_type(**data)
        return _Result(result_obj)

    async def run(self, prompt: str, deps: Any) -> _Result:
        if hasattr(self.model, "function") and callable(self.model.function):
            tools_view = {name: fn for name, fn in self._tools.items()}
            text = self.model.function([type("Msg", (), {"content": prompt})()], tools_view)
            try:
                data = json.loads(text)
            except Exception:
                data = self._default_response(prompt, [])
        else:
            call_tools = getattr(self.model, "call_tools", [])
            used_tools = await self._invoke_tools_async(deps, call_tools)
            custom_text = getattr(self.model, "custom_output_text", None)
            if custom_text:
                try:
                    data = json.loads(custom_text)
                except Exception:
                    data = self._default_response(prompt, used_tools)
            else:
                data = self._default_response(prompt, used_tools)
        result_obj = self.result_type(**data)
        return _Result(result_obj)

