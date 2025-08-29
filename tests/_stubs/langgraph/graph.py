"""Minimal stubs for langgraph.graph to satisfy tests.

Implements START/END, StateGraph, and a compiled Graph object exposing .nodes.
"""

from typing import Any, Callable, Dict, List, Tuple, Optional


class _Sentinel:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name


START = _Sentinel("START")
END = _Sentinel("END")


class StateGraph:
    def __init__(self, _state_type: Any):
        self.nodes: Dict[str, Callable] = {}
        self.edges: List[Tuple[Any, Any]] = []

    def add_node(self, name: str, fn: Callable) -> None:
        self.nodes[name] = fn

    def add_edge(self, from_node: Any, to_node: Any) -> None:
        self.edges.append((from_node, to_node))

    def compile(self, checkpointer: Optional[Any] = None) -> "_Graph":
        return _Graph(self.nodes, self.edges)


class _Graph:
    def __init__(self, nodes: Dict[str, Callable], edges: List[Tuple[Any, Any]]):
        self.nodes = dict(nodes)
        self._edges = list(edges)

    def _ordered_nodes(self) -> List[str]:
        order: List[str] = []
        next_map: Dict[Any, Any] = {}
        for a, b in self._edges:
            next_map[a] = b
        cur = START
        seen = set()
        while cur in next_map and next_map[cur] not in (END, None):
            nxt = next_map[cur]
            if isinstance(nxt, str):
                if nxt in seen:
                    break
                seen.add(nxt)
                order.append(nxt)
            cur = nxt
        return order

    def invoke(self, state: Dict[str, Any], _config: Optional[Dict[str, Any]] = None):
        result = dict(state)
        for name in self._ordered_nodes():
            fn = self.nodes.get(name)
            if callable(fn):
                updates = fn(result)
                if isinstance(updates, dict):
                    result.update(updates)
        return result

    async def ainvoke(self, state: Dict[str, Any], _config: Optional[Dict[str, Any]] = None):
        return self.invoke(state, _config)

