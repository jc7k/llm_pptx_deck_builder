"""Minimal message classes used by tests."""

from dataclasses import dataclass
from typing import Union


@dataclass
class HumanMessage:
    content: str


@dataclass
class AIMessage:
    content: str


AnyMessage = Union[HumanMessage, AIMessage]

