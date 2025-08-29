"""A minimal FakeListLLM compatible with tests."""

from typing import List


class _Response:
    def __init__(self, content: str):
        self.content = content


class FakeListLLM:
    def __init__(self, responses: List[str]):
        self._responses = list(responses)
        self._i = 0

    def invoke(self, *_args, **_kwargs):
        if not self._responses:
            return _Response("")
        resp = self._responses[self._i % len(self._responses)]
        self._i += 1
        return _Response(resp)

