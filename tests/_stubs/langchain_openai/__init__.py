"""Stub for langchain_openai.ChatOpenAI compatible with tests."""


class _Resp:
    def __init__(self, content: str):
        self.content = content


class ChatOpenAI:
    def __init__(self, api_key: str, model: str, temperature: float = 0.1):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def invoke(self, prompt: str) -> _Resp:  # pragma: no cover
        return _Resp("{}")

