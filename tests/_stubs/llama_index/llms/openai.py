"""Stub for LlamaIndex OpenAI LLM wrapper."""


class OpenAI:
    def __init__(self, api_key: str, model: str, temperature: float = 0.1):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

