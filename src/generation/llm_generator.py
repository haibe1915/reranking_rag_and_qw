from typing import List, Any

class LLMGenerator:
    """Minimal LLMGenerator stub used by main.py.

    Methods
    -------
    generate(query: str, context: list) -> str
        Return a generated answer (stub returns a fixed placeholder string).
    """

    def __init__(self, model_name: str = "gpt-4") -> None:
        self.model_name = model_name

    def generate(self, query: str, context: List[Any]) -> str:
        # Simple placeholder generation
        context_text = " ".join([str(d.get("text", "")) if isinstance(d, dict) else str(d) for d in context])
        return f"[stub answer using model={self.model_name}] Context: {context_text}"
