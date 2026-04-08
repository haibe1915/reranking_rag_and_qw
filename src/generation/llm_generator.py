from typing import List, Dict, Optional
import requests
import torch

from src.rewriting.prompt_templates import GENERATION_PROMPT

class LLMGenerator:
    def __init__(self, provider: str = "ollama", **kwargs):
        self.provider = provider
        self.config = kwargs
        
        if provider == "ollama":
            self.host = kwargs.get("ollama_host", "http://localhost:11434")
            self.model = kwargs.get("ollama_model", "phi")
        elif provider == "groq":
            self.api_key = kwargs.get("groq_api_key", "")
            self.model = kwargs.get("groq_model", "llama2-70b-4096")
        elif provider == "openai":
            self.api_key = kwargs.get("openai_api_key", "")
            self.model = kwargs.get("openai_model", "gpt-4o")
        elif provider == "huggingface":
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_name = kwargs.get("hf_model", kwargs.get("hf_model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"Loading {model_name} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            print("✅ Model loaded")
    
    def generate(self, 
                 query: str, 
                 context: List[Dict], 
                 max_tokens: int = 500,
                 temperature: float = 0.7) -> str:
        """Generate answer based on query and context"""
        
        context_text = self._build_context(context)
        prompt = GENERATION_PROMPT.format(context=context_text, question=query)
        
        if self.provider == "ollama":
            return self._ollama(prompt, max_tokens, temperature)
        elif self.provider == "groq":
            return self._groq(prompt, max_tokens, temperature)
        elif self.provider == "openai":
            return self._openai(prompt, max_tokens, temperature)
        elif self.provider == "huggingface":
            return self._huggingface(prompt, max_tokens, temperature)
        else:
            return "Không hỗ trợ provider: " + self.provider
    
    def _build_context(self, context: List[Dict]) -> str:
        """Build context string from documents"""
        context_parts = []
        for i, doc in enumerate(context[:3], 1):
            text = doc.get("text", "")[:400]
            context_parts.append(f"[{i}] {text}")
        return "\n".join(context_parts) if context_parts else "Không có tài liệu liên quan."
    
    def _ollama(self, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature,
                    "num_predict": max_tokens
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get("response", "")
        except Exception as e:
            print(f"❌ Ollama error: {e}")
        return "Không thể kết nối Ollama"
    
    def _groq(self, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            from groq import Groq

            client = Groq(api_key=self.api_key)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"❌ Groq error: {e}")
        return ""

    def _openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ OpenAI error: {e}")
        return ""
    
    def _huggingface(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using HuggingFace (Qwen or other models)"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            return self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
        except Exception as e:
            print(f"❌ HuggingFace error: {e}")
        return ""