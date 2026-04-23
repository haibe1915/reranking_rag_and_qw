from typing import List, Dict, Optional
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class LLMGenerator:
    def __init__(self, provider: str = "huggingface", **kwargs):
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
            model_name = kwargs.get("hf_model", kwargs.get("hf_model_name", "Qwen/Qwen2.5-3B-Instruct"))
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )

    def generate(
        self,
        prompt: str,
        context: Optional[List[Dict]] = None,
        max_tokens: int = 256,
        temperature: float = 0.3
    ) -> str:
        if context: 
            ctx_text = "\n".join([d.get("text","")[:300] for d in context if isinstance(d, dict)])[:1500]
            from src.rewriting.prompt_templates import GENERATION_PROMPT
            prompt = GENERATION_PROMPT.format(context=ctx_text, question=prompt)
            
        if self.provider == "ollama":
            return self._ollama(prompt, max_tokens, temperature)
        elif self.provider == "groq":
            return self._groq(prompt, max_tokens, temperature)
        elif self.provider == "openai":
            return self._openai(prompt, max_tokens, temperature)
        elif self.provider == "huggingface":
            return self._huggingface(prompt, max_tokens, temperature)
        return ""

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
        except Exception:
            pass
        return ""

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
        except Exception:
            return ""

    def _openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception:
            return ""

    def _huggingface(self, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            return self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
        except Exception:
            return ""