from typing import List
import re

from src.rewriting.prompt_templates import (
    MULTI_QUERY_PROMPT,
    HYDE_PROMPT,
    CODE_SWITCHING_PROMPT,
)

class QueryRewriter:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    def rewrite(self, query: str, method: str = "multi-query-hyde") -> List[str]:
        """Rewrite query using multiple methods"""
        queries = [query]  # Original query
        
        if method in ["multi-query", "multi-query-hyde"]:
            queries.extend(self.multi_query(query))
        
        if method in ["hyde", "multi-query-hyde"]:
            hyde = self._hyde(query)
            if hyde:
                queries.extend(hyde)
        
        # Code-switching detection
        if self.has_code_switching(query):
            queries.extend(self.code_switching_handle(query))
        
        # Remove duplicates while preserving order
        seen: set = set()
        result: List[str] = []
        for q in queries:
            q = q.strip()
            if q and q not in seen:
                seen.add(q)
                result.append(q)
        return result
    
    def multi_query(self, query: str, num_variations: int = 2) -> List[str]:
        """Generate variations of the query"""
        if self.llm_client:
            prompt = MULTI_QUERY_PROMPT.format(
                num_queries=num_variations, question=query
            )
            try:
                raw = self.llm_client.generate(prompt, [], max_tokens=200)
                parsed = self._parse_answer_block(raw)
                lines = [l.strip() for l in parsed.splitlines() if l.strip()]
                if lines:
                    return lines[:num_variations]
            except Exception:
                pass

        # Rule-based fallback
        return [
            self._apply_synonyms(query),
            self._reorder_words(query),
        ][:num_variations]

    
    def _hyde(self, query: str) -> List[str]:
        """HyDE - Generate hypothetical documents"""
        if not self.llm_client:
            return []
        prompt = HYDE_PROMPT.format(question=query)
        try:
            raw = self.llm_client.generate(prompt, [], max_tokens=150)
            passage = self._parse_answer_block(raw).strip()
            
            if passage and len(passage.split()) <= 80:
                return [passage]
        except Exception:
            pass
        return []
        
    # keep old name as alias so notebooks don't break
    def hyde_rewrite(self, query: str) -> List[str]:
        return self._hyde(query)

    def multi_query_rewrite(self, query: str, num_variations: int = 2) -> List[str]:
        return self._multi_query(query, num_variations)
    
    def has_code_switching(self, text: str) -> bool:
        has_vn = bool(re.search(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]", text))
        has_en = bool(re.search(r"\b[a-zA-Z]{2,}\b", text))
        return has_vn and has_en

    def code_switching_handle(self, query: str) -> List[str]:
        """Try LLM translation; fallback to bracketing English terms."""
        if self.llm_client:
            prompt = CODE_SWITCHING_PROMPT.format(question=query)
            try:
                raw = self.llm_client.generate(prompt, [], max_tokens=150)
                translated = self._parse_answer_block(raw).strip()
                
                # Validate: output phải ngắn hơn 3x query, và khác query gốc
                if (translated
                    and translated != query
                    and len(translated) < len(query) * 3   # không quá dài
                    and len(translated.split()) <= len(query.split()) + 5  # không thêm quá nhiều từ
                ):
                    return [translated]
            except Exception:
                pass

        # Fallback heuristic: bracket English terms
        english_terms = re.findall(r"\b[a-zA-Z]{2,}\b", query)
        if english_terms:
            bracketed = query
            for term in english_terms[:3]:
                bracketed = bracketed.replace(term, f"({term})")
            return [bracketed]
        return []
    
    @staticmethod
    def _parse_answer_block(text: str) -> str:
        """Extract content inside <answer>...</answer> if present."""
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        
        # Fallback: chỉ có <answer> mà không có </answer> (bị cắt do max_tokens)
        m_open = re.search(r"<answer>(.*)", text, re.DOTALL | re.IGNORECASE)
        if m_open:
            return m_open.group(1).strip()
    
        # Fallback: không có tag nào
        return text.strip()

    @staticmethod
    def _apply_synonyms(text: str) -> str:
        synonyms = {
            "là gì": "định nghĩa là gì",
            "như thế nào": "cách thực hiện như thế nào",
            "tại sao": "lý do tại sao",
            "cách": "phương pháp",
        }
        result = text
        lower = text.lower()
        for key, value in synonyms.items():
            if key in lower:
                result = lower.replace(key, value, 1)
                break
        return result

    @staticmethod
    def _reorder_words(text: str) -> str:
        words = text.split()
        if len(words) > 4:
            return " ".join(words[2:] + words[:2])
        return text