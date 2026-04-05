import re
from .prompt_templates import MULTI_QUERY_PROMPT, HYDE_PROMPT

class QueryRewriter:
    def __init__(self, llm_client):
        self.llm = llm_client

    def rewrite(self, question, method="multi-query"):
        """Thực hiện viết lại truy vấn dựa trên phương pháp lựa chọn [5, 8]."""
        if method == "multi-query":
            prompt = MULTI_QUERY_PROMPT.format(question=question)
        elif method == "hyde":
            prompt = HYDE_PROMPT.format(question=question)
        
        response = self.llm.generate(prompt)
        return self._extract_answer(response)

    def _extract_answer(self, response):
        """Trích xuất nội dung bên trong thẻ <answer> [6]."""
        match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Nếu là multi-query, tách thành list
            return [q.strip() for q in content.split('\n') if q.strip()]
        return [response] # Fallback nếu không có thẻ
