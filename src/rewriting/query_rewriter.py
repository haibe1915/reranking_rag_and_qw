import re
from typing import List

from src.rewriting.prompt_templates import (
    MULTI_QUERY_PROMPT,
    HYDE_PROMPT,
    CODE_SWITCHING_PROMPT,
)


class QueryRewriter:
    def __init__(self, llm_client=None, embedding_fn=None):
        self.llm_client = llm_client
        self.embedding_fn = embedding_fn

    def rewrite(self, query: str, method: str = "multi-query-hyde") -> List[str]:
        queries = [query]

        if method in ["multi-query", "multi-query-hyde"]:
            queries.extend(self.multi_query(query))

        if method in ["hyde", "multi-query-hyde"]:
            queries.extend(self._hyde(query))

        if self.has_code_switching(query):
            queries.extend(self.code_switching_handle(query))

        queries = self._filter_relevance(query, queries)
        seen = set()
        result = []
        for q in queries:
            q = q.strip()
            if q and q not in seen:
                seen.add(q)
                result.append(q)

        return result[:5]

    def multi_query(self, query: str, num_variations: int = 2) -> List[str]:
        if self.llm_client:
            prompt = MULTI_QUERY_PROMPT.format(
                num_queries=num_variations, question=query
            )
            try:
                raw = self.llm_client.generate(prompt, max_tokens=128, temperature=0.3)
                parsed = self._parse_answer_block(raw)

                lines = [
                    l.strip()
                    for l in parsed.splitlines()
                    if l.strip()
                ]

                lines = [l for l in lines if self._is_relevant(query, l)]
                if lines:
                    return lines[:num_variations]
            except Exception:
                pass

        return [self._apply_synonyms(query)]

    def _hyde(self, query: str) -> List[str]:
        if not self.llm_client:
            return []

        prompt = HYDE_PROMPT.format(question=query)

        try:
            raw = self.llm_client.generate(prompt, max_tokens=128, temperature=0.3)
            passage = self._parse_answer_block(raw).strip()

            if (
                passage
                and len(passage.split()) <= 60
                and self._is_relevant(query, passage)
            ):
                return [passage]

        except Exception:
            pass

        return []

    def hyde_rewrite(self, query: str) -> List[str]:
        return self._hyde(query)

    def multi_query_rewrite(self, query: str, num_variations: int = 2) -> List[str]:
        return self.multi_query(query, num_variations)

    # FIX 1: Removed misplaced `import re` that was here inside the class body.
    # FIX 2: Fixed indentation to consistent 4-space throughout.
    def has_code_switching(self, text: str) -> bool:
        words = re.findall(r"\b[a-z]+\b", text.lower())
        if not words:
            return False

        vn_common = {
            "di","an","va","co","la","gi","cua","nay","tai","cho","trong","duoc",
            "lam","ra","den","theo","nhu","nhung","mot","hai","ba","bon","nam",
            "bay","tam","chin","muoi","ong","anh","chi","em","con","me","cha","bo",
            "da","dang","se","moi","xong","vua","khi","sau","ngu","dau","bung",
            "ho","sot","can","benh","thuoc","kham","vien","bac","si","gio","ngay",
            "tuan","thang","sang","trua","chieu","toi","uong","tiem","huyet","ap",
            "nhip","tim","phoi","gan","than","mat","mui","chan","tay","lung","hong",
            "mieng","rang","toc","mong","bap","xuong","khop","vet","thuong","sung",
            "viem","nhiem","trung","mau","trang","vang","xanh","nhanh","cham",
            "manh","yeu","nhieu","lon","nho","cao","thap","nang","nhe","tot","xau",
            "nghi","ve","vao","len","mo","dong","bat","tat","lay","dat","giu",
            "bam","nhan","keo","day","cat","bop","phai","trai","tren","duoi",
            "ngoai","qua","roi","hay","neu","vi","de","tu","voi","that","rat",
            "kha","hon","nhat","cung","ca","thi","ma","hoac","deu","cac","vai",
            "san","pham","dich","vu","cong","ty","khach","hang","gia","tien","phi",
            "thanh","toan","mua","ban","giao","hang","van","chuyen","don","dat",
            "phong","ngu","bep","tam","nha","xe","duong","pho","quan","huyen",
            "tinh","viet","nam","bai","bao","hoc","sinh","giao","vien","truong",
            "lop","mon","kiem","tra","thi","diem","ket","qua","bao","cao",
            "he","thong","may","tinh","mang","phan","mem","ung","dung","cai","dat",
            "chay","lenh","dong","code","lap","trinh","bien","bien","so","ham",
            "lap","trinh","bien","dich","ket","noi","cau","hinh","thu","vien",
        }

        en_words = {
            "machine","learning","deep","neural","network","model","training",
            "dataset","algorithm","feature","input","output","layer","weight",
            "docker","ubuntu","linux","windows","macos","server","cloud","deploy",
            "container","image","volume","port","network","compose","kubernetes",
            "api","rest","graphql","endpoint","request","response","json","xml",
            "setup","install","config","update","upgrade","download","build","run",
            "python","java","javascript","typescript","golang","rust","kotlin","swift",
            "react","vue","angular","node","django","flask","spring","laravel",
            "database","mysql","postgres","mongodb","redis","sqlite","query","table",
            "git","github","gitlab","commit","push","pull","branch","merge","clone",
            "function","class","object","method","variable","array","string","int",
            "loop","if","else","return","import","export","module","package","library",
            "framework","backend","frontend","fullstack","devops","agile","scrum",
            "test","debug","error","bug","fix","refactor","review","deploy","release",
            "the","and","for","are","but","not","you","can","was","his","her",
            "she","they","our","who","get","has","him","how","man","new",
            "use","okay","ok","yes","hello","bye","thanks","thank","please",
            "with","from","have","this","that","will","been","were","said",
            "good","nice","great","very","much","many","some","more","most",
            "well","just","also","back","after","first","long","own","right",
            "big","high","small","large","next","early","bad","same","able",
            "call","feel","keep","let","put","seem","tell","try","turn","ask",
            "need","play","move","live","change","even","old","see","way",
            "online","offline","free","open","close","start","stop","new","hot",
            "top","best","like","love","follow","share","post","page","group",
            "chat","call","meet","zoom","team","slack","email","mail","send",
            "check","click","link","file","data","info","system","user","admin",
            "password","login","logout","account","profile","setting","menu",
            "home","search","filter","sort","list","detail","view","edit","delete",
            "save","upload","download","export","import","print","copy","paste",
            "phone","mobile","app","web","site","store","shop","order","pay",
            "price","sale","deal","offer","brand","product","service","support",
            "review","rate","star","comment","feed","news","blog","video","photo",
            "game","play","score","level","win","lose","team","player","match",
        }

        # Vietnamese-exclusive character patterns (never appear in English)
        VN_EXCLUSIVE = re.compile(
            r"uo[ci]|oai|uoi|uon|uong|ieng|oanh|"
            r"[aeiou]nh$|anh$|inh$|ung$|ong$|eng$|ang$|"
            r"^(nh|kh|gh|ngh|qu)[aeiou]|"
            r"^gi[aeiou]|^ph[aeiou]"
        )

        # Clearly English suffixes
        EN_SUFFIX = re.compile(
            r"(tion|sion|ment|ness|ful|less|ing|ings|ed|er|ers|est|ly|"
            r"ize|ise|ify|ous|ious|ive|able|ible|"
            r"ight|ck$|tch$)$"
        )

        # Consonant clusters that don't occur in Vietnamese
        EN_CLUSTER = re.compile(
            r"(str|spr|scr|spl|thr|phr|wr|kn|[bcdfghjklmnpqrstvwxyz]{3,})"
        )

        en_terms = []
        for w in words:
            if len(w) <= 1:
                continue
            if w in vn_common:
                continue
            if VN_EXCLUSIVE.search(w):
                continue

            if w in en_words:
                en_terms.append(w)
                continue

            if EN_SUFFIX.search(w):
                en_terms.append(w)
                continue

            if len(w) >= 4 and EN_CLUSTER.search(w):
                en_terms.append(w)

        return len(en_terms) > 0

    def code_switching_handle(self, query: str) -> List[str]:
        if self.llm_client:
            prompt = CODE_SWITCHING_PROMPT.format(question=query)
            try:
                raw = self.llm_client.generate(prompt, max_tokens=128, temperature=0.3)
                translated = self._parse_answer_block(raw).strip()

                # Sanity guards: reject if the result looks like prompt bleed-through
                # (contains template markers), is longer than 2x the original, or is
                # a multi-line blob (the model resumed generating extra examples).
                is_too_long = len(translated.split()) > len(query.split()) * 2 + 5
                has_prompt_bleed = any(
                    marker in translated
                    for marker in ("Câu hỏi gốc", "<answer", "Ví dụ", "QUY TẮC")
                )
                is_multiline = "\n" in translated.strip()

                if translated and translated != query and not is_too_long \
                        and not has_prompt_bleed and not is_multiline:
                    return [translated]
            except Exception:
                pass

        # FIX 3: Use has_code_switching's en_words logic to find ONLY English terms,
        # not any ASCII sequence. We reuse the same vn_common guard to avoid
        # falsely bracketing romanized Vietnamese like 'trong', 'tren', 'cach'.
        vn_common = {
            "di","an","va","co","la","gi","cua","nay","tai","cho","trong","duoc",
            "lam","ra","den","theo","nhu","nhung","mot","hai","ba","bon","nam",
            "bay","tam","chin","muoi","ong","anh","chi","em","con","me","cha","bo",
            "da","dang","se","moi","xong","vua","khi","sau","ngu","dau","bung",
            "ho","sot","can","benh","thuoc","kham","vien","bac","si","gio","ngay",
            "tuan","thang","sang","trua","chieu","toi","phong","nha","xe","duong",
            "pho","quan","tinh","viet","nam","hoc","sinh","truong","lop","mon",
            "he","thong","may","mang","phan","mem","ung","dung","cai","chay",
            "lenh","dong","lap","trinh","bien","ket","noi","cau","hinh","thu",
            "tren","cach","sau","truoc","vao","ra","len","xuong","qua","lai",
        }
        # Extract tokens that are all-ASCII-alpha (candidate English words)
        candidate_terms = re.findall(r"\b[a-zA-Z]{2,}\b", query)
        english_terms = [t for t in candidate_terms if t.lower() not in vn_common]

        if english_terms:
            bracketed = query
            for term in english_terms[:2]:
                bracketed = bracketed.replace(term, f"({term})")
            return [bracketed]

        return []

    def _filter_relevance(self, original: str, queries: List[str]) -> List[str]:
        return [q for q in queries if self._is_relevant(original, q)]

    def _is_relevant(self, original: str, candidate: str) -> bool:
        if self.embedding_fn:
            try:
                score = self.embedding_fn(original, candidate)
                return score > 0.7
            except Exception:
                pass

        # Fallback: lexical overlap on ASCII tokens only (diacritic-safe)
        o_tokens = set(re.findall(r"[a-z]+", original.lower()))
        c_tokens = set(re.findall(r"[a-z]+", candidate.lower()))

        overlap = len(o_tokens & c_tokens)
        return overlap >= max(1, len(o_tokens) // 3)

    @staticmethod
    def _parse_answer_block(text: str) -> str:
        # Strict extraction: match only up to the FIRST </answer> closing tag.
        # This prevents bleed-through when the model keeps generating extra
        # "Câu hỏi gốc:" examples after the closing tag.
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: if the model forgot to close the tag, grab everything after
        # <answer> but stop at the first blank line or "Câu hỏi" continuation.
        open_match = re.search(r"<answer>(.*)", text, re.DOTALL)
        if open_match:
            raw = open_match.group(1)
            # Truncate at any sign the model looped back into the prompt template
            cutoff = re.search(
                r"\n\s*\n|Câu hỏi gốc|<answer|Câu hỏi của câu", raw
            )
            chunk = raw[: cutoff.start()].strip() if cutoff else raw.strip()
            if chunk:
                return chunk

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