import re
from typing import Tuple, List, Dict
from underthesea import sent_tokenize, word_tokenize


class CodeSwitchingProcessor:
    """Detect and handle Vietnamese-English code-switching"""
    
    def __init__(self):
        # Common English words and technical terms
        self.english_words = {
            "data", "model", "ai", "machine", "learning", "deep", "neural",
            "network", "code", "python", "java", "sql", "database", "api",
            "server", "client", "cloud", "docker", "kubernetes", "gpu",
            "cpu", "memory", "storage", "software", "hardware", "system",
            "application", "web", "mobile", "frontend", "backend", "framework",
            "library", "function", "variable", "class", "object", "method",
            "algorithm", "optimization", "performance", "security", "privacy"
        }
    
    def detect_code_switching(self, text: str) -> Tuple[bool, List[str]]:
        """
        Detect if text contains code-switching (VN-EN mix)
        Returns: (has_code_switching, list_of_english_words)
        """
        # Tokenize
        words = word_tokenize(text.lower())
        
        english_words_found = []
        for word in words:
            # Remove punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.english_words:
                english_words_found.append(word)
        
        has_switching = len(english_words_found) > 0
        return has_switching, english_words_found
    
    def get_english_segments(self, text: str) -> List[str]:
        """Extract English word/phrase segments from Vietnamese text"""
        # Match sequences of English-like characters
        pattern = r'\b[a-zA-Z]+(?:\s+[a-zA-Z]+)*\b'
        matches = re.findall(pattern, text)
        return matches
    
    def translate_english_to_vietnamese(self, word: str) -> str:
        """Simple translation dictionary for common technical terms"""
        translation_dict = {
            "data": "dữ liệu",
            "model": "mô hình",
            "ai": "trí tuệ nhân tạo",
            "machine": "máy",
            "learning": "học",
            "deep": "sâu",
            "neural": "thần kinh",
            "network": "mạng",
            "code": "mã",
            "python": "python",  # Keep as-is
            "database": "cơ sở dữ liệu",
            "api": "api",
            "server": "máy chủ",
            "client": "máy khách",
            "cloud": "đám mây",
            "gpu": "gpu",
            "cpu": "cpu",
            "software": "phần mềm",
            "hardware": "phần cứng",
            "system": "hệ thống",
            "algorithm": "thuật toán",
            "optimization": "tối ưu hóa",
            "performance": "hiệu suất"
        }
        
        lower_word = word.lower()
        return translation_dict.get(lower_word, word)
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize query with code-switching handling
        - Detect English segments
        - Translate to Vietnamese if possible
        - Return normalized query
        """
        # Detect code-switching
        has_switching, english_words = self.detect_code_switching(query)
        
        if not has_switching:
            return query
        
        # Get English segments
        english_segments = self.get_english_segments(query)
        
        normalized = query
        for segment in english_segments:
            translation = self.translate_english_to_vietnamese(segment)
            if translation != segment:
                # Replace English with Vietnamese, case-insensitive
                normalized = re.sub(
                    rf'\b{re.escape(segment)}\b',
                    translation,
                    normalized,
                    flags=re.IGNORECASE
                )
        
        return normalized
    
    def expand_code_switching_queries(self, query: str) -> List[str]:
        """
        Generate alternative queries for code-switching handling
        - Original query
        - Normalized query (English -> Vietnamese)
        - English-only segments as separate queries
        """
        variants = [query]
        
        # Add normalized query
        normalized = self.normalize_query(query)
        if normalized != query:
            variants.append(normalized)
        
        # Extract and add English segments as queries
        english_segments = self.get_english_segments(query)
        for segment in english_segments:
            variants.append(segment)
        
        return list(set(variants))  # Remove duplicates


class LanguageDetector:
    """Detect language of text"""
    
    @staticmethod
    def detect_language(text: str) -> str:
        """
        Simple language detection based on character patterns
        Returns: "vi", "en", or "mixed"
        """
        # Count Vietnamese-specific characters
        vietnamese_chars = len(re.findall(r'[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', text, re.IGNORECASE))
        
        # Count English-like characters
        english_words = len(re.findall(r'\b[a-z]+\b', text, re.IGNORECASE))
        
        total_words = len(text.split())
        
        if vietnamese_chars > total_words * 0.3:
            if english_words > total_words * 0.2:
                return "mixed"
            return "vi"
        else:
            return "en"


if __name__ == "__main__":
    processor = CodeSwitchingProcessor()
    
    # Test examples
    test_queries = [
        "Machine learning là gì?",
        "Hãy giải thích về neural network",
        "Cách tối ưu hóa database",
        "Python dùng để làm gì?"
    ]
    
    for query in test_queries:
        has_switching, english_words = processor.detect_code_switching(query)
        normalized = processor.normalize_query(query)
        variants = processor.expand_code_switching_queries(query)
        
        print(f"Original: {query}")
        print(f"Code-switching detected: {has_switching}")
        print(f"English words: {english_words}")
        print(f"Normalized: {normalized}")
        print(f"Variants: {variants}")
        print()