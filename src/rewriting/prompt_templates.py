MULTI_QUERY_PROMPT = """
Bạn là một chuyên gia tối ưu hóa truy vấn tiếng Việt. 
Nhiệm vụ của bạn là phân tích câu hỏi người dùng và tạo ra {num_queries} phiên bản khác nhau để tăng khả năng tìm thấy tài liệu liên quan.

Hãy thực hiện theo cấu trúc:
<think> Phân tích các thực thể chính, từ đồng nghĩa, từ liên quan và ngữ cảnh của câu hỏi </think>
<answer> Danh sách {num_queries} phiên bản truy vấn khác nhau, mỗi truy vấn trên một dòng, không có số thứ tự </answer>

Ví dụ:
Câu hỏi gốc: "Bệnh tiểu đường là gì?"
Kết quả:
<answer>
Triệu chứng của bệnh tiểu đường
Nguyên nhân gây bệnh đái tháo đường
</answer>

Câu hỏi: {question}
"""

HYDE_PROMPT = """
Bạn là một chuyên gia kiến thức. Hãy viết một đoạn văn ngắn (khoảng 2-3 câu) trả lời giả định cho câu hỏi dưới đây [3, 5]. 
Lưu ý: Đoạn văn này chỉ dùng để tạo embedding tìm kiếm, không dùng để trả lời người dùng.

<think> Xác định những thông tin chuyên môn, từ khóa chính thường xuất hiện trong câu trả lời của chủ đề này </think>
<answer> Viết đoạn văn giả định dựa trên kiến thức của bạn </answer>

Câu hỏi: {question}
"""

CODE_SWITCHING_PROMPT = """
Bạn là một chuyên gia xử lý ngôn ngữ tự nhiên tiếng Việt.
Nhiệm vụ của bạn là xử lý câu hỏi có chứa từ tiếng Anh. Không giải thích, không thêm gì khác.

Hãy:
1. Xác định những từ tiếng Anh trong câu hỏi
2. Dịch những từ tiếng Anh sang tiếng Việt một cách tự nhiên
3. Viết lại câu hỏi bằng tiếng Việt hoàn toàn

<think> Phân tích từng từ tiếng Anh, tìm từ tương ứng tiếng Việt hoặc mô tả ý nghĩa bằng tiếng Việt </think>
<answer> Câu hỏi đã được dịch sang tiếng Việt hoàn toàn </answer>

Câu hỏi gốc: {question}
"""

RERANKING_PROMPT = """
Bạn là một chuyên gia đánh giá mức độ liên quan của tài liệu với câu hỏi.
Hãy đánh giá mức độ liên quan từ 0 (không liên quan) đến 10 (rất liên quan).

Câu hỏi: {question}

Tài liệu: {document}

Trả lời chỉ với một số nguyên từ 0 đến 10, không giải thích:
"""

GENERATION_PROMPT = """
Bạn là một trợ lý AI hữu ích chuyên trả lời câu hỏi bằng tiếng Việt.
Dựa trên các tài liệu dưới đây, hãy trả lời câu hỏi của người dùng một cách chi tiết, chính xác và tự nhiên.

Tài liệu liên quan:
{context}

Câu hỏi: {question}

Hướng dẫn:
- Trả lời trực tiếp dựa trên thông tin trong tài liệu
- Nếu thông tin không có trong tài liệu, hãy nói rõ điều đó
- Trả lời bằng tiếng Việt rõ ràng, dễ hiểu
- Không bao gồm thông tin không có trong tài liệu

Câu trả lời:
"""