MULTI_QUERY_PROMPT = """
Bạn là một chuyên gia tối ưu hóa truy vấn tiếng Việt. 
Nhiệm vụ của bạn là phân tích câu hỏi người dùng và tạo ra 3-5 phiên bản khác nhau để tăng khả năng tìm thấy tài liệu liên quan [5, 8].

Hãy thực hiện theo cấu trúc:
<think> Phân tích các thực thể chính, từ đồng nghĩa và ngữ cảnh của câu hỏi </think>
<answer> Danh sách các truy vấn mở rộng, mỗi truy vấn trên một dòng </answer>

Câu hỏi: {question}
"""

HYDE_PROMPT = """
Bạn là một chuyên gia kiến thức. Hãy viết một đoạn văn ngắn (khoảng 2-3 câu) trả lời giả định cho câu hỏi dưới đây [3, 5]. 
Lưu ý: Đoạn văn này chỉ dùng để tạo embedding tìm kiếm, không dùng để trả lời người dùng.

<think> Xác định những thông tin chuyên môn thường xuất hiện trong câu trả lời của chủ đề này </think>
<answer> {question} </answer>
"""