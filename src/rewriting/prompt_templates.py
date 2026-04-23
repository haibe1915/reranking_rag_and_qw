MULTI_QUERY_PROMPT = """
Bạn là chuyên gia tối ưu truy vấn tìm kiếm.

Nhiệm vụ:
Tạo {num_queries} truy vấn khác nhau từ câu hỏi gốc.

QUY TẮC QUAN TRỌNG:

- Nếu câu hỏi chứa dấu hiệu cụ thể (triệu chứng, lỗi, hiện tượng cụ thể):
  → PHẢI giữ nguyên các từ khóa quan trọng, chỉ được paraphrase.

- Nếu câu hỏi mang tính khái niệm chung:
  → có thể mở rộng theo các khía cạnh (định nghĩa, nguyên nhân, ứng dụng).

- KHÔNG được làm mất thông tin quan trọng trong câu hỏi gốc.

FORMAT:
- Mỗi truy vấn trên 1 dòng
- Không đánh số, không giải thích

Ví dụ:

Câu hỏi gốc: Trẻ sốt, chướng bụng và ăn kém là dấu hiệu bệnh gì?
<answer>
Trẻ bị sốt, bụng phình to và biếng ăn là biểu hiện của bệnh gì
Sốt kèm chướng bụng và ăn kém ở trẻ cảnh báo bệnh gì
Triệu chứng sốt, bụng to và ăn uống kém ở trẻ là bệnh gì
</answer>

Câu hỏi gốc: {question}
<answer>"""


HYDE_PROMPT = """
Bạn là một chuyên gia kiến thức đa lĩnh vực (y tế, công nghệ, khoa học và đời sống). Hãy viết một đoạn văn ngắn (khoảng 2-3 câu) chứa những thông tin cốt lõi để trả lời trực tiếp cho câu hỏi dưới đây.
Đoạn văn này sẽ được dùng làm dữ liệu ảo để máy học tìm kiếm tài liệu tương đồng.

QUY TẮC BẮT BUỘC:
1. Sử dụng văn phong khách quan, học thuật giống như bách khoa toàn thư.
2. KHÔNG sử dụng đại từ nhân xưng (tôi, bạn).
3. KHÔNG có câu chào hỏi hay câu dẫn (VD: không viết "Câu trả lời là...").
4. Viết trực tiếp vào vấn đề ngay lập tức.

Câu hỏi: {question}
<answer>
"""

CODE_SWITCHING_PROMPT = """
Bạn là chuyên gia ngôn ngữ học tiếng Việt. Nhiệm vụ của bạn là chuẩn hóa câu hỏi có chứa từ tiếng Anh xen lẫn tiếng Việt (Code-Switching).

QUY TẮC BẮT BUỘC:
1. Dịch các từ tiếng Anh thông dụng sang tiếng Việt một cách tự nhiên nhất.
2. GIỮ NGUYÊN các từ là tên riêng, tên công nghệ, tên phần mềm (Ví dụ: Docker, Ubuntu, Python, API, AI, GPU, Covid-19).
3. Chỉ xuất ra MỘT câu duy nhất là kết quả đã chuẩn hóa.
4. TUYỆT ĐỐI KHÔNG giải thích, KHÔNG thêm tiền tố như "Dịch:", "Câu trả lời:". KHÔNG dùng dấu ngoặc đơn.
5. Sau thẻ </answer> KHÔNG được viết thêm bất kỳ nội dung nào. Dừng lại hoàn toàn.

Ví dụ:
Câu hỏi gốc: Machine learning là gì?
<answer>
Học máy là gì?
</answer>

Câu hỏi gốc: Cách setup Docker trên Ubuntu
<answer>
Cách cài đặt Docker trên Ubuntu
</answer>

Bây giờ hãy xử lý câu hỏi SAU ĐÂY (chỉ câu này, không tạo thêm ví dụ):
Câu hỏi gốc: {question}
<answer>
"""

RERANKING_PROMPT = """
Bạn là một chuyên gia đánh giá mức độ liên quan của tài liệu đa lĩnh vực.
Nhiệm vụ của bạn là chấm điểm mức độ tài liệu dưới đây có thể dùng để trả lời câu hỏi hay không.
Thang điểm từ 0 (Hoàn toàn không liên quan) đến 10 (Chứa câu trả lời hoàn hảo).

Câu hỏi: {question}

Tài liệu: {document}

QUY TẮC: Chỉ xuất ra MỘT SỐ NGUYÊN TỪ 0 ĐẾN 10. Tuyệt đối không giải thích, không viết thêm bất kỳ chữ nào khác.
Điểm số: """

GENERATION_PROMPT = """
Bạn là một trợ lý AI thông minh và đáng tin cậy. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng bằng tiếng Việt, DỰA HOÀN TOÀN VÀO CÁC TÀI LIỆU được cung cấp dưới đây.

TÀI LIỆU LIÊN QUAN:
{context}

CÂU HỎI CỦA NGƯỜI DÙNG: {question}

QUY TẮC BẮT BUỘC:
1. Chỉ sử dụng thông tin CÓ TRONG TÀI LIỆU LIÊN QUAN để trả lời. 
2. Tuyệt đối không tự suy diễn, không dùng kiến thức bên ngoài (không Hallucination).
3. Nếu tài liệu KHÔNG CHỨA đủ thông tin để trả lời, HÃY TRẢ LỜI ĐÚNG CÂU SAU: "Dựa trên các tài liệu được cung cấp, tôi không tìm thấy thông tin để trả lời câu hỏi này."
4. Trả lời trực tiếp, rõ ràng, mạch lạc bằng tiếng Việt.

Câu trả lời của bạn:
"""