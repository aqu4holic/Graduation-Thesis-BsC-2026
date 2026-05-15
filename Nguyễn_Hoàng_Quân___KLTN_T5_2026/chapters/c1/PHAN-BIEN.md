# Phản biện Chương 1: Đặt vấn đề

**Người phản biện:** Nguyễn Thành Phát  
**Ngày:** 2026  
**Các file đã xem xét:** c1_introduction.tex, c1_context.tex, c1_problem.tex, c1_purpose.tex

---

## 1. Đánh giá tổng thể

**Mức độ mạnh:** ⭐⭐⭐⭐☆ (4/5)

Chương 1 trình bày phát biểu bài toán rõ ràng với luồng logic tốt từ bối cảnh → vấn đề → cách tiếp cận giải pháp. Văn phong kỹ thuật tốt bằng tiếng Việt.

---

## 2. Phân tích chi tiết từng phần

### 2.1 c1_context.tex - Giới thiệu về bài toán

**Điểm mạnh:**
- ✅ Giới thiệu rõ ràng về bối cảnh CI/CD và tầm quan trọng của unit testing
- ✅ Sử dụng trích dẫn tốt (`\cite{unittesing, unittesting1}`)
- ✅ Giới thiệu công cụ Qodo Cover phù hợp
- ✅ Giải thích rõ vấn đề cốt lõi: tối ưu hóa cơ chế tự phục hồi cho các bài kiểm thử do LLM tạo ra

**Gợi ý cải thiện:**
1. **Thêm mở đầu thu hút:** Cân nhắc bắt đầu bằng ví dụ cụ thể hoặc thống kê về tỷ lệ thất bại khi tạo test để thu hút sự chú ý
2. **Làm rõ dòng thời gian:** Ngày "15/06/2025" xuất hiện sau trong problem.tex là ngày ngừng phát triển Qodo Cover - cần nhất quán nếu tham chiếu ở đây
3. **Mở rộng từ viết tắt:** Lần đầu nhắc đến CI/CD cần viết đầy đủ "Continuous Integration/Continuous Deployment"

**Chất lượng viết:** Tiếng Việt kỹ thuật tốt, giọng văn trang trọng phù hợp

---

### 2.2 c1_problem.tex - Những thách thức lớn trong bài toán

**Điểm mạnh:**
- ✅ Cấu trúc rõ ràng với 3 thách thức chính:
  1. Hạn chế CLI (không tích hợp IDE)
  2. Lãng phí tài nguyên từ việc loại bỏ test
  3. Sự không ổn định của LLM với context window lớn
- ✅ Phân tích kỹ thuật tốt về lý do xử lý theo lô thất bại
- ✅ Trích dẫn tài liệu liên quan (ảo giác, vấn đề context window)
- ✅ Thuật ngữ kỹ thuật cụ thể: "không gian tên", "ảo giác"

**Vấn đề quan trọng:**
1. **Trùng lặp:** Ngày ngừng phát triển "15/06/2025" và hạn chế CLI xuất hiện ở đây VÀ trong c3_analysis_and_evaluation.tex (văn bản tương tự). Cần xác định vị trí phù hợp nhất.

**Gợi ý:**
1. **Lượng hóa tác động:** Thêm số liệu thống kê nếu có (ví dụ: "40% các test được tạo bị lỗi do lỗi cú pháp")
2. **Tham chiếu hình vẽ:** Văn bản tham chiếu Hình \ref{fig:batch_processing} nhưng nó nằm trong c3_analysis_and_evaluation.tex - đảm bảo các hình theo thứ tự logic
3. **Đơn giản hóa cấu trúc câu:** Một số câu rất dài (ví dụ: dòng 3 có 5+ mệnh đề). Tách thành câu ngắn hơn để dễ đọc.

**Ví dụ câu phức tạp (dòng 3):**
> "Thách thức đầu tiên và rõ nét nhất xuất phát từ rào cản nền tảng của các công cụ tiền nhiệm..."

Có thể tách thành 2-3 câu.

---

### 2.3 c1_purpose.tex - Hướng tiếp cận và đóng góp

**Điểm mạnh:**
- ✅ Tham chiếu sơ đồ hệ thống rõ ràng (Hình \ref{fig:intro})
- ✅ Danh sách đóng góp có cấu trúc tốt (4 bullet points)
- ✅ Đề cập kết quả định lượng: "5-36% độ bao phủ dòng lệnh, 10-47% độ bao phủ nhánh"
- ✅ Giải thích rõ ràng về vòng lặp phản hồi self-healing
- ✅ Lộ trình chương cuối rõ ràng

**Vấn đề quan trọng:**
1. **Sai lệch:** Dòng 10 nói Qodo Cover ngừng phát triển "15/06/2025" nhưng đã xuất hiện trong c1_problem.tex
2. **Nhãn hình:** Dòng 6 tham chiếu `intro.png` nhưng chú thích chung chung ("Sơ đồ hệ thống Qodo Plus") - nên mô tả cụ thể hơn

**Gợi ý:**
1. **Chuyển lộ trình sang introduction.tex:** Phần dàn bài chương ở cuối (dòng 22-28) có thể trùng lặp với nội dung trong c1_introduction.tex - kiểm tra và tập hợp lại
2. **Thêm số liệu cụ thể:** Đề cập "max-fix-attempts" - cần chỉ rõ giá trị thực tế (ví dụ: 3 lần?)
3. **Củng cố các tuyên bố đóng góp:**
   - "Xây dựng nền tảng sửa lỗi cục bộ tự động" → Tốt
   - "Cải thiện tính ổn định" → Đo lường như thế nào?
   - "Tối ưu hóa tài nguyên" → Tiết kiệm token cụ thể bao nhiêu?

**Vấn đề trích dẫn:** Dòng 10 có `\cite{qodo}` - cần kiểm tra trong references.bib

---

## 3. Các vấn đề xuyên suốt

### 3.1 Trùng lặp nội dung
Nội dung sau xuất hiện trong nhiều file:
- Ngày ngừng phát triển Qodo Cover (15/06/2025)
- Giải thích hạn chế CLI
- Mô tả vấn đề xử lý theo lô

**Khuyến nghị:** Giữ trong c1_problem.tex (vị trí logic nhất) và tham chiếu ngắn gọn ở nơi khác.

### 3.2 Tham chiếu hình ảnh
- \ref{fig:intro} - trong c1_purpose.tex (phù hợp)
- \ref{fig:batch_processing} - tham chiếu trong c1_problem.tex nhưng định nghĩa trong c3_analysis_and_evaluation.tex

**Sửa:** Hoặc di chuyển hình sang Chương 1, hoặc bỏ tham chiếu cho đến Chương 3.

### 3.3 Tính nhất quán trích dẫn
- `\cite{qodo}` xuất hiện hai lần - cần xác nhận trong file .bib
- `\cite{fan2023automatedrepairprogramslarge}` trong c1_purpose - trích dẫn gần đây tốt

---

## 4. KHUYẾN NGHỊ

### Sửa ngay:
1. [ ] Bỏ ngày ngừng phát triển trùng lặp khỏi c1_purpose.tex
2. [ ] Sửa tham chiếu hình \ref{fig:batch_processing} trong c1_problem.tex
3. [ ] Xác nhận tất cả trích dẫn tồn tại trong references.bib

### Cải thiện:
1. [ ] Thêm thống kê thu hút vào đầu c1_context.tex
2. [ ] Tách các câu dài trong c1_problem.tex
3. [ ] Thêm số liệu tiết kiệm token/cost cụ thể vào danh sách đóng góp
4. [ ] Cân nhắc tập hợp c1_context.tex và c1_problem.tex (cả hai đều tương đối ngắn)

### Câu hỏi cho tác giả:
1. Ngày 15/06/2025 là đã xác nhận hay ước tính?
2. Giá trị max_fix_attempts thực tế là bao nhiêu?
3. Có số liệu định lượng nào về tỷ lệ thất bại tạo test không?

---

## 5. SO SÁNH VỚI LUẬN VĂN CỦA TÔI (PHÁT)

| Tiêu chí | Luận văn Quân | Luận văn Phát | Ghi chú |
|----------|---------------|---------------|---------|
| **Ngôn ngữ** | Tiếng Việt | Tiếng Anh | |
| **Cấu trúc** | Context→Vấn đề→Mục đích | Vấn đề→Động lực→Đóng góp | Cả hai đều logic |
| **Độ dài** | ~35 dòng qua 3 file | 214 dòng 1 file | Quân module hơn |
| **Định lượng** | Đề cập % độ bao phủ | Nhiều % chỉ số | Cả hai đều dựa trên dữ liệu |
| **Hình ảnh** | 1 hình | Nhiều hình | |
| **Trích dẫn** | ~10 trích dẫn | ~15 trích dẫn | Mật độ tốt |

**Tôi có thể học được:**
- Cách tiếp cận module file con sạch và dễ bảo trì
- Cấu trúc 3-thách thức rõ ràng trong phát biểu bài toán
- Sử dụng tốt thuật ngữ kỹ thuật tiếng Việt

---

## 6. KẾT LUẬN CUỐI CÙNG

Chương 1 thiết lập bài toán nghiên cứu hiệu quả. Các vấn đề chính là:
1. Trùng lặp nội dung qua các file
2. Một số câu cấu trúc phức tạp
3. Thiếu số liệu cụ thể trong tuyên bố đóng góp

**Thời gian sửa ước tính:** 1-2 giờ

**Ưu tiên sửa:** Trùng lặp nội dung, tham chiếu hình

**Đánh giá:** ⭐⭐⭐⭐☆ (4/5) - Cần sửa ít để đạt 5/5
