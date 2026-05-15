# BÁO CÁO PHẢN BIỆN LUẬN VĂN

**Tên đề tài:** Nghiên cứu và cải tiến công cụ tự động sinh test Qodo Cover  
**Tác giả:** Nguyễn Hoàng Quân  
**Người phản biện:** Nguyễn Thành Phát  
**Thời gian:** 2026  

---

## 1. TỔNG QUAN

Đây là một luận văn công nghệ phần mềm tốt với xác thực thực nghiệm xuất sắc. Đóng góp cốt lõi - cơ chế tự phục hồi cho các bài kiểm thử được tạo bởi LLM - được động lực hóa rõ ràng, đúng đắn về mặt kỹ thuật, và được chứng minh thuyết phục qua 10 dự án mã nguồn mở.

**Đánh giá tổng thể:** ⭐⭐⭐⭐☆ (4/5) - Có thể đạt ⭐⭐⭐⭐⭐ sau khi sửa vấn đề trùng lặp C2-C3

---

## 2. ĐÁNH GIÁ TỪNG CHƯƠNG

| Chương | Đánh giá | Điểm mạnh chính | Vấn đề chính |
|--------|----------|-----------------|--------------|
| **C1: Đặt vấn đề** | ⭐⭐⭐⭐☆ | Phát biểu bài toán rõ ràng | Nội dung trùng lặp nhẹ |
| **C2: Cơ sở lý thuyết** | ⭐⭐⭐⭐⭐ | Nền tảng lý thuyết toàn diện | Trùng lặp với C3 |
| **C3: Giải pháp** | ⭐⭐⭐☆☆ | Chi tiết kỹ thuật xuất sắc | **TRÙNG LẶP NỘI DUNG** |
| **C4: Thực nghiệm** | ⭐⭐⭐⭐⭐ | Xác thực xuất sắc | Cần chỉnh sửa nhỏ |
| **Kết luận** | ⭐⭐⭐⭐☆ | Thành thật, hướng tới tương lai | Thêm chi tiết cụ thể |

---

## 3. VẤN ĐỀ NGHIÊM TRỌNG (PHẢI SỬA)

### 🔴 NGHIÊM TRỌNG: Trùng lặp nội dung Chương 2 và Chương 3

**Vấn đề:** File `c3_analysis_and_evaluation.tex` chứa ~88 dòng gần như **GIỐNG HỆT** file `c2_analysis_and_evaluation.tex`

**Nội dung trùng lặp bao gồm:**
- Giới thiệu công cụ Qodo Cover (dòng 1-6)
- Mô tả quy trình 3 giai đoạn (dòng 27-61)
- Phần hạn chế với xử lý theo lô (dòng 59-88)

**Ảnh hưởng:**
- Người đọc gặp lại nội dung hai lần
- Gợi ý chỉnh sửa/planning kém
- Lãng phí không gian luận văn

**Cách sửa:**
1. **XÓA** nội dung trùng lặp khỏi `c3_analysis_and_evaluation.tex`
2. **THAY THẾ** bằng đoạn tóm tắt ngắn tham chiếu C2:
   ```latex
   \subsection{Tóm tắt hạn chế Qodo Cover}
   Như đã phân tích chi tiết ở Mục \ref{sec:problems_of_qodo}, 
   Qodo Cover gặp ba vấn đề chính: (1) cơ chế xử lý theo lô, 
   (2) lãng phí tài nguyên, và (3) dừng phát triển CLI.
   ```

**Thời gian sửa:** ~2 giờ

---

## 4. CÁC VẤN ĐỀ KHÁC

### 🟡 Trung bình: Trùng lặp nội dung khác

**Vấn đề:**
1. Ngày ngừng phát triển Qodo Cover (15/06/2025) xuất hiện 3+ lần
2. Giải thích hạn chế CLI được lặp lại
3. Phê phán xử lý theo lô xuất hiện hai lần

**Cách sửa:** Tập trung vào C1 hoặc C2, tham chiếu ở nơi khác

### 🟡 Code bị comment cần dọn dẹp

**Các file có nội dung comment nhiều:**
- c3_chapter.tex (30 dòng comment template)
- c2_ai_agent.tex (15 dòng mô tả LLM)
- c3_solution.tex (code hình ảnh bị comment)
- c4_chapter.tex (30 dòng comment template)

**Cách sửa:** Xóa trước khi nộp

---

## 5. ĐIỂM MẠNH (GIỮ NGUYÊN)

### 1. Thiết kế thực nghiệm (C4) ⭐⭐⭐⭐⭐
- **10 dự án đa dạng:** Flask, Django REST, HanLP, Gymnasium, LocalStack, Locust, Pipenv, Scrapy, tqdm, OpenAI-Python
- **3 LLM được kiểm thử:** DeepSeek-V3.2, Qwen3 Coder, GPT-OSS-120B
- **Chỉ số rõ ràng:** Độ bao phủ dòng lệnh + độ bao phủ nhánh
- **Kết quả định lượng:** Tăng 5-36% dòng lệnh, 10-47% nhánh

### 2. Chi tiết kỹ thuật (C3) ⭐⭐⭐⭐⭐
- **Hai trigger self-healing:** Lỗi thực thi HOẶC độ bao phủ không tăng
- **Kiến trúc prompt 3 lớp:** System + Ngữ cảnh động + Neo thực thi
- **6 ví dụ code cụ thể:** So sánh trước/sau
- **Xử lý edge case:** Phát hiện flaky test, lọc test vô giá trị

### 3. Nền tảng lý thuyết (C2) ⭐⭐⭐⭐⭐
- **Cơ chế pytest:** Discovery, Fixture, Assertion
- **Thành phần AI Agent:** Agent, Planning, Memory, Tools
- **Sự tiến hóa self-healing:** Infrastructure → UI → Code level
- **Vòng lặp MAPE-K:** Giải thích chi tiết

### 4. Đánh giá thành thật (C4/Kết luận) ⭐⭐⭐⭐⭐
- Thừa nhận các trường hợp regression
- Thảo luận vấn đề non-determinism
- Liệt kê 4 hạn chế cụ thể
- Hướng phát triển tương lai rõ ràng

---

## 6. SO SÁNH VỚI LUẬN VĂN KHÁC

| Tiêu chí | Quân | Minh | Phát (tôi) |
|----------|------|------|-------------|
| **Lĩnh vực** | Kiểm thử phần mềm | Nền tảng EdTech | Phát hiện nguyên nhân |
| **Ngôn ngữ** | Tiếng Việt | Tiếng Anh | Tiếng Anh |
| **Cấu trúc** | Module nhất (34 file tex) | Module | Trung bình |
| **Hình ảnh** | 17 sơ đồ workflow | Biểu đồ khảo sát | TikZ diagrams |
| **Trích dẫn** | 70+ (LLM/kiểm thử) | 25 (giáo dục/AI) | 32 (causal ML) |
| **Dữ liệu** | 10 dự án mã nguồn mở | Khảo sát 48 giáo viên | 47K tập dữ liệu ML |
| **Vấn đề chính** | Trùng lặp C2-C3 | Dữ liệu C4 thiếu | Chưa hoàn thành C5 |

**Điểm độc đáo của luận văn Quân:**
- Cấu trúc module nhất
- Xác thực thực nghiệm mạnh nhất
- Chi tiết kỹ thuật code xuất sắc
- Thành thật về hạn chế

---

## 7. KHUYẾN NGHỊ CHO TÁC GIẢ

### Hành động ngay (Trước bảo vệ):

**Tuần 1:**
1. [ ] **NGHIÊM TRỌNG:** Sửa trùng lặp C2-C3 (2 giờ)
2. [ ] Xóa tất cả text comment template (1 giờ)
3. [ ] Kiểm tra/sửa các tham chiếu hình ảnh (30 phút)
4. [ ] Sửa các entry trùng lặp trong .bib (30 phút)

**Tuần 2:**
5. [ ] Thêm pseudocode thuật toán cho vòng lặp self-healing (1 giờ)
6. [ ] Tách các câu dài trong C1 (1 giờ)
7. [ ] Tập hợp các giải thích regression (30 phút)
8. [ ] Đọc lại toàn bộ (2 giờ)

### Cải thiện tùy chọn:
- Thêm biểu đồ cột so sánh 3 LLM
- Thêm ảnh chụp Extension
- Thêm phân tích chi phí (token API)

---

## 8. NHỮNG GÌ TÔI HỌC ĐƯỢC (CHO LUẬN VĂN CỦA TÔI)

### Kỹ thuật:
1. **So sánh code trước/sau** - rất hiệu quả để chỉ ra cải tiến
2. **Xác thực đa mô hình** - tăng độ tin cậy đáng kể
3. **Thiết kế trigger kép** - cách tiếp cận tinh vi cho edge case
4. **Kiến trúc prompt engineering** - phân tích hệ thống các lớp

### Viết luận văn:
1. **Cấu trúc module** - dễ bảo trì hơn file đơn
2. **Thảo luận hạn chế thành thật** - tăng độ tin cậy
3. **Ví dụ code cụ thể** - làm khái niệm trừu tượng trở nên hữu hình
4. **Tập dữ liệu thực tế** - 10 dự án production rất thuyết phục

### Tổ chức:
1. **Chương nền tảng chi tiết** - C2 là tài liệu tham khảo toàn diện
2. **Chương thí nghiệm riêng** - C4 đứng độc lập như xác thực
3. **Lộ trình chương rõ ràng** - giúp người đọc định hướng

---

## 9. KẾT LUẬN CUỐI CÙNG

**Sẵn sàng bảo vệ?** Gần như - cần sửa trùng lặp C2-C3 trước

**Thời gian cần để sẵn sàng:** 5-6 giờ

**Điểm mạnh nhất:** Xác thực thực nghiệm (C4)

**Điểm yếu nhất:** Tổ chức nội dung (trùng lặp C2-C3)

**Khuyến nghị:** Sửa vấn đề trùng lặp, sau đó đây là luận văn ⭐⭐⭐⭐⭐

---

**Chữ ký người phản biện:** Nguyễn Thành Phát  
**Ngày:** 2026  
**Liên hệ:** Sẵn sàng trả lời câu hỏi bổ sung

---

## PHỤ LỤC: CÁC FILE PHẢN BIỆN ĐÃ TẠO

| File | Vị trí | Nội dung |
|------|--------|---------|
| Phản biện C1 | `chapters/c1/REVIEW.md` | Bối cảnh, bài toán, mục tiêu |
| Phản biện C2 | `chapters/c2/REVIEW.md` | Nền tảng, pytest, AI Agent |
| Phản biện C3 | `chapters/c3/REVIEW.md` | **Ghi chú trùng lặp C2-C3** |
| Phản biện C4 | `chapters/c4/REVIEW.md` | Kết quả thực nghiệm |
| Phản biện Kết luận | `chapters/REVIEW-CONCLUSION.md` | Tóm tắt & tương lai |
| Phản biện Tổng | `MASTER-REVIEW.md` | Đánh giá toàn diện |

*Tất cả file phản biện chi tiết đã được lưu trong thư mục luận văn.*
