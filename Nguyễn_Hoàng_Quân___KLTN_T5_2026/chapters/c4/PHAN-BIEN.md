# Phản biện Chương 4: Thực nghiệm và đánh giá

**Người phản biện:** Nguyễn Thành Phát  
**Ngày:** 2026  
**Các file đã xem xét:** c4_chapter.tex, c4_dataset.tex, c4_test_script.tex, c4_test_result.tex, c4_analysis_and_discussion.tex

---

## 1. Đánh giá tổng thể

**Mức độ mạnh:** ⭐⭐⭐⭐⭐ (5/5)

Chương thí nghiệm xuất sắc. Lựa chọn tập dữ liệu mạnh (10 dự án mã nguồn mở đa dạng), kiểm thử toàn diện trên 3 LLM, chỉ số rõ ràng (dòng + nhánh), và phân tích kết quả chi tiết với ví dụ code cụ thể.

---

## 2. Phân tích chi tiết từng phần

### 2.1 c4_dataset.tex - Dữ liệu thực nghiệm

**Điểm mạnh:**
- ✅ Lựa chọn dự án xuất sắc: 10 dự án Python hàng đầu trên GitHub
- ✅ Phủ đa dạng miền:
  - Web/API: Flask, Django REST
  - AI/NLP: HanLP, Gymnasium, OpenAI-Python
  - DevOps: LocalStack, Locust, Pipenv
  - Dữ liệu/Tiện ích: Scrapy, tqdm
- ✅ Triển khai longtable cho khả năng tương thích nhiều trang
- ✅ Cột "Thách thức kiểm thử" chi tiết
- ✅ Phân tích nhóm 4 loại cấu trúc tốt

**Độ sâu kỹ thuật:** Xuất sắc. Vượt xa liệt kê đơn giản để giải thích TẠI SAO mỗi dự án có thách thức:
- Flask: Phức tạp cơ chế state machine session
- DRF: Serialization và nhầm lẫn kiểu dữ liệu
- Gymnasium: Cô lập trạng thái pseudo-random
- OpenAI-Python: Xung đột đồng bộ/bất đồng bộ
- LocalStack: Cấu trúc OOP và phân giải namespace
- Scrapy: Import module và vấn đề monkeypatch
- tqdm: Cutoff dữ liệu huấn luyện (K vs KB)

**Vấn đề nhỏ:**
1. **Định dạng bảng:** Môi trường `longtable` xuất sắc, nhưng:
   - `\midrule` sau mỗi hàng là quá mức - cân nhắc chỉ giữa các dự án
   - Chiều rộng cột có thể điều chỉnh (6cm cho thách thức là tốt)

2. **Trích dẫn xếp hạng GitHub:** Dòng 2 trích dẫn `\cite{gitstarranking}` - xác nhận tham chiếu này

3. **Bảng bị comment:** Dòng 59-99 có triển khai bảng thay thế bị comment toàn bộ. Xóa trước khi nộp.

**Gợi ý:**
1. **Thêm số liệu kích thước:** Cho mỗi dự án, thêm xấp xỉ:
   - Số dòng code
   - Số file kiểm thử
   - % độ bao phủ ban đầu (đã có trong bảng kết quả, nhưng có thể xem trước ở đây)

2. **Lý do chọn 10 dự án:** Thêm đoạn giải thích tại sao 10 dự án là đủ (về mặt thống kê, thực tế)

3. **Ghi chú hạt giải ngẫu nhiên:** Đối với phần PRNG Gymnasium, đề cập nếu thí nghiệm kiểm soát seed

---

### 2.2 c4_test_script.tex - Quy trình thực nghiệm

**Điểm mạnh:**
- ✅ Mô tả thiết lập thí nghiệm rõ ràng
- ✅ Thông số phần cứng được cung cấp (AMD Ryzen 7, 32GB RAM)
- ✅ Môi trường phần mềm được chỉ định (Python 3.11)
- ✅ Ba LLM đa dạng được kiểm thử:
  - DeepSeek-V3.2 (Non-thinking Mode) - thương mại
  - Qwen3 Coder 480B A35B Instruct - mã nguồn mở
  - GPT-OSS-120B - mã nguồn mở qua Fireworks AI
- ✅ Công thức độ bao phủ dòng lệnh và nhánh được định nghĩa

**Vấn đề quan trọng:**

1. **Câu không đầy đủ tại dòng 6:**
   > "Riêng 2 model Qwen3 Coder 480B A35B Instruct và GPT-OSS-120B được thử nghiệm thông qua nền tảng Fireworks AI."

   Câu dường như đầy đủ nhưng kiểm tra xem cần thêm ngữ cảnh không.

2. **Tham chiếu phương trình:** Dòng 11 nói `\ref{eq:branch_coverage}` nhưng phương trình ở dòng 12 - xác nhận tham chiếu hoạt động

**Gợi ý:**
1. **Thêm giá trị max_fix_attempts:** "Cấu hình các tham số hoàn toàn giống nhau, chỉ trừ tham số max_fix_attempts" - chỉ rõ giá trị thực tế được kiểm thử (ví dụ: 3, 5, 10)

2. **Thêm tính thống kê nghiêm ngặt:**
   - Chạy bao nhiêu lần cho mỗi dự án?
   - Có phương sai qua các lần chạy không (non-determinism được đề cập)?
   - Kiểm định ý nghĩa thống kê?

3. **Phân tích chi phí:** Dòng 4 đề cập "chi phí" (cost) - thêm bảng so sánh chi phí API nếu có

4. **Thêm so sánh baseline:** Làm rõ "Ban đầu" (Initial) nghĩa là gì - test hiện có? Không có test?

---

### 2.3 c4_test_result.tex - Kết quả thực nghiệm

**Điểm mạnh:**
- ✅ Kết quả toàn diện trên 3 mô hình
- ✅ 6 bảng tổng (dòng + nhánh cho mỗi mô hình)
- ✅ Cả 10 dự án được bao phủ
- ✅ Cột "Tăng" (Increase) hiển thị rõ ràng cải thiện
- ✅ Định dạng bảng nhất quán

**Chất lượng dữ liệu:**
- DeepSeek: Tăng độ bao phủ dòng 2-37%, nhánh 3-47%
- Qwen: Tăng độ bao phủ dòng 3-20%, nhánh 3-27%
- GPT-OSS: Tăng độ bao phủ dòng 3-20%, nhánh 2-31%

Cải thiện ấn tượng nhất:
- Localstack +36.86% dòng (DeepSeek)
- HanLP +30.83% dòng (DeepSeek)
- Localstack +47.14% nhánh (DeepSeek)

**Vấn đề nhỏ:**

1. **Không nhất quán chú thích:**
   - Bảng 1,3,5: "Bảng so sánh trung bình độ bao phủ dòng lệnh"
   - Bảng 2,4,6: "Bảng so sánh trung bình độ bao phủ nhánh"
   
   Cân nhắc thêm tên mô hình vào tất cả chú thích để dễ nhìn.

2. **Bất thường Tqdm:** Qwen độ bao phủ dòng cho Tqdm hiển thị 71.42% (Bảng 3) so với 87.15% DeepSeek (Bảng 1) - chênh lệch lớn đáng thảo luận

3. **Độ bao phủ thấp hơn Pipenv:** Tất cả mô hình hiển thị độ bao phủ thấp hơn cho Pipenv (69-74% dòng) - giải thích tại sao trong phần thảo luận

**Gợi ý:**
1. **Thêm bảng tóm tắt:** Tạo bảng meta hiển thị:
   | Mô hình | TB Dòng Δ | TB Nhánh Δ | Dự án tốt nhất | Dự án tệ nhất |

2. **Biểu diễn trực quan:** Cân nhắc thêm biểu đồ cột so sánh ba mô hình

3. **Ý nghĩa thống kê:** Thêm dấu sao (*) cho cải thiện có ý nghĩa thống kê

---

### 2.4 c4_analysis_and_discussion.tex - Phân tích và thảo luận kết quả

**Điểm mạnh:**
- ✅ Phân tích định lượng chi tiết
- ✅ Ba ví dụ code cụ thể (Gymnasium, OpenAI-Python, Scrapy)
- ✅ So sánh code trước/sau với giải thích
- ✅ Thảo luận thành thật về hạn chế (trường hợp regression)
- ✅ Phân tích hành vi từng mô hình
- ✅ Giải thích rõ TẠI SAO có cải thiện

**Chất lượng ví dụ code:**

1. **Gymnasium PRNG (xuất sắc):**
   - Hiển thị lỗi chia sẻ trạng thái
   - Sửa chữa rõ ràng với tạo đối tượng độc lập
   - Giải thích lỗi AssertionError

2. **OpenAI-Python async (xuất sắc):**
   - Hiển thị lỗi đồng bộ/bất đồng bộ
   - Giải pháp async generator
   - Giải thích kỹ thuật `__aiter__`

3. **Scrapy monkeypatch (xuất sắc):**
   - Hiển thị nhầm lẫn import module
   - Phân biệt setattr vs setitem
   - Kịch bản debug thực tế

**Quan sát quan trọng:**

1. **Phân tích bị comment:** Dòng 3-9 chứa phân tích chi tiết bị comment. Hoặc:
   - Bỏ comment và tích hợp
   - Xóa hoàn toàn
   - Hoặc chuyển sang phụ lục

2. **Trùng lặp:** Giải thích "thoái lui" (regression) xuất hiện nhiều lần (dòng 7, 123, 131) với từ ngữ tương tự. Tập hợp lại.

3. **Thảo luận độ bao phủ nhánh:** Giải thích tốt tại sao cải thiện nhánh > dòng (dòng 121-122)

**Gợi ý:**
1. **Thêm tóm tắt thống kê:**
   - "Qodo Plus đạt cải thiện trung bình X% trên Y dự án"
   - "Z% dự án cho thấy cải thiện >20%"

2. **Phân tích chi phí-lợi ích:** Thêm thảo luận về tiêu thụ token so với cải thiện độ bao phủ

3. **Trường hợp thất bại:** Thảo luận 1-2 trường hợp Qodo Plus không cải thiện (nếu có)

---

## 3. Các vấn đề xuyên suốt

### 3.1 Kiểm tra tham chiếu bảng

Tất cả bảng được gán nhãn đúng:
- \label{deepseek_line}, \label{deepseek_branch}
- \label{qwen_line}, \label{qwen_branch}
- \label{gpt_line}, \label{gpt_branch}

Tham chiếu chính xác trong phần phân tích.

### 3.2 Chất lượng danh sách code

Định nghĩa style Python nhất quán. Cả 6 danh sách code (3 cặp trước/sau) đều:
- Gán nhãn đúng
- Chú thích rõ ràng
- Highlight cú pháp phù hợp
- Giải thích tốt trong văn bản

### 3.3 Tham chiếu hình trong C4

| Hình | Trạng thái |
|------|------------|
| \ref{deepseek_line} v.v. | ✅ Cả 6 bảng |
| \ref{lst:PRNG} v.v. | ✅ Cả 6 danh sách code |

---

## 4. KHUYẾN NGHỊ

### Sửa ngay:
1. [ ] Xóa phân tích chi tiết bị comment (c4_analysis dòng 3-9)
2. [ ] Tập hợp các giải thích regression
3. [ ] Xóa bảng thay thế bị comment trong c4_dataset
4. [ ] Thêm \label{eq:branch_coverage} vào phương trình

### Cải thiện:
1. [ ] Thêm bảng thống kê tóm tắt trên 3 mô hình
2. [ ] Thêm phần đe dọa tính hợp lệ (threats to validity)
3. [ ] Thêm phân tích chi phí (token API đã chi)
4. [ ] Giải thích bất thường Tqdm và độ bao phủ thấp Pipenv
5. [ ] Thêm hình/biểu đồ trực quan

### Câu hỏi cho tác giả:
1. Chạy bao nhiêu lần cho mỗi dự án? Dữ liệu có lấy trung bình không?
2. Các giá trị max_fix_attempts là gì?
3. Có dự án nào Qodo Plus thực hiện kém hơn không?
4. Tổng chi phí API cho thí nghiệm?

---

## 5. SO SÁNH VỚI LUẬN VĂN CỦA TÔI (PHÁT)

| Tiêu chí | Luận văn Quân | Luận văn Phát | Ghi chú |
|----------|---------------|---------------|---------|
| **Tập dữ liệu** | 10 dự án mã nguồn mở | 47K tập tổng hợp | Khác miền |
| **Mô hình** | 3 LLM | 1 kiến trúc | Quân đa dạng hơn |
| **Chỉ số** | Dòng + nhánh | Cân bằng | Phù hợp miền |
| **Ví dụ code** | 6 ví dụ thực | Tối thiểu | Quân mạnh hơn |
| **Bảng** | 6 bảng dữ liệu | Nhiều bảng kết quả | Tương đương |
| **Độ sâu phân tích** | Từng dự án + từng mô hình | Phân tích ablation | Cả hai đều kỹ lưỡng |

**Tôi có thể học được:**
- Kiểm thử dự án mã nguồn mở thực tế
- Kỹ thuật so sánh code trước/sau
- Tiếp cận xác thực đa mô hình
- Giải thích kỹ thuật cụ thể cho cải thiện
- Thảo luận thành thật về trường hợp regression

---

## 6. KẾT LUẬN CUỐI CÙNG

Chương 4 là chương mạnh nhất trong luận văn. Thiết kế thí nghiệm xuất sắc, kết quả toàn diện, phân tích trung thực.

**Tính nghiêm ngặt kỹ thuật:** ⭐⭐⭐⭐⭐
**Trình bày:** ⭐⭐⭐⭐⭐
**Độ sâu phân tích:** ⭐⭐⭐⭐⭐

**Tổng thể:** ⭐⭐⭐⭐⭐ (5/5)

**Cần chỉnh sửa nhỏ:**
- Xóa các phần bị comment
- Tập hợp giải thích trùng lặp
- Thêm đe dọa tính hợp lệ

**Thời gian sửa ước tính:** 1 giờ

---

## 7. TÓM TẮT CÁC ĐÓNG GÓP CHÍNH

Từ Chương 4, các đóng góp có thể chứng minh bao gồm:

1. **Cải thiện định lượng:** Tăng 5-36% dòng, 10-47% nhánh
2. **Xác thực đa mô hình:** Hoạt động trên DeepSeek, Qwen, và GPT-OSS
3. **Kiểm thử thực tế:** 10 dự án production-grade
4. **Độ sâu kỹ thuật:** Ví dụ code trước/sau cụ thể chỉ TẠI SAO hoạt động
5. **Đánh giá trung thực:** Thừa nhận trường hợp regression và hạn chế

Các kết quả này hỗ trợ mạnh mẽ các tuyên bố luận văn và chứng minh tính thực tiễn.
