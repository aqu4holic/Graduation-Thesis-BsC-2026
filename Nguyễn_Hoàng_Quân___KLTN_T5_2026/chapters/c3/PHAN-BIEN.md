# Phản biện Chương 3: Giải pháp Qodo Plus

**Người phản biện:** Nguyễn Thành Phát  
**Ngày:** 2026  
**Các file đã xem xét:** c3_chapter.tex, c3_approach.tex, c3_solution.tex, c3_analysis_and_evaluation.tex

---

## 1. Đánh giá tổng thể

**Mức độ mạnh:** ⭐⭐⭐☆☆ (3/5)

Chương 3 có nội dung kỹ thuật xuất sắc nhưng bị ảnh hưởng nghiêm trọng bởi sự trùng lặp nội dung với Chương 2. Mô tả giải pháp chi tiết và cấu trúc tốt sau khi tách biệt nội dung duy nhất.

**VẤN ĐỀ NGHIÊM TRỌNG:** ~88 dòng trong c3_analysis_and_evaluation.tex gần như giống hệt c2_analysis_and_evaluation.tex

---

## 2. Phân tích chi tiết từng phần

### 2.1 c3_chapter.tex - Wrapper chương chính

**Đánh giá:** File wrapper chương chuẩn.

**Nội dung:**
- ✅ Comment template phù hợp từ hướng dẫn luận văn
- ✅ Tuyên bố mục tiêu chương rõ ràng
- ✅ Chuỗi `\input{}` đúng cho các file con

**Ghi chú:** Dòng 3-33 chứa text template comment. Cân nhắc xóa trước khi nộp.

---

### 2.2 c3_approach.tex - Hướng tiếp cận

**Điểm mạnh:**
- ✅ 4 nguyên tắc thiết kế Qodo Plus rõ ràng:
  1. Vòng lặp phản hồi cục bộ để sửa lỗi
  2. Dự phòng tổng hợp ngữ cảnh để đảm bảo ổn định
  3. Xử lý các trường hợp biên và kiểm soát tài nguyên
  4. Kế thừa và phát triển kiến trúc lõi
- ✅ Tiến triển logic từ vấn đề đến nguyên tắc giải pháp
- ✅ Liên kết tốt với phát biểu bài toán C1

**Vấn đề nhỏ:**
1. **Tham chiếu C1:** Dòng 2 nói "Từ những vấn đề được đề cập ở Mục \ref{sec:problems_of_qodo}" - xác nhận nhãn này tồn tại (có thể trong C2)
2. **Nguyên tắc 3 bị cắt:** Dòng 7 dường như kết thúc đột ngột: "Hệ thống cần được trang bị các cơ chế phòng vệ" - kiểm tra xem đã đầy đủ chưa

**Gợi ý:**
1. **Lượng hóa nguyên tắc 2:** Đề cập "max-fix-attempts" - cho biết số thực tế (3? 5?)
2. **Thêm sơ đồ:** Sơ đồ đơn giản hiển thị 4 nguyên tắc làm nền tảng có thể giúp
3. **Liên kết MAPE-K:** Đề cập ngắn gọn cách 4 nguyên tắc ánh xạ tới Monitor/Analyze/Plan/Execute

---

### 2.3 c3_solution.tex - Thiết kế giải pháp

**Điểm mạnh:**
- ✅ Cấu trúc 4 thành phần xuất sắc:
  1. Quy trình tự phục hồi (cục bộ + toàn cục)
  2. Cải thiện độ bao phủ qua kỹ thuật prompt
  3. Tối ưu hóa vận hành và xử lý ngoại lệ
  4. Tích hợp lõi và phát triển Extension
- ✅ Chi tiết kỹ thuật cụ thể xuyên suốt
- ✅ Sử dụng tốt danh sách code (cú pháp Python được highlight)
- ✅ Nhiều hình ảnh được tham chiếu phù hợp

**Điểm nổi bật kỹ thuật:**
- Self-healing cục bộ với hai trigger: lỗi thực thi VÀ độ bao phủ không tăng
- Kiến trúc prompt 3 lớp (tuyệt vời)
- Phát hiện flaky test và lọc test vô giá trị

**Vấn đề:**

1. **Vị trí hình:** Dòng 14-19 có hình bị comment:
   ```latex
   % \begin{figure}[htpb]
   %     \centering
   %     \includegraphics[width=0.9\textwidth]{figures/new_flows.png}
   % ...
   ```
   - Hoặc bỏ comment nếu hình tồn tại, hoặc xóa hoàn toàn

2. **Hình bị thiếu:** Dòng tham chiếu `\ref{fig:self_healing_pipeline}` nhưng bị comment

3. **Định nghĩa style code:** Dòng 20-45 định nghĩa style Python ở giữa nội dung. Cân nhắc:
   - Chuyển lên phần đầu thesis.tex
   - Hoặc dùng style nhất quán với các danh sách code khác

4. **Đoạn văn dài:** Dòng 86-94 (phần Self-healing prompt) rất dài

**Phản hồi kỹ thuật cụ thể:**

**Thiết kế trigger self-healing (xuất sắc):**
- Trigger thực thi: exit code ≠ 0
- Trigger độ bao phủ: exit code = 0 nhưng độ bao phủ không tăng
- Thiết kế hai trigger này tinh vi - cần làm nổi bật hơn

**Kiến trúc prompt 3 lớp:**
1. System prompt (vai trò + ràng buộc + heuristic logic)
2. Dynamic context injection (Jinja2 templates)
3. Task execution anchor

Đây là đóng góp mạnh - cân nhắc làm thành tiểu mục hoặc hình riêng

**Gợi ý cải thiện:**

1. **Thêm giả mã thuật toán:** Vòng lặp self-healing có thể được hưởng lợi từ môi trường Algorithm:
   ```latex
   \begin{algorithm}
   \caption{Vòng lặp Self-Healing Cục bộ}
   \begin{algorithmic}
   \FOR{each test case}
       \STATE Chạy pytest
       \IF{exit $\neq$ 0 OR độ bao phủ không đổi}
           \STATE Kích hoạt self-healing
           \FOR{i = 1 to max\_attempts}
               \STATE Gọi LLM với ngữ cảnh lỗi
               \STATE Cập nhật file test
               \STATE Chạy lại pytest
               \IF{pass VÀ độ bao phủ tăng}
                   \STATE Lưu test; \textbf{break}
               \ENDIF
           \ENDFOR
       \ENDIF
   \ENDFOR
   \end{algorithmic}
   \end{algorithm}
   ```

2. **Ví dụ template prompt:** Hiển thị text prompt thực tế (ẩn danh) để minh họa cách tiếp cận

3. **Ảnh chụp Extension:** Nếu có, ảnh chụp IDE extension sẽ củng cố phần này đáng kể

---

### 2.4 c3_analysis_and_evaluation.tex - Các vấn đề hiện tại của Qodo Cover

**VẤN ĐỀ NGHIÊM TRỌNG:** File này chứa ~88 dòng gần như giống hệt c2_analysis_and_evaluation.tex

**Nội dung trùng lặp bao gồm:**
- Mô tả Qodo Cover (dòng 1-18)
- Mô tả quy trình 3 giai đoạn (dòng 21-61)
- Phần hạn chế với xử lý theo lô (dòng 59-88)

**Ảnh hưởng:**
- Người đọc gặp lại nội dung hai lần
- Lãng phí không gian luận văn
- Gợi ý chỉnh sửa/planning kém

**Khuyến nghị:** 
1. **XÓA** nội dung trùng lặp khỏi c3_analysis_and_evaluation.tex
2. **THAY THẾ** bằng đoạn tóm tắt 5 dòng tham chiếu C2:
   ```latex
   \subsection{Tóm tắt hạn chế Qodo Cover}
   Như đã phân tích chi tiết ở Mục \ref{sec:problems_of_qodo}, 
   Qodo Cover gặp ba nhóm vấn đề chính: (1) cơ chế xử lý 
   theo lô gây mất ổn định, (2) lãng phí tài nguyên do 
   rollback tức thời, và (3) dừng phát triển kèm hạn chế CLI.
   ```
3. **Chỉ giữ nội dung duy nhất** trong c3_analysis_and_evaluation.tex

**Nội dung duy nhất trong c3_analysis_and_evaluation.tex:**
- Kiểm tra xem có phần nào KHÔNG trong phiên bản C2
- Nếu toàn bộ file là trùng lặp, cân nhắc xóa file hoàn toàn và bỏ `\input{}`

---

## 3. Các vấn đề xuyên suốt

### 3.1 Bản đồ trùng lặp C2-C3

| Nội dung | File C2 | File C3 | Hành động |
|----------|---------|---------|-----------|
| Giới thiệu Qodo Cover | c2_analysis (dòng 1-6) | c3_analysis (dòng 1-6) | Giữ C2, xóa C3 |
| Quy trình 3 giai đoạn | c2_analysis (dòng 27-61) | c3_analysis (dòng 21-56) | Giữ C2, xóa C3 |
| Hạn chế | c2_analysis (dòng 59-88) | c3_analysis (dòng 59-88) | Giữ C2, xóa C3 |
| Cách tiếp cận giải pháp | - | c3_approach | Giữ |
| Thiết kế kỹ thuật | - | c3_solution | Giữ |

### 3.2 Kiểm tra tham chiếu hình

| Hình | Tham chiếu trong | File tồn tại? | Trạng thái |
|------|------------------|---------------|------------|
| \ref{fig:self_healing_pipeline} | c3_solution | new_flows.png? | ⚠️ Kiểm tra |
| \ref{fig:Workflow_self_healing} | c3_solution | ✅ | Tốt |
| \ref{fig:simple_workflow_qodoplus} | c3_solution | ✅ | Tốt |
| \ref{fig:prompt} | c3_solution | ✅ | Tốt |
| \ref{lst:PRNG} | c3_solution | Code trong file | Tốt |
| \ref{lst:PRNG_plus} | c3_solution | Code trong file | Tốt |
| \ref{lst:sync_error} | c3_solution | Code trong file | Tốt |
| \ref{lst:sync_plus} | c3_solution | Code trong file | Tốt |
| \ref{lst:dictionary_error} | c3_solution | Code trong file | Tốt |
| \ref{lst:dictionary_plus} | c3_solution | Code trong file | Tốt |

### 3.3 Chất lượng danh sách code

**Điểm tốt:**
- Highlight cú pháp Python được định nghĩa nhất quán
- So sánh trước/sau rõ ràng
- Số dòng để tham chiếu dễ dàng
- Chú thích giải thích sự khác biệt chính

**Gợi ý:**
1. Cân nhắc làm nổi bật các dòng thay đổi bằng comment hoặc màu
2. Thêm chú thích ngắn giải thích tại sao thay đổi khắc phục vấn đề

---

## 4. KHUYẾN NGHỊ

### Sửa ngay (trước bảo vệ):
1. [ ] **NGHIÊM TRỌNG:** Xóa/tóm tắt nội dung trùng lặp trong c3_analysis_and_evaluation.tex
2. [ ] Bỏ comment hoặc xóa code hình bị comment (c3_solution.tex dòng 14-19)
3. [ ] Sửa hoặc xóa tham chiếu \ref{fig:self_healing_pipeline}
4. [ ] Xác minh vị trí định nghĩa style Python

### Cải thiện:
1. [ ] Thêm giả mã thuật toán cho vòng lặp self-healing
2. [ ] Hiển thị ví dụ prompt thực tế (ẩn danh)
3. [ ] Thêm ảnh chụp Extension nếu có
4. [ ] Tách đoạn văn dài c3_solution.tex dòng 86-94
5. [ ] Thêm ánh xạ MAPE-K cho 4 nguyên tắc

### Câu hỏi cho tác giả:
1. figures/new_flows.png có tồn tại không hay nên xóa tham chiếu?
2. Giá trị max_fix_attempts thực tế là bao nhiêu?
3. Các danh sách code trong c3_solution.tex là ví dụ thực tế được tạo hay đã được đơn giản hóa?

---

## 5. SO SÁNH VỚI LUẬN VĂN CỦA TÔI (PHÁT)

| Tiêu chí | Luận văn Quân | Luận văn Phát | Ghi chú |
|----------|---------------|---------------|---------|
| **Cấu trúc** | 4 file con | 1 file | Quân module hơn |
| **Độ dài** | ~250 dòng (có trùng) | 307 dòng | Tương đương |
| **Trùng lặp** | ~88 dòng trùng | Không | Vấn đề lớn cho Quân |
| **Hình ảnh** | 4 hình + 6 danh sách code | Nhiều TikZ | Chi tiết kỹ thuật |
| **Giải thuật** | Mô tả văn xuôi | Giả mã | Cả hai có thể thêm |
| **Ví dụ code** | 6 danh sách Python | Tối thiểu | Quân mạnh hơn |

**Tôi có thể học được:**
- So sánh code trước/sau xuất sắc
- Xác thực đa mô hình
- Thiết kế hai trigger (thực thi + độ bao phủ)
- Kiến trúc prompt engineering mạnh mẽ
- Tiếp cận thực tế với Extension

---

## 6. KẾT LUẬN CUỐI CÙNG

Chương 3 có nội dung kỹ thuật xuất sắc nhưng bị ảnh hưởng nghiêm trọng bởi trùng lặp C2-C3. Sau khi giải quyết, đây sẽ là chương mạnh.

**Chất lượng kỹ thuật:** ⭐⭐⭐⭐⭐ (5/5)
**Tổ chức:** ⭐⭐⭐☆☆ (3/5)
**Tính duy nhất:** ⭐⭐☆☆☆ (2/5) - do trùng lặp

**Tổng thể:** ⭐⭐⭐☆☆ (3/5) - Có thể đạt ⭐⭐⭐⭐⭐ sau khi sửa trùng lặp

**Thời gian sửa ước tính:** 2-3 giờ (chủ yếu giải quyết trùng lặp C2-C3)

**Ưu tiên:** CAO - Trùng lặp nội dung là vấn đề luận văn nghiêm trọng

---

## 7. ĐỀ XUẤT TÁI CẤU TRÚC C3

**Cấu trúc hiện tại:**
1. c3_chapter.tex (wrapper)
2. c3_approach.tex (4 nguyên tắc)
3. c3_solution.tex (thiết kế chi tiết)
4. c3_analysis_and_evaluation.tex (hầu hết trùng lặp)

**Cấu trúc đề xuất:**
1. c3_chapter.tex (wrapper)
2. c3_approach.tex (giữ - nội dung duy nhất)
3. c3_solution.tex (giữ - nội dung duy nhất)
4. c3_analysis_and_evaluation.tex → **THAY THẾ bằng tóm tắt 5 dòng tham chiếu C2**

Hoặc: Xóa hoàn toàn c3_analysis_and_evaluation.tex và chuyển nội dung duy nhất (nếu có) sang c3_solution.tex
