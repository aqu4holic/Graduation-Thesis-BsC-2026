# Phản biện Chương 4: Kết quả và Đánh giá

**Người phản biện:** Nguyễn Thành Phát  
**Ngày:** 2026  
**File đã xem xét:** c4/c4_chapter.tex (233 dòng)

---

## 1. Đánh giá tổng thể

**Mức độ mạnh:** ⭐⭐⭐☆☆ (3/5)

Cấu trúc tốt bao phủ kiểm thử phần mềm, đánh giá AI, và kiểm thử người dùng - nhưng **có vấn đề nghiêm trọng với dữ liệu placeholder**. Nhiều bảng hiển thị "TBD" (chưa xác định) hoặc "Không đo trong lần chạy này". Đây là chương yếu nhất và cần chú ý khẩn cấp trước bảo vệ.

---

## 2. Phân tích chi tiết từng phần

### 2.1 Kết quả Kiểm thử Phần mềm (Dòng 9-81)

**Cấu trúc Kim tự tháp Kiểm thử (Tốt):**
- Nền tảng: Unit test (nhanh, cô lập)
- Giữa: Integration test (xác thực ranh giới)
- Trên cùng: E2E test (hành trình người dùng)

**Kết quả Độ bao phủ (Bảng \ref{tab:coverage_results}):**

| Hệ thống con | Độ bao phủ Dòng | Độ bao phủ Nhánh |
|--------------|-----------------|------------------|
| Product Backend | 86.56% | 69.04% |
| AI Module | 89.12% | 73.11% |
| Web Frontend | 75.91% | 63.71% |

**Phân tích:**
- ✅ Backend và AI có độ bao phủ mạnh (86-89% dòng, 69-73% nhánh)
- ⚠️ Độ bao phủ nhánh Frontend 63.71% thấp hơn
- ✅ Thảo luận thành thật về các nhánh không được bao phủ (dòng 49-50)

**Kết quả E2E (Bảng \ref{tab:e2e_validation_matrix}) - VẤN ĐỀ NGHIÊM TRỌNG:**

| UC | Kịch bản | Kết quả |
|----|----------|---------|
| UC-02 | Chuyển hướng đến cấp học | **Thất bại (Timeout)** |
| UC-02 | Lưu chọn sách giáo khoa | **Bị chặn** |
| UC-03 | Tạo bài giảng mới | **Bị chặn** |
| UC-03 | Hiển thị lỗi xung đột | **Bị chặn** |
| UC-04 | Thu thập đầu vào thiếu | **Bị chặn** |
| UC-04 | Báo cáo lỗi AI gracefully | **Bị chặn** |
| UC-05 | Điều hướng scenes | **Thiếu phủ sóng** |

**Vấn đề lớn:**
- Test đầu tiên (UC-02 timeout) chặn tất cả test tiếp theo
- Báo cáo thành thật (dòng 78-80) nhưng nâng lo ngại về độ ổn định hệ thống
- "Hệ thống hiện thiếu tính nhất quán pass đáng tin cậy"

**Đề xuất:**
Phần này cần hoặc:
1. Sửa vấn đề cơ sở hạ tầng và chạy lại test với kết quả đạt
2. Hoặc định khung tích cực hơn: "cơ sở hạ tầng kiểm thử E2E đã thiết lập, các lần chạy ban đầu tiết lộ vấn đề ổn định môi trường đang được giải quyết"

---

### 2.2 Kết quả Đánh giá AI (Dòng 82-233)

#### 2.2.1 Quy trình và Chỉ số (Tốt)

**Khung LLM-as-Judge:**
- Thư viện OpenEvals cho các hàm đánh giá
- LangSmith để thực thi thí nghiệm
- Chuẩn hóa đầu vào (các trường có giới hạn, an toàn)

**Chỉ số RAG (Tốt):**
- Context Precision: R_r / R_t (đã truy xuất với marker / tổng truy xuất)
- Context Recall: M_m / M_t (marker khớp / tổng marker)
- Hit Rate, MRR, Các phân vị độ trễ

**Phương trình:**
```latex
Context Precision = R_r / R_t
Context Recall = M_m / M_t
```

#### 2.2.2 So sánh Baseline AI (Bảng \ref{tab:ai_baseline_comparison}) - **NGHIÊM TRỌNG**

| Trục Chỉ số | Mô hình Vanilla | Mô hình Agentic | Chênh lệch |
|-------------|-----------------|-----------------|------------|
| Reply helpfulness | **TBD-LANGSMITH-VANILLA-HELPFULNESS** | **TBD-LANGSMITH-AGENTIC-HELPFULNESS** | **TBD** |
| Reference alignment | **TBD-LANGSMITH-VANILLA-REFALIGN** | **TBD-LANGSMITH-AGENTIC-REFALIGN** | **TBD** |
| Story framing | **TBD-LANGSMITH-VANILLA-STORY** | **TBD-LANGSMITH-AGENTIC-STORY** | **TBD** |
| Scene adherence | N/A | **TBD-LANGSMITH-AGENTIC-SCENEPLAN** | N/A |
| Storytelling arc | **TBD-LANGSMITH-VANILLA-ARC** | **TBD-LANGSMITH-AGENTIC-ARC** | **TBD** |
| Storytelling engagement | **TBD-LANGSMITH-VANILLA-ENGAGEMENT** | **TBD-LANGSMITH-AGENTIC-ENGAGEMENT** | **TBD** |

**🔴 VẤN ĐỀ NGHIÊM TRỌNG:**
Tất cả giá trị đều là placeholder "TBD"! Bảng này không thể nộp cho bảo vệ.

**Giải thích trong văn bản (dòng 196-197):**
"Cuối cùng, những cải thiện định lượng này chứng minh kỹ thuật kiến trúc đã được đền đáp."

**Nhưng không có cải thiện định lượng nào được hiển thị!**

#### 2.2.3 Chỉ số Đánh giá RAG (Bảng \ref{tab:rag_evaluation_metrics}) - **NGHIÊM TRỌNG**

| Chỉ số | Giá trị |
|--------|---------|
| Context Precision | **Không đo trong lần chạy này (benchmark bị bỏ qua)** |
| Context Recall | **Không đo trong lần chạy này (benchmark bị bỏ qua)** |
| Hit Rate | **Không đo trong lần chạy này (benchmark bị bỏ qua)** |
| MRR | **Không đo trong lần chạy này (benchmark bị bỏ qua)** |
| Độ trễ (Mean, p50, p95, p99) | **Không đo** |

**Giải thích (dòng 203-204):**
"Hiện tại, việc thực thi benchmark này yêu cầu khóa provider bên ngoài hợp lệ, và lần chạy được tài liệu phản ánh trạng thái bỏ qua chủ ý đang chờ cấu hình môi trường production cuối cùng."

**🔴 NGHIÊM TRỌNG:** Benchmark RAG - một tuyên bố cốt lõi của luận văn - chưa được chạy.

---

## 3. TÓM TẮT CÁC VẤN ĐỀ NGHIÊM TRỌNG

### 🔴 Phải Sửa Trước Bảo vệ:

1. **Bảng Đánh giá AI (Dòng 172-193):**
   - 6 chỉ số × 2 mô hình = 12 giá trị "TBD"
   **Hành động:** Chạy đánh giá LangSmith và điền số thực

2. **Bảng Benchmark RAG (Dòng 206-231):**
   - 8 chỉ số = 8 giá trị "Không đo"
   **Hành động:** Thực thi benchmark RAG với API key hợp lệ

3. **Kết quả E2E (Dòng 53-76):**
   - 6/7 kịch bản thất bại hoặc bị chặn
   **Hành động:** Sửa cơ sở hạ tầng và chạy lại, hoặc định khung lại câu chuyện

### 🟡 Nên Sửa:

4. **Phân tích Độ bao phủ:**
   - Độ bao phủ nhánh Frontend 63.71% là thấp nhất
   - Thêm giải thích tại sao điều này có thể chấp nhận được

---

## 4. KHUYẾN NGHỊ

### Hành động Ngay (Trước Bảo vệ):

**Tuần 1:**
1. [ ] **NGHIÊM TRỌNG:** Chạy đánh giá LangSmith AI
   - Mô hình vanilla baseline
   - Mô hình agentic (hệ thống của bạn)
   - Tính toán các chênh lệch
   - Điền Bảng \ref{tab:ai_baseline_comparison}

2. [ ] **NGHIÊM TRỌNG:** Thực thi benchmark RAG
   - Lấy API key hợp lệ
   - Chạy tập 5 truy vấn
   - Điền Bảng \ref{tab:rag_evaluation_metrics}

3. [ ] **NGHIÊM TRỌNG:** Sửa hoặc định khung lại kết quả E2E
   - Lựa chọn A: Sửa cơ sở hạ tầng, chạy lại test, hiển thị kết quả đạt
   - Lựa chọn B: Trình bày là "cơ sở hạ tầng đã thiết lập, cải tiến ổn định đang tiến hành"

**Tuần 2:**
4. [ ] Thêm kết quả đánh giá người dùng định tính (được đề cập trong C1 nhưng không hiển thị)
5. [ ] Thêm phân tích độ trễ RAG (p95/p99 quan trọng cho UX)
6. [ ] Thêm so sánh: 86.56% độ bao phủ backend có tốt không? (so với tiêu chuẩn ngành)

### Nếu Không thể Lấy Dữ liệu Thực:

**Lựa chọn Hạt nhân:** Nếu đánh giá không thể hoàn thành:
1. Xóa các tuyên bố về cải thiện định lượng
2. Định khung C4 là "Khung đánh giá đã thiết lập"
3. Trình bày phương pháp luận như đóng góp
4. Chuyển kết quả chi tiết sang Hướng phát triển

**⚠️ Cảnh báo:** Làm giảm đáng kể luận văn. **Khuyến nghị mạnh mẽ** hoàn thành các đánh giá.

---

## 5. SO SÁNH VỚI LUẬN VĂN KHÁC

| Tiêu chí | Luận văn Minh | Luận văn Quân | Luận văn Phát (tôi) |
|----------|---------------|---------------|---------------------|
| **Trạng thái** | ⚠️ Dữ liệu placeholder | ✅ Kết quả hoàn chỉnh | ✅ Kết quả hoàn chỉnh |
| **Kiểm thử Phần mềm** | 86.56% độ bao phủ (thực) | Không có | Không có |
| **Đánh giá AI** | ❌ Placeholder TBD | ✅ 3 mô hình được kiểm thử | Tối thiểu |
| **Benchmark RAG** | ❌ Bị bỏ qua | Không có | Không có |
| **Kết quả E2E** | ❌ Hầu hết thất bại | Không có | Không có |
| **Định lượng** | ❌ Thiếu | ✅ Mạnh | ✅ Mạnh |

**Luận văn Quân mạnh hơn nhiều** - dữ liệu thực trên 10 dự án và 3 LLM.

**Luận văn Phát mạnh hơn** - tất cả các bảng được điền với kết quả thí nghiệm thực.

---

## 6. KẾT LUẬN CUỐI CÙNG

Chương 4 có **khoảng trống dữ liệu nghiêm trọng** ngăn bảo vệ.

**Đánh giá hiện tại:** ⭐⭐⭐☆☆ (3/5)

**Với Dữ liệu Thực:** ⭐⭐⭐⭐⭐ (5/5)

**Thời gian sửa ước tính:** 1-2 tuần (phụ thuộc vào việc thực thi đánh giá)

**Ưu tiên:** **NGHIÊM TRỌNG** - Không thể bảo vệ với giá trị "TBD"

---

## 7. CÂU HỎI CHO TÁC GIẢ

1. **Tại sao đánh giá AI chưa hoàn thành?** Vấn đề thiết lập LangSmith? Ngân sách? Thời gian?
2. **Tại sao benchmark RAG bị bỏ qua?** Vấn đề API key? Vấn đề kỹ thuật?
3. **Tại sao test E2E thất bại?** Vấn đề cơ sở hạ tầng? Lỗi ứng dụng?
4. **Có phản hồi định tính nào từ 48 nhà giáo dục** được đề cập trong C1 không?
5. **Timeline:** Khi nào các đánh giá này có thể hoàn thành?

---

## 8. CHIẾN LƯỢC BẢO VỆ ĐỀ XUẤT

**Lựa chọn A - Hoàn thành Đánh giá (Khuyến nghị):**
- Hoãn bảo vệ 1-2 tuần
- Hoàn thành tất cả đánh giá
- Điền tất cả các bảng với dữ liệu thực
- **Kết quả:** Luận văn mạnh

**Lựa chọn B - Định khung (Khẩn cấp):**
- Trình bày C4 là "Khung đánh giá đã thiết lập"
- Chuyển kết quả sang Hướng phát triển
- Giảm nhẹ các tuyên bố kết luận
- **Kết quả:** Chấp nhận được nhưng yếu

**Lựa chọn C - Dữ liệu Một phần (Rủi ro):**
- Trình bày bất kỳ dữ liệu nào có sẵn
- Thảo luận thành thật về hạn chế
- Chỉ ra khung hoạt động
- **Rủi ro:** Hội đồng có thể đặt câu hỏi về độ nghiêm ngặt

**Khuyến nghị:** Chọn Lựa chọn A nếu có thể. Luận văn xứng đáng có dữ liệu đầy đủ.
