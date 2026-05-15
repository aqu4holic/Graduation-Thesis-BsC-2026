# Phản biện Chương 3: Nền tảng Phát triển Bài giảng Edlora

**Người phản biện:** Nguyễn Thành Phát  
**Ngày:** 2026  
**File đã xem xét:** c3/c3_chapter.tex (1193 dòng)

---

## 1. Đánh giá tổng thể

**Mức độ mạnh:** ⭐⭐⭐⭐⭐ (5/5)

Tài liệu kỹ thuật phần mềm xuất sắc. Đặc tả use case chuyên nghiệp (5 UC), khung quản trị AI toàn diện (7 đặc tả), tài liệu kiến trúc multi-agent chi tiết (12 node với bảng đầy đủ), lược đồ cơ sở dữ liệu hoàn chỉnh, và hướng dẫn triển khai. Chương này phục vụ như **mẫu cho luận văn kỹ thuật phần mềm**.

---

## 2. Phân tích chi tiết từng phần

### 2.1 Phân tích Yêu cầu - Các Use Case (Dòng 6-209)

**Tài liệu Đặc tả Use Case Xuất sắc:**

**Năm Use Case Tuần tự:**
1. **UC-01: Xác thực** - Xác minh danh tính với Clerk
2. **UC-02: Onboarding** - Chọn cấp học và sách giáo khoa
3. **UC-03: Quản lý bài giảng** - CRUD trên thư viện bài giảng
4. **UC-04: Chỉnh sửa bài giảng** - Tạo nội dung hỗ trợ AI (giá trị cốt lõi)
5. **UC-05: Trình bày bài giảng** - Trình chiếu phòng học toàn màn hình

**Chất lượng Bảng Use Case:**
Tất cả 5 use case được tài liệu trong định dạng `longtable` với:
- Use Case ID
- Tên
- Tác nhân
- Mục tiêu
- Điều kiện tiên quyết
- Kích hoạt
- Luồng thành công
- Ngoại lệ
- Kết quả

**Điểm mạnh quan trọng:**
- ✅ **Chuỗi điều kiện tiên quyết:** UC-01 → UC-02 → UC-03 → UC-04 → UC-05 (phụ thuộc logic)
- ✅ **Định nghĩa tác nhân:** Teacher, Teaching Assistant (AI), Clerk (dịch vụ xác thực)
- ✅ **Xử lý ngoại lệ:** Các kịch bản lỗi cụ thể được tài liệu
- ✅ **Điều kiện trước/sau:** Trạng thái rõ ràng được yêu cầu

**UC-04 (Chỉnh sửa bài giảng) - Giá trị Cốt lõi:**
```
Mục tiêu: Biến đổi bản nháp một phần thành trải nghiệm story-driven
Điều kiện tiên quyết: UC-03 hoàn thành, người dùng sở hữu bài giảng
Tác nhân: Teacher, Teaching Assistant (AI)
```

Đây là phần tạo ra doanh thu - nơi AI thực sự tạo nội dung.

---

### 2.2 Đặc tả AI (Dòng 212-365)

**Khung Quản trị AI Xuất sắc:**

**Bảy Đặc tả Hành vi:**

1. **Neo phản hồi trong đầu vào đã xác minh** - Không ảo giác curriculum/tải lên giả
2. **Thực thi nghiêm ngặt hợp đồng cấu trúc bài giảng** - 15-20 phân đoạn, 3 hoạt động bắt buộc (đối đầu, vault, boss), không hoạt động liền kề
3. **Giữ slide dạy học súc tích** - Tối đa 150 từ/1000 ký tự, mô tả hình ảnh bằng lời
4. **Yêu cầu xác nhận rõ ràng trước khi xây dựng** - Ngăn thao tác tốn kém tình cờ
5. **Neo phản hồi trong bằng chứng curriculum đã truy xuất** - Tích hợp RAG với thuộc tính
6. **Duy trì minh bạch về giới hạn kiến thức** - Thừa nhận truy xuất thiếu/không đầy đủ
7. **Căn chỉnh theo thứ bậc ưu tiên nghiêm ngặt** - Trung thực > Cấu trúc > Mục tiêu người dùng > Curriculum > Tông sáng tạo

**Chất lượng Ví dụ:**
Mỗi đặc tả bao gồm ví dụ **Tuân thủ** và **Vi phạm** với lý do:

```
Người dùng: Làm nền trông như rừng. (không có hình đính kèm)

Tuân thủ: "Tôi đã đặt chủ đề bài giảng là rừng. Không có hình nền nào được đính kèm với tin nhắn này, nên nền trình bày sẽ giữ nguyên như cũ."

Vi phạm: "Cảm ơn hình rừng bạn đã tải lên! Tôi đã thêm nó vào sau mọi slide." (Ảo giác)
```

**Thứ bậc Ưu tiên (Quyết định thiết kế Quan trọng):**
1. Trung thực về bằng chứng (cao nhất)
2. Ràng buộc cấu trúc và an toàn bài giảng
3. Mục tiêu và thủ tục hiện tại của giáo viên
4. Đoạn curriculum chính thức đã truy xuất
5. Tông và lựa chọn storytelling (thấp nhất)

Ngăn AI nói dối để hữu ích.

---

### 2.3 Thiết kế Hệ thống - Kiến trúc Multi-Agent (Dòng 390-913)

**Độ sâu Kỹ thuật Xuất sắc:**

**Tổng quan Kiến trúc:**
- **Root Graph:** Điều phối viên
- **Lesson Director Subgraph:** Hiểu ý định, lập kế hoạch, xác định sẵn sàng
- **Lesson Developer Subgraph:** Kế hoạch → Vật liệu hóa scene

**12 Node Hoạt động được Tài liệu:**

| Node ID | Agent | Mục đích |
|---------|-------|----------|
| `lesson_director` | Root | Điểm vào, gọi subgraph |
| `lesson_developer` | Root | Gọi subgraph developer |
| `detect_generation_intent` | Director | Quyết định nếu người dùng yêu cầu tạo rõ ràng |
| `route_director` | Director | Chọn nhánh tiếp theo (trả lời/lập kế hoạch/phát triển) |
| `extract_lesson_spec` | Director | Trích xuất mục tiêu và thủ tục |
| `assess_input_comprehensiveness` | Director | Kiểm tra thông tin đầy đủ để tạo |
| `draft_plan` | Director | Tạo creation_plan chuẩn |
| `delegate_to_developer` | Director | Chuẩn bị gói trạng thái để chuyển giao |
| `create_lesson` | Developer | Vật liệu hóa kế hoạch thành scenes |
| `report_to_director` | Developer | Tóm tắt kết quả tạo |
| `reply_to_user` | Director | Soạn tin nhắn cuối cùng cho giáo viên |
| `finish_turn` | Director | Kết thúc an toàn với giá trị mặc định |

**Định dạng Tài liệu Node:**
Mỗi bảng node bao gồm:
- ID, Agent (root/director/developer)
- Vai trò (bullet points)
- Kích hoạt Thực thi (khi nào chạy)
- Phục hồi/Bối cảnh & Kết quả (xử lý lỗi)

**Tài liệu Trạng thái Chia sẻ (Dòng 877-913):**
```
- Lịch sử hội thoại
- Ngữ cảnh sách giáo khoa (đã truy xuất RAG)
- lesson_plan (mục tiêu đã trích xuất)
- procedure (các pha dạy học)
- creation_plan (dàn ý chuẩn)
- scenes_data (slide có thể chơi cuối cùng)
- generation_status (trạng thái vòng đời)
```

---

### 2.4 Kiến trúc RAG (Dòng 915-931)

**Pipeline Ba Giai đoạn:**
1. **Tiếp nhận:** PDF → Phân đoạn theo tiêu đề → LLM nhiệt độ thấp → Bản ghi chunk có cấu trúc
2. **Lưu trữ:** Embeddings → PostgreSQL + pgvector
3. **Truy xuất:** Tìm kiếm tương tự → Ngữ cảnh sách giáo khoa → Trạng thái AI

**Trích dẫn:** `\cite{lewis2020retrievalAugmentedGeneration}`

---

### 2.5 Backend và Cơ sở dữ liệu (Dòng 933-1084)

**Danh mục Endpoint HTTP Hoàn chỉnh (Dòng 942-967):**

Tài liệu 12 endpoint HTTP với:
- Phương thức, Đường dẫn, Yêu cầu Auth, Vai trò

Ví dụ:
```
POST /api/lessons/create - Bearer required - Tạo bài giảng mới với khóa idempotency
POST /api/lessons/chat - Bearer required - Proxy đến AI với SSE streaming
POST /api/storage/upload - Bearer required - Cloudflare R2 multipart upload
```

**Sơ đồ ER (Hình \ref{fig:backend_er_model}):**
- Các thực thể User, Lesson, Scene, Chat History, Textbook Chunk

**Các Bảng Lược đồ Cơ sở dữ liệu:**

**Bảng User (Dòng 982-1002):**
- id (định danh Clerk, PK)
- email (duy nhất, đã index)
- full_name
- education_level
- textbook_set
- is_active (khóa mềm)

**Bảng Lesson (Dòng 1004-1021):**
- id (UUIDv7)
- name
- owner_id (FK đến user)
- background_image_url
- background_audio_url

**Bảng Scene (Dòng 1023-1040):**
- id, lesson_id (FK), name, order_index
- content (JSONB - lược đồ linh hoạt cho các loại trò chơi)

**Lịch sử Chat (Dòng 1042-1061):**
- Tách biệt khỏi bảng lesson để hiệu suất
- threads (mảng JSONB tin nhắn)

**Textbook Chunk (Dòng 1063-1083):**
- id, textbook_id, unit_name, skill_type
- content (text), embedding (vector(1536))
- Chỉ mục IVFFlat cho tìm kiếm tương tự nhanh

**Lý do Thiết kế Lược đồ:**
Giải thích tốt tại dòng 1042 và 1082:
- Tách lịch sử chat "đảm bảo truy vấn danh sách bài giảng vẫn cực kỳ nhanh"
- Tách chunk sách giáo khoa "có thể phục vụ hàng ngàn giáo viên đồng thời mà không nhân bản dữ liệu vector nặng một cách không cần thiết"

---

### 2.6 Triển khai - Hướng dẫn từng Use Case (Dòng 1085-1193)

**Hình chụp UI + Sơ đồ trình tự cho cả 5 Use Case:**

**Định dạng cho mỗi UC:**
- Hình chụp màn hình placeholder (mockup UI)
- Sơ đồ trình tự placeholder (tương tác hệ thống)
- Giải thích văn bản chi tiết

**Ví dụ: UC-04 Tạo bài giảng (Dòng 1153-1173):**
- UI: Bố cục ba cột (scenes, preview, chat)
- Trình tự: GET lesson → POST chat với SSE → Truy xuất RAG → AI streaming → PATCH updates
- Chi tiết kỹ thuật: Server-Sent Events, streaming tăng dần

**Các tính năng đáng chú ý:**
- Khóa idempotency để tạo bài giảng (ngăn trùng lặp)
- Webhooks Clerk cho sự kiện vòng đời
- Cloudflare R2 cho lưu trữ multimedia
- Cập nhật UI lạc quan được đề cập

---

## 3. Các vấn đề xuyên suốt

### 3.1 Tham chiếu Hình ảnh

| Hình | Tham chiếu | Trạng thái |
|------|-----------|------------|
| Sơ đồ use case | `\ref{fig:use_case_diagram}` | ✅ |
| Kiến trúc high-level | `\ref{fig:high_level_architecture}` | ⚠️ Dòng 383: Hình bị comment |
| Root graph | `\ref{fig:multi_agent_root_graph}` | ✅ |
| Lesson Director | `\ref{fig:lesson_director_mas_architecture}` | ✅ |
| Lesson Developer | `\ref{fig:lesson_developer_mas_architecture}` | ✅ |
| Thiết kế RAG | `\ref{fig:rag_design_architecture}` | ✅ |
| Sơ đồ ER | `\ref{fig:backend_er_model}` | ✅ |

**Vấn đề quan trọng:** Dòng 383 có hình kiến trúc high-level **bị comment**:
```latex
% \includegraphics[width=1\textwidth]{figures/high_level_architecture.png}
```

### 3.2 Tham chiếu Bảng

**Tất cả các bảng được gán nhãn đúng:**
- 5 bảng use case (UC-01 đến UC-05)
- 12 bảng mô tả node multi-agent
- 1 bảng trạng thái chia sẻ
- 5 bảng lược đồ cơ sở dữ liệu

**Định dạng nhất quán:** Sử dụng `longtable` với header/footer phù hợp cho nhiều trang

### 3.3 Các Placeholder Triển khai

**Dòng 1095-1193:** Tất cả các phần triển khai đều có hình placeholder:
```latex
\fbox{\parbox{0.92\textwidth}{\centering Placeholder screenshot: ...}}
```

**Trạng thái:**
- ⚠️ Cần thay thế bằng hình chụp/sơ đồ thực tế trước bảo vệ
- Nhưng placeholder được đánh dấu rõ ràng và sẽ không làm hỏng biên dịch

---

## 4. KHUYẾN NGHỊ

### Sửa ngay:
1. [ ] **Bỏ comment hoặc xóa** hình kiến trúc high-level (dòng 383)
2. [ ] Thay thế tất cả hình chụp màn hình placeholder bằng hình thực tế
3. [ ] Thay thế tất cả sơ đồ trình tự placeholder bằng sơ đồ thực tế

### Cải thiện:
1. [ ] Thêm ước tính chi phí cho thao tác AI (sử dụng token cho mỗi lần tạo bài giảng)
2. [ ] Thêm benchmark độ trễ cho mỗi use case
3. [ ] Cân nhắc thêm số liệu tỷ lệ lỗi từ kiểm thử

### Câu hỏi cho tác giả:
1. Các hình chụp màn hình placeholder cần được chụp từ production hay staging?
2. Các sơ đồ trình tự có sẵn trong tài liệu kỹ thuật để vẽ lại không?
3. Chi phí token thực tế cho mỗi lần tạo bài giảng là bao nhiêu?

---

## 5. SO SÁNH VỚI LUẬN VĂN KHÁC

| Tiêu chí | Luận văn Minh | Luận văn Quân | Luận văn Phát (tôi) |
|----------|---------------|---------------|---------------------|
| **Độ dài** | 1193 dòng | ~250 dòng | 307 dòng |
| **Use Case** | 5 đặc tả chi tiết | Không | Tối thiểu |
| **Quy tắc AI** | 7 đặc tả hành vi | Prompt engineering | Kiến trúc |
| **Kiến trúc** | 12 node multi-agent | Vòng lặp self-healing | Dual-pipeline NN |
| **CSDL** | Lược đồ hoàn chỉnh | Tối thiểu | Không |
| **Triển khai** | Hướng dẫn 5 UC | Kiến trúc | Giả mã |

**Điểm độc đáo của luận văn Minh:**
1. Đặc tả use case chuyên nghiệp
2. Khung quản trị AI (ràng buộc đạo đức)
3. Tài liệu node-level multi-agent đầy đủ
4. Thiết kế lược đồ cơ sở dữ liệu với lý do
5. Hướng dẫn triển khai với UI + sơ đồ trình tự

---

## 6. KẾT LUẬN CUỐI CÙNG

Chương 3 là **tài liệu kỹ thuật phần mềm chất lượng xuất bản**.

**Điểm mạnh:**
- ✅ Phân tích use case cấp doanh nghiệp
- ✅ Quản trị AI đạo đức (ràng buộc trung thực)
- ✅ Tài liệu kiến trúc hệ thống đầy đủ
- ✅ Thiết kế lược đồ cơ sở dữ liệu sẵn sàng production
- ✅ Hướng dẫn triển khai

**Vấn đề:**
- Hình kiến trúc high-level bị comment
- Placeholder triển khai cần hình thực tế

**Thời gian sửa ước tính:** 3-4 giờ (chủ yếu chụp hình)

**Đánh giá:** ⭐⭐⭐⭐⭐ (5/5) - Một khi placeholder được điền, đây là chương luận văn vàng

---

## 7. BÀI HỌC CHÍNH CHO LUẬN VĂN CỦA TÔI

**Tài liệu Use Case:**
- Mẫu: ID, Tên, Tác nhân, Mục tiêu, Điều kiện tiên quyết, Kích hoạt, Luồng, Ngoại lệ, Kết quả
- Chuỗi điều kiện tiên quyết để hiển thị phụ thuộc logic

**Quản trị AI:**
- Thứ bậc ưu tiên ngăn ảo giác có hại
- Ví dụ tuân thủ/vi phạm cụ thể cho mỗi quy tắc

**Thiết kế Cơ sở dữ liệu:**
- Phân tách mối quan tâm (lịch sử chat tách khỏi bài giảng)
- JSONB cho lược đồ linh hoạt có thể tiến hóa
- Giải thích lý do thiết kế

**Phần Triển khai:**
- Hình chụp UI + sơ đồ trình tự + văn bản cho mỗi use case
- Chi tiết kỹ thuật rõ ràng (khóa idempotency, SSE, webhooks)
