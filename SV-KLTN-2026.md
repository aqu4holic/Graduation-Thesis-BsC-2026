## **HƯỚNG DẪN VIẾT** 

## **KHÓA LUẬN/ĐỒ ÁN TỐT NGHIỆP**

## 

## 

## **LIÊM CHÍNH HỌC THUẬT & SỬ DỤNG AI CÓ TRÁCH NHIỆM**

Sinh viên được phép sử dụng công cụ hỗ trợ nhưng phải minh bạch.

### **1\. Quy tắc về Đạo văn**

* **Công cụ kiểm tra:** Báo cáo sẽ được quét qua phần mềm bởi Khoa CNTT.  
* **Ngưỡng cho phép:** Tỷ lệ trùng lặp không được vượt quá **20%** (trừ phần tài liệu tham khảo và các định nghĩa kinh điển).

### **2\. Quy tắc sử dụng AI (ChatGPT, GitHub Copilot, Claude...)**

Sử dụng các công cụ Gen AI (ChatGPT, Copilot, Gemini...) được cho phép nhưng phải tuân thủ nguyên tắc **"AI là trợ lý, không phải người làm thay"**.

* **ĐƯỢC PHÉP:**  
  * Sử dụng AI để gợi ý sửa lỗi ngữ pháp tiếng Anh, cấu trúc câu văn.  
  * Sử dụng AI để giải thích lỗi code (debug), gợi ý thuật toán tối ưu.  
  * Tạo dữ liệu mẫu (dummy data) để kiểm thử.  
* **KHÔNG ĐƯỢC PHÉP:**  
  * Sao chép nguyên văn toàn bộ các nội dung lý thuyết hoặc kết luận từ AI.  
  * Nhờ AI viết toàn bộ mã nguồn của các chức năng cốt lõi (core features) mà sinh viên không hiểu cơ chế hoạt động.  
* Ngoài các việc ĐƯỢC PHÉP và KHÔNG ĐƯỢC PHÉP bên trên, SV cần tuân thủ quy định về sử dụng AI của GV hướng dẫn.

### **3\. Quy trình kiểm tra (Vấn đáp trực tiếp được thực hiện bởi Khoa CNTT)**

Để đảm bảo sinh viên thực sự hiểu những gì mình viết và code:

* **Kiểm tra lý thuyết:** Giảng viên sẽ hỏi sâu về các khái niệm được viết trong báo cáo. Nếu viết nhưng không giải thích được → Coi là sao chép/AI viết → **Vi phạm quy định**.  
* **Kiểm tra Code (Live Demo):** Giảng viên sẽ chỉ định đoạn code bất kỳ trong Github và yêu cầu giải thích luồng chạy, hoặc yêu cầu sửa đổi nhỏ trực tiếp (live coding). Nếu không làm được → Code không chính chủ → **Vi phạm quy định**.

## 

## 

## **CÔNG CỤ VÀ HÌNH THỨC TRÌNH BÀY**

### **1\. Báo cáo Khóa luận/Đồ án** 

* **Công cụ:** **LaTeX**  
* **Template:** Sinh viên tải và sử dụng template chính thức tại: [**https://tinyurl.com/fit-kltn-template**](https://tinyurl.com/fit-kltn-template)  
* **Yêu cầu:**  
  * Không tự ý thay đổi cấu trúc file style (`.sty`) hoặc font chữ quy định.  
  * Công thức, code, và tài liệu tham khảo (BibTeX) phải định dạng chuẩn LaTeX.  
  * Xuất bản phẩm cuối cùng là file .pdf.  
* **Thư mục KLTN tham khảo: [https://tinyurl.com/fit-kltn-mau](https://tinyurl.com/fit-kltn-mau)**  
* **Một số hướng dẫn và gợi ý tham khảo của PGS. TS. Nguyễn Việt Hà: [https://uet.vnu.edu.vn/\~hanv/htwthesis.pdf](https://uet.vnu.edu.vn/~hanv/htwthesis.pdf)**

### **2\. Quản lý Mã nguồn** 

* **Công cụ bắt buộc với KL/ĐA không thuần lý thuyết:** **GitHub**.  
* **Yêu cầu:**  
  * Mã nguồn phải được đưa lên (push) thường xuyên trong suốt quá trình làm bài, thể hiện lịch sử commit (commit history), ít nhất 2 tháng kể từ khi hoàn thiện.  
  * **Cấm:** Chỉ upload 1 lần duy nhất (upload code zip) vào cuối kỳ. Điều này sẽ bị đánh giá là không minh bạch về tiến độ, nhận điểm F.  
  * Repository phải bao gồm file README.md hướng dẫn chi tiết cách cài đặt, triển khai (deploy) và chạy thử phần mềm.  
* **Cấu trúc Repository:**  
  * `/src`: Mã nguồn chương trình.  
  * `/references`: Các bài báo, tài liệu tham khảo (PDF).  
  * `README.md`: Buộc phải có. Hướng dẫn chi tiết cách cài đặt, chạy code và demo.

## 

## **CẤU TRÚC VÀ NỘI DUNG**

Báo cáo KL/ĐA TN cần tuân thủ cấu trúc logic của một bài nghiên cứu khoa học/kỹ thuật. Dưới đây là cấu trúc tham khảo cho KLTN 

**—\> Xây dựng hệ thống/sản phẩm**

1. **Tóm tắt:** Viết cuối cùng. Phải trả lời được 3 câu hỏi trong 1 trang: Bạn giải quyết vấn đề gì? Bạn dùng phương pháp nào? Kết quả đạt được là gì (con số cụ thể)?  
2. **Mở đầu:** Đặt vấn đề, lý do chọn đề tài, mục tiêu, phạm vi nghiên cứu.  
* **Đặt vấn đề:** Tại sao đề tài này quan trọng? (Ví dụ: Sự bùng nổ của AI, nhu cầu bảo mật...). Có những hướng giải quyết chính nào và limitations là gì?  
* **Mục tiêu:** Khóa luận này định làm gì? (Xây dựng app, cải thiện thuật toán, so sánh hiệu năng...). Ý tưởng chính là gì? Cụ thể các bước lớn? Kết quả đáng chú ý  
* **Phạm vi:** Giới hạn của đề tài (chỉ làm trên dữ liệu tiếng Việt, chỉ chạy trên Android...).  
* **Cấu trúc khóa luận:** Tóm tắt ngắn gọn nội dung các chương sau.  
3. **Chương 1: Cơ sở lý thuyết:** Các kiến thức nền tảng, các công trình liên quan.  
* **Lý thuyết nền:** Giải thích các khái niệm/công nghệ *cốt lõi* mà bạn dùng (Ví dụ: React Native là gì? Transformer Architecture là gì?).  
* **Các công trình liên quan (Related Works):** Tóm tắt các nghiên cứu trước đây. Họ đã làm gì? Họ còn hạn chế gì? \-\> Dẫn dắt đến lý do tại sao giải pháp của bạn cần thiết.  
4. **Chương 2: Phân tích & Thiết kế hệ thống:** Yêu cầu hệ thống, kiến trúc, sơ đồ thiết kế  
5. **Chương 3: Hiện thực & Đánh giá:** Công nghệ sử dụng, kết quả demo, kịch bản kiểm thử (test cases), đánh giá hiệu năng/độ chính xác và các kết quả khác.  
* **Môi trường:** Cấu hình máy, dataset sử dụng, các thư viện/framework.  
* **Kịch bản kiểm thử:** Bạn test những trường hợp nào?  
* **Kết quả:** Biểu đồ (Chart), Bảng số liệu so sánh.  
* **Nhận xét:** Tại sao kết quả lại như vậy? (Tốt hơn vì sao? Kém hơn ở đâu?)  
6. **Kết luận & Hướng phát triển:** Tổng kết kết quả đạt được, hạn chế và hướng mở rộng.

Các phần khác đã được tự động xử lý bằng template latex. Một số hướng dẫn và gợi ý rất hữu ích có thể tham khảo ở Hướng dẫn của PGS. TS. Nguyễn Việt Hà: [**https://uet.vnu.edu.vn/\~hanv/htwthesis.pdf**](https://uet.vnu.edu.vn/~hanv/htwthesis.pdf)

### 

### **—\> Cấu trúc Khóa luận Nghiên cứu**

1. **Tóm tắt:** Viết cuối cùng. Phải trả lời được 3 câu hỏi trong 1 trang: Bạn giải quyết vấn đề gì? Bạn dùng phương pháp nào? Kết quả đạt được là gì (con số cụ thể)?

2. #### **Mở đầu (Introduction)**

* **Đặt vấn đề (Problem Statement):** Giới thiệu bối cảnh. Vấn đề thực tế là gì? Tại sao các giải pháp hiện tại chưa tốt?  
* **Giải pháp:** Ý tưởng chính là gì? Cụ thể các bước lớn? Kết quả tiêu biểu  
* **Liệt kê đống góp:** Liệt kê 3-4 gạch đầu dòng những gì bạn làm được (Ví dụ: Đề xuất thuật toán X, Cải thiện độ chính xác Y%, Xây dựng bộ dữ liệu Z).  
* **Cấu trúc khóa luận:** Tóm tắt các chương sau.

3. #### **Chương 2: Cơ sở lý thuyết & Các công trình liên quan (Background & Related Work)**

* **Cơ sở lý thuyết:** Các kiến thức toán học/kỹ thuật nền tảng để hiểu giải pháp của bạn (Ví dụ: Support Vector Machine, Transformers, Digital Signature schemes...).  
  * Sử dụng nhiều công thức toán (equation) để định nghĩa bài toán một cách hình thức (Formal Definition).  
* **Các công trình liên quan (Related Work):** Phân nhóm các nghiên cứu trước đây.  
  * Nhóm phương pháp A (ưu/nhược điểm).  
  * Nhóm phương pháp B (ưu/nhược điểm).  
* **Phân tích khoảng trống (Gap Analysis):** Kết luận rằng các phương pháp trên vẫn còn hạn chế gì (chậm, tốn tài nguyên, độ chính xác thấp...) → Đây là lý do phương pháp của bạn ra đời.

4. #### **Chương 3: Phương pháp đề xuất (Proposed Method)**

* **Tổng quan mô hình (Overview/Architecture):** Sơ đồ tổng quát của phương pháp.  
  * Dùng TikZ trong LaTeX hoặc vẽ hình bên ngoài chèn vào để có hình ảnh sắc nét nhất.  
* **Mô tả chi tiết từng thành phần**

5. #### **Chương 4: Phương pháp Đánh giá (Evaluation Methodology)**

   * **Dataset:** Mô tả bộ dữ liệu (nguồn, số lượng, phân bố).  
   * **Metrics:** Các độ đo sử dụng (Accuracy, F1-Score, PSNR, Latency...). Giải thích công thức của các độ đo này.  
   * **Baseline:** Bạn so sánh với những phương pháp nào? (Phải so sánh với các phương pháp State-of-the-art \- SOTA).  
6. **Chương 5: Kết quả**  
* Kết quả so sánh:  
  * Dùng Bảng (Table) để so sánh số liệu trực tiếp.  
  * Dùng Biểu đồ (Chart) để trực quan hóa.  
  * Mô tả và phân tích  
* **Phân tích chuyên sâu (In-depth Analysis / Ablation Study):**   
  * **Ablation Study:** Thử bỏ đi từng thành phần trong đề xuất của bạn xem hiệu quả giảm thế nào \-\> Chứng minh sự cần thiết của từng module.  
  * **Error Analysis:** Phân tích các trường hợp sai (Fail cases). Tại sao mô hình lại sai ở những mẫu này?

7. #### **Chương 5: Kết luận & Hướng phát triển**

* **Tổng kết:** Nhắc lại đóng góp.  
* **Hạn chế:** Thành thật về những điểm chưa làm được.  
* **Hướng phát triển:** Ý tưởng mở rộng trong tương lai.

## 

## **MỘT SỐ QUY TẮC VỀ TRÌNH BÀY**

#### **A. Toán học & Công thức (Math & Equations)**

* **Quy tắc:** Mọi biến số (x, y, n, i) trong đoạn văn phải được đặt trong dấu $ (ví dụ: $x$, $y$).  
* **Phương trình:** Với công thức quan trọng, dùng môi trường equation để tự động đánh số.  
  * *Ví dụ:* Hàm loss function $L$ được định nghĩa như sau:  
    $$L \= \\frac{1}{n} \\sum\_{i=1}^{n} (y\_i \- \\hat{y}\_i)^2$$  
* **Giải thích:** Sau mỗi công thức, phải giải thích các biến số (trong đó $y\_i$ là..., $n$ là...).

#### **B. Hình ảnh (Figures)**

* **Chất lượng:** Tuyệt đối không dùng ảnh chụp màn hình (screenshot) bị mờ, vỡ hạt.  
  * *Biểu đồ:* Xuất ra file .pdf, .eps hoặc .svg từ Python (Matplotlib), Excel hoặc các tool vẽ để khi zoom không bị vỡ.  
  * *Sơ đồ hệ thống:* Dùng draw.io (xuất PDF) hoặc Visio (công cụ tương tự).  
* **Caption:** Luôn có \\caption{} mô tả chi tiết hình (Ví dụ: "Hình 3.2: Kiến trúc mạng CNN đề xuất"). Caption đặt **dưới** hình.  
* **Cross-reference:** Trong văn bản, **bắt buộc** phải nhắc đến hình (Ví dụ: "Như thể hiện tại Hình \\ref{fig:model\_arch}..."). Không dùng từ "hình dưới đây" hay "hình trên".

#### **C. Bảng biểu (Tables)**

* **Caption:** Caption của bảng thường đặt **trên** bảng.  
* **Trình bày:** Tránh kẻ quá nhiều đường dọc (vertical lines). Trong nghiên cứu khoa học, người ta ưu tiên dùng booktabs (chỉ kẻ đường ngang đậm ở đầu và cuối).

#### **D. Thuật toán & Mã nguồn (Algorithms & Code)**

* **Giả mã (Pseudocode):** Ưu tiên dùng giả mã để mô tả thuật toán thay vì copy code thật. Sử dụng gói algorithm2e hoặc algorithmicx.  
* **Code Snippet:** Nếu bắt buộc phải đưa code vào (chỉ đưa những đoạn xử lý cốt lõi, không đưa cả file), hãy dùng gói listings hoặc minted.  
  * Cần có highlight syntax màu sắc rõ ràng.  
  * Dùng font monospaced (Courier, Consolas).

#### **E. Tài liệu tham khảo (Citations)**

* **BibTeX:** Hãy quản lý file `.bib` thật kỹ.  
* **Cách trích dẫn:**  
  * Mọi khẳng định, số liệu không phải của bạn đều phải trích dẫn (Ví dụ: "Theo nghiên cứu của Google \[1\]...").  
  * Dùng lệnh `\cite{key}`.  
  * Sử dụng Google Scholar để lấy bibtex

#### **E. Văn phong học thuật (Academic Writing Style)**

* **Ngôi xưng:** Tuyệt đối không dùng "em", "mình". Dùng câu bị động hoặc chủ ngữ giả định.  
  * *Sai:* "Em đã xây dựng hệ thống..."  
  * *Đúng:* "Hệ thống được xây dựng...", "Khóa luận đề xuất phương pháp...", "Nhóm tác giả thực hiện..."  
  * Trong một số trường hợp có thể xưng "tôi"  
* **Thuật ngữ tiếng Anh:**  
  * Với thuật ngữ chuyên ngành quá phổ biến (Deep Learning, Framework, API), có thể giữ nguyên.  
  * Với thuật ngữ có thể dịch, hãy dịch lần đầu và mở ngoặc tiếng Anh. Ví dụ: "Học máy (Machine Learning \- ML)".  
* **Tính khách quan:** Tránh các từ cảm xúc như "rất tuyệt vời", "cực kỳ nhanh". Hãy dùng số liệu: "độ chính xác đạt 98%", "thời gian phản hồi giảm 20%".

## 