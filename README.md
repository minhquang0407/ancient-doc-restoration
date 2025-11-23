# 📜 Ancient Document Restoration & Digitization System (S-Tier Project)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org)

> **Một hệ thống xử lý ảnh toàn diện giúp phục hồi, làm phẳng và số hóa các tài liệu cổ bị hư hỏng, cong vênh, ố vàng thành văn bản kỹ thuật số chất lượng cao.**

---

## 🌟 Giới thiệu (Introduction)

Dự án này giải quyết các thách thức trong việc bảo tồn và số hóa tài liệu lịch sử. Không sử dụng các mô hình Deep Learning "hộp đen" (Black-box AI), chúng tôi xây dựng một pipeline xử lý dựa trên **Toán học (Mathematics)** và **Xử lý ảnh Cổ điển (Classical Computer Vision)** để đảm bảo tính minh bạch, tốc độ và khả năng kiểm soát cao nhất.

### ✨ Tính năng nổi bật (Key Features)
* **🔄 Làm phẳng 3D (3D Dewarping):** Tự động phát hiện đường cong văn bản và "trải phẳng" trang sách bị cong gáy.
* **🔍 Khảo cổ số (Forensic Ink Recovery):** Sử dụng thuật toán PCA để tách và khôi phục các nét mực bị phai màu mắt thường khó thấy.
* **🧼 Phục hồi & Làm sạch (Restoration):** Khử nhiễu muối tiêu, nhiễu hạt, khử bóng đổ (Shadow Removal) và vá lỗ thủng (Inpainting).
* **📐 Vectơ hóa (Vectorization):** Chuyển đổi văn bản bitmap sang định dạng Vector (SVG) sắc nét ở mọi mức phóng to.
* **📄 Số hóa (Digitization):** Tách chữ thông minh (Sauvola Thresholding) và xuất ra PDF Searchable (tích hợp OCR).

---

## 🚀 Cài đặt & Sử dụng (Installation & Usage)

### 1. Yêu cầu hệ thống
* Python 3.8 trở lên
* Tesseract OCR (cần cài đặt riêng trên máy)

### 2. Cài đặt
```bash
# Clone repository
git clone [https://github.com/username/ancient-doc-restoration.git](https://github.com/username/ancient-doc-restoration.git)
cd ancient-doc-restoration

# Tạo môi trường ảo (Khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
