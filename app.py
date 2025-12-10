import streamlit as st
import numpy as np
from PIL import Image
import cv2  # OpenCV for conversions
from src.pipeline import DocumentRestorationPipeline

# --- Kích hoạt cache cho object nặng ---
@st.cache_resource
def get_pipeline():
    return DocumentRestorationPipeline()

pipeline = get_pipeline()

st.set_page_config(layout="wide", page_title="Hệ thống Số hóa Tài liệu Cổ")
st.title("Hệ thống Phục hồi Tài liệu Cổ")

# Sidebar controls
st.sidebar.header("⚙️ Điều chỉnh Pipeline")

st.sidebar.subheader("Giai đoạn 3: Khảo cổ & Phục hồi")
forensic_ink_enabled = st.sidebar.checkbox(
    'Bật Khôi phục Mực phai (Forensic Ink)', value=True,
    help="Sử dụng Decorrelation Stretch (PCA) để tăng cường mực phai. Cần ảnh màu."
)
median_ksize = st.sidebar.slider(
    'Kích thước Kernel Median (Median filter)', min_value=3, max_value=7, step=2, value=3
)

# File uploader
uploaded_file = st.file_uploader("🖼️ Tải lên ảnh tài liệu (.jpg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert('RGB')
    image = np.array(pil_img).astype(np.uint8)

    st.header("🔍 Kết quả Xử lý Pipeline")

    # Chuẩn bị params để gửi chính xác vào pipeline
    params = {
        "forensic_ink": forensic_ink_enabled,   # nếu pipeline dùng key khác => đổi tương ứng
        "denoise": True,
        "denoise_method": "median",
        "median_ksize": int(median_ksize),
        # bạn có thể thêm các key mặc định khác ở đây
    }

    with st.spinner('Đang xử lý tài liệu...'):
        processed_results = pipeline.run(image, params)

    if processed_results.get("status") == "ok":
        # Hiển thị ảnh gốc
        st.subheader("Ảnh Gốc")
        st.image(image, use_column_width=True)

        # Hiển thị bước Forensic & Denoise
        st.subheader("Các Bước Phục hồi (Tuần 2)")
        col_f, col_d = st.columns(2)

        # Forensic Ink
        with col_f:
            ink = processed_results["images"].get("ink_restored") or processed_results["images"].get("ink")
            if ink is not None:
                # convert single-channel -> 3-channel nếu cần
                if isinstance(ink, np.ndarray):
                    if ink.ndim == 2:
                        ink_show = cv2.cvtColor(ink, cv2.COLOR_GRAY2RGB)
                    else:
                        ink_show = ink
                else:
                    ink_show = ink
                st.image(ink_show, caption="Mực phai đã Khôi phục (Forensic Ink)", use_column_width=True)
            else:
                st.info("Bước Khôi phục Mực phai đã bị bỏ qua hoặc không trả về ảnh.")

        # Denoised
        with col_d:
            den = processed_results["images"].get("denoised")
            if den is not None:
                if isinstance(den, np.ndarray) and den.ndim == 2:
                    den_show = cv2.cvtColor(den, cv2.COLOR_GRAY2RGB)
                else:
                    den_show = den
                st.image(den_show, caption=f"Ảnh sau Khử nhiễu (Median k={median_ksize})", use_column_width=True)
            else:
                st.warning("Thiếu ảnh sau Denoise. Kiểm tra lại logic pipeline.")

        #hiển thị final
        if "final" in processed_results["images"]:
            st.subheader("Kết quả Cuối (Final)")
            fin = processed_results["images"]["final"]
            if isinstance(fin, np.ndarray) and fin.ndim == 2:
                fin_show = cv2.cvtColor(fin, cv2.COLOR_GRAY2RGB)
            else:
                fin_show = fin
            st.image(fin_show, use_column_width=True)

    else:
        st.error(f"Đã xảy ra lỗi: {processed_results.get('error', 'Unknown error')}")

st.sidebar.markdown("---")
st.sidebar.info("Chạy: `streamlit run app.py`")
