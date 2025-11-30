import streamlit as st
import cv2
import numpy as np
import time
from streamlit_image_comparison import image_comparison

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Ancient Doc Restore",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TÙY CHỈNH (Làm đẹp giao diện) ---
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        h1 {
            color: #4b2e2e;
            text-align: center;
        }
        .stButton>button {
            width: 100%;
            background-color: #ff4b4b;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Bảng Điều Khiển")
        st.info("Tinh chỉnh các thông số thuật toán tại đây.")
        
        # Nhóm 1: Khử nhiễu (Denoiser)
        st.subheader("1. Khử Nhiễu (Denoise)")
        median_k = st.slider("Median Kernel (Diệt đốm)", 1, 7, 3, step=2, help="Kích thước cửa sổ lọc trung vị. Lớn quá sẽ mất nét nhỏ.")
        gaussian_k = st.slider("Gaussian Kernel (Mịn nền)", 1, 9, 3, step=2)
        
        # Nhóm 2: Tăng cường (Enhancer)
        st.subheader("2. Tăng Cường (Enhance)")
        clip_limit = st.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0, help="Giới hạn độ tương phản cục bộ.")
        sharp_amount = st.slider("Độ nét (Sharpen)", 0.0, 3.0, 1.0, help="Cường độ làm nét Unsharp Mask.")

        # Nhóm 3: Tách chữ (Segmentation)
        st.subheader("3. Tách Chữ (Binarize)")
        sauvola_k = st.slider("Sauvola k-factor", 0.01, 0.5, 0.2, help="Độ nhạy của ngưỡng tách chữ.")

        st.markdown("---")
        st.caption("Nhóm thực hiện: ")
        
        # Trả về một dictionary chứa các tham số để main dùng
        return {
            "median_k": median_k,
            "gaussian_k": gaussian_k,
            "clip_limit": clip_limit,
            "sharp_amount": sharp_amount,
            "sauvola_k": sauvola_k
        }

def to_rgb(img):
    """convert màu cho đúng chuẩn hiển thị Web."""
    if img is None: return None
    if len(img.shape) == 2: return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def main():
    # --- HEADER ---
    st.title("📜 HỆ THỐNG PHỤC HỒI & SỐ HÓA TÀI LIỆU CỔ")
    st.markdown("---")

    params = render_sidebar()

    # --- MAIN AREA: INPUT & PROCESS ---
    col_upload, col_action = st.columns([3, 2])
    
    with col_upload:
        uploaded_file = st.file_uploader("📂 Tải lên ảnh tài liệu (JPG, PNG)...", type=['jpg', 'png', 'jpeg'])

    # Session State để lưu kết quả (tránh mất khi reload trang)
    if 'results' not in st.session_state:
        st.session_state.results = None

    if uploaded_file is not None:
        # 1. Đọc ảnh vào bộ nhớ
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_input = cv2.imdecode(file_bytes, 1) # BGR format

        # Hiển thị ảnh gốc
        with col_action:
            st.write("Preview Ảnh Gốc:")
            st.image(to_rgb(img_input),use_container_width=True)
            
            # Nút chạy xử lý
            if st.button("🚀 BẮT ĐẦU PHỤC HỒI", type="primary"):
                with st.spinner("Đang khởi động Pipeline..."):
                    # --- CHỖ NÀY SẼ GỌI PIPELINE CỦA BẠN ---
                    # pipeline = DocumentRestorationPipeline()
                    # st.session_state.results = pipeline.run(img_input, params)
                    
                    # (Giả lập kết quả để test giao diện khi chưa có pipeline)
                    time.sleep(1) # Giả vờ đang chạy
                    dummy_res = {
                        '1_Input': img_input,
			'2_Gauss':cv2.GaussianBlur(img_input, ksize = (5,5), sigmaX = 2),
                        '7_Final_Crop': cv2.bitwise_not(img_input) # Nghịch đảo màu làm ví dụ
                    }
                    st.session_state.results = dummy_res
                
                st.success("Xử lý hoàn tất!")

    # --- KẾT QUẢ (HIỂN THỊ SAU KHI CÓ DỮ LIỆU) ---
    if st.session_state.results is not None:
        results = st.session_state.results
        
        st.markdown("### 📊 KẾT QUẢ XỬ LÝ")
        
        # Tạo Tabs chức năng
        tab_compare, tab_detail, tab_showcase, tab_export = st.tabs([
            "🔍 So sánh Trực quan", 
            "🛠️ Giải phẫu Chi tiết", 
            "🎥 Trình diễn (Showcase)", 
            "💾 Xuất bản"
        ])

        # === TAB 1: SO SÁNH (IMAGE COMPARISON) ===
        with tab_compare:
            st.markdown("#### So sánh Trước & Sau")
            
            img_before = to_rgb(results.get('1_Input', img_input))
            img_after = to_rgb(results.get('Output_Visual', results.get('7_Final_Crop')))

            if img_before is not None and img_after is not None:
                image_comparison(
                    img1=img_before,
                    img2=img_after,
                    label1="Ảnh Gốc (Nhiễu/Cong)",
                    label2="Kết Quả (Phẳng/Sạch)",
                    width=700,
                    starting_position=50,
                    show_labels=True,
                    make_responsive=True,
                    in_memory=True
                )

        # === TAB 2: CHI TIẾT TỪNG BƯỚC ===
        with tab_detail:
            st.markdown("#### Các bước trung gian trong Pipeline")
            # (Bạn có thể thêm code hiển thị các bước trung gian ở đây khi có pipeline thực)
            st.info("Chưa có dữ liệu chi tiết (Cần kết nối Pipeline).")

        # === TAB 3: TRÌNH DIỄN (ANIMATION & SLIDESHOW) ===
        with tab_showcase:
            st.markdown("#### Xem lại quá trình biến đổi")
            
            col_anim_btn, col_anim_view = st.columns([1, 4])
            
            with col_anim_btn:
                run_anim = st.button("▶️ Chạy Timelapse")
                st.info("Bấm nút để xem ảnh biến đổi từ từ.")

            with col_anim_view:
                placeholder_img = st.empty()
                placeholder_txt = st.empty()
                progress_bar = st.empty()

                if run_anim:
                    # Danh sách các bước giả lập để test giao diện
                    steps = [
                        ("1. Ảnh Đầu Vào", results.get('1_Input')),
			("2. Làm mờ", results.get('2_Gauss')),
                        ("✅ HOÀN THÀNH!", results.get('7_Final_Crop'))
                    ]

                    for i, (text, img_step) in enumerate(steps):
                        if img_step is not None:
                            placeholder_txt.markdown(f"**{text}**")
                            placeholder_img.image(to_rgb(img_step), use_container_width=True)
                            progress_bar.progress((i + 1) / len(steps))
                            time.sleep(1.0)
                    
                    st.balloons() 

        # === TAB 4: XUẤT BẢN (DOWNLOAD) ===
        with tab_export:
            st.markdown("#### Tải xuống Kết quả")
            
            final_img = results.get('7_Final_Crop')
            
            if final_img is not None:
                is_success, buffer = cv2.imencode(".png", final_img)
                if is_success:
                    st.download_button(
                        label="📥 Tải ảnh PNG (Chất lượng cao)",
                        data=buffer.tobytes(),
                        file_name="restored_document.png",
                        mime="image/png"
                    )

if __name__ == "__main__":
	main()
