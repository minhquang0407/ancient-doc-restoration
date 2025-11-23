
class Preprocessor:
    def __init__(self):
        pass

    def to_grayscale(self, image):
        """
        Chuyển ảnh sang thang xám chuẩn (Luma coding).
        Công thức: Y = 0.299R + 0.587G + 0.114B
        """


    def resize_image(self, image, target_width=None):
        """
        Thay đổi kích thước ảnh, giữ nguyên tỷ lệ khung hình.
        Mục đích: Chuẩn hóa kích thước đầu vào để các thuật toán chạy ổn định.
        """


    def compute_histogram(self, image):
        """
        Tính histogram của ảnh xám.
        Trả về mảng 256 phần tử đếm số lượng pixel cho mỗi mức xám.
        """
        # Có thể dùng cv2.calcHist hoặc np.histogram
