
from src.utils.math_ops import calculate_gradient_sobel


class Deskewer:
    def __init__(self):
        pass

    def detect_skew_angle(self, image):
        """
        Tự động phát hiện góc nghiêng của văn bản.
        Phương pháp: Hough Line Transform hoặc Projection Profile.
        """
        # 1. Canny Edge Detection để tìm biên chữ
        # 2. Hough Line Transform (Probabilistic)
        # Tìm các đoạn thẳng trong ảnh (thường là dòng kẻ hoặc chân dòng chữ)
        # 3. Tính góc trung bình
                # Chỉ lấy các góc gần 0 (ngang) hoặc 90 (dọc) để tránh nhiễu
        # Trả về trung vị của các góc (median để loại bỏ nhiễu ngoại lai)


    def deskew(self, image):
        """
        Thực hiện xoay ảnh thẳng lại.
        """
        # 1. Tìm góc
        # 2. Tính tâm ảnh
        # 3. Tạo ma trận xoay (Rotation Matrix)

        # 4. Xoay ảnh (Warp Affine)
        # borderMode=cv2.BORDER_REPLICATE để lấp đầy viền đen bằng pixel rìa
