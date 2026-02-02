import cv2
import numpy as np
import math
# Import hàm từ utils nếu cần dùng cho các biến thể khác (theo yêu cầu của bạn)
from src.utils.math_ops import calc_gradient_sobel
import matplotlib.pyplot as plt

from src.core.geometry import GeometryCorrector
class Deskewer:
    def __init__(self):
        # Gọi ông thợ cơ khí vào
        self.geometry = GeometryCorrector()
    def detect_skew_angle(self, image: np.ndarray) -> float:
        """
        Tự động phát hiện góc nghiêng của văn bản.
        Phương pháp: Hough Line Transform (Probabilistic).
        """
        # 0. Tiền xử lý: Chuyển xám nếu cần
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 1. Canny Edge Detection để tìm biên chữ
        # Dùng GaussianBlur nhẹ trước để giảm nhiễu hạt, giúp Canny bắt biên mượt hơn
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blur, 50, 150, apertureSize=3)

        # 2. Hough Line Transform (Probabilistic)
        # minLineLength: Độ dài tối thiểu của đoạn thẳng (khoảng 1/4 chiều rộng ảnh là ổn)
        # maxLineGap: Khoảng cách tối đa để nối các nét đứt thành 1 dòng (cho phép chữ đứt quãng)
        h, w = gray.shape
        min_line_len = w // 10  # Dòng phải dài ít nhất 10% chiều rộng ảnh
        max_line_gap = 20  # Cho phép nét đứt cách nhau 20px

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=min_line_len,
            maxLineGap=max_line_gap
        )
        # [[x1,y1,x2,y2], [x1,x2....
        if lines is None:
            print("   [Deskewer] Warning: No lines detected. Assuming 0 skew.")
            return 0.0

        # 3. Tính góc trung bình
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Tính góc (radians) -> đổi sang độ (degrees)
            # atan2 trả về giá trị từ -pi đến pi
            angle_rad = math.atan2(y2 - y1, x2 - x1)
            angle_deg = math.degrees(angle_rad)

            # Chỉ lấy các góc gần 0 (ngang) để tránh nhiễu từ các nét sổ dọc hoặc khung tranh
            # Ta giả định văn bản không bao giờ nghiêng quá 45 độ
            if -45 < angle_deg < 45:
                angles.append(angle_deg)

        if not angles:
            return 0.0

        # Trả về trung vị (Median) để loại bỏ nhiễu ngoại lai (Outliers)
        # Mean (Trung bình cộng) rất dễ bị sai nếu có 1 đường kẻ bậy bạ
        skew_angle = np.median(angles)

        print(f"   [Deskewer] Detected Angle: {skew_angle:.2f} degrees")
        return skew_angle

    def deskew(self, image):
        """
        Sử dụng GeometryCorrector để xoay.
        """

        angle = self.detect_skew_angle(image)

        if abs(angle) < 0.1:
            print(f"   [Deskewer] Angle {angle:.2f} is negligible. Skipping.")
            return image

        print(f"   [Deskewer] Rotating image by {angle:.2f} degrees...")
        rotated_img = self.geometry.rotate_image(image, angle, keep_size=True)

        return rotated_img

