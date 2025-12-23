import numpy as np
import cv2 as cv
class PageDewarper:
    def get_text_lines(self, binary_image):
        """
        Phát hiện các dòng văn bản cong.
        Dùng Morphology Dilate (kernel ngang dài) để nối chữ thành dòng.
        """
        pass

    def fit_polynomial(self, points, degree=3):
        """
        Hồi quy đa thức tìm phương trình đường cong y = f(x).
        Input: Các điểm trên dòng chữ.
        Output: Hệ số đa thức.
        """
        pass

    def generate_mesh(self, image_shape, top_curve, bottom_curve):
        """
        Tạo lưới tọa độ nguồn (Source Mesh) dựa trên đa thức đường cong trên và dưới.
        Dùng cho cv2.remap để làm phẳng trang cong.
        """
        height, width = image_shape[:2]

        # Tạo trục tọa độ
        x_range = np.arange(width, dtype=np.float32)
        y_range = np.arange(height, dtype=np.float32)

        # Tính y của đường cong trên & dưới
        top_y = np.polyval(top_curve, x_range)
        bottom_y = np.polyval(bottom_curve, x_range)

        # Giới hạn biên an toàn
        top_y = np.clip(top_y, 0, height - 1)
        bottom_y = np.clip(bottom_y, 0, height - 1)

        # Đảm bảo bottom luôn dưới top
        bottom_y = np.maximum(bottom_y, top_y + 1)

        # Lưới tọa độ
        target_x, target_y = np.meshgrid(x_range, y_range)

        # Chuẩn hóa chiều cao
        relative_height = target_y / max(height - 1, 1)

        # Nội suy tuyến tính theo chiều dọc
        top_y_row = top_y.reshape(1, -1)
        bottom_y_row = bottom_y.reshape(1, -1)

        # Công thức: source_y = top + tỷ lệ * khoảng cách
        source_y = top_y_row + relative_height*(bottom_y_row - top_y_row)

        # Xử lý source_x
        source_x = target_x

        return source_x.astype(np.float32), source_y.astype(np.float32)

    def dewarp(self, image):
        """
        Hàm chính thực hiện làm phẳng.
        Gọi các hàm con trên -> Tạo map_x, map_y -> Remap.
        """
        pass
