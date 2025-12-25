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

    def generate_mesh(self, image_shape, top_curve, bottom_curve, output_size=None):
        """
        Tạo lưới tọa độ nguồn (Source Mesh) dựa trên đa thức đường cong trên và dưới.
        Dùng cho cv2.remap để làm phẳng trang cong.
        """
        if output_size is not None:
            out_h, out_w = output_size
        else:
            out_h, out_w = image_shape[:2]

        src_h, src_w = image_shape[:2]

        # Tạo trục tọa độ
        x_range = np.linspace(0, src_w - 1, num=out_w, dtype=np.float32)
        y_ratio = np.linspace(0, 1, num=out_h, dtype=np.float32).reshape(-1, 1)

        # Tính y của đường cong trên & dưới
        top_y = np.clip(np.polyval(top_curve, x_range), 0, src_h - 1)
        bottom_y = np.clip(np.polyval(bottom_curve, x_range), 0, src_h - 1)

        # Đảm bảo bottom luôn dưới top
        bottom_y = np.maximum(bottom_y, top_y + 1)

        # Công thức: source_y = top + tỷ lệ * khoảng cách
        # (H, 1) x (1, W) -> (H, W)
        source_y = top_y + y_ratio*(bottom_y - top_y)

        # Broadcast x_range thành mảng 2D
        source_x = np.broadcast_to(x_range, (out_h, out_w))

        return source_x.astype(np.float32), source_y.astype(np.float32)

    def dewarp(self, image):
        """
        Hàm chính thực hiện làm phẳng.
        Gọi các hàm con trên -> Tạo map_x, map_y -> Remap.
        """
        pass
