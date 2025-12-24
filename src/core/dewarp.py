import cv2
import numpy as np
from typing import List, Tuple, Optional


class PageDewarper:
    """
    Class xử lý làm phẳng tài liệu cong 3D (Dewarping).
    Thực hiện logic Giai đoạn 2 của dự án.
    """

    def __init__(self):
        # Cấu hình kernel cho phép toán hình thái học (Morphology)
        # Kernel chữ nhật dài (25x3) giúp nối các chữ cái trên cùng 1 dòng
        # nhưng không làm dính các dòng trên/dưới lại với nhau.
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))

    def get_text_lines(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """
        Phát hiện các dòng văn bản trong ảnh nhị phân.

        Logic Kỹ thuật[cite: 119]:
        1. Dilate: Làm dày chữ để nối thành dải băng (text blobs).
        2. Contour: Tìm biên của các dải băng đó.

        Args:
            binary_image: Ảnh đầu vào đã nhị phân hóa (đen trắng).

        Returns:
            List[np.ndarray]: Danh sách các contour hợp lệ đại diện cho dòng chữ.
        """
        # 1. Phép toán Dilate (Nở): Nối chữ rời rạc thành dòng liền mạch
        dilated = cv2.dilate(binary_image, self.morph_kernel, iterations=1)

        # 2. Tìm Contour (đường bao)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 3. Lọc nhiễu: Chỉ lấy các khối có diện tích > 500 pixel
        # Loại bỏ các dấu chấm, vết bẩn nhỏ không phải là dòng chữ
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

        return valid_contours

    def fit_polynomial(self, points: np.ndarray, degree: int = 3) -> np.poly1d:
        """
        Hồi quy đa thức để mô hình hóa đường cong dòng chữ.

        Logic Toán học[cite: 104]:
        - Sử dụng phương pháp Bình phương tối thiểu (Least Squares).
        - Tìm hệ số cho đa thức bậc 3: y = ax^3 + bx^2 + cx + d.

        Args:
            points: Mảng tọa độ (x, y) của các điểm trên dòng chữ.
            degree: Bậc của đa thức (mặc định là 3 cho đường cong chữ S).

        Returns:
            np.poly1d: Hàm đa thức f(x) dùng để tính y từ x.
        """
        if len(points) == 0:
            return None

        x = points[:, 0]
        y = points[:, 1]

        # np.polyfit giải hệ phương trình Ax = b để tìm bộ hệ số tối ưu
        coeffs = np.polyfit(x, y, degree)

        # Tạo đối tượng hàm số để dễ dàng gọi f(x) sau này
        poly_func = np.poly1d(coeffs)

        return poly_func

    def generate_mesh(self, image_shape: Tuple[int, int], top_poly: np.poly1d, bottom_poly: np.poly1d) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Tạo lưới tọa độ biến đổi (Remap Map).

        Logic Kỹ thuật[cite: 107]:
        - Tạo lưới đích (thẳng).
        - Tính ngược vị trí tương ứng trên ảnh nguồn (cong) bằng nội suy.

        Args:
            image_shape: Kích thước ảnh (cao, rộng).
            top_poly: Hàm đa thức của dòng trên cùng.
            bottom_poly: Hàm đa thức của dòng dưới cùng.

        Returns:
            Tuple[map_x, map_y]: Hai ma trận dùng cho hàm cv2.remap.
        """
        h, w = image_shape[:2]

        # Map X: Tọa độ x không đổi theo chiều dọc (vì ta chỉ nắn chỉnh chiều y là chủ yếu)
        # np.tile nhân bản dòng 0..w cho đủ h dòng
        map_x = np.tile(np.arange(w), (h, 1)).astype(np.float32)

        # Map Y: Cần tính toán nội suy
        map_y = np.zeros((h, w), dtype=np.float32)

        # Tính giá trị y của biên trên và biên dưới tại mọi điểm x
        x_range = np.arange(w)
        top_y_curve = top_poly(x_range)  # y_top
        bottom_y_curve = bottom_poly(x_range)  # y_bottom

        # Duyệt qua từng dòng y của ảnh đích (ảnh phẳng)
        for y in range(h):
            # Alpha: Tỷ lệ vị trí tương đối (0.0 ở đỉnh, 1.0 ở đáy)
            alpha = y / h

            # Công thức nội suy tuyến tính:
            # y_nguồn = y_top + alpha * (khoảng cách giữa top và bottom)
            map_y[y, :] = top_y_curve + alpha * (bottom_y_curve - top_y_curve)

        return map_x, map_y

    def dewarp(self, image: np.ndarray) -> np.ndarray:
        """
        Hàm chính thực hiện quy trình làm phẳng ảnh (Pipeline).
        """
        h, w = image.shape[:2]

        # B1: Tiền xử lý (Preprocessing)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Dùng Otsu để tự động tìm ngưỡng tối ưu tách nền/chữ
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # B2: Lấy các dòng văn bản (Text Line Detection) [cite: 119]
        contours = self.get_text_lines(binary)

        if not contours:
            print("Cảnh báo: Không tìm thấy dòng văn bản nào để làm phẳng.")
            return image

        # B3: Tìm Keylines (Dòng đầu và dòng cuối)
        # Sắp xếp contour theo trục y (từ trên xuống dưới)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

        top_cnt = contours[0]  # Dòng trên cùng
        bottom_cnt = contours[-1]  # Dòng dưới cùng

        # Reshape contour (N, 1, 2) -> (N, 2) để đưa vào hàm fit
        top_points = top_cnt.reshape(-1, 2)
        bottom_points = bottom_cnt.reshape(-1, 2)

        # B4: Mô hình hóa đường cong (Polynomial Regression) [cite: 104]
        top_poly = self.fit_polynomial(top_points)
        bottom_poly = self.fit_polynomial(bottom_points)

        # B5: Tạo lưới biến đổi (Mesh Generation)
        map_x, map_y = self.generate_mesh((h, w), top_poly, bottom_poly)

        # B6: Kéo phẳng ảnh (Remapping)
        # Dùng INTER_LINEAR (Bilinear Interpolation) để ảnh mượt, không vỡ hạt
        dewarped_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

        return dewarped_image