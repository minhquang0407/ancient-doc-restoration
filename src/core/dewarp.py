import cv2
import numpy as np

class PageDewarper:
    def __init__(self):
        pass

    def get_text_lines(self, binary_image):
        """
        Phát hiện các dòng văn bản cong.
        Logic: Dùng Morphology Dilate (kernel ngang dài) để nối chữ thành dòng.
        """
        # 1. Tạo kernel hình chữ nhật dài (ngang >> dọc) để nối các chữ cái liền nhau
        # Kích thước (25, 3) giúp nối chữ trên cùng 1 dòng nhưng không nối các dòng với nhau
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))

        # 2. Dilate để làm dày và nối liền các ký tự
        dilated = cv2.dilate(binary_image, kernel, iterations=1)

        # 3. Tìm Contour (đường bao) của các dòng
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lọc bỏ các nhiễu quá nhỏ, giữ lại các dòng văn bản chính
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

        return valid_contours

    def fit_polynomial(self, points, degree=3):
        """
        Hồi quy đa thức tìm phương trình đường cong y = f(x).
        Logic: Dùng Least Squares giải hệ Ax=b để tìm hệ số đa thức bậc 3.
        Input: points (list các điểm x,y trên dòng chữ).
        Output: Hàm đa thức (polynomial function).
        """
        if len(points) == 0:
            return None

        # Chuyển đổi points thành mảng numpy
        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]

        # Dùng numpy.polyfit (Least Squares)
        # y = ax^3 + bx^2 + cx + d
        coeffs = np.polyfit(x, y, degree)

        # Tạo hàm f(x) từ hệ số để dễ tính toán sau này
        poly_func = np.poly1d(coeffs)

        return poly_func

    def generate_mesh(self, image_shape, top_poly, bottom_poly):
        """
        Tạo lưới tọa độ nguồn (Source Mesh) dựa trên các đường cong.
        Logic: Biến đổi lưới cong thành lưới thẳng qua nội suy[cite: 107].
        """
        h, w = image_shape[:2]

        # Tạo lưới tọa độ đích (Destination Mesh) - là lưới thẳng
        # map_x: tọa độ x không đổi theo chiều dọc
        map_x = np.tile(np.arange(w), (h, 1)).astype(np.float32)

        # map_y: cần tính toán dựa trên đường cong trên và dưới
        map_y = np.zeros((h, w), dtype=np.float32)

        # Tính giá trị y của đường cong trên và dưới tại mọi điểm x
        x_range = np.arange(w)
        top_y_curve = top_poly(x_range)  # y = f_top(x)
        bottom_y_curve = bottom_poly(x_range)  # y = f_bottom(x)

        # Nội suy tuyến tính cho từng cột x
        # Tại mỗi cột x, pixel ở dòng y (0->h) sẽ tương ứng với vị trí nào trong ảnh cong?
        for y in range(h):
            # Tỷ lệ tương đối của dòng y hiện tại so với chiều cao ảnh phẳng (0.0 -> 1.0)
            alpha = y / h

            # Công thức nội suy: y_src = y_top + alpha * (y_bottom - y_top)
            map_y[y, :] = top_y_curve + alpha * (bottom_y_curve - top_y_curve)

        return map_x, map_y

    def dewarp(self, image):
        """
        Hàm chính thực hiện làm phẳng.
        Logic: Tạo Source Mesh → Destination Mesh → Bilinear Interpolation[cite: 107].
        """
        h, w = image.shape[:2]

        # B1: Tiền xử lý để lấy ảnh nhị phân (dùng cho việc phát hiện dòng)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Sử dụng Sauvola hoặc Otsu để nhị phân hóa (ở đây dùng Otsu đơn giản để demo)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # B2: Lấy các dòng văn bản
        contours = self.get_text_lines(binary)

        if not contours:
            print("Không tìm thấy dòng văn bản nào.")
            return image

        # B3: Tìm đường cong trên cùng và dưới cùng (Top & Bottom Curves)
        # Sắp xếp contour theo trục y để tìm dòng đầu và dòng cuối
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

        top_cnt = contours[0]  # Dòng trên cùng
        bottom_cnt = contours[-1]  # Dòng dưới cùng

        # Lấy các điểm (x, y) thuộc contour để hồi quy
        # shape của cnt là (N, 1, 2) -> reshape thành (N, 2)
        top_points = top_cnt.reshape(-1, 2)
        bottom_points = bottom_cnt.reshape(-1, 2)

        # B4: Hồi quy đa thức
        # Tìm hàm f(x) cho cạnh trên và cạnh dưới
        top_poly = self.fit_polynomial(top_points, degree=3)
        bottom_poly = self.fit_polynomial(bottom_points, degree=3)

        # B5: Tạo lưới biến đổi (Remap Map) [cite: 81]
        map_x, map_y = self.generate_mesh((h, w), top_poly, bottom_poly)

        # B6: Áp dụng Remap (Kéo phẳng)
        # Dùng Bilinear Interpolation như kế hoạch [cite: 107]
        dewarped_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

        return dewarped_image