import cv2
import numpy as np
import svgwrite  # Cần: pip install svgwrite


class Vectorizer:
    def __init__(self, smoothness: int = 2):
        """
        Args:
            smoothness (int): Độ mượt (epsilon cho approxPolyDP). 
                              Càng lớn -> càng ít điểm -> càng mượt nhưng mất chi tiết.
        """
        self.smoothness = smoothness

    def _get_cubic_bezier_control_points(self, p_minus_1, p, p_plus_1, p_plus_2):
        """
        Tính toán 2 điểm điều khiển (Control Points) cho đoạn cong từ P đến P+1
        dựa trên thuật toán Catmull-Rom Spline.

        Math:
            CP1 = P + (P_next - P_prev) / 6
            CP2 = P_next - (P_next_next - P) / 6
        """
        # Chuyển về float để tính toán vector
        p_minus_1 = p_minus_1.astype(np.float32)
        p = p.astype(np.float32)
        p_plus_1 = p_plus_1.astype(np.float32)
        p_plus_2 = p_plus_2.astype(np.float32)

        # Tension factor (1/6 ~ 0.166 là chuẩn cho Catmull-Rom)
        tension = 6.0

        # Tính toán vector tiếp tuyến
        cp1 = p + (p_plus_1 - p_minus_1) / tension
        cp2 = p_plus_1 - (p_plus_2 - p) / tension

        return cp1, cp2

    def fit_bezier(self, points: np.ndarray) -> str:
        """
        Khớp chuỗi điểm thành chuỗi lệnh SVG Path (Cubic Bezier).

        Args:
            points: Mảng (N, 2) các điểm tọa độ (x, y).

        Returns:
            str: Chuỗi lệnh path (VD: "M 10 10 C 12 15, 18 20, 20 20 ...")
        """
        if len(points) < 3:
            return ""

        # Đóng vòng lặp kính: Thêm điểm cuối vào đầu và điểm đầu vào cuối
        # để tính tiếp tuyến cho các điểm biên được mượt mà.
        pts = np.vstack([points[-1], points, points[0], points[1]])

        # Bắt đầu path tại điểm đầu tiên
        path_str = f"M {points[0][0]:.2f},{points[0][1]:.2f} "

        # Duyệt qua từng đoạn
        for i in range(1, len(pts) - 2):
            p_prev = pts[i - 1]
            p_curr = pts[i]  # Điểm bắt đầu đoạn
            p_next = pts[i + 1]  # Điểm kết thúc đoạn
            p_next2 = pts[i + 2]

            # Tính điểm điều khiển
            cp1, cp2 = self._get_cubic_bezier_control_points(p_prev, p_curr, p_next, p_next2)

            # Lệnh C (Cubic Bezier): C cp1_x cp1_y, cp2_x cp2_y, end_x end_y
            path_str += (f"C {cp1[0]:.2f},{cp1[1]:.2f} "
                         f"{cp2[0]:.2f},{cp2[1]:.2f} "
                         f"{p_next[0]:.2f},{p_next[1]:.2f} ")

        path_str += "Z"  # Đóng path
        return path_str

    def image_to_svg(self, binary_image: np.ndarray, output_path: str):
        """
        Chuyển ảnh bitmap nhị phân sang file SVG vector.

        Pipeline:
            1. Find Contours (OpenCV)
            2. Simplify Contours (Douglas-Peucker) -> Giảm số lượng điểm thừa.
            3. Fit Bezier -> Làm mềm các góc cạnh.
            4. Export SVG.
        """
        h, w = binary_image.shape[:2]

        # 1. Tìm contours
        # Dùng RETR_LIST hoặc RETR_TREE đều được. CHAIN_APPROX_SIMPLE để nén các đoạn thẳng.
        contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Tạo file SVG
        dwg = svgwrite.Drawing(output_path, profile='tiny', size=(w, h))

        print(f"🔄 Đang vector hóa {len(contours)} contours...")

        for cnt in contours:
            # Lọc nhiễu (bỏ các contour quá nhỏ)
            if cv2.contourArea(cnt) < 20:
                continue

            # 2. Làm đơn giản hóa đường cong (Simplify)
            # Rất quan trọng: Giảm số điểm để file SVG nhẹ và đường cong mượt hơn
            epsilon = self.smoothness  # Tham số độ mượt
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Reshape về (N, 2)
            pts = approx.reshape(-1, 2)

            # 3. Fit Bezier & Tạo Path
            if len(pts) > 2:
                d_str = self.fit_bezier(pts)

                # Thêm path vào bản vẽ
                # fill='black' vì đây là ảnh nhị phân text
                path = dwg.path(d=d_str, fill='black', stroke='none')
                dwg.add(path)

        # 4. Lưu file
        dwg.save()
        print(f"✅ Đã lưu file SVG tại: {output_path}")


# --- DEMO CÁCH DÙNG ---
if __name__ == "__main__":
    # Giả sử bạn đã có ảnh 'binary_result' từ bước trước
    # binary_result = cv2.imread('binary_sauvola.png', 0)

    # Tạo ảnh giả lập để test
    dummy_img = np.zeros((500, 500), dtype=np.uint8)
    cv2.circle(dummy_img, (250, 250), 100, 255, -1)  # Hình tròn đặc
    cv2.putText(dummy_img, "A", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, 0, 10)  # Chữ A rỗng

    vectorizer = Vectorizer(smoothness=2)
    vectorizer.image_to_svg(dummy_img, "output_vector.svg")