import cv2
import numpy as np

class GeometryCorrector:
    def rotate_image(self, image, angle):
        """
        Xoay ảnh một góc angle (độ) quanh tâm.
        Dùng Ma trận Affine.
        """
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Ma trận xoay
        # 1. Lấy ma trận xoay (Rotation Matrix)
        # cv2.getRotationMatrix2D(center, angle, scale)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 2. Thực hiện biến đổi Affine
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def _order_points(self, pts):
        """
        Hàm phụ trợ: Sắp xếp 4 điểm theo thứ tự chuẩn:
        0: Top-Left (TL)
        1: Top-Right (TR)
        2: Bottom-Right (BR)
        3: Bottom-Left (BL)
        """
        rect = np.zeros((4, 2), dtype="float32")

        # Top-Left có tổng (x + y) nhỏ nhất
        # Bottom-Right có tổng (x + y) lớn nhất
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Top-Right có hiệu (y - x) nhỏ nhất
        # Bottom-Left có hiệu (y - x) lớn nhất
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect



    def four_point_transform(self, image, pts):
        """
        Biến đổi phối cảnh (Perspective Transform) từ 4 điểm góc.
        Input: pts là mảng numpy shape (4, 2) chứa tọa độ 4 góc của vùng cần cắt.
        """
        # 1. Sắp xếp lại các điểm để đảm bảo thứ tự
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        # 2. Tính toán độ rộng (width) mới của ảnh đích
        # Width là khoảng cách lớn nhất giữa (bottom-right & bottom-left) hoặc (top-right & top-left)
        # Công thức khoảng cách Euclid: sqrt((x2-x1)^2 + (y2-y1)^2)
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # 3. Tính toán độ cao (height) mới của ảnh đích
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # 4. Tạo tập hợp điểm đích (Destination points)
        # Đây là toạ độ của ảnh chữ nhật mới ("bird's eye view")
        dst = np.array([
            [0, 0],  # Top-Left
            [maxWidth - 1, 0],  # Top-Right
            [maxWidth - 1, maxHeight - 1],  # Bottom-Right
            [0, maxHeight - 1]],  # Bottom-Left
            dtype="float32")

        # 5. Tính ma trận biến đổi phối cảnh M và áp dụng
        # getPerspectiveTransform giải hệ phương trình để tìm ma trận 3x3
        M = cv2.getPerspectiveTransform(rect, dst)

        # warpPerspective thực hiện biến đổi
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

