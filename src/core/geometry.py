import cv2
import numpy as np

class GeometryCorrector:
    def rotate_image(self, image, angle, keep_size=False):
        """
        Xoay ảnh một góc angle (độ).
        - keep_size=True: Giữ nguyên kích thước khung ảnh cũ (bị cắt góc).
        - keep_size=False: Tự động mở rộng khung ảnh để chứa đủ hình (không mất dữ liệu).
        """
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # 1. Lấy ma trận xoay (Rotation Matrix) chuẩn
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        if not keep_size:
            # --- LOGIC TÍNH KHUNG HÌNH MỚI ---

            # Lấy cos và sin từ ma trận M (đã được OpenCV tính sẵn)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            # Tính kích thước Bounding Box mới
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

            # Điều chỉnh lại tâm xoay (Translation adjustment)
            # Vì ảnh to ra, tâm ảnh mới thay đổi so với tâm cũ.
            # Ta cần dời ảnh đi một đoạn offset để nó nằm giữa khung mới.
            M[0, 2] += (nW / 2) - center[0]
            M[1, 2] += (nH / 2) - center[1]

            # Cập nhật kích thước khung hình đích
            w, h = nW, nH

        # 2. Thực hiện biến đổi Affine
        # Lưu ý: Luôn để borderValue màu trắng (255, 255, 255) cho tài liệu
        rotated = cv2.warpAffine(image, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
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
        [Updated]:
        - Sử dụng np.linalg.norm để tính khoảng cách Euclid.
        - Sử dụng INTER_CUBIC để ảnh sắc nét hơn.
        """
        # 1. Sắp xếp lại các điểm để đảm bảo thứ tự
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        # 2. Tính toán độ rộng (width) mới của ảnh đích
        # np.linalg.norm(a - b) tương đương sqrt((x2-x1)^2 + (y2-y1)^2)
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        # 3. Tính toán độ cao (height) mới của ảnh đích
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        # 4. Tạo tập hợp điểm đích (Destination points)
        # Đây là toạ độ của ảnh chữ nhật mới ("bird's eye view")
        dst = np.array([
            [0, 0],                         # Top-Left
            [maxWidth - 1, 0],              # Top-Right
            [maxWidth - 1, maxHeight - 1],  # Bottom-Right
            [0, maxHeight - 1]],            # Bottom-Left
            dtype="float32")

        # 5. Tính ma trận biến đổi phối cảnh M và áp dụng
        M = cv2.getPerspectiveTransform(rect, dst)

        # 6. Thực hiện biến đổi (Warp)
        # Thêm flags=cv2.INTER_CUBIC để nội suy ảnh sắc nét hơn (tốt cho OCR sau này)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)

        return warped

