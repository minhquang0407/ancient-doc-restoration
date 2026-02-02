import numpy as np
import cv2
import matplotlib.pyplot as plt

class ForensicInk:

    def _decorrelation_stretch(self, image) -> np.ndarray:
        """
        Thuật toán Decorrelation Stretch (Dựa trên PCA).
        Mục tiêu: Loại bỏ sự tương quan giữa các kênh màu (R, G, B) để làm nổi bật
        các chi tiết màu sắc nhỏ nhất (như mực phai, con dấu mờ).
        """
        # 1. Kiểm tra đầu vào: Chỉ xử lý ảnh màu 3 kênh (RGB/BGR)
        if image is None or len(image.shape) != 3:
            return image

        h, w, c = image.shape

        # 2. Chuẩn bị dữ liệu (Flattening)
        # Biến đổi ảnh từ không gian (Height, Width, 3) thành ma trận dữ liệu (N_pixels, 3)
        # Mỗi pixel là một điểm dữ liệu trong không gian 3 chiều.
        # R G B
        # 1 row,
        # 20 30 40
        # 10 20 34
        X = image.reshape(-1, 3).astype(np.float64)

        # 3. Tính toán Thống kê cơ bản
        N = X.shape[0]  # Tổng số pixel

        # Tính Vector trung bình màu (Mean vector) mu = [mean_R, mean_G, mean_B]
        mu = np.mean(X, axis=0)

        # Center data: Đưa đám mây dữ liệu về gốc tọa độ (0,0,0)
        X_centered = X - mu

        # 4. Tính Ma trận Hiệp phương sai (Covariance Matrix)
        # Ma trận này (3x3) cho biết các kênh màu R, G, B "phụ thuộc" vào nhau như thế nào.
        # Nếu cov cao -> Màu này tăng thì màu kia cũng tăng (ít thông tin riêng biệt).
        cov_matrix = np.dot(X_centered.T, X_centered) / (N - 1)

        # 5. Phân tích Eigen (PCA - Principal Component Analysis)
        # Tìm các hướng biến thiên chính (Eigenvectors) và độ lớn biến thiên (Eigenvalues)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 6. Tính toán Ma trận Co giãn (Scaling Matrix)
        # Mục tiêu: Chuẩn hóa phương sai của tất cả các kênh về 1 (Whitening).
        sigma = np.sqrt(eigenvalues)

        # Tránh lỗi chia cho 0 (nếu ảnh quá đơn điệu)
        sigma = np.where(sigma < 1e-6, 1e-6, sigma)

        # Ma trận đường chéo chứa 1/sigma để co giãn dữ liệu
        scaling_matrix = np.diag(1.0 / sigma)

        # 7. Tạo Ma trận Biến đổi (Transformation Matrix)
        # Công thức: T = V * S * V^T
        # Ý nghĩa: Xoay trục theo PCA -> Co giãn (Scale) -> Xoay ngược lại trục gốc.
        # Việc xoay ngược lại giúp giữ được "cảm giác màu" gốc (không bị loạn màu như PCA thuần).
        transform_matrix = np.dot(np.dot(eigenvectors, scaling_matrix), eigenvectors.T)

        # 8. Áp dụng biến đổi lên dữ liệu
        # Dữ liệu lúc này đã được "gỡ rối" (decorrelated), sự khác biệt màu sắc được khuếch đại.
        X_transformed = np.dot(X_centered, transform_matrix)

        # 9. Tái tạo và Chuẩn hóa (Min-Max Normalization)
        # Cộng lại trung bình (để ảnh không bị đen thui)
        X_final = X_transformed + mu


        # Ép giá trị về khoảng [0, 255] để hiển thị được trên màn hình
        _min, _max = np.min(X_final), np.max(X_final)
        if _max > _min:
            X_final = 255 * (X_final - _min) / (_max - _min)

        # Trả về kích thước ảnh ban đầu (H, W, C)
        return X_final.reshape(h, w, c).astype(np.uint8)

    def _extract_raw_mask_zca(self,stretched_image: np.ndarray) -> np.ndarray:
        """
        Tự động chọn kênh tốt nhất từ ảnh ZCA và tách mask.
        Logic: Kênh chứa chữ sẽ có độ tương phản (Standard Deviation) cao nhất.
        """
        if stretched_image is None: return None

        # 1. Tách 3 kênh màu (Blue, Green, Red)
        channels = cv2.split(stretched_image)

        best_channel = None
        max_std = -1

        # 2. Duyệt qua từng kênh để tìm "Ứng cử viên vô địch"
        for i, ch in enumerate(channels):
            # Tính độ lệch chuẩn (Standard Deviation) - Đo độ tương phản
            # Mẹo: Bỏ qua vùng viền ảnh để tránh nhiễu viền làm sai lệch
            h, w = ch.shape
            crop = ch[10:h - 10, 10:w - 10]
            std_val = np.std(crop)

            print(f"   [ZCA] Channel {i} Std Dev: {std_val:.2f}")

            if std_val > max_std:
                max_std = std_val
                best_channel = ch

        # 3. Xử lý kênh tốt nhất để lấy Mask nhị phân
        # Làm mờ nhẹ để nối liền các hạt mực đứt gãy
        blurred = cv2.GaussianBlur(best_channel, (3, 3), 0)

        # Dùng Otsu Threshold để máy tự quyết định ngưỡng
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Kiểm tra đảo ngược (Đảm bảo Chữ = Trắng, Nền = Đen)
        # Nếu số pixel trắng > 50% tổng số pixel -> Nghĩa là nền đang màu trắng -> Đảo ngược lại
        if cv2.countNonZero(mask) > (mask.size / 2):
            mask = cv2.bitwise_not(mask)
        # Lọc bỏ các đốm đen (vết bẩn) nhỏ hơn kích thước chữ
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        sizes = stats[1:, -1]  # Cột cuối cùng là diện tích (Area)
        nb_components = nb_components - 1

        min_size = 50  # Ngưỡng diện tích (tùy chỉnh theo ảnh)
        clean_mask = np.zeros((mask.shape), dtype=np.uint8)

        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                clean_mask[output == i + 1] = 255
        return clean_mask

    def extract_masks_zca(self,stretched_image: np.ndarray):
        """
        Trả về CẢ Mask Chữ (Text) và Mask Vết Ố (Stain).
        """
        if stretched_image is None: return None, None

        # 1. Tách kênh
        channels = cv2.split(stretched_image)

        # 2. Tìm kênh Chữ (Độ tương phản cao nhất - như cũ)
        std_devs = [np.std(ch) for ch in channels]
        text_idx = np.argmax(std_devs)
        text_channel = channels[text_idx]

        # 3. Tìm kênh Vết Ố (Stain Channel)
        # Mẹo: Vết ố thường nằm ở kênh có độ sáng trung bình cao thứ nhì,
        # hoặc đơn giản là kênh còn lại khác kênh chữ.
        # Ở đây ta dùng logic loại trừ: Lấy kênh có độ tương phản thấp hơn text nhưng ko phải noise
        stain_idx = (text_idx + 1) % 3  # Lấy kênh khác kênh text
        stain_channel = channels[stain_idx]

        # --- Xử lý Mask Chữ ---
        blurred_text = cv2.GaussianBlur(text_channel, (3, 3), 0)
        _, text_mask = cv2.threshold(blurred_text, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if cv2.countNonZero(text_mask) > (text_mask.size / 2):
            text_mask = cv2.bitwise_not(text_mask)

        # --- Xử lý Mask Vết Ố ---
        # Vết ố trên ZCA thường có màu xám/trung tính so với nền cực trị
        # Ta dùng Adaptive Threshold để bắt nó
        stain_mask = cv2.adaptiveThreshold(stain_channel, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY_INV, 45, 5)

        # Lọc nhiễu cho stain mask (chỉ lấy đốm to)
        kernel = np.ones((5, 5), np.uint8)
        stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_OPEN, kernel)

        # Mở rộng stain mask ra một chút để bao trùm vùng mép
        stain_mask = cv2.dilate(stain_mask, np.ones((15, 15), np.uint8))

        return text_mask, stain_mask
    def process(self, image):
        stretched = self._decorrelation_stretch(image)
        raw_mask = self._extract_raw_mask_zca(stretched)
        return raw_mask

