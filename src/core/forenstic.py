import numpy as np
import cv2


class ForensicInk:
    @staticmethod
    def decorrelation_stretch(image: np.ndarray) -> np.ndarray:
        """
        Kéo giãn màu sắc để tách mực phai ra khỏi nền.
        """
        if image is None or len(image.shape) != 3:
            return image

        h, w, c = image.shape
        X = image.reshape(-1, 3).astype(np.float64)
        N = X.shape[0]

        mu = np.sum(X, axis=0) / N
        X_centered = X - mu
        cov_matrix = np.dot(X_centered.T, X_centered) / (N - 1)

        # Dùng eigh cho nhanh và ổn định
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sigma = np.sqrt(eigenvalues)
        sigma = np.where(sigma < 1e-6, 1e-6, sigma)
        scaling_matrix = np.diag(1.0 / sigma)

        transform_matrix = np.dot(np.dot(eigenvectors, scaling_matrix), eigenvectors.T)
        X_transformed = np.dot(X_centered, transform_matrix)

        X_final = X_transformed + mu
        _min, _max = np.min(X_final), np.max(X_final)
        if _max > _min:
            X_final = 255 * (X_final - _min) / (_max - _min)

        return X_final.reshape(h, w, c).astype(np.uint8)

    @staticmethod
    def extract_ink_mask(stretched_image: np.ndarray) -> np.ndarray:
        """
        Hàm tự động tạo Mask Mực.
        """
        # 1. Chuyển sang LAB để lấy kênh 'b' (Vàng vs Xanh)
        lab = cv2.cvtColor(stretched_image, cv2.COLOR_RGB2LAB)
        _, _, b_channel = cv2.split(lab)

        # 2. Gaussian Blur để làm hạt mực dính vào nhau (Khắc phục lỗi rỗ)
        # Kernel (5,5) là đủ để nối các điểm đứt gãy nhỏ

        blurred = cv2.GaussianBlur(b_channel, (5, 5), 0)

        # 3. Otsu Threshold tự động
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Logic tự động đảo màu: Đảm bảo Mực luôn là màu Trắng (255)
        # Giả định nền giấy chiếm diện tích lớn hơn nét mực
        if cv2.countNonZero(mask) > (mask.size / 2):
            mask = cv2.bitwise_not(mask)

        # 5. Morphological Close để lấp đầy các khe hở bên trong nét mực
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return mask

