import numpy as np
class ForensicInk:
    def decorrelation_stretch(self, image):

        """
        Tách lớp mực phai bằng PCA (Principal Component Analysis).
        Input: Ảnh màu RGB.
        Output: Ảnh xám làm nổi bật mực.
        """
        if image is None:
            return None

        if len(image.shape) != 3:
            print("Warning: Decorrelation Stretch requires RGB image to separate ink layers.")
            return image
        h, w, c = image.shape
        # 1. Reshape ảnh thành mảng (N, 3)
        X = image.reshape(-1, 3).astype(np.float64)
        N = X.shape[0]
        # 2. Tính Covariance Matrix
        mu = np.sum(X, axis=0).astype(np.float64) / N
        X_centered = X - mu
        cov_matrix = np.dot(X_centered.T, X_centered) / (N - 1)
        # 3. Tính Eigenvalues & Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Chiếu dữ liệu và chuẩn hóa
        sigma = np.sqrt(eigenvalues)
        sigma = np.where(sigma < 1e-6, 1e-6, sigma)  # Tránh chia 0
        scaling_matrix = np.diag(1.0 / sigma)
        transform_matrix = np.dot(np.dot(eigenvectors, scaling_matrix), eigenvectors.T)
        X_transformed = np.dot(X_centered, transform_matrix)

        X_final = X_transformed + mu
        _min = np.min(X_final)
        _max = np.max(X_final)
        if _max - _min > 0:
            X_final = 255 * (X_final - _min) / (_max - _min)

        return X_final.reshape(h, w, c).astype(np.uint8)

    def inpaint_holes(image: np.ndarray, mask: np.ndarray, iterations: int = 5) -> np.ndarray:
        """
        Vá lỗ thủng thủ công (Diffusion based).
        Có thể chạy trên cả ảnh xám hoặc     ảnh màu.
        """
        if mask is None: return image

        # Xử lý cho cả ảnh xám (2D) và màu (3D)
        is_color = len(image.shape) == 3
        h, w = image.shape[:2]

        restored = image.copy().astype(np.float32)
        hole_pixels = np.argwhere(mask > 0)

        for _ in range(iterations):
            for y, x in hole_pixels:
                if y <= 0 or y >= h - 1 or x <= 0 or x >= w - 1: continue

                neighbors = []
                # Kiểm tra 4 hướng
                coords = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
                for ny, nx in coords:
                    if mask[ny, nx] == 0:  # Chỉ lấy pixel tốt
                        neighbors.append(restored[ny, nx])

                if len(neighbors) > 0:
                    # Tính trung bình cộng vector (nếu màu) hoặc scalar (nếu xám)
                    val = np.mean(neighbors, axis=0)
                    restored[y, x] = val

        return restored.astype(np.uint8)