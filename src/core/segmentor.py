import numpy as np
import cv2 as cv
class DocumentSegmentor:
    def binarize_sauvola(self, image, window_size=25, k=0.2, R=128):
        """
        Nhị phân hóa thích nghi Sauvola.
        T = mean * (1 + k * (std / R - 1))
        Args:
            image (np.ndarray) : ảnh grayscale (8-bit) hoặc ảnh màu.
            window_size (int) : kích thước cửa sổ trượt (phải là số lẻ)
            k (float) : Hằng số điều chỉnh
            R (int) : Độ lệch chuẩn tối đa (128 cho anrh 8-bit)
        Returns: 
            np.ndarray : ảnh nhị phân (0-255)
        """
        # Tính Mean và Std cục bộ (có thể dùng integral image để tối ưu)
        # Áp dụng công thức

        # --- 1. Chuyển ảnh về grayscale nếu cần ---
        if image.ndim > 2:
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            image_gray = image.copy()
        # Ép kiểu float32 sớm để tránh mất mát số lẻ
        image_gray = image_gray.astype(np.float32)

        # --- 2. Đảm bảo window_size là số lẻ ---
        if window_size % 2 == 0:
            window_size += 1

        # --- 3. Tính Local Mean (m) ---
        mean = cv.boxFilter(image_gray, ddepth=-1,
                            ksize=(window_size, window_size),
                            normalize=True)
        
        # --- 4. Tính Local Standard Deviation (Std - s) ---
        image_sq = image_gray ** 2
        mean_sq = cv.boxFilter(image_sq, ddepth=-1,
                               ksize=(window_size, window_size),
                               normalize=True)
        # Đảm bảo phương sai không âm trước khi lấy căn
        variance = np.maximum(mean_sq - mean**2, 0)
        std = np.sqrt(variance)

        # --- 5. Tính ngưỡng Sauvola ---
        thresholes = mean * (1 + k * ((std/R) - 1.0))

        # --- 6. Giới hạn ngưỡng trong [0, 255] cho an toàn
        thresholes = np.clip(thresholes, 0, 255)

        # --- 7. Nhị phân hóa ---
        binary_image = np.where(image_gray < thresholes, 0, 255).astype(np.uint8)

        return binary_image