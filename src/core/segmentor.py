import cv2
import numpy as np
import matplotlib.pyplot as plt

class Segmentor:
    def binarization(self, image: np.ndarray, line_mask: np.ndarray, dilation_h: int = 40):
        """
        Dùng Mask dòng để hướng dẫn cắt chữ.

        Args:
            image: Ảnh gốc (Grayscale hoặc Màu).
            line_mask: Mask dòng từ AI (Nền đen, Dòng trắng).
            dilation_h: Độ mở rộng chiều cao của dòng (để bao trọn chữ h, g, y...).
                        Tùy vào cỡ chữ to hay nhỏ mà chỉnh số này (30-50 px).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # --- BƯỚC 1: TẠO "ĐƯỜNG ỐNG" AN TOÀN (ROI) ---
        # Mask dòng của AI thường mỏng, ta cần nở nó ra theo chiều dọc
        # để đảm bảo không cắt mất đầu chữ 'h' hay đuôi chữ 'g'.
        # Kernel (1, dilation_h): Chỉ nở theo chiều dọc, không nở chiều ngang
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, dilation_h))

        # Đảm bảo line_mask là binary chuẩn (0 và 255)
        _, line_mask_bin = cv2.threshold(line_mask, 127, 255, cv2.THRESH_BINARY)

        # Nở mask ra
        expanded_mask = cv2.dilate(line_mask_bin, kernel, iterations=1)

        # --- BƯỚC 2: NHỊ PHÂN HÓA "DỄ TÍNH" (SENSITIVE THRESHOLD) ---
        # Vì đã có mask bảo kê, ta dùng Adaptive Threshold với C thấp (hoặc âm)
        # để bắt trọn vẹn nét mờ nhất.
        # C=2 (Rất nhạy). Bình thường C=15 để lọc nhiễu, giờ C=2 để lấy hết nét.
        sensitive_binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 15
        )

        # --- BƯỚC 3: HỢP NHẤT (MASKING) ---
        # Logic:
        # - Pixel nào nằm TRONG mask mở rộng -> Giữ nguyên giá trị nhị phân.
        # - Pixel nào nằm NGOÀI mask -> Cho thành TRẮNG (255) tuyệt đối.

        final_result = np.ones_like(gray) * 255 # Tạo nền trắng tinh

        # Copy vùng chữ vào nền trắng
        # (expanded_mask == 255) trả về True ở chỗ có dòng
        final_result[expanded_mask == 255] = sensitive_binary[expanded_mask == 255]

        return final_result, expanded_mask

