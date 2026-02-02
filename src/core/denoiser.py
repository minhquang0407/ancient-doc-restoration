import numpy as np
import cv2
from numpy.lib.stride_tricks import sliding_window_view

class ImageDenoiser:
    """
    Module khử nhiễu (Denoising) - Đã tối ưu hóa Vectorization.
    """

    # --- PHẦN 1: CÀI ĐẶT THỦ CÔNG (OPTIMIZED) ---

    def manual_median_filter(self, image: np.ndarray, ksize: int = 3) -> np.ndarray:
        """
        Bộ lọc Median thủ công nhưng dùng Numpy Strides để tăng tốc.
        Nhanh hơn vòng lặp for gấp 100 lần.
        """
        if len(image.shape) != 2:
            raise TypeError("Image must be grayscale")

        pad = ksize // 2
        # Padding biên (Reflect tốt hơn Edge cho ảnh tự nhiên)
        img_padded = np.pad(image, ((pad, pad), (pad, pad)), mode='reflect')

        # Tạo các cửa sổ trượt (Windows)
        # Shape output: (H, W, ksize, ksize)
        windows = sliding_window_view(img_padded, window_shape=(ksize, ksize))

        # Tính median trên trục cuối cùng (axis -1 và -2)
        # Median của từng cửa sổ
        output = np.median(windows, axis=(-2, -1))

        return output.astype(np.uint8)

    def create_gaussian_kernel(self, ksize: int, sigma: float) -> np.ndarray:
        """
        Tạo Gaussian Kernel (Giữ nguyên logic của bạn vì đã chuẩn).
        """
        if ksize % 2 == 0: ksize += 1
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        center = ksize // 2

        # Dùng mgrid để tránh vòng lặp for (tối ưu code gọn hơn)
        x, y = np.mgrid[-center:center+1, -center:center+1]
        exponent = -(x**2 + y**2) / (2 * sigma**2)
        kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(exponent)

        return kernel / kernel.sum()

    def manual_convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Tích chập 2D thủ công (Vectorized).
        """
        if len(image.shape) != 2: raise TypeError("Image must be grayscale")

        ksize = kernel.shape[0]
        pad = ksize // 2

        # Padding
        img_padded = np.pad(image, ((pad, pad), (pad, pad)), mode='reflect')

        # Tạo Windows: (H, W, K, K)
        windows = sliding_window_view(img_padded, window_shape=(ksize, ksize))

        # Nhân chập: (H, W, K, K) * (K, K) -> Sum over last 2 axes
        # Đây là bản chất của Convolution: Nhân tương ứng rồi cộng lại
        output = np.sum(windows * kernel, axis=(-2, -1))

        return np.clip(output, 0, 255).astype(np.uint8)

    def apply_gaussian(self, image: np.ndarray, ksize: int = 3, sigma: float = 1.0) -> np.ndarray:
        kernel = self.create_gaussian_kernel(ksize, sigma)
        return self.manual_convolution(image, kernel)

    # --- PHẦN 2: UTILS & MORPHOLOGY (ĐIỀU CHỈNH LOGIC) ---

    def clean_binary_noise(self, mask: np.ndarray, min_area: int = 20) -> np.ndarray:
        """Lọc nhiễu muối tiêu trên ảnh nhị phân (Giữ nguyên, rất tốt)."""
        if mask is None: return None
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean_mask[labels == i] = 255
        return clean_mask

    def safe_morphology_inpaint(self, mask: np.ndarray) -> np.ndarray:
        """
        Logic an toàn hơn cho chữ viết tay mảnh:
        Thay vì Erode trước (dễ mất nét), ta dùng Close để nối nét đứt trước.
        """
        # Kernel nhỏ để nối các điểm đứt gãy nhỏ
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Kernel lớn hơn chút để làm sạch
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 1. CLOSE: Nối các vết đứt nét (quan trọng nhất cho chữ viết tay)
        # Nối trước khi làm gì khác để bảo toàn cấu trúc
        connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_connect, iterations=2)

        # 2. OPEN: Chỉ dùng sau khi đã nối nét, để loại bỏ gai/nhiễu thừa ra ngoài
        cleaned = cv2.morphologyEx(connected, cv2.MORPH_OPEN, kernel_clean, iterations=1)

        return cleaned