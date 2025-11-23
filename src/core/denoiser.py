class ImageDenoiser:
    def manual_median_filter(self, image: np.ndarray, ksize: int = 3) -> np.ndarray:
        """Tự cài đặt bộ lọc Median (Sắp xếp mảng)."""
        pass

    def create_gaussian_kernel(self, ksize: int, sigma: float) -> np.ndarray:
        """Tự tạo ma trận Gaussian Kernel."""
        pass

    def manual_convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Tự thực hiện tích chập 2D."""
        pass

    def apply_gaussian(self, image: np.ndarray, ksize: int = 3, sigma: float = 1.0) -> np.ndarray:
        """Wrapper gọi hàm convolution hoặc GaussianBlur."""
        pass

    def remove_bleed_through(self, image, mask=None):
        """
        Khử mực thấm mặt sau.
        Logic: Mực thấm thường nhạt hơn mực chính.
        Dùng threshold phụ để loại bỏ các pixel xám nhạt.
        """
        pass

    def inpaint_holes(self, image, mask):
        """
        Vá lỗ thủng/vết rách.
        Dùng cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        """
        pass