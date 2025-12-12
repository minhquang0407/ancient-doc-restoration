import numpy as np

class ImageDenoiser:
    def manual_median_filter(self, image: np.ndarray, ksize: int = 3) -> np.ndarray:
        """Tự cài đặt bộ lọc Median (Sắp xếp mảng)."""
        if len(image.shape) != 2:
            raise TypeError("Image must be grayscale")

        h, w = image.shape
        output = np.zeros_like(image)
        pad = ksize // 2
        img_padded = np.pad(image, ((pad, pad), (pad, pad)), mode='edge')


        for i in range(h):
            for j in range(w):
                window = img_padded[i:i + ksize, j:j + ksize]

                sorted_pixels = np.sort(window)

                output[i, j] = np.median(sorted_pixels)

        return output

    def create_gaussian_kernel(self, ksize: int, sigma: float) -> np.ndarray:
        """
        Tự tạo ma trận Gaussian Kernel.
        G(x,y) = (1 / 2*pi*sigma^2) * exp(-(x^2 + y^2) / 2*sigma^2)
        Theo G ở trên thì G đạt max tại (0,0) và đó phải là tâm kernel.
        Nhưng vì theo lập trình thì tâm ở (ksize//2,    ksize//2).
        Do đó phải chuẩn hóa lại tọa độ để tính.
        """
        kernel = np.zeros((ksize,ksize), dtype= np.float32)
        center = ksize // 2
        sum_val = 0.0
        for x in range(ksize):
            for y in range(ksize):
                # Chuẩn hóa tọa độ.
                rel_x = x- center
                rel_y = y- center

                exponent = -(rel_x**2 + rel_y**2) / (2 * sigma**2)

                val = (1 / (2 * np.pi * sigma**2)) * np.exp(exponent)

                kernel[x, y] = val

                sum_val += val

        return kernel / sum_val

    def manual_convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Tự thực hiện tích chập 2D."""
        if len(image.shape) != 2:
            raise TypeError("Image must be grayscale")

        ksize = kernel.shape[0]
        pad = ksize // 2
        h, w = image.shape

        img_padded = np.pad(image, ((pad, pad), (pad, pad)), mode='constant')

        output = np.zeros_like(image, dtype=np.float32)
        for i in range(h):
            for j in range(w):
                region = img_padded[i:i + ksize, j:j + ksize]

                value = np.sum(region * kernel)

                output[i, j] = value

        return np.clip(output, 0, 255).astype(np.uint8)

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

    def clean_binary_noise(self, mask: np.ndarray, min_area: int = 20) -> np.ndarray:
        """
        Dọn rác nhiễu (mực thấm, đốm bẩn) dựa trên diện tích.
        """
        if mask is None: return None
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean_mask[labels == i] = 255
        return clean_mask

    def morphology_inpaint(self, mask: np.ndarray, close_iter: int = 2) -> np.ndarray:
        """
        Tách chữ ra trước -> Vá lỗ mạnh tay -> Trả lại độ dày.
        """
        # Kernel cơ bản 3x3
        kernel = np.ones((3, 3), np.uint8)

        # 1: ERODE (Co chữ lại)
        eroded = cv2.erode(mask, kernel, iterations=1)

        # 2: CLOSE (Vá lỗ mạnh tay)
        restored = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

        # 3: DILATE (Phình ra lại)
        final_result = cv2.dilate(restored, kernel, iterations=1)

        # 4: OPEN (Dọn gai lần cuối)
        final_result = cv2.morphologyEx(final_result, cv2.MORPH_OPEN, kernel, iterations=1)

        return final_result

    def remove_bleed_through(mask: np.ndarray, min_area: int = 10) -> np.ndarray:
        """
        Khử nhiễu thấm và rác dựa trên diện tích (CC Analysis).

        Input: mask (Ảnh nhị phân: Mực=255, Nền=0)
        min_area: Ngưỡng diện tích. Những vết nhỏ hơn số pixel này sẽ bị xóa.
        """
        if mask is None: return None

        # 1. Tìm tất cả các vùng liên thông (Connected Components)
        # num_labels: Số lượng vùng tìm thấy
        # labels: Bản đồ nhãn
        # stats: Thông tin [x, y, width, height, AREA] của từng vùng
        # centroids: Tâm của từng vùng
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Tạo mask kết quả sạch sẽ
        clean_mask = np.zeros_like(mask)

        # 2. Duyệt qua từng vùng (Bỏ qua nhãn 0 là background)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            # 3. Logic: Chỉ giữ lại những vùng đủ lớn (là nét chữ)
            # Vết thấm/Nhiễu thường là các đốm nhỏ rời rạc
            if area >= min_area:
                clean_mask[labels == i] = 255