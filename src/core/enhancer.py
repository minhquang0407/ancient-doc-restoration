import numpy as np
import cv2
class ImageEnhancer:
    def remove_shadow(self, image: np.ndarray) -> np.ndarray:
        """
        Khử bóng đổ bằng phương pháp chia nền (Background Division).

        Logic: Ảnh I = Phản xạ R * Chiếu sáng L. Ta ước lượng L và tính R = I / L
        1. Ước lượng nền L (ánh sáng) bằng Morphological Closing (kernel lớn)
        2. Chia ảnh gốc cho nền để lấy Phản xạ R.

        Args:
            image (np.ndarray): Ảnh đầu vào (thường là ảnh xám 8-bit).

        Returns:
            np.ndarray: Ảnh đã khử bóng (dạng 8-bit).
        """
        if len(image.shape) == 3:
            # Chuyển về ảnh xám nếu là ảnh màu để xử lý nền
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # 1. Ước lượng nền L bằng Morphological Closing
        # Kernel size lớn để ước lượng background (L) mượt mà, bỏ qua chi tiết chữ viết.
        kernel_size = 51  # Kích thước có thể tùy chỉnh
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Áp dụng Closing: dilation theo sau erosion. Nó giúp lấp đầy các vùng tối nhỏ (chữ)
        # và ước lượng nền sáng (L)
        background_L = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)

        # 2. Chia ảnh gốc cho nền: R = I / L. Cần chuyển sang float.
        # Thêm một epsilon nhỏ (1e-6) vào background để tránh chia cho 0.
        # Công thức: result = (image / background) * 255

        # Chuyển ảnh xám và nền sang kiểu float
        I_float = gray_image.astype(np.float32)
        L_float = background_L.astype(np.float32)

        # Thực hiện phép chia: Lấy phản xạ R
        # Phản xạ R sẽ nằm trong khoảng 0-1
        R_float = I_float / (L_float + 1e-6)

        # Chuẩn hóa về khoảng 0-255 và chuyển lại về kiểu 8-bit
        # Cắt giá trị trên 1.0 (vì R thường <= 1)
        R_norm = np.clip(R_float, 0, 1)

        result_image = (R_norm * 255).astype(np.uint8)

        # Có thể áp dụng lại Sauvola hoặc Binarization sau bước này để tăng cường độ rõ

        return result_image

    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
        """
        Cân bằng histogram thích nghi cục bộ (CLAHE)

        Args:
            image (np.ndarray): Ảnh đầu vào (xám hoặc màu BGR).
            clip_limit (float): Ngưỡng cắt histogram để tránh khuếch đại nhiễu.
            tile_grid_size (tuple): Kích thước lưới (m, n) chia ảnh.

        Returns:
            np.ndarray: Ảnh đã được cân bằng histogram.
        """

        # CLAHE hoạt động hiệu quả nhất trên không gian màu L (Luminosity)
        if len(image.shape) == 3:
            # Chuyển sang không gian màu LAB
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            # Tách kênh L (Luminosity/Độ sáng)
            l_channel, a_channel, b_channel = cv2.split(lab)

            # Khởi tạo CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

            # Áp dụng CLAHE cho kênh L
            cl = clahe.apply(l_channel)

            # Gộp lại kênh L đã xử lý với các kênh A và B
            merged_lab = cv2.merge([cl, a_channel, b_channel])

            # Chuyển lại về BGR
            result_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        else:
            # Xử lý cho ảnh xám
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            result_image = clahe.apply(image)

        return result_image

    def unsharp_mask(self, image: np.ndarray, kernel_size: tuple = (5, 5), sigma: float = 1.0, amount: float = 1.5,
                     threshold: int = 0) -> np.ndarray:
        """
        Làm nét ảnh (Sharpening) bằng Unsharp Masking
        Công thức: Output = Input + (Input - Blurred) * amount

        Args:
            image (np.ndarray): Ảnh đầu vào (xám hoặc màu BGR).
            kernel_size (tuple): Kích thước kernel Gaussian.
            sigma (float): Độ lệch chuẩn cho Gaussian Blur.
            amount (float): Độ lớn (cường độ) của hiệu ứng làm nét.
            threshold (int): Ngưỡng, chỉ áp dụng làm nét cho các biên có giá trị trên ngưỡng.

        Returns:
            np.ndarray: Ảnh đã được làm nét.
        """

        # Chuyển ảnh sang float để tính toán
        float_image = image.astype(np.float32)

        # 1. Làm mờ ảnh (Blurred)
        blurred = cv2.GaussianBlur(float_image, kernel_size, sigma)

        # 2. Tính Mask (phần chi tiết biên): Mask = Input - Blurred
        mask = float_image - blurred

        # 3. Áp dụng ngưỡng (threshold) để chỉ làm nét các cạnh rõ ràng (tùy chọn)
        if threshold > 0:
            mask[np.abs(mask) < threshold] = 0

        # 4. Tính Output: Output = Input + Mask * amount
        sharpened = float_image + mask * amount

        # Giới hạn giá trị trong khoảng 0-255
        sharpened = np.clip(sharpened, 0, 255)

        # Chuyển lại về kiểu 8-bit
        result_image = sharpened.astype(np.uint8)

        return result_image



        