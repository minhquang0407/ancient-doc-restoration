import numpy as np


class DataAugmentor:
    def __init__(self,
                 # Noise Params
                 noise_mean: float = 0,
                 noise_std: float = 25,
                 sp_prob: float = 0.05,
                 salt_ratio: float = 0.5,
                 # Shadow Params
                 shadow_amount: float = 0.5,  # 0.0 (đen) -> 1.0 (không đổi)
                 # Rotation Params
                 max_rotation_angle: int = 15,
                 # Cylinder Warp Params
                 cylinder_mag: float = 10.0):

        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.sp_prob = sp_prob
        self.salt_ratio = salt_ratio
        self.shadow_amount = shadow_amount
        self.max_rotation_angle = max_rotation_angle
        self.cylinder_mag = cylinder_mag

    def add_noise_gaussian(self, image: np.ndarray) -> np.ndarray:
        """Thêm nhiễu Gaussian (Additive Noise)"""
        # Chuyển sang float để tính toán
        img_float = image.astype(np.float32)

        # Tạo nhiễu
        noise = np.random.normal(self.noise_mean, self.noise_std, image.shape)

        # Cộng nhiễu và clip giá trị
        noisy_img = np.clip(img_float + noise, 0, 255)

        return noisy_img.astype(np.uint8)

    def add_noise_sp(self, image: np.ndarray) -> np.ndarray:
        """Thêm nhiễu Muối Tiêu (Impulse Noise)"""
        output = image.copy()

        # Tạo ma trận xác suất ngẫu nhiên
        probs = np.random.random(output.shape[:2])  # Chỉ cần shape HxW

        # Nếu ảnh màu (3 kênh), ta cần mở rộng dimension của probs để mask áp dụng lên cả 3 kênh
        if output.ndim == 3:
            probs = np.expand_dims(probs, axis=-1)

        # Salt (Trắng)
        output[probs < (self.sp_prob * self.salt_ratio)] = 255

        # Pepper (Đen)
        # Ngưỡng dưới cho pepper: 1 - prob * (1-salt)
        pepper_thresh = 1 - self.sp_prob * (1 - self.salt_ratio)
        output[probs > pepper_thresh] = 0

        return output

    def add_shadow(self, image: np.ndarray) -> np.ndarray:
        """Tạo bóng râm tuyến tính ngẫu nhiên"""
        h, w = image.shape[:2]

        # Tạo lưới toạ độ
        y_grid, x_grid = np.indices((h, w))

        # Chọn đường thẳng ngẫu nhiên cắt qua ảnh
        x1, y1 = np.random.randint(0, w), 0
        x2, y2 = np.random.randint(0, w), h

        # Tính phương trình đường thẳng (Cross product 2D)
        # > 0 là một bên, < 0 là bên kia
        mask = (x_grid - x1) * (y2 - y1) - (y_grid - y1) * (x2 - x1)

        # Chọn ngẫu nhiên 1 bên để làm tối
        is_upper = np.random.choice([True, False])
        shadow_mask = mask > 0 if is_upper else mask < 0

        # Xử lý channel dimension cho ảnh màu
        if image.ndim == 3:
            shadow_mask = shadow_mask[:, :, np.newaxis]

        # Áp dụng bóng
        img_float = image.astype(np.float32)
        # Giảm độ sáng vùng có bóng
        img_float[shadow_mask] *= self.shadow_amount

        return np.clip(img_float, 0, 255).astype(np.uint8)

    def add_rotation(self, image: np.ndarray) -> np.ndarray:
        """Xoay ảnh dùng Nearest Neighbor Interpolation"""
        h, w = image.shape[:2]
        angle = np.random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
        rad = np.deg2rad(angle)

        # Tính sin, cos
        c, s = np.cos(rad), np.sin(rad)
        cx, cy = w // 2, h // 2  # Tâm xoay

        # Tạo lưới toạ độ đích (Destination Grid)
        y_dst, x_dst = np.indices((h, w))

        # Inverse Mapping: Tìm toạ độ nguồn (Source) từ toạ độ đích
        # Công thức xoay ngược:
        # x_src = (x-cx)cos + (y-cy)sin + cx
        # y_src = -(x-cx)sin + (y-cy)cos + cy
        x_src = (x_dst - cx) * c + (y_dst - cy) * s + cx
        y_src = -(x_dst - cx) * s + (y_dst - cy) * c + cy

        # Làm tròn về index gần nhất (Nearest Neighbor)
        x_src = np.round(x_src).astype(int)
        y_src = np.round(y_src).astype(int)

        # Tạo mask các pixel hợp lệ (nằm trong ảnh gốc)
        valid_mask = (x_src >= 0) & (x_src < w) & (y_src >= 0) & (y_src < h)

        output = np.zeros_like(image)

        # Map giá trị pixel
        # Numpy broadcasting tự động xử lý kênh màu nếu image là 3D
        output[y_dst[valid_mask], x_dst[valid_mask]] = image[y_src[valid_mask], x_src[valid_mask]]

        return output

    def warp_cylinder(self, image: np.ndarray) -> np.ndarray:
        """Giả lập độ cong trang sách (Vertical Cylinder Warp)"""
        h, w = image.shape[:2]

        # Tạo lưới toạ độ
        y_dst, x_dst = np.indices((h, w))

        # Tính toán biến dạng (cong theo trục y dựa trên vị trí x)
        # Omega: tần số sóng (1 chu kỳ trên chiều rộng ảnh)
        omega = 2 * np.pi / w

        # Offset y: pixel bị dịch chuyển lên/xuống theo hình sin
        offset_y = self.cylinder_mag * np.sin(x_dst * omega)

        # Toạ độ nguồn
        y_src = (y_dst + offset_y).astype(int)
        x_src = x_dst  # X giữ nguyên

        # Mask hợp lệ
        valid_mask = (y_src >= 0) & (y_src < h)

        output = np.zeros_like(image)

        # Map giá trị
        output[y_dst[valid_mask], x_dst[valid_mask]] = image[y_src[valid_mask], x_src[valid_mask]]

        return output


# --- Ví dụ sử dụng ---
if __name__ == "__main__":
    # 1. Khởi tạo Augmentor
    aug = DataAugmentor(
        noise_std=30,
        sp_prob=0.1,
        shadow_amount=0.4,  # Bóng khá đậm
        max_rotation_angle=20,
        cylinder_mag=15.0
    )

    # 2. Tạo ảnh giả lập (100x100 pixels, 3 channels)
    # Tạo ảnh caro để dễ nhìn thấy biến dạng
    fake_img = np.zeros((200, 200, 3), dtype=np.uint8)
    fake_img[::20, :] = 255  # Kẻ sọc ngang
    fake_img[:, ::20] = 255  # Kẻ sọc dọc

    # 3. Chạy thử các hàm
    res_gauss = aug.add_noise_gaussian(fake_img)
    res_sp = aug.add_noise_sp(fake_img)
    res_shadow = aug.add_shadow(fake_img)
    res_rot = aug.add_rotation(fake_img)
    res_warp = aug.warp_cylinder(fake_img)

    print("Gaussian shape:", res_gauss.shape)
    print("Shadow shape:", res_shadow.shape)
    print("Rotation shape:", res_rot.shape)
    # Nếu muốn xem ảnh, bạn có thể lưu ra file hoặc dùng matplotlib