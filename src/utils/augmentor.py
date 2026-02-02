import numpy as np
import cv2
import random
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

    def add_random_stamp(self, image):
        """
        Đóng con dấu giả (Màu đỏ/Xanh) lên ảnh.
        Lưu ý: Chỉ tác động lên Image, KHÔNG tác động lên Mask.
        """
        h, w = image.shape[:2]

        # 1. Tạo layer con dấu (để blend màu cho tự nhiên)
        overlay = image.copy()

        # 2. Random thông số
        # Vị trí ngẫu nhiên
        cx = random.randint(50, w - 50)
        cy = random.randint(50, h - 50)

        # Màu sắc (Ưu tiên màu Đỏ của dấu, hoặc Xanh của mực)
        # BGR format
        if random.random() < 0.8:
            color = (random.randint(0, 50), random.randint(0, 50), random.randint(150, 255))  # Đỏ
        else:
            color = (random.randint(150, 255), random.randint(0, 50), random.randint(0, 50))  # Xanh

        # Loại con dấu (0: Tròn, 1: Vuông, 2: Elip)
        stamp_type = random.choice([0, 1, 2])

        # Kích thước
        radius = random.randint(30, 60)

        # 3. Vẽ con dấu
        thickness = random.randint(2, 5)

        if stamp_type == 0:  # Tròn
            cv2.circle(overlay, (cx, cy), radius, color, thickness)
            # Vẽ thêm vòng tròn nhỏ bên trong
            cv2.circle(overlay, (cx, cy), radius - 10, color, 1)

        elif stamp_type == 1:  # Vuông (Dấu chức danh)
            x1, y1 = cx - radius, cy - radius // 2
            x2, y2 = cx + radius, cy + radius // 2
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        elif stamp_type == 2:  # Elip (Dấu công ty)
            axes = (radius, radius // 2 + random.randint(0, 10))
            angle = random.randint(0, 180)
            cv2.ellipse(overlay, (cx, cy), axes, angle, 0, 360, color, thickness)

        # 4. Thêm "Nội dung" nhăng cuội vào giữa con dấu
        # Vẽ vài nét gạch loằng ngoằng giả làm chữ ký
        for _ in range(3):
            pt1 = (cx + random.randint(-20, 20), cy + random.randint(-20, 20))
            pt2 = (cx + random.randint(-20, 20), cy + random.randint(-20, 20))
            cv2.line(overlay, pt1, pt2, color, 2)

        # 5. Blend màu (Trộn đè lên ảnh gốc)
        # alpha=0.6 nghĩa là con dấu hơi trong suốt, vẫn nhìn thấy chữ bên dưới
        alpha = random.uniform(0.5, 0.8)
        image_stamped = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        return image_stamped
    def add_rotation(self, image: np.ndarray, angle: float = None, border_color: int = 255) -> np.ndarray:
        """
        Xoay ảnh để giả lập đặt giấy bị lệch.

        Args:
            angle: Góc xoay (độ). Nếu None, sẽ random trong khoảng [-max_angle, max_angle].
            border_color: Màu viền thừa ra (255 là trắng - giống scan, 0 là đen - giống chụp đt).
        """
        h, w = image.shape[:2]

        # Nếu không chỉ định góc, random góc ngẫu nhiên
        if angle is None:
            angle = np.random.uniform(-self.max_rotation_angle, self.max_rotation_angle)

        # Tính tâm xoay
        center = (w // 2, h // 2)

        # Tạo ma trận xoay
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Xoay ảnh (Dùng borderValue để lấp đầy góc trống bằng màu giấy)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_color
        )

        return rotated, angle  # Trả về cả góc để lát còn so sánh (Ground Truth)

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

    def add_stains(self, image, num_stains=None):
        """
        Giả lập vết ố, nấm mốc (Phiên bản An toàn, hỗ trợ cả Xám và Màu).
        """
        h, w = image.shape[:2]

        # 1. Tạo layer chứa các vết ố (Toàn bộ là màu trắng 1.0)
        # Kiểu float32 để nhân ma trận
        stain_map = np.full((h, w), 1.0, dtype=np.float32)

        if num_stains is None:
            num_stains = random.randint(3, 10)

        for _ in range(num_stains):
            # Random vị trí và kích thước
            cx = random.randint(0, w)
            cy = random.randint(0, h)
            radius = random.randint(10, 60)

            # Random độ đậm của vết ố (Càng nhỏ càng tối)
            darkness = random.uniform(0.5, 0.9)

            # Thay vì tính toán cắt vùng (dễ lỗi), ta vẽ trực tiếp lên mask tạm
            # Dùng cv2.circle vẽ hình tròn đặc màu đen (giá trị darkness)
            # Lưu ý: cv2.circle cần tọa độ (int, int)
            cv2.circle(stain_map, (cx, cy), radius, darkness, -1)

        # 2. Làm nhòe stain_map để vết ố trông loang lổ tự nhiên (như nước thấm)
        # Kernel size lớn (ví dụ 51, 51) để biên cực mềm
        stain_map = cv2.GaussianBlur(stain_map, (51, 51), 0)

        # 3. Xử lý Broadcasting (Quan trọng để không lỗi Dimension)
        # Nếu ảnh đầu vào là 3 kênh (Màu), stain_map cần thêm trục thứ 3
        if len(image.shape) == 3:
            stain_map = stain_map[:, :, np.newaxis]

        # 4. Áp dụng lên ảnh gốc
        # Ảnh gốc (uint8) -> float -> nhân với stain -> uint8
        image_float = image.astype(np.float32)

        # Phép nhân này làm tối những vùng có vết ố
        output = image_float * stain_map

        return np.clip(output, 0, 255).astype(np.uint8)

        return np.clip(output, 0, 255).astype(np.uint8)
    def add_forensic_style(self, image):
        """
        Thêm hiệu ứng "Sách Cổ" chỉ cho ảnh (không thêm vào Mask).
        Mask phải sạch để AI học.
        """
        # 1. Đổi màu giấy (Vàng ố)
        # Chuyển sang BGR để tô màu
        color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Màu vàng ngẫu nhiên (R=Cao, G=Cao, B=Thấp)
        bg_color = np.array([random.randint(180, 220), random.randint(210, 240), random.randint(230, 255)]) # BGR

        # Blend màu giấy vào nền trắng
        # Những chỗ màu đen (chữ) giữ nguyên, chỗ trắng thành vàng
        normalized = image.astype(float) / 255.0
        for c in range(3):
            color_img[:, :, c] = (normalized * bg_color[c] + (1 - normalized) * 50).astype(np.uint8)

        # 2. Thêm nhiễu Gaussian
        noise = np.random.normal(0, 15, color_img.shape).astype(np.int16)
        color_img = np.clip(color_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 3. Thêm vết ố (Foxing) - Tùy chọn
        # (Bạn có thể copy hàm add_foxing_stains từ bài trước vào đây)

        return color_img
    def apply_synced_warp(self, image, mask):
        """
        [QUAN TRỌNG] Uốn cong cả ảnh và mask với cùng tham số.
        """
        h, w = image.shape[:2]

        # --- 1. BIẾN DẠNG HÌNH TRỤ (CYLINDER WARP) ---
        # Tạo lưới toạ độ (Map) một lần dùng cho cả 2
        y_dst, x_dst = np.indices((h, w))

        # Random tham số uốn
        mag = random.uniform(5.0, 20.0) # Độ cong
        phase = random.uniform(0, 2 * np.pi) # Vị trí đỉnh sóng
        freq = random.uniform(0.5, 1.5) # Tần số sóng

        omega = freq * 2 * np.pi / w
        offset_y = mag * np.sin(x_dst * omega + phase)

        y_src = (y_dst + offset_y).astype(np.float32)
        x_src = x_dst.astype(np.float32)

        # Áp dụng Remap (Biến dạng)
        # Image: Dùng nội suy Linear/Cubic cho đẹp
        warped_img = cv2.remap(image, x_src, y_src, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        # Mask: Dùng nội suy Nearest để giữ giá trị 0/255 dứt khoát (không bị mờ biên)
        warped_mask = cv2.remap(mask, x_src, y_src, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # --- 2. XOAY NGẪU NHIÊN (ROTATION) ---
        angle = random.uniform(-10, 10)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        rot_img = cv2.warpAffine(warped_img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
        rot_mask = cv2.warpAffine(warped_mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

        return rot_img, rot_mask
    def simulate_faded_ink(self, image: np.ndarray, fade_strength: float = 0.5) -> np.ndarray:
        """
        Làm phai màu mực.
        Input: Ảnh có nền trắng/vàng, chữ đen/xanh.
        fade_strength: 0.0 (giữ nguyên) -> 1.0 (mất hẳn chữ).
        """
        # 1. Chuyển sang không gian màu HSV để xử lý Saturation và Value
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)

        # 2. Phát hiện mực (vùng tối)
        # Giả định mực là những pixel có độ sáng (Value) thấp
        ink_mask = v < 150

        # 3. Làm phai (Tăng độ sáng V, Giảm độ bão hòa S)
        # Mực càng phai thì càng sáng lên (tiến về 255) và mất màu (S tiến về 0)
        v[ink_mask] = v[ink_mask] * (1 - fade_strength) + 255 * fade_strength
        s[ink_mask] = s[ink_mask] * (1 - fade_strength)

        # 4. Gộp lại
        hsv_faded = cv2.merge([h, s, v])
        img_faded = cv2.cvtColor(hsv_faded.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return img_faded

    def simulate_bleed_through(self, image: np.ndarray, opacity: float = 0.3) -> np.ndarray:
        """
        Giả lập mực thấm từ mặt sau.
        Cách làm: Lật ngược ảnh, làm mờ, rồi trộn đè lên ảnh gốc.
        """
        # 1. Tạo "mặt sau" giả: Lật ngang ảnh
        back_side = cv2.flip(image, 1)  # Flip Horizontal

        # 2. Làm mờ và giảm độ tương phản (để giống mực thấm)
        back_side = cv2.GaussianBlur(back_side, (5, 5), 0)

        # 3. Trộn (Blend)
        # Công thức: Result = Image * (1 - opacity) + BackSide * opacity
        # Nhưng mực thấm là "trừ đi" độ sáng (Subtractive), nên ta dùng phép nhân (Multiply)
        # Hoặc đơn giản là weighted sum

        # Cách trộn kiểu "thấm": Chỉ làm tối đi, không làm sáng lên
        # Chuyển về float [0, 1]
        img_f = image.astype(float) / 255.0
        back_f = back_side.astype(float) / 255.0

        # Giả lập thấm: Pixel thấm = Pixel gốc * (1 - opacity * (1 - Pixel_sau))
        # Tức là: Chỗ nào mặt sau ĐEN thì mặt trước bị tối đi. Chỗ nào mặt sau TRẮNG thì giữ nguyên.
        bleed_effect = 1.0 - (1.0 - back_f) * opacity

        result = img_f * bleed_effect

        return np.clip(result * 255, 0, 255).astype(np.uint8)

    def apply_heavy_augmentation(self, image, mask, augmentor):
        """
        Quy trình biến đổi tổng hợp: Hình học -> Chất liệu -> Camera.
        """
        h, w = image.shape[:2]

        # ==================================================
        # PHASE 1: HÌNH HỌC (GEOMETRY) -> Ảnh + Mask
        # ==================================================

        # 1. Cylinder Warp (Cong gáy sách) - Luôn luôn có hoặc xác suất cao
        if random.random() < 0.9:
            image, mask = self.apply_synced_warp(image, mask) # Hàm cũ đã viết

        # 2. Perspective (Phối cảnh) - Thỉnh thoảng chụp nghiêng
        if random.random() < 0.3:
            pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
            # Dời 4 góc đi một chút ngẫu nhiên
            shift = w * 0.1
            pts2 = np.float32([[random.uniform(0, shift), random.uniform(0, shift)],
                               [w - random.uniform(0, shift), random.uniform(0, shift)],
                               [random.uniform(0, shift), h - random.uniform(0, shift)],
                               [w - random.uniform(0, shift), h - random.uniform(0, shift)]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            image = cv2.warpPerspective(image, M, (w, h), borderValue=255)
            mask = cv2.warpPerspective(mask, M, (w, h), borderValue=0, flags=cv2.INTER_NEAREST)

        # ==================================================
        # PHASE 2: CHẤT LIỆU (TEXTURE) -> Chỉ Ảnh
        # ==================================================

        # 3. Bleed-through (Mực thấm) - Trước khi đổi màu giấy
        if random.random() < 0.4:
            image = augmentor.simulate_bleed_through(image, opacity=random.uniform(0.1, 0.3))

        if random.random() < 0.5:
            num = random.choice([random.randint(20, 50), random.randint(2, 5)])
            image = augmentor.add_stains(image, num_stains=num) # Hàm này cũng chạy ngon trên ảnh xám


        # 5. Stains (Vết ố/Mốc)
        if random.random() < 0.5:
            image = augmentor.add_stains(image, num_stains=random.randint(1, 5))

        # 6. Shadow (Bóng râm) - BẮT BUỘC để trị bóng
        if random.random() < 0.8:
            augmentor.shadow_amount = random.uniform(0.2, 0.7) # Random độ đậm
            image = augmentor.add_shadow(image)

        # ==================================================
        # PHASE 3: CAMERA & OPTICS -> Chỉ Ảnh
        # ==================================================

        # 7. Blur (Mờ) - Giả lập mất nét
        if random.random() < 0.3:
            k = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (k, k), 0)

        # 8. Noise (Nhiễu) - Giả lập ISO cao
        if random.random() < 0.5:
            augmentor.noise_std = random.uniform(5, 20)
            image = augmentor.add_noise_gaussian(image)

        image = self.add_forensic_style(image)
        # 9. Brightness/Contrast (Chỉnh sáng tối ngẫu nhiên)
        alpha = random.uniform(0.8, 1.2) # Contrast
        beta = random.uniform(-30, 30)   # Brightness
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        return image, mask