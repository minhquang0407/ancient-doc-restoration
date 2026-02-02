import cv2
import numpy as np


class SmartRouter:
    def __init__(self):
        # Ngưỡng bão hòa màu: Nếu > ngưỡng này -> Giấy màu/ố vàng -> Dùng ZCA
        self.SATURATION_THRESHOLD = 20.0

        # Ngưỡng độ lệch pha kênh màu: Nếu chênh lệch R-B cao -> Sepia -> Dùng ZCA
        self.CHANNEL_DIFF_THRESHOLD = 15.0

        # Ngưỡng tương phản: Nếu tương phản quá thấp (mực mờ) -> Dùng ZCA
        self.CONTRAST_THRESHOLD = 30.0

    def analyze(self, image: np.ndarray) -> dict:
        """
        Phân tích ảnh để quyết định phương pháp xử lý.
        Trả về dict chứa quyết định và các chỉ số.
        """
        result = {
            "mode": "standard",  # Mặc định là ca dễ
            "reason": [],
            "metrics": {}
        }

        if image is None:
            return result

        # 1. Kiểm tra ảnh xám đầu vào
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            result["mode"] = "standard"
            result["reason"].append("Input is grayscale")
            return result

        # 2. Phân tích không gian màu HSV (Saturation)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        mean_sat = np.mean(sat)
        result["metrics"]["mean_saturation"] = mean_sat

        # 3. Phân tích độ lệch kênh (Channel Divergence)
        # Giúp phát hiện ảnh Sepia hoặc giấy ố vàng đậm
        b, g, r = cv2.split(image)
        diff_rb = np.abs(r.astype(float) - b.astype(float))
        mean_diff = np.mean(diff_rb)
        result["metrics"]["channel_diff"] = mean_diff

        # 4. Phân tích độ tương phản (Global Contrast)
        # Chuyển xám để tính độ lệch chuẩn
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        result["metrics"]["contrast"] = contrast

        # --- LOGIC QUYẾT ĐỊNH ---

        # Case 1: Giấy quá ố vàng hoặc màu mè (Saturation cao)
        if mean_sat > self.SATURATION_THRESHOLD:
            result["mode"] = "zca"
            result["reason"].append(f"High saturation ({mean_sat:.1f} > {self.SATURATION_THRESHOLD})")

        # Case 2: Ảnh Sepia/Mực màu (Kênh R và B lệch nhau nhiều)
        elif mean_diff > self.CHANNEL_DIFF_THRESHOLD:
            result["mode"] = "zca"
            result["reason"].append(f"High channel diff ({mean_diff:.1f})")

        # Case 3: Mực quá mờ (Tương phản thấp)
        # ZCA giỏi lôi đầu mực mờ lên hơn là Division Norm
        elif contrast < self.CONTRAST_THRESHOLD:
            result["mode"] = "zca"
            result["reason"].append(f"Low contrast/Faded ink ({contrast:.1f})")

        else:
            result["mode"] = "standard"
            result["reason"].append("Standard document detected")

        print(f"   [SmartRouter] Decision: {result['mode'].upper()} | Reasons: {', '.join(result['reason'])}")
        return result