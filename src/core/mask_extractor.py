import numpy as np
import cv2
import torch
class AIResizer:
    def __init__(self, target_size=(512, 512)):
        self.target_w, self.target_h = target_size

    def preprocess(self, image):
        """
        Bước 1: Resize ảnh gốc về 512x512 nhưng GIỮ TỶ LỆ (Letterbox).
        Trả về: Ảnh 512, tỉ lệ scale, và phần đệm (padding) đã thêm.
        """
        h, w = image.shape[:2]
        scale = min(self.target_w / w, self.target_h / h)
        nw, nh = int(w * scale), int(h * scale)

        # Resize ảnh nhỏ lại
        resized_img = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Tạo khung canvas đen 512x512
        canvas = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)

        # Tính toán padding để đặt ảnh vào giữa
        pad_w = (self.target_w - nw) // 2
        pad_h = (self.target_h - nh) // 2

        # Dán ảnh vào giữa canvas
        canvas[pad_h:pad_h+nh, pad_w:pad_w+nw] = resized_img

        meta = {
            "scale": scale,
            "pad": (pad_w, pad_h),
            "original_dim": (w, h),
            "new_dim": (nw, nh)
        }

        return canvas, meta

    def postprocess(self, mask_512, meta):
        """
        Bước 2: Phóng to Mask 512x512 về kích thước gốc.
        Loại bỏ padding và resize ngược lại.
        """
        pad_w, pad_h = meta["pad"]
        nw, nh = meta["new_dim"]
        orig_w, orig_h = meta["original_dim"]

        # 1. Cắt bỏ phần viền đen (Un-pad)
        # Lấy phần ảnh mask thực sự ở giữa
        mask_cropped = mask_512[pad_h:pad_h+nh, pad_w:pad_w+nw]

        # 2. Resize ngược lại về kích thước gốc
        # Quan trọng: Dùng INTER_NEAREST cho mask nhị phân để viền sắc nét, không bị mờ (anti-aliasing)
        mask_original = cv2.resize(mask_cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return mask_original

class MaskExtractor:
    def __init__(self):
        self.ai_resizer = AIResizer((512, 512))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def extract(self, image, model):
        img_512, meta = self.ai_resizer.preprocess(image)
        img = cv2.cvtColor(img_512, cv2.COLOR_BGR2GRAY)

        # 3. Preprocess
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = model(img_tensor)
            pred = torch.sigmoid(pred)
            mask = (pred > 0.5).float().cpu().numpy().squeeze() * 255
            mask = mask.astype(np.uint8)

        # 5. Lưu kết quả
        mask = self.ai_resizer.postprocess(mask, meta)
        if mask.max() <= 1:
            mask_bw = (mask * 255).astype(np.uint8)
        else:
            mask_bw = mask.astype(np.uint8)
        return mask_bw