import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.morphology import skeletonize
from scipy.interpolate import UnivariateSpline

# ======================================================
# 1. UTILS (Hàm hỗ trợ)
# ======================================================
def tensor_to_img_numpy(img_tensor):
    """Chuyển đổi Tensor/Array về định dạng ảnh Numpy chuẩn (H, W, C) hoặc (H, W)"""
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.detach().cpu().numpy()
    else:
        img = img_tensor.copy()

    # Bỏ batch dim
    if img.ndim == 4: img = img[0]

    # Chuyển (C, H, W) -> (H, W, C)
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 3 and img.shape[1] == 3: # Case lỗi lạ (512, 3, 512)
        img = np.transpose(img, (0, 2, 1))

    # Squeeze channel 1
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(2)

    # Chuẩn hóa về uint8 [0-255]
    if img.dtype != np.uint8:
        # Nếu là float 0-1
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    return img

# ======================================================
# 2. MODULE: SKELETON (Tạo khung xương)
# ======================================================
class SkeletonExtractor:
    def __init__(self):
        pass

    def extract(self, mask_input):
        """
        Input: Mask (Tensor hoặc Numpy)
        Output: Ảnh Skeleton (uint8, 0-255)
        """
        mask = tensor_to_img_numpy(mask_input)

        # Binary hóa
        mask_binary = (mask > 127).astype(np.uint8)

        # Skeletonize (scikit-image yêu cầu input 0-1 hoặc bool)
        skeleton = skeletonize(mask_binary)

        # Chuyển lại về uint8 để dùng với OpenCV
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)

        return skeleton_uint8

# ======================================================
# 3. MODULE: FIT (Tìm đường cong)
# ======================================================
class CurveFitter:
    def __init__(self, min_len=20, points_threshold=10):
        self.min_len = min_len
        self.points_threshold = points_threshold

    def fit(self, skeleton_img):
        """
        Input: Ảnh Skeleton
        Output: List các đường cong (Spline objects, x_min, x_max)
        """
        # Tìm contours
        contours, _ = cv2.findContours(skeleton_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_curves = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # --- BỘ LỌC RÁC ---
            if w < self.min_len: continue     # Quá ngắn
            if h > w: continue                # Dựng đứng (nhiễu dọc)
            if len(cnt) < self.points_threshold: continue

            # Lấy tọa độ
            points = cnt.squeeze()
            if points.ndim < 2: continue

            # Sắp xếp theo trục X
            points = points[points[:, 0].argsort()]
            x_coords = points[:, 0]
            y_coords = points[:, 1]

            # --- LỌC TRÙNG X (Bắt buộc cho Spline/Interpolation) ---
            _, unique_indices = np.unique(x_coords, return_index=True)
            x_unique = x_coords[unique_indices]
            y_unique = y_coords[unique_indices]

            if len(x_unique) < 5: continue

            try:
                # Dùng UnivariateSpline (Mượt hơn Polyfit)
                # k=3: Cubic Spline
                # s: Smoothing factor (càng lớn càng thẳng). Để None tự động hoặc set len(x)*hệ số
                spline = UnivariateSpline(x_unique, y_unique, k=3, s=len(x_unique)*2)

                x_min, x_max = np.min(x_unique), np.max(x_unique)
                valid_curves.append((spline, x_min, x_max))
            except Exception:
                pass

        return valid_curves

# ======================================================
# 4. MODULE: TPS (Biến đổi hình học)
# ======================================================
class TPSDewarper:
    def __init__(self, samples_per_line=60, regularization=0.8):
        self.samples = samples_per_line
        self.reg = regularization

    def dewarp(self, image, curves):
        """
        Input: Ảnh gốc (Màu/Xám), List các đường cong
        Output: Ảnh đã nắn thẳng
        """
        if not curves:
            print("Warning: No curves detected. Returning original image.")
            return image

        H, W = image.shape[:2]
        src_points = []
        dst_points = []

        # 1. Tạo điểm neo từ đường cong
        for spline, x_start, x_end in curves:
            x_samp = np.linspace(x_start, x_end, self.samples)
            y_samp = spline(x_samp)
            y_flat = np.mean(y_samp) # Kéo về đường trung bình phẳng

            for x, y in zip(x_samp, y_samp):
                if 0 <= x < W and 0 <= y < H:
                    src_points.append([x, y])
                    dst_points.append([x, y_flat])

        # 2. Tạo điểm neo biên (Giữ cố định khung ảnh)
        anchors = [
            [0,0], [W//2, 0], [W-1,0],         # Top
            [0,H-1], [W//2, H-1], [W-1,H-1],   # Bottom
            [0, H//2], [W-1, H//2]             # Mid Left/Right
        ]
        for pt in anchors:
            src_points.append(pt)
            dst_points.append(pt)

        # 3. Chuẩn bị dữ liệu cho OpenCV TPS
        src_arr = np.array(src_points, dtype=np.float32).reshape(1, -1, 2)
        dst_arr = np.array(dst_points, dtype=np.float32).reshape(1, -1, 2)

        # Chuẩn hóa về [0, 1] (QUAN TRỌNG)
        src_norm = src_arr.copy(); dst_norm = dst_arr.copy()
        src_norm[...,0] /= W; src_norm[...,1] /= H
        dst_norm[...,0] /= W; dst_norm[...,1] /= H

        # 4. Tính toán TPS
        matches = [cv2.DMatch(i, i, 0) for i in range(src_arr.shape[1])]
        tps = cv2.createThinPlateSplineShapeTransformer()
        tps.setRegularizationParameter(self.reg)

        try:
            tps.estimateTransformation(dst_norm, src_norm, matches)

            # Tạo lưới pixel
            grid_y, grid_x = np.mgrid[0:H, 0:W]
            grid_x = grid_x.astype(np.float32) / W
            grid_y = grid_y.astype(np.float32) / H
            pts_in = np.column_stack((grid_x.ravel(), grid_y.ravel())).reshape(1, -1, 2)

            # Biến đổi
            _, pts_out = tps.applyTransformation(pts_in)
            map_x = (pts_out[...,0] * W).reshape(H, W).astype(np.float32)
            map_y = (pts_out[...,1] * H).reshape(H, W).astype(np.float32)

            # Remap
            dewarped = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return dewarped

        except Exception as e:
            print(f"TPS Failed: {e}")
            return image
    def process(self, img_input, mask_input):
        skeletonizer = SkeletonExtractor()
        fitter = CurveFitter()
        dewarper = TPSDewarper(60, 0.01)

        # Bước 0: Chuẩn hóa ảnh đầu vào
        # Ảnh này dùng để nắn (nên là ảnh gốc RGB hoặc ảnh xám rõ nét)
        original_img = tensor_to_img_numpy(img_input)

        # Bước 1: Skeletonize
        print("1. Extracting Skeleton...")
        skeleton = skeletonizer.extract(mask_input)

        # Bước 2: Fit Curves
        print("2. Fitting Curves...")
        curves = fitter.fit(skeleton)
        print(f"   -> Detected {len(curves)} lines.")
        cv2.imwrite('skeleton.png', skeleton)
        # Bước 3: TPS Dewarp
        print("3. Applying TPS...")
        result = dewarper.dewarp(original_img, curves)

        return result
