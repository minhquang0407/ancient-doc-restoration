import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.interpolate import Rbf


def convert_mask_to_points(mask_path):
    """
    Bước 1: Chuyển đổi Mask AI thành tập hợp các vùng điểm (contours).
    Xử lý lỗi mask giá trị thấp (0-1) thành chuẩn (0-255).
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    # Fix: Nếu mask là dạng class index (0-1), scale lên 255
    if np.max(mask) <= 1:
        mask = mask * 255

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Lấy tất cả các contour (đại diện cho các dòng chữ)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lọc bỏ nhiễu (các chấm quá nhỏ)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 50]

    return binary, valid_contours


def get_skeleton(binary_mask, contours):
    """
    Bước 2: 'get_skeleton'
    Từ tập điểm các dòng -> Tìm xương sống (skeleton) và tính điểm điều khiển (src -> dst).
    Bao gồm cả việc tạo khung bao ảo (virtual box) để neo 4 góc.
    """
    if not contours:
        return None, None, None

    src_points = []
    dst_points = []

    # --- Phần A: Tạo điểm Neo (Anchors) từ khung bao ảo ---
    # Gộp tất cả điểm lại để tìm hình chữ nhật bao quanh toàn bộ văn bản
    all_points = np.vstack([c.reshape(-1, 2) for c in contours])
    rect = cv2.minAreaRect(all_points)
    box = np.int0(cv2.boxPoints(rect))

    # Sắp xếp 4 góc: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    def order_corners(pts):
        res = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        res[0] = pts[np.argmin(s)]  # TL
        res[2] = pts[np.argmax(s)]  # BR
        diff = np.diff(pts, axis=1)
        res[1] = pts[np.argmin(diff)]  # TR
        res[3] = pts[np.argmax(diff)]  # BL
        return res

    src_corners = order_corners(box)

    # Tính kích thước ảnh phẳng đầu ra
    (tl, tr, br, bl) = src_corners
    width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))

    dst_corners = np.array([
        [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]
    ], dtype="float32")

    # Thêm 4 góc và trung điểm cạnh vào list điểm điều khiển
    for i in range(4):
        src_points.append(src_corners[i])
        dst_points.append(dst_corners[i])
        # Thêm trung điểm cạnh
        next_i = (i + 1) % 4
        src_points.append((src_corners[i] + src_corners[next_i]) / 2)
        dst_points.append((dst_corners[i] + dst_corners[next_i]) / 2)

    M = cv2.getPerspectiveTransform(src_corners, dst_corners)

    # --- Phần B: Lấy Skeleton từng dòng ---
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = binary_mask[y:y + h, x:x + w]

        # Tạo skeleton (trục giữa dòng)
        skeleton = skeletonize(roi // 255)
        y_loc, x_loc = np.where(skeleton > 0)

        if len(x_loc) < 10: continue

        global_pts = np.column_stack((x_loc + x, y_loc + y))

        # Sampling (lấy mẫu điểm thưa ra để giảm tải)
        indices = np.linspace(0, len(global_pts) - 1, min(8, len(global_pts)), dtype=int)
        sampled = global_pts[indices]

        # Map sang hệ tọa độ phẳng để tính Y đích
        pts_array = np.array([sampled], dtype='float32')
        mapped = cv2.perspectiveTransform(pts_array, M)[0]
        avg_y = np.mean(mapped[:, 1])

        for i, pt_src in enumerate(sampled):
            src_points.append(pt_src)
            dst_points.append([mapped[i][0], avg_y])  # Ép thẳng hàng

    return np.array(src_points), np.array(dst_points), (width, height)


def tpa_dewarp(img_path, src_points, dst_points, out_size):
    """
    Bước 3: TPA (Text Point Alignment / TPS Warping)
    Biến đổi ảnh gốc dựa trên các điểm kiểm soát.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    try:
        # Sử dụng Thin Plate Spline
        tps_x = Rbf(dst_points[:, 0], dst_points[:, 1], src_points[:, 0], function='thin_plate', smooth=0.5)
        tps_y = Rbf(dst_points[:, 0], dst_points[:, 1], src_points[:, 1], function='thin_plate', smooth=0.5)

        grid_y, grid_x = np.mgrid[0:out_size[1], 0:out_size[0]]

        map_x = tps_x(grid_x, grid_y).astype(np.float32)
        map_y = tps_y(grid_x, grid_y).astype(np.float32)

        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    except Exception as e:
        print(f"TPA Error: {e}")
        return img


# Main Flow theo đúng yêu cầu
if __name__ == "__main__":
    mask_file = '1_19_4-ec_Page_244-qut0001_mask.png'
    real_file = '1_19_4-ec_Page_244-qut0001.png'

    print("Running Dewarping Pipeline...")

    # 1. Mask -> Points
    binary, contours = convert_mask_to_points(mask_file)

    # 2. Points -> Skeleton
    src, dst, size = get_skeleton(binary, contours)

    if src is not None and len(src) > 0:
        # 3. Skeleton -> TPA
        result = tpa_dewarp(real_file, src, dst, size)

        # Hiển thị
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(cv2.imread(real_file), cv2.COLOR_BGR2RGB))
        plt.scatter(src[:, 0], src[:, 1], c='r', s=1)
        plt.title("Detected Skeleton & Anchors")

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title("Final Dewarped Result")
        plt.show()
    else:
        print("Failed: No text lines detected.")