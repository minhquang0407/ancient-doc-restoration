import numpy as np


def calc_gradient_sobel(image: np.ndarray):
    """
    Tính Magnitude và Angle của Gradient sử dụng toán tử Sobel thủ công.
    Input: Ảnh xám (Grayscale) 2D numpy array.
    Output: (magnitude, angle)
    """
    # 1. Định nghĩa Kernel Sobel thủ công
    # Kx: Phát hiện biên dọc
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)

    # Ky: Phát hiện biên ngang
    Ky = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=np.float32)

    # 2. Thực hiện Convolution (Tích chập)
    # Lưu ý: Để tối ưu tốc độ trong Python thuần, ta có thể dùng mẹo slicing
    # hoặc hàm convolve2d của scipy, nhưng ở đây dùng logic trượt cửa sổ cơ bản
    # (pad ảnh để giữ kích thước).

    rows, cols = image.shape
    # Pad ảnh thêm 1 viền số 0 xung quanh để xử lý biên
    padded_img = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0).astype(np.float32)

    gx = np.zeros_like(image, dtype=np.float32)
    gy = np.zeros_like(image, dtype=np.float32)

    # Vectorized Sliding Window (Nhanh hơn vòng lặp for lồng nhau)
    # Lấy các vùng lân cận
    top_left = padded_img[:-2, :-2]
    top_center = padded_img[:-2, 1:-1]
    top_right = padded_img[:-2, 2:]

    mid_left = padded_img[1:-1, :-2]
    mid_right = padded_img[1:-1, 2:]

    bot_left = padded_img[2:, :-2]
    bot_center = padded_img[2:, 1:-1]
    bot_right = padded_img[2:, 2:]

    # Áp dụng công thức Convolution với Kernel Kx và Ky
    gx = (1 * top_right + 2 * mid_right + 1 * bot_right) - (1 * top_left + 2 * mid_left + 1 * bot_left)
    gy = (1 * bot_left + 2 * bot_center + 1 * bot_right) - (1 * top_left + 2 * top_center + 1 * top_right)

    # 3. Tính Magnitude và Angle
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # Chuyển về khoảng [0, 255]
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    # Tính góc (rad)
    angle = np.arctan2(gy, gx)

    return magnitude, angle


def bilinear_interpolation(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """
    Nội suy song tuyến tính thủ công (Thay thế cv2.remap).
    Dùng cho giai đoạn Dewarping để tính giá trị pixel từ lưới tọa độ thực.

    Args:
        image: Ảnh gốc (Input Source).
        map_x: Ma trận tọa độ X cần lấy mẫu (float).
        map_y: Ma trận tọa độ Y cần lấy mẫu (float).
    """
    H, W = image.shape[:2]

    # 1. Tìm toạ độ nguyên (Integer coordinates) của 4 điểm lân cận
    # (x0, y0) là góc trên trái, (x1, y1) là góc dưới phải
    x0 = np.floor(map_x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(map_y).astype(np.int32)
    y1 = y0 + 1

    # 2. Giới hạn toạ độ trong khung ảnh (Clipping)
    # Đảm bảo không truy cập ra ngoài mảng
    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)

    # 3. Lấy giá trị pixel tại 4 điểm lân cận (Ia, Ib, Ic, Id)
    # Ia: Top-Left, Ib: Top-Right, Ic: Bottom-Left, Id: Bottom-Right
    Ia = image[y0, x0]
    Ib = image[y0, x1]
    Ic = image[y1, x0]
    Id = image[y1, x1]

    # 4. Tính trọng số (khoảng cách thập phân)
    wa = (x1 - map_x) * (y1 - map_y)
    wb = (map_x - x0) * (y1 - map_y)
    wc = (x1 - map_x) * (map_y - y0)
    wd = (map_x - x0) * (map_y - y0)

    # Nếu ảnh là ảnh màu (3 kênh), cần mở rộng chiều trọng số để nhân
    if image.ndim == 3:
        wa = wa[..., np.newaxis]
        wb = wb[..., np.newaxis]
        wc = wc[..., np.newaxis]
        wd = wd[..., np.newaxis]

    # 5. Tính giá trị nội suy cuối cùng
    output = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return output.astype(np.uint8)