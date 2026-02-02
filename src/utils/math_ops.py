import numpy as np

import cv2
import numpy as np


def calc_gradient_sobel(image: np.ndarray, ksize: int = 3):
    """
    Tính Gradient (Đạo hàm bậc nhất) của ảnh bằng toán tử Sobel.

    Args:
        image: Ảnh đầu vào (Nên là ảnh xám - Grayscale).
        ksize: Kích thước kernel (thường là 1, 3, 5, 7). Default = 3.

    Returns:
        magnitude (np.ndarray): Độ lớn của biên (Cường độ thay đổi màu).
        angle (np.ndarray): Góc hướng của biên (Đơn vị: Độ, 0-360).
    """
    # 1. Kiểm tra đầu vào
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Chuyển sang float64 để tránh tràn số (overflow) khi tính đạo hàm âm
    # Nếu để uint8, các giá trị âm (gradient chuyển từ sáng -> tối) sẽ bị gán về 0 -> Sai lệch.
    img_float = image.astype(np.float64)

    # 2. Tính đạo hàm theo phương X (Gx) - Phát hiện nét dọc
    # dx=1, dy=0
    gx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=ksize)

    # 3. Tính đạo hàm theo phương Y (Gy) - Phát hiện nét ngang
    # dx=0, dy=1
    gy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=ksize)

    # 4. Tính Magnitude và Angle
    # Cách thủ công:
    # magnitude = sqrt(gx^2 + gy^2)
    # angle = arctan(gy / gx)
    # Nhưng OpenCV có hàm tối ưu hơn là cartToPolar:

    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    return magnitude, angle
def bilinear_interpolation(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Nội suy song tuyến tính thủ công (Thay thế cv2.remap)."""
    pass