import numpy as np
import cv2

def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Tính Mean Squared Error (MSE)
    """
    # 1. Kiểm tra kích thước
    if img1.shape != img2.shape:
        raise ValueError(f"Không cùng kích thước: {img1.shape} vs {img2.shape}")

    # 2. Chuyển sang float để tính toán không bị tràn số
    # (Vì uint8 - uint8 có thể ra âm hoặc wrap-around)
    arr1 = img1.astype(np.float64)
    arr2 = img2.astype(np.float64)

    # 3. Tính hiệu bình phương
    diff = arr1 - arr2
    err = np.mean(diff ** 2)
    return float(err)

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Tính Peak Signal-to-Noise Ratio (PSNR).
    Đơn vị: dB.
    """
    # 1. Tính MSE trước
    mse = calculate_mse(img1, img2)

    # 2. Xử lý trường hợp suy biến (MSE = 0)
    if mse == 0:
        return 100.0

    # 3. Giá trị pixel tối đa (với ảnh 8-bit là 255)
    max_pixel = 255.0

    # 4. Công thức PSNR = 10 * log10(MAX^2/MSE) = 20 * log10(MAX / sqrt(MSE))
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return float(psnr)

def calculate_blur_score(image: np.ndarray) -> float:
    """
    Tính độ mờ của ảnh (Blur Score) bằng phương sai của Laplacian.
    """
    # 1. Chuyển ảnh màu sang ảnh xám (Grayscale)
    # Vì độ nét thường được tính trên cường độ sáng (luminance)
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # 2. Áp dụng bộ lọc Laplacian (Laplacian Filter)

    laplacian_result = cv2.Laplacian(gray_image, cv2.CV_64F)

    # 3. Tính phương sai (Variance) của ảnh biên
    # Phương sai càng cao -> Càng nhiều cạnh sắc nét -> Ảnh nét
    score = np.var(laplacian_result)

    return float(score)