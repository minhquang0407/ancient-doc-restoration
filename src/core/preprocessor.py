import cv2 as cv
import numpy as np
class Preprocessor:
    def __init__(self):
        pass

    def to_grayscale(self, image:np.ndarray, assume_rgb=False):
        """
        Chuyển ảnh sang thang xám chuẩn (Luma coding).
        Công thức: Y = 0.299R + 0.587G + 0.114B
        """
        if image is None:
            raise ValueError("Input image is None!")
        
        if image.ndim == 2:
            return image
        
        img = cv.COLOR_RGB2GRAY if assume_rgb else cv.COLOR_BGR2GRAY

        return cv.cvtColor(image, img)


    def resize_image(self, image:np.ndarray, target_width=None):
        """
        Thay đổi kích thước ảnh, giữ nguyên tỷ lệ khung hình.
        Mục đích: Chuẩn hóa kích thước đầu vào để các thuật toán chạy ổn định.
        """
        if image is None:
            raise ValueError("Input image is None!")
        
        if target_width is None:
            return image
        
        h, w = image.shape[:2]
        ratio = target_width/w
        target_height = int(h * ratio)

        # Chọn nội suy 
        inter = cv.INTER_AREA if target_width < w else cv.INTER_CUBIC

        return cv.resize(image, (target_width, target_height), interpolation=inter)


    def compute_histogram(self, image:np.ndarray):
        """
        Tính histogram của ảnh xám.
        Trả về mảng 256 phần tử đếm số lượng pixel cho mỗi mức xám.
        """
        # Có thể dùng cv2.calcHist hoặc np.histogram
        if image is None:
            raise ValueError("Input image is None!")
        
        if image.ndim > 2:
            image = self.to_grayscale(image)

        # Chuẩn hóa dữ liệu
        if image.dtype != np.uint8:
            image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX).astype(np.unit8)

        hist = cv.calcHist([image], [0], None, [256], [0, 256]).flatten()
        return hist