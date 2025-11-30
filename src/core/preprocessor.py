import cv2 as cv
import numpy as np
from typing import Optional
class Preprocessor:
    def __init__(self):
        pass

    def to_grayscale(self, image:np.ndarray, assume_rgb:bool=False) -> np.ndarray:
        """
        Chuyển ảnh sang thang xám chuẩn (Luma coding).
        Công thức: Y = 0.299R + 0.587G + 0.114B
        """
        if image is None:
            raise ValueError("Input image is None!")
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a np.ndarray!")
        if not isinstance(assume_rgb, bool):
            raise TypeError("assume_rgb must be a boolean value!")

        if image.ndim == 2:
            return image
        
        # Nếu shape (H, W, 1) -> squeeze về 2D
        if image.ndim == 3 and image.shape[2] == 1:
            return np.squeeze(image, axis=2)
        
        # Xử lý kênh alpha nếu có
        if image.ndim == 3 and image.shape[2] == 4:
            conversion_code = cv.COLOR_RGBA2GRAY if assume_rgb else cv.COLOR_BGRA2GRAY
            return cv.cvtColor(image, conversion_code)
        
        # 3 channels thường
        if image.ndim == 3 and image.shape[2] == 3:
            conversion_code = cv.COLOR_RGB2GRAY if assume_rgb else cv.COLOR_BGR2GRAY
            return cv.cvtColor(image, conversion_code)
        
        raise ValueError(f"Unsupported number of channels: {image.shape[2] if image.ndim == 3 else 'ndim=' + str(image.ndim)}")


    def resize_image(self, image:np.ndarray, 
                    target_width:Optional[int]=None, 
                    target_height:Optional[int]=None) -> np.ndarray:
        """
        Thay đổi kích thước ảnh, giữ nguyên tỷ lệ khung hình.
        Mục đích: Chuẩn hóa kích thước đầu vào để các thuật toán chạy ổn định.
        """
        if image is None:
            raise ValueError("Input image is None!")
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy ndarray!")
        
        h, w = image.shape[:2]
        
        if target_width is None and target_height is None:
            return image
        
        if target_width is not None and (not isinstance(target_width, int) 
                                         or target_width <= 0):
            raise ValueError("target_width must be a positive integer or None")
        
        if target_height is not None and (not isinstance(target_height, int)
                                          or target_height <= 0):
            raise ValueError("target_height must be a positive integer or None")
        
        # Nếu chỉ truyền width -> giữ tỷ lệ theo width
        if target_width is not None and target_height is not None:
            ratio = target_width / float(w)
            new_w = target_width
            new_h = max(1, int(round(h * ratio)))
        # Nếu chỉ truyền height -> giữ tỷ lệ theo height
        elif target_height is not None and target_width is None:
            ratio = target_height / float(h)
            new_h = target_height
            new_w = max(1, int(round(w * ratio)))
        else:
            # Cả hai chiều được truyền -> resize trực tiếp
            new_w, new_h = target_width, target_height
            # Tính ratio_w và ratio_h riêng, chọn ratio tổng quát để chọn nội suy:
            ratio_w = target_width / float(w)
            ratio_h = target_height / float(h)
            # Chọn ratio tổng quát để quyết interpolation:
            # nếu một chiều phóng to thì coi là phóng to -> dùng INTER_CUBIC
            # nếu cả hai < 1 -> thu nhỏ -> INTER_AREA
            ratio = max(ratio_w, ratio_h)
 
        # Quy tắc: INTER_AREA khi ratio < 1 (thu nhỏ), còn lại INTER_CUBIC
        inter = cv.INTER_AREA if ratio < 1.0 else cv.INTER_CUBIC

        return cv.resize(image, (int(new_w), int(new_h)), interpolation=inter)


    def compute_histogram(self, image:np.ndarray,
                          mask:Optional[np.ndarray]=None) -> np.ndarray:
        """
        Tính histogram của ảnh xám.
        Trả về mảng 256 phần tử đếm số lượng pixel cho mỗi mức xám.
        """
        # Có thể dùng cv2.calcHist hoặc np.histogram

        if image is None:
            raise ValueError("Input image is None!")
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy ndarray!")
        
        if image.ndim > 2:
            image = self.to_grayscale(image)

        # Loại NaN/inf trước
        if not np.isfinite(image).all():
            image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)

        # Nếu là float & max < 1.0 -> scale lên 0-255
        if np.issubdtype(image.dtype, np.floating):
            max_val = float(np.max(image)) if image.size > 0 else 0.0
            if max_val <= 1.0:
                image = (image * 255.0).round()
            # Nếu max_val > 1.0 -> giữ nguyên, (không áp NORM_MINMAX mặc định)
            image = np.clip(image, 0, 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            # Nếu integer nhưng không là uint8 -> chuyển về uint8 bằng clip
            image = np.clip(image, 0, 255).astype(np.uint8)

        # calcHist chấp nhận mask; mask phải là uint8, cùng kích thước HxW, giá trị 0 hoặc 255
        mask_proc = None
        if mask is not None:
            if not isinstance(mask, np.ndarray):
                raise TypeError("Mask must be a numpy ndarray or None!")
            if mask.dtype != np.uint8:
                # Chuẩn hóa về 0/255 uint8
                mask_proc = (mask > 0).astype(np.uint8) * 255
            else:
                mask_proc = mask
            if mask_proc.shape != image.shape[:2]:
                raise ValueError("Mask must have same HxW as image!")
            
        hist = cv.calcHist([image], [0], mask_proc, [256], [0, 256]).flatten()
        return hist