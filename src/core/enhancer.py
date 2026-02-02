import numpy as np
import cv2

class ImageEnhancer:
  def division_norm(self, image: np.ndarray, kernel_size: tuple = (31, 31), sigma: float = 30) -> np.ndarray:
        if len(image.shape) == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img = image

        # 1. Ước lượng nền (Background estimation)
        smooth = cv2.GaussianBlur(img, kernel_size, sigma)

        # 2. Division: Ảnh gốc / Nền
        img_float = img.astype(np.float32)
        smooth_float = smooth.astype(np.float32)

        # Thêm 1.0 để tránh chia cho 0
        # Kết quả: Vùng nền sẽ xấp xỉ 1.0, vùng chữ sẽ < 1.0
        division = img_float / (smooth_float + 1.0)

        # 3. Rescale và Clip (SỬA Ở ĐÂY)
        # Thay vì normalize min-max, ta nhân thẳng với 255.
        # Lý do: Giá trị nền ~ 1.0 * 255 = 255 (Trắng).
        # Ta nhân thêm một chút gain (ví dụ 10-20) để đảm bảo nền trắng hẳn.
        division = division * 255 + 10

        # Cắt giá trị vượt quá 255 về 255 (Trắng tuyệt đối)
        division = np.clip(division, 0, 255)

        return division.astype(np.uint8)

  def remove_shadow(self, image: np.ndarray) -> np.ndarray:
      """
      Khử bóng đổ bằng phương pháp chia nền (Morphology based).
      Tốt cho việc loại bỏ các vùng tối cục bộ phức tạp.
      """
      if len(image.shape) == 3:
          gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      else:
          gray_image = image

      kernel_size = 51
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

      # Morphology Close bắt background tốt hơn với các văn bản dày đặc
      background_L = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)

      I_float = gray_image.astype(np.float32)
      L_float = background_L.astype(np.float32)

      R_float = I_float / (L_float + 1e-6)
      R_norm = np.clip(R_float, 0, 1)

      result_image = (R_norm * 255).astype(np.uint8)
      return result_image

  def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
      if len(image.shape) == 3:
          lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
          l, a, b = cv2.split(lab)
          clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
          cl = clahe.apply(l)
          result_image = cv2.cvtColor(cv2.merge([cl, a, b]), cv2.COLOR_LAB2BGR)
      else:
          clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
          result_image = clahe.apply(image)
      return result_image

  def unsharp_mask(self, image: np.ndarray, kernel_size: tuple = (5, 5), sigma: float = 1.0, amount: float = 1.5, threshold: int = 0) -> np.ndarray:
      float_image = image.astype(np.float32)
      blurred = cv2.GaussianBlur(float_image, kernel_size, sigma)
      mask = float_image - blurred
      if threshold > 0:
          mask[np.abs(mask) < threshold] = 0
      sharpened = float_image + mask * amount
      sharpened = np.clip(sharpened, 0, 255)
      return sharpened.astype(np.uint8)