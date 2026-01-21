import cv2
import numpy as np

def auto_crop(image_path, output_path):
    # 1. Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print("Không thể mở ảnh!")
        return

    # 2. Chuyển sang ảnh xám và làm mờ nhẹ để giảm nhiễu
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Phân ngưỡng (Thresholding) hoặc Canny Edge Detection
    # Dùng Otsu's thresholding để tự động tìm ngưỡng phân tách vật thể và nền
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Tìm các đường biên (Contours)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Tìm đường biên có diện tích lớn nhất (giả sử đó là nội dung chính)
        c = max(contours, key=cv2.contourArea)

        # Lấy tọa độ hình chữ nhật bao quanh đường biên đó
        x, y, w, h = cv2.boundingRect(c)

        # 5. Cắt ảnh (Crop)
        cropped_img = img[y:y + h, x:x + w]

        # 6. Lưu kết quả
        cv2.imwrite(output_path, cropped_img)
        print(f"Đã cắt và lưu ảnh tại: {output_path}")
    else:
        print("Không tìm thấy vùng nội dung nào để cắt.")