# src/utils/io.py

import cv2
import numpy as np
import os


class IOManager:
    """
    Class quản lý việc Đọc/Ghi file, hỗ trợ Unicode và các định dạng đặc biệt.
    """

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """
        Đọc ảnh từ đường dẫn (Hỗ trợ Tiếng Việt/Unicode).
        Thay thế cho cv2.imread vốn hay lỗi với path có dấu.

        Args:
            path (str): Đường dẫn tới file ảnh.

        Returns:
            np.ndarray: Ảnh BGR (OpenCV format) hoặc None nếu lỗi.
        """
        try:
            with open(path, "rb") as f:
                file_bytes = bytearray(f.read())
                numpy_array = np.asarray(file_bytes, dtype=np.uint8)
                image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            print(f"[IO Error] Không thể đọc file {path}: {e}")
            return None

    @staticmethod
    def save_image(image: np.ndarray, path: str) -> bool:
        """
        Lưu ảnh ra đường dẫn (Hỗ trợ Tiếng Việt/Unicode).
        Thay thế cho cv2.imwrite.

        Args:
            image (np.ndarray): Ảnh cần lưu.
            path (str): Đường dẫn đích.

        Returns:
            bool: True nếu thành công.
        """
        try:
            # Tách đuôi file (ví dụ .jpg)
            ext = os.path.splitext(path)[1]

            # Encode ảnh sang định dạng mong muốn trong bộ nhớ
            success, buffer = cv2.imencode(ext, image)

            if success:
                # Ghi buffer ra file
                with open(path, mode='wb') as f:
                    buffer.tofile(f)
                return True
            return False
        except Exception as e:
            print(f"[IO Error] Không thể lưu file {path}: {e}")
            return False

    @staticmethod
    def save_text(text: str, path: str) -> bool:
        """
        Lưu chuỗi văn bản (kết quả OCR) ra file .txt.
        """
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            return True
        except Exception as e:
            print(f"[IO Error] Lỗi ghi text: {e}")
            return False

    @staticmethod
    def save_svg(paths_list: list, output_path: str, width: int, height: int):
        """
        Lưu danh sách các đường cong (Bezier) thành file SVG.
        Dùng cho module Vectorizer.

        Args:
            paths_list (list): List các chuỗi lệnh SVG path (ví dụ: "M 10 10 L 20 20...").
            output_path (str): Đường dẫn file .svg.
            width, height: Kích thước canvas.
        """
        svg_header = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        svg_footer = '</svg>'

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(svg_header + "\n")
                # Ghi từng đường path
                for p in paths_list:
                    # Style cơ bản: Fill đen, không stroke
                    line = f'<path d="{p}" fill="black" stroke="none"/>'
                    f.write(line + "\n")
                f.write(svg_footer)
            return True
        except Exception as e:
            print(f"[IO Error] Lỗi ghi SVG: {e}")
            return False


