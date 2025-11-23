
class DocumentSegmentor:
    def binarize_sauvola(self, image, window_size=25, k=0.2, r=128):
        """
        Nhị phân hóa thích nghi Sauvola.
        T = mean * (1 + k * (std / r - 1))
        """
        # Tính Mean và Std cục bộ (có thể dùng integral image để tối ưu)
        # Áp dụng công thức
        pass