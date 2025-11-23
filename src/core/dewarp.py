
class PageDewarper:
    def get_text_lines(self, binary_image):
        """
        Phát hiện các dòng văn bản cong.
        Dùng Morphology Dilate (kernel ngang dài) để nối chữ thành dòng.
        """
        pass

    def fit_polynomial(self, points, degree=3):
        """
        Hồi quy đa thức tìm phương trình đường cong y = f(x).
        Input: Các điểm trên dòng chữ.
        Output: Hệ số đa thức.
        """
        pass

    def generate_mesh(self, image_shape, top_curve, bottom_curve):
        """
        Tạo lưới tọa độ nguồn (Source Mesh) dựa trên các đường cong.
        """
        pass

    def dewarp(self, image):
        """
        Hàm chính thực hiện làm phẳng.
        Gọi các hàm con trên -> Tạo map_x, map_y -> Remap.
        """
        pass
