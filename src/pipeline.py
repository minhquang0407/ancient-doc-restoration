from src.core.preprocessor import Preprocessor
from src.core.denoiser import ImageDenoiser
from src.core.enhancer import ImageEnhancer
from src.core.geometry import GeometryCorrector
from src.core.dewarp import PageDewarper
from src.core.segmentor import DocumentSegmentor
from src.core.layout import LayoutAnalyzer



class DocumentRestorationPipeline:
    def __init__(self):
        # Khởi tạo các worker
        self.prep = Preprocessor()
        self.denoiser = ImageDenoiser()
        self.geo = GeometryCorrector()
        self.dewarp = PageDewarper()
        self.enhancer = ImageEnhancer()
        self.seg = DocumentSegmentor()
        self.layout = LayoutAnalyzer()

    def run(self, image, params={}):
        """Chạy luồng xử lý chính"""
        results = {}

        # 1. Preprocess
        gray = self.prep.to_grayscale(image)
        results['gray'] = gray

        # 2. Geometry (Deskew -> Dewarp)
        # ...

        # 3. Restore (Denoise -> Shadow)
        # ...

        # 4. Enhance & Digitize
        # ...

        return results  # Trả về dict chứa các ảnh ở từng bước