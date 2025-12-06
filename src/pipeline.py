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
        """Chạy luồng xử lý chính cho 1 ảnh tài liệu
        Trả về dict chứa ảnh/intermediate results và các metadata (sizes, times)
        params: dict (tùy chỉnh ngưỡng/bật tắt các bước)"""
        import time
        if params is None:
            params = {}
        results = {"meta": {}, "images": {}}
        t0 = time.time()

        try:

            # ------ 1. Preprocess ------
            img = self.prep.to_grayscale(image, 
                                          assume_rgb=params.get("assume_rgb", True))
            results["images"]["gray"] = img

            if params.get("resize_max"):
                img = self.prep.resize(img, max_size=params["resize_max"])
                results["images"]["gray_resized"] = img

            if params.get("equalize", True):
                img = self.prep.equalize_histogram(img)
                results["images"]["hist_equalized"] = img
            results["meta"]["t_preprocess"] = time.time() - t0

            # 2. ------ Geometry correction (Deskew -> Dewarp) ------
            t_geo = time.time()
            if params.get("deskew", True):
                img = self.geo.deskew(img)
                results["images"]["deskewed"] = img

            if params.get("dewarp", True):
                img = self.dewarp.dewarp_page(img,
                                            method=params.get("dewarp_method", "mesh"))
                results["images"]["dewarped"] = img
            results["meta"]["t_geometry"] = time.time() - t_geo

            # 3. ------ Restore (Denoise -> Shadow) ------
            t_den = time.time()

            if params.get("denoise", True):
                img = self.denoiser.denoise(img,
                                            method=params.get("denoise_method", "median"),
                                            strength=params.get("denoise_strength", 1.0))
                results["images"]["denoised"] = img

            if params.get("inpaint", False):
                img = self.denoiser.inpaint_holes(img,
                                                  mask=params.get("inpaint_mask", None))
                results["images"]["inpainted"] = img

            # Shadow removal after denoise
            if params.get("remove_shadows", True):
                img = self.enhancer.remove_shadows(img)
                results["images"]["no_shadows"] = img
            results["meta"]["t_restore"] = time.time() - t_den

            # 4. Enhance & Digitize
            t_en = time.time()

            if params.get("enhance_constrast", True):
                img = self.enhancer.enhance_constrast(img,
                                                    clip_limit=params.get("clip_limit", 2.0))
                results["images"]["enhanced"] = img

            if params.get("binarize", True):
                binary = self.prep.adaptive_threshold(img,
                                                   block_size=params.get("block_size", 35),
                                                   C=params.get("threshold_C", 10))
                results["images"]["binary"] = binary
                final_img = binary
            else:
                final_img = img

            segments = self.seg.segment(final_img,
                                        min_area=params.get("seg_min_area", 500))
            results["images"]["segments"] = segments
            results["meta"]["layout"] = self.layout.analyze(segments,
                                                            image_shape=final_img.shape)
            results["images"]["final"] = final_img
            results["meta"]["total_time"] = time.time() - t0
            results["status"] = "ok"

        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)

        return results  # Trả về dict chứa các ảnh ở từng bước