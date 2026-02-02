# src/pipeline.py (Updated)

import time
import cv2
import numpy as np

# Import các core modules
from src.core.preprocessor import Preprocessor
from src.core.denoiser import ImageDenoiser
from src.core.enhancer import ImageEnhancer
from src.core.geometry import GeometryCorrector
from src.core.dewarp import PageDewarper
from src.core.segmentor import DocumentSegmentor
from src.core.layout import LayoutAnalyzer
from src.core.deskewer import Deskewer
from src.core.ai_model import TinyUNet
from src.core.mask_extractor import MaskExtractor

# Giả sử bạn lưu class SmartRouter ở src/core/router.py
# from src.core.router import SmartRouter 

class DocumentRestorationPipeline:
    def __init__(self):
        self.prep = Preprocessor()
        self.denoiser = ImageDenoiser()
        self.geo = GeometryCorrector()
        self.dewarper = PageDewarper()
        self.deskewer = Deskewer()
        self.enhancer = ImageEnhancer()
        self.forensic = ForensicInk()  # Module ZCA
        self.seg = DocumentSegmentor()
        self.layout = LayoutAnalyzer()

        # Khởi tạo Router (Dùng class mới hoặc hàm cũ đều được)
        # self.router = SmartRouter() 

    def run(self, image, params={}):
        results = {"meta": {}, "images": {}}
        t0 = time.time()

        # Mặc định param
        if params is None: params = {}

        try:
            # ----------------------------------------
            # BƯỚC 1: GEOMETRY CORRECTION (Trên ảnh gốc)
            # ----------------------------------------
            current_img = image.copy()

            # 1.1 Resize nếu ảnh quá lớn để xử lý nhanh (tùy chọn)
            if params.get("resize_max"):
                current_img = self.prep.resize_image(current_img, target_width=params["resize_max"][0])

            # 1.2 Deskew (Xoay thẳng)
            if params.get("deskew", True):
                current_img = self.deskewer.deskew(current_img)
                results["images"]["deskewed"] = current_img

            # 1.3 Dewarp (Nắn cong) - Bước này quan trọng nhất
            if params.get("dewarp", True):
                # Lưu ý: Dewarp thường cần ảnh Gray để detect dòng, nhưng warp trên ảnh màu
                # Hàm dewarp của bạn nên hỗ trợ trả về ảnh màu
                current_img = self.dewarper.dewarp(current_img)
                results["images"]["dewarped"] = current_img

            # ----------------------------------------
            # BƯỚC 2: SMART ROUTING (Phân loại Ca)
            # ----------------------------------------
            # Nếu chưa có class SmartRouter, dùng hàm analyze_and_route cũ của bạn
            # mode = self.router.analyze(current_img)["mode"]

            # Tạm dùng logic đơn giản từ detector.py của bạn:
            from src.core.detector import analyze_and_route
            mode = analyze_and_route(current_img)

            # Cho phép ghi đè mode từ params
            if params.get("force_mode"):
                mode = params["force_mode"]

            results["meta"]["mode"] = mode

            # ----------------------------------------
            # BƯỚC 3: RESTORATION & ENHANCEMENT
            # ----------------------------------------
            if mode == 'zca':
                # --- NHÁNH KHÓ (ZCA) ---
                print(">>> Executing ZCA Pipeline...")

                # 1. Decorrelation Stretch & Mask Extraction
                # forensic.process trả về raw_mask (đen trắng)
                processed_mask = self.forensic.process(current_img)

                # 2. Refine Mask (Khử nhiễu trên mask)
                cleaned_img = self.denoiser.clean_binary_noise(processed_mask, min_area=20)

                # ZCA trả về ảnh nhị phân nền đen chữ trắng, ta đảo lại cho giống tài liệu
                # Hoặc giữ nguyên tùy nhu cầu. Ở đây giả sử cần nền trắng chữ đen:
                final_img = cv2.bitwise_not(cleaned_img)

            else:
                # --- NHÁNH DỄ (DIVISION NORM) ---
                print(">>> Executing Standard Pipeline...")

                # 1. Chuyển xám (nếu chưa)
                if len(current_img.shape) == 3:
                    gray_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
                else:
                    gray_img = current_img

                # 2. Khử nhiễu (Denoise)
                if params.get("denoise", True):
                    gray_img = self.denoiser.manual_median_filter(gray_img, ksize=3)

                # 3. Khử bóng (Shadow Removal) - Quan trọng!
                no_shadow = self.enhancer.remove_shadow(gray_img)
                results["images"]["no_shadow"] = no_shadow

                # 4. Tăng tương phản (CLAHE)
                enhanced = self.enhancer.apply_clahe(no_shadow, clip_limit=2.0)

                # 5. Nhị phân hóa (Binarization) - Sauvola hoặc Otsu
                # (Ở đây dùng Adaptive cho an toàn)
                final_img = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, 31, 10)

            results["images"]["enhanced"] = final_img

            # ----------------------------------------
            # BƯỚC 4: FINAL REFINEMENT (Chung cho cả 2 nhánh)
            # ----------------------------------------

            # Segmentation (Cắt dòng/chữ) nếu cần
            # segments = self.seg.segment(final_img)
            # results["images"]["segments"] = segments

            results["images"]["final"] = final_img
            results["meta"]["total_time"] = time.time() - t0
            results["status"] = "ok"

        except Exception as e:
            import traceback
            traceback.print_exc()
            results["status"] = "error"
            results["error"] = str(e)

        return results