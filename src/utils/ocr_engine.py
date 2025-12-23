import pytesseract
from PIL import Image

def extract_text(image, lang='vie', config='--psm 3'):
    """
    Trích xuất văn bản từ ảnh bằng Tesseract OCR.
    Dùng --psm cho việc đọc text ra string.
    """
    try:
        if isinstance(image, str):
            with Image.open(image) as img:
                text = pytesseract.image_to_string(img, lang=lang, config=config)
        else:
            text = pytesseract.image_to_string(image, lang=lang, config=config)
        return text.strip()
    
    except Exception as e:
        raise RuntimeError(f"OCR extract_text failed: {e}")
    
def export_pdf(image, output_path, lang='vie'):
    """
    Tạo PDF Searchable từ ảnh (có text layer ẩn).
    """
    try:
        if isinstance(image, str):
            with Image.open(image) as img:
                pdf_bytes = pytesseract.image_to_pdf_or_hocr(img, lang=lang, extension='pdf')
        else:
            pdf_bytes = pytesseract.image_to_pdf_or_hocr(image, lang=lang, extension='pdf')
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)

        return True
    
    except Exception as e:
        raise RuntimeError(f"OCR export_pdf failed: {e}")