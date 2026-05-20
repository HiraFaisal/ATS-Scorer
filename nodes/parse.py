import io
import pytesseract
from PIL import Image
import pdfplumber
import fitz
from state import ResumeState
from config import TESSERACT_PATH

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def parse_resume_node(state: ResumeState) -> ResumeState:
    if not state.get("resume_bytes"):
        return {**state, "error": "No resume file provided."}
    file_bytes = state["resume_bytes"]
    try:
        text = ""
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if len(text.strip()) < 100:
            ocr_text = ""
            pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text += pytesseract.image_to_string(img) + "\n"
            pdf_document.close()
            text = ocr_text

        # Extract page images as base64 for Multimodal analysis
        resume_images = []
        try:
            import base64
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                img_data = pix.tobytes("jpeg")
                img_b64 = base64.b64encode(img_data).decode("utf-8")
                resume_images.append(img_b64)
            pdf_doc.close()
        except Exception as img_err:
            print(f"Failed to generate page images: {img_err}")

        return {
            **state, 
            "resume_text": text.strip(),
            "resume_images": resume_images
        }
    except Exception as e:
        return {**state, "error": f"Parsing failed: {str(e)}"}