import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Dell\Downloads\tesseract-ocr-w64-setup-5.5.0.20241111.exe"
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import logging
from dotenv import load_dotenv
load_dotenv()

from pypdf import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import requests
from bs4 import BeautifulSoup

from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------- PDF (Text + OCR) ----------------
def load_pdf(pdf_file):
    logger.info(f"Reading PDF: {pdf_file.name}")
    docs = []

    reader = PdfReader(pdf_file)
    pdf_file.seek(0)
    images = convert_from_bytes(pdf_file.read())

    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        if text and len(text.strip()) > 50:
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": pdf_file.name,
                    "page": i + 1,
                    "type": "pdf"
                }
            ))
        else:
            ocr_text = pytesseract.image_to_string(images[i])
            if ocr_text.strip():
                docs.append(Document(
                    page_content=ocr_text,
                    metadata={
                        "source": pdf_file.name,
                        "page": i + 1,
                        "type": "pdf_ocr"
                    }
                ))
    return docs


# ---------------- Website ----------------
def load_website(url):
    logger.info(f"Scraping website: {url}")
    html = requests.get(url, timeout=10).text
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = " ".join(soup.get_text().split())

    return [Document(
        page_content=text,
        metadata={
            "source": url,
            "type": "website"
        }
    )]


# ---------------- Image OCR ----------------
def load_image(image_file):
    logger.info(f"OCR image: {image_file.name}")
    img = Image.open(image_file)
    text = pytesseract.image_to_string(img)

    if not text.strip():
        return []

    return [Document(
        page_content=text,
        metadata={
            "source": image_file.name,
            "type": "image"
        }
    )]


# ---------------- Chunking ----------------
def split_documents(docs):
    splitter = CharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)


# ---------------- Vector DB ----------------
def create_vector_db(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)
