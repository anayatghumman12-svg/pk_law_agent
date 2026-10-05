"""One-time script: PDF -> OCR -> clean -> section-aware chunks -> Chroma.

This is NOT part of the agent's runtime graph. It runs once, offline, to build
the vector store the agent's Retriever tool will later query.
"""
import io
import logging
import os
import re
from typing import List

import pymupdf
import pytesseract
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


def ocr_pdf(pdf_path: str, dpi: int = 300) -> str:
    """
    Runs OCR on every page of the scanned PDF using Tesseract.

    Args:
        pdf_path (str): Path to the scanned PDF file.
        dpi (int, optional): Render resolution before OCR. Defaults to 300.

    Returns:
        str: OCR text of all pages, joined by newlines.

    Raises:
        FileNotFoundError: If the PDF does not exist.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file '{pdf_path}' not found.")

    pages_text = []
    with pymupdf.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            png_bytes = page.get_pixmap(dpi=dpi).tobytes("png")
            image = Image.open(io.BytesIO(png_bytes))
            text = pytesseract.image_to_string(image, config="--psm 6")
            pages_text.append(text)
            logger.info(f"OCR done for page {page_number}/{len(doc)}")

    return "\n".join(pages_text)
def clean_text(raw_text: str) -> str:
    """
    Removes Gazette headers, fixes OCR bracket errors, and drops the
    unrelated Banks Act that is printed after PECA in this PDF.

    Args:
        raw_text (str): Raw OCR output from ocr_pdf().

    Returns:
        str: Cleaned text containing only PECA 2016.

    Raises:
        ValueError: If section 1's heading cannot be found.
    """
    text = re.sub(r"(?im)^.*GAZETTE\s+OF\s+PAKISTAN.*$", "", raw_text)
    text = text.replace("{", "(").replace("}", ")")

    match = re.search(r"(?m)^\D{0,5}1\s*[.,]\s+Short\s+title", text)
    if not match:
        raise ValueError("Could not find 'Section 1. Short title' - OCR text may be unusable.")
    text = text[match.start():]

    unrelated_marker = text.find("Banks (Nationalization)")
    if unrelated_marker != -1:
        cut_point = text.rfind("ACT NO", 0, unrelated_marker)
        if cut_point != -1:
            text = text[:cut_point]

    return text.strip()


# Characters OCR sometimes confuses with digits in this scanned document.
DIGIT_LOOKALIKES = {"5": "5§S", "8": "8B"}


def fuzzy_number_pattern(number: int) -> str:
    """
    Builds a regex fragment for a number that tolerates OCR digit misreads.

    Args:
        number (int): The section number to search for, e.g. 8.

    Returns:
        str: A regex character-class fragment, e.g. "[8B]" for 8.
    """
    return "".join(f"[{DIGIT_LOOKALIKES.get(d, d)}]" for d in str(number))


def split_into_sections(text: str, total_sections: int = 55) -> dict:
    """
    Splits cleaned law text into individual sections by searching for
    heading numbers in sequence (1, then 2, then 3, ...).

    Args:
        text (str): Cleaned text from clean_text().
        total_sections (int, optional): Total number of sections. Defaults to 55.

    Returns:
        dict: Mapping of {section_number: section_text}.
    """
    positions = []
    search_from = 0
    for number in range(1, total_sections + 1):
        pattern = re.compile(
            rf"(?m)^\D{{0,5}}{fuzzy_number_pattern(number)}[B]?\s*[.,]\s+(?=[A-Z])"
        )
        match = pattern.search(text, search_from)
        if not match:
            logger.warning(f"Section {number} heading not found - skipping")
            continue
        positions.append((number, match.start()))
        search_from = match.end()

    sections = {}
    for i, (number, start) in enumerate(positions):
        end = positions[i + 1][1] if i + 1 < len(positions) else len(text)
        sections[number] = text[start:end].strip()
    return sections
def extract_and_chunk_pdf(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Extracts PECA using OCR, cleans it, splits it into legal sections,
    and creates section-aware chunks with metadata.

    Args:
        pdf_path (str): Path to the scanned PDF file.
        chunk_size (int, optional): Max characters per chunk. Defaults to 1000.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 200.

    Returns:
        List[Document]: Section-aware chunks, each with section metadata.

    Raises:
        FileNotFoundError: If the PDF does not exist.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"File not found at path: {pdf_path}")
        raise FileNotFoundError(f"PDF file '{pdf_path}' not found.")

    logger.info("Running OCR on PDF...")
    raw_text = ocr_pdf(pdf_path)

    logger.info("Cleaning OCR text...")
    cleaned_text = clean_text(raw_text)

    logger.info("Splitting text into legal sections...")
    sections = split_into_sections(cleaned_text)
    logger.info(f"Found {len(sections)} legal sections.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []
    for section_number, section_text in sections.items():
        section_document = Document(
            page_content=section_text,
            metadata={"source": pdf_path, "section": section_number}
        )
        chunks.extend(text_splitter.split_documents([section_document]))

    logger.info(f"Total {len(chunks)} section-aware chunks created.")
    return chunks


def build_vector_store(
    pdf_path: str = "data/datapeca_2016.pdf",
    persist_directory: str = "chroma_db"
) -> Chroma:
    """
    Generates Gemini embeddings for section-aware chunks and saves them to Chroma.

    This is the ONE-TIME indexing step. It is never called from inside the
    agent's LangGraph - the Retriever tool only ever reads the persisted
    vector store this function builds.

    Args:
        pdf_path (str, optional): Path to the law PDF. Defaults to data/datapeca_2016.pdf.
        persist_directory (str, optional): Where to persist Chroma. Defaults to "chroma_db".

    Returns:
        Chroma: The populated, persisted vector store.
    """
    chunks = extract_and_chunk_pdf(pdf_path=pdf_path)

    logger.info("Initializing Gemini Embeddings model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

    logger.info(f"Generating embeddings and saving Chroma DB to '{persist_directory}'...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    logger.info("Vector store successfully created and persisted!")

    logger.info("--- Running Sanity-Check Query ---")
    sample_query = "What is the punishment for unauthorized access to information system?"
    results = vector_db.similarity_search(sample_query, k=3)

    print("\n================ Sanity Check Search Results ================")
    print(f"Query: '{sample_query}'\n")
    for i, doc in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Section: {doc.metadata.get('section')}")
        print(doc.page_content[:300] + "...\n")
    print("=============================================================\n")

    return vector_db


if __name__ == "__main__":
     build_vector_store()