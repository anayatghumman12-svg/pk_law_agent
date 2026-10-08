"""Central configuration for the PECA 2016 law agent.

Every tunable value lives here so no function retypes literals.
Secrets (the Gemini API key) are NOT stored here; they come from the
environment / .env file and are read by langchain-google-genai directly.
"""

# --- The law this agent covers -------------------------------------------- #
LAW_NAME = "Prevention of Electronic Crimes Act, 2016"
TOTAL_SECTIONS = 55  # PECA 2016 has sections 1..55

# --- Models ---------------------------------------------------------------- #
LLM_MODEL = "gemini-3.8-flash"
LLM_TEMPERATURE = 0.0  # deterministic: scope decisions and answers should be repeatable
EMBEDDING_MODEL = "gemini-embedding-2-preview"

# --- Paths ----------------------------------------------------------------- #
PDF_PATH = "data/datapeca_2016.pdf"
CHROMA_DIR = "chroma_db"

# --- Indexing (offline, one-time) ------------------------------------------ #
CHUNK_SIZE = 1000     # characters; most sections fit in one chunk
CHUNK_OVERLAP = 200   # keeps context across a cut inside a long section
OCR_DPI = 300         # the scan is low resolution; 300 DPI gives much cleaner OCR
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows default

# --- Retrieval ------------------------------------------------------------- #
RETRIEVAL_TOP_K = 3
# Cosine similarity of the best chunk must reach this, otherwise the agent declines.
# Calibrated on real questions: a clearly answerable question scored 0.797, while an
# in-scope question the Act does not answer (prosecution statistics) scored 0.659.
RETRIEVAL_MIN_SCORE = 0.68

# --- User-facing messages -------------------------------------------------- #
DECLINE_OUT_OF_SCOPE = (
    f"I can only answer questions about the {LAW_NAME}. "
    "That question is outside what I have information on."
)
DECLINE_NO_RESULTS = (
    f"I don't have enough information in the {LAW_NAME} to answer that question."
)