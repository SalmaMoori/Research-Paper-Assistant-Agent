# ☰ 
import os
import datetime
from typing import Generator, List, Dict, Any
import re
import arxiv
import requests
import fitz
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from fastapi import FastAPI, Form, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, func, Index, event, or_
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pdfplumber
import shutil
from urllib.parse import urlparse, quote
from functools import lru_cache
import hashlib
import asyncio

DATABASE_URL = "sqlite:///./research_papers.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

Base = declarative_base()
class Paper(Base):
    __tablename__ = "papers"
    id = Column(Integer, primary_key=True)
    title = Column(Text, nullable=False)
    abstract = Column(Text)
    source = Column(String)
    file_path = Column(String)
    source_path = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Add to the existing memory system
class ConversationState:
    def __init__(self):
        self.uploaded_papers = []  # List of {id, title, source_path} for uploaded papers
        self.retrieved_papers = []  # List of {id/url, title, source, abstract} for retrieved papers
        self.last_action = None    # Track the last action performed
        
    def add_uploaded_paper(self, paper_id, title, source_path):
        self.uploaded_papers.append({
            "id": paper_id,
            "title": title,
            "source_path": source_path
        })
        self.last_action = "upload"
        
    def add_retrieved_papers(self, papers_list, source="web"):
        # papers_list should be a list of dicts with at least id/url, title, and abstract
        print("Adding retrieved papers:") # Debugging
        for paper in papers_list:
            print(f"  - {paper}") # Debugging
        self.retrieved_papers.extend([{**paper, "source": source} for paper in papers_list])
        self.last_action = "retrieve"
        print(f"Retrieved papers added: {self.retrieved_papers}") # Debugging
        
    def get_latest_uploaded_paper(self):
        return self.uploaded_papers[-1] if self.uploaded_papers else None
        
    def get_retrieved_papers(self, limit=5):
        papers = self.retrieved_papers[-limit:] if self.retrieved_papers else []
        print("Getting retrieved papers:") # Debugging
        for paper in papers: # Debugging
            print(f"  - {paper}") # Debugging
        return papers

Index('idx_paper_title', Paper.title)
Index('idx_paper_source', Paper.source)

Base.metadata.create_all(engine)

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")  
    cursor.execute("PRAGMA synchronous=NORMAL") 
    cursor.close()

SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine
)

def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(engine)
    print("Database initialized.")

chroma_client = chromadb.PersistentClient(path="chroma_db")
try:
    chroma_client.get_collection(name="research_papers")
except ValueError:
    chroma_client.create_collection(name="research_papers")

instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"}
)

memory_index_path = "faiss_memory_index"
if os.path.exists(memory_index_path):
    faiss_store = FAISS.load_local(
        memory_index_path,
        instructor_embeddings,
        allow_dangerous_deserialization=True
    )
else:
    faiss_store = None

memory = None

if faiss_store:
    memory = VectorStoreRetrieverMemory(retriever=faiss_store.as_retriever(search_kwargs={"k": 5}))

os.environ["OLLAMA_KEEP_ALIVE"] = "-1"
llm = Ollama(
    model="mistral:7b-instruct",
    temperature=0.0,
    timeout=300,
    # num_gpu=1,  # Ensure GPU acceleration if available
)

def extract_pdf_content(pdf_path: str) -> Dict[str, str]:
    """
    Extracts title, abstract, and full text from a PDF file,
    with improved title extraction.
    """
    print(f"Extracting content from PDF: {pdf_path}")
    title = ""
    abstract = ""
    full_text = ""

    try:
        doc = fitz.open(pdf_path)

        # 1. Metadata Extraction (with encoding checks)
        meta_title = doc.metadata.get("title", "")
        if meta_title:
            try:
                # Try decoding if it's byte-encoded
                if isinstance(meta_title, bytes):
                    meta_title = meta_title.decode('utf-8', errors='ignore')
            except Exception as e:
                print(f"Metadata decode error: {e}")

            meta_title = meta_title.strip()

            # Check title quality.  More stringent than the initial check
            if (len(meta_title) > 5 and
                not ("untitled" in meta_title.lower() or "microsoft word" in meta_title.lower()) and
                len(re.findall(r'[A-Z]', meta_title)) >= 2 and  # At least two uppercase letters
                len(re.findall(r'[a-z]', meta_title)) >= 2):  # At least two lowercase letters
                title = meta_title

        # 2. Extract Full Text (early)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text("text") + "\n"

        # 3. Heuristics from First Page (if metadata fails)
        if not title:
            first_page_text = doc[0].get_text("text")
            lines = first_page_text.split('\n')
            cleaned_lines = [line.strip() for line in lines if line.strip()]

            potential_titles = []
            for i, line in enumerate(cleaned_lines[:20]):
                # Enhanced Title Filtering Criteria
                if (5 < len(line) < 200 and
                    not line.lower().startswith("abstract") and
                    not line.lower().startswith("introduction") and
                    not line.lower().startswith("keywords") and
                    not re.match(r"^\d+$", line) and
                    not "arxiv" in line.lower() and
                    not "http" in line.lower() and
                    not "@" in line and
                    len(re.findall(r'[A-Z]', line)) >= 2 and
                    len(re.findall(r'[a-z]', line)) >= 2 and
                    not any(word in line.lower() for word in ["preprint", "copyright", "all rights reserved"])):
                    potential_titles.append(line)
            # Selecting the best potential title
            if potential_titles:
                # Prioritize longer titles
                title = max(potential_titles, key=len)  # Pick longest
                # Further refining - remove potential affiliation lines below the title
                if len(cleaned_lines) > potential_titles.index(title)+1:
                    if (len(cleaned_lines[potential_titles.index(title)+1]) < 50 and
                        (len(re.findall(r'[A-Z]', cleaned_lines[potential_titles.index(title)+1])) < 3 or
                         cleaned_lines[potential_titles.index(title)+1].count(',') > 2)):
                         title = title
                    else:
                        title = potential_titles[0]

        # 4. Abstract Extraction (Regex-based) - largely unchanged, but simplified

        abstract_match = re.search(
            r"(Abstract|Summary)\b[\s\:]*([\s\S]+?)(?:(?:\n\s*\n)|(?:I\.\s*INTRODUCTION|1\.\s*Introduction|Keywords|Index Terms|\n\s*Introduction\b))",
            full_text,
            re.IGNORECASE
        )

        if abstract_match:
            abstract = abstract_match.group(2).strip().replace('\n', ' ')
            abstract = ' '.join(abstract.split()[:300])
            if len(abstract) < 30 and "abstract" in abstract.lower():
                abstract = ""

        if not abstract and title:  # Simpler fallback if regex fails
            try:
                first_page_text_lower = doc[0].get_text("text").lower()
                abstract_start_idx = first_page_text_lower.find("abstract")
                if abstract_start_idx != -1:
                    # Take text after "abstract" keyword for a certain length
                    potential_abstract = doc[0].get_text("text")[abstract_start_idx + len("abstract"):].strip()
                    potential_abstract = potential_abstract.split("\n\n")[0] # Take first paragraph
                    if len(potential_abstract) > 50:
                        abstract = ' '.join(potential_abstract.replace('\n', ' ').split()[:250])
            except Exception:
                pass

        doc.close()

    except Exception as e_fitz:
        print(f"PyMuPDF error: {e_fitz}. Falling back to pdfplumber.")
        full_text = ""  # Reset full_text for pdfplumber
        try:
            with pdfplumber.open(pdf_path) as pdf:
                first_page_text = ""
                if pdf.pages:
                    first_page_text = pdf.pages[0].extract_text() or ""

                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
                # Heuristics for title from first page text (similar to PyMuPDF)
                if not title and first_page_text:  # if title not found by PyMuPDF metadata
                    lines = first_page_text.split('\n')
                    cleaned_lines = [line.strip() for line in lines if line.strip()]
                    potential_titles = []
                    for i, line in enumerate(cleaned_lines[:20]):
                         if 5 < len(line) < 200 and \
                           not line.lower().startswith("abstract") and \
                           not line.lower().startswith("introduction"):
                            potential_titles.append(line)
                    if potential_titles:
                        title = potential_titles[0] # Simplistic for fallback

                # Heuristics for abstract (similar to PyMuPDF)
                if not abstract and first_page_text: # if abstract not found by PyMuPDF
                    abstract_match = re.search(
                        r"(Abstract|Summary)\b[\s\:]*([\s\S]+?)(?:(?:\n\s*\n)|(?:I\.\s*INTRODUCTION|1\.\s*Introduction|Keywords))",
                        first_page_text, # Search only in first page for pdfplumber for simplicity
                        re.IGNORECASE
                    )
                    if abstract_match:
                        abstract = abstract_match.group(2).strip().replace('\n', ' ')
                        abstract = ' '.join(abstract.split()[:300])

        except Exception as e_plumber:
            print(f"pdfplumber error: {e_plumber}")

    # --- Final Cleanups and Fallbacks ---
    if not title or len(title) < 5:  # If title is still missing or too short
        title = os.path.splitext(os.path.basename(pdf_path))[0].replace("_", " ").replace("-", " ")
        title = title.strip() if title else "Untitled Paper"

    if not abstract or len(abstract) < 20:
        if full_text and len(full_text) > 500:  # Try to grab some initial text if abstract is bad
            # A very naive abstract: first few sentences of the full text after potential title.
            start_search_text = full_text
            if title:
                title_idx = full_text.lower().find(title.lower())
                if title_idx != -1:
                    start_search_text = full_text[title_idx + len(title):]

            # Remove common non-abstract starts
            start_search_text = re.sub(r"^(authors|affiliations|keywords|introduction|index terms|[\s\d\.\,\*†‡§])*\b", "", start_search_text.strip(), flags=re.IGNORECASE|re.MULTILINE).strip()

            # Split into sentences (basic) and take first few
            sentences = re.split(r'(?<=[.!?])\s+', start_search_text)
            potential_abstract_text = " ".join(sentences[:3])  # Take first 3 sentences
            if len(potential_abstract_text) > 50:
                 abstract = ' '.join(potential_abstract_text.split()[:150]) # Limit words
            else:
                abstract = "Abstract not found or too short."
        else:
             abstract = "Abstract not found or too short."

    # Sanitize title and abstract (remove excessive whitespace)
    title = ' '.join(title.split()) if title else "Untitled Paper"
    abstract = ' '.join(abstract.split()) if abstract else "No abstract available."

    print(f"Extracted Title: {title[:100]}...")
    print(f"Extracted Abstract: {abstract[:150]}...")

    return {"title": title, "abstract": abstract, "full_text": full_text}

def embed_and_store(paper_id: int, text: str):
    global faiss_store, memory
    try:
        doc = Document(page_content=text, metadata={"paper_id": str(paper_id)}) # Ensure paper_id is string for some vector stores

        if faiss_store is None:
            faiss_store = FAISS.from_documents([doc], instructor_embeddings)
        else:
            faiss_store.add_documents([doc])

        faiss_store.save_local(memory_index_path)

        if memory is None: 
             memory = VectorStoreRetrieverMemory(retriever=faiss_store.as_retriever(search_kwargs={"k": 5}))
        else: # If memory exists, update its retriever (though FAISS updates in place)
             memory.retriever = faiss_store.as_retriever(search_kwargs={"k": 5})

        print(f"[FAISS] Paper ID {paper_id} embedded and FAISS index updated. Memory (re)initialized.")

    except Exception as e:
        print(f"[FAISS] Error embedding and storing text: {e}")

# (download_arxiv_pdf - unchanged)
def download_arxiv_pdf(arxiv_url, save_dir='uploaded_papers'):
    # Extract the arXiv ID from the URL
    parsed = urlparse(arxiv_url)
    path = parsed.path
    paper_id_match = re.search(r'(?:abs/|pdf/)?(\d{4}\.\d{4,5}(v\d+)?|[a-zA-Z\-]+/\d{7}(v\d+)?)', path)

    if paper_id_match:
        paper_id = paper_id_match.group(1)
        print(f"Extracted paper ID: {paper_id}")
    else:
        print(f"Could not extract paper ID from path: {path}")
        raise ValueError("Invalid arXiv URL format or could not extract paper ID.")

    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    print(f"Attempting to download PDF from: {pdf_url}")
    response = requests.get(pdf_url, headers={'User-Agent': 'ResearchAssistant/1.0'}) # Add User-Agent
    print(f"Response status code: {response.status_code}")

    if response.status_code == 200:
        safe_paper_id = paper_id.replace('/', '_')
        file_path = os.path.join(save_dir, f"{safe_paper_id}.pdf")
        print(f"Saving PDF to: {file_path}")
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {file_path}")
        return file_path, paper_id 
    else:
        print(f"Failed to download PDF (status code {response.status_code}). Response: {response.text[:200]}")
        return None, None

# --- Tools ---
# (upload_pdf_tool - unchanged)
def upload_pdf(source_path_input: str, background_tasks=None) -> str:
    """
    Uploads a PDF research paper from a local file path or a direct web URL.
    It cleans the input string, downloads if it's a URL, extracts metadata,
    stores the paper, and indexes its content.
    Raises FileNotFoundError if a local path is given but the file doesn't exist.
    """
    # 1. Clean the input string to get the actual path/URL
    cleaned_source_path = source_path_input
    # Remove potential "source_path =" or "path =" prefixes and surrounding quotes/spaces
    # This regex is more general for "key = value" type inputs
    match_prefix = re.match(r"^\s*(?:source_path|path|url|uri|file)\s*=\s*(.*)$", cleaned_source_path, re.IGNORECASE)
    if match_prefix:
        cleaned_source_path = match_prefix.group(1)
    
    # Strip leading/trailing quotes and spaces
    cleaned_source_path = cleaned_source_path.strip("'\" ")
    
    source_path = cleaned_source_path # Use this cleaned path from now on
    print(f"Upload tool: Processing source path: '{source_path}'")

    db = next(get_db()) # Assuming get_db() is your SQLAlchemy session generator
    is_web_source = source_path.startswith("http://") or source_path.startswith("https://")
    
    source_type = "web_Search" if is_web_source else "internal_upload"
    
    original_filename = os.path.basename(source_path) if not is_web_source else "web_downloaded_paper.pdf"

    downloaded_filepath = None  # Path of the initially downloaded file (if web) or source_path (if local)
    final_paper_filepath = None # Path where the paper is stored after processing/renaming

    try:
        if is_web_source:
            print(f"Attempting to download from web source: {source_path}")
            if "arxiv.org" in source_path.lower():
                # Assuming download_arxiv_pdf returns (filepath, arxiv_id) or (None, None)
                temp_filepath, _ = download_arxiv_pdf(source_path, UPLOAD_DIR)
                if not temp_filepath:
                    return f"Failed to download paper from ArXiv URL: {source_path}"
                downloaded_filepath = temp_filepath
                original_filename = os.path.basename(downloaded_filepath) # Use the name from download_arxiv_pdf
            else: # Generic web download
                response = requests.get(source_path, stream=True, headers={'User-Agent': 'ResearchAssistant/1.0'}, timeout=20)
                response.raise_for_status() # Will raise HTTPError for bad responses (4xx or 5xx)
                
                # Try to get a filename from Content-Disposition or URL
                cd = response.headers.get('content-disposition')
                if cd:
                    fname_match = re.findall('filename="?([^"]+)"?', cd) # More robust regex for filename
                    if fname_match:
                        original_filename = fname_match[0]
                
                # Ensure filename ends with .pdf, if not, generate one
                if not original_filename.lower().endswith(".pdf"):
                     original_filename = "downloaded_paper_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".pdf"
                
                # Sanitize original_filename for path safety
                original_filename = re.sub(r'[^\w\s.-]', '_', os.path.basename(original_filename))

                temp_filepath = os.path.join(UPLOAD_DIR, f"temp_{original_filename}")
                with open(temp_filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                downloaded_filepath = temp_filepath
            
            if not downloaded_filepath or not os.path.exists(downloaded_filepath): # Double check
                return f"File download failed or file not found after download for: {source_path}"
            
            print(f"File downloaded to: {downloaded_filepath}")
            paper_data = extract_pdf_content(downloaded_filepath)
            # Use extracted title, or fallback to a generated name from the original filename if title is poor
            title = paper_data.get("title", "")
            if not title or title == "N/A (Extraction Error)" or title == "Untitled Paper" or len(title) < 5:
                title = os.path.splitext(original_filename)[0].replace("_", " ").replace("-", " ")

        else: # Local file upload
            print(f"Processing local file: {source_path}")
            if not os.path.exists(source_path):
                # Raise FileNotFoundError as discussed, so AgentExecutor might handle it differently
                print(f"Error: Local file not found at {source_path}. Raising FileNotFoundError.")
                raise FileNotFoundError(f"Local file specified for upload not found: {source_path}")
            
            downloaded_filepath = source_path # For local files, the source is the "downloaded" path
            paper_data = extract_pdf_content(downloaded_filepath)
            title = paper_data.get("title", "")
            if not title or title == "N/A (Extraction Error)" or title == "Untitled Paper" or len(title) < 5:
                title = os.path.splitext(os.path.basename(source_path))[0].replace("_", " ").replace("-", " ")

        # Sanitize title for filename
        clean_title_base = re.sub(r'[^\w\s-]', '', title).strip() # Remove special chars except underscore, hyphen
        clean_title_base = re.sub(r'\s+', '_', clean_title_base)    # Replace spaces with underscores
        if not clean_title_base: # Handle empty title after cleaning
            clean_title_base = "untitled_paper_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Truncate long titles to prevent overly long filenames
        final_filename_base = clean_title_base[:100]
        final_filename = f"{final_filename_base}.pdf"
        final_paper_filepath = os.path.join(UPLOAD_DIR, final_filename)
        print(f"Final filename will be: {final_filename}")

        # Move/Copy file to final location with new name
        # Ensure UPLOAD_DIR exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        if is_web_source:
            # For web sources, downloaded_filepath is a temp file. Rename it.
            if os.path.exists(final_paper_filepath) and downloaded_filepath != final_paper_filepath:
                print(f"Warning: Final paper path {final_paper_filepath} already exists. Removing it before renaming.")
                os.remove(final_paper_filepath)
            shutil.move(downloaded_filepath, final_paper_filepath) # Use shutil.move for robustness
            print(f"Web file moved from {downloaded_filepath} to {final_paper_filepath}")
        else: # Local file, copy it to UPLOAD_DIR
            # downloaded_filepath is the original source_path here
            if os.path.abspath(downloaded_filepath) == os.path.abspath(final_paper_filepath):
                print(f"Source and destination are the same ({final_paper_filepath}). No copy needed.")
            else:
                if os.path.exists(final_paper_filepath):
                     print(f"Warning: Destination path {final_paper_filepath} already exists for copy. Overwriting.")
                shutil.copyfile(downloaded_filepath, final_paper_filepath)
                print(f"Local file copied from {downloaded_filepath} to {final_paper_filepath}")

        abstract = paper_data.get("abstract", "No abstract available.")
        full_text = paper_data.get("full_text", "")

        # Store metadata in the database
        new_paper = Paper(
            title=title,
            abstract=abstract,
            source=source_type,
            file_path=final_filename, # Store just the filename relative to UPLOAD_DIR
            source_path=source_path    # Store original link/path as given by user
        )
        db.add(new_paper)
        db.commit()
        db.refresh(new_paper)
        
        text_to_embed = full_text if full_text and len(full_text) > 50 else (title + "\n" + abstract)
    
        if background_tasks:
            # Schedule embedding as a background task
            background_tasks.add_task(background_embed_and_store, new_paper.id, text_to_embed)
            print(f"Scheduled background embedding for paper ID {new_paper.id}")
        else:
            # Fall back to synchronous processing if no background_tasks available
            embed_and_store(new_paper.id, text_to_embed)
        
        success_message = f"Paper '{title}' (ID: {new_paper.id}) uploaded successfully from {source_type} and stored as '{final_filename}'."
        
        conversation_state.add_uploaded_paper(new_paper.id, title, source_path)

        print(success_message)
        return success_message

    except requests.exceptions.HTTPError as e_http:
        print(f"HTTP error downloading web file {source_path}: {e_http}")
        return f"Error downloading paper from {source_path}: {e_http.response.status_code} {e_http.response.reason}"
    except requests.exceptions.RequestException as e_req:
        print(f"Request error downloading web file {source_path}: {e_req}")
        return f"Error accessing URL {source_path}: {e_req}"
    except FileNotFoundError as e_fnf:
        raise e_fnf 
    except Exception as e:
        db.rollback()
        error_message = f"An unexpected error occurred during upload of '{source_path}': {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        # Cleanup temp file if web download failed mid-process
        if is_web_source and downloaded_filepath and os.path.exists(downloaded_filepath) and downloaded_filepath != final_paper_filepath:
            try:
                os.remove(downloaded_filepath)
                print(f"Cleaned up temporary file: {downloaded_filepath}")
            except Exception as e_clean:
                print(f"Error cleaning up temp file {downloaded_filepath}: {e_clean}")
        return error_message
    finally:
        db.close()

upload_pdf_tool = Tool(
    name="upload_paper_from_path_or_url",
    func=upload_pdf,
    return_direct=True, 
    description="Uploads a PDF research paper from a local file path (e.g., '/path/to/paper.pdf') or a direct web URL (e.g., 'https://arxiv.org/pdf/xxxx.xxxx.pdf'). Extracts metadata, stores the paper, and indexes its content. Use this when the user explicitly provides a path or URL to a specific paper.",
)

def background_embed_and_store(paper_id: int, text: str):
    """Background task for embedding and storing paper content"""
    global faiss_store, memory
    try:
        doc = Document(page_content=text, metadata={"paper_id": str(paper_id)})

        if faiss_store is None:
            faiss_store = FAISS.from_documents([doc], instructor_embeddings)
        else:
            faiss_store.add_documents([doc])

        faiss_store.save_local(memory_index_path)

        if memory is None: 
             memory = VectorStoreRetrieverMemory(retriever=faiss_store.as_retriever(search_kwargs={"k": 5}))
        else:
             memory.retriever = faiss_store.as_retriever(search_kwargs={"k": 5})

        print(f"[FAISS] Paper ID {paper_id} embedded and FAISS index updated. Memory (re)initialized.")

    except Exception as e:
        print(f"[FAISS] Error embedding and storing text: {e}")

def cache_search_result(func):
    # Create an LRU cache with a size limit of 100 items
    cache = {}
    max_size = 100
    
    def wrapper(*args, **kwargs):
        # Create a key from the function arguments
        query = args[0] if args else kwargs.get('query', '')
        key = hashlib.md5(query.encode()).hexdigest()
        
        # Check if result is in cache
        if key in cache:
            print(f"Cache hit for query: {query}")
            return cache[key]
        
        # Call the original function
        result = func(*args, **kwargs)
        
        # Store result in cache
        cache[key] = result
        
        # Limit cache size
        if len(cache) > max_size:
            # Remove the first item (oldest)
            cache.pop(next(iter(cache)))
        
        return result
    
    return wrapper

@cache_search_result
def search_internal_papers(query: str) -> str:
    """Searches the internal paper database using keyword search and semantic search (vector store),
    combines results, ranks them, and returns the top 3-5 papers."""
    print(f"Searching internal papers with query: '{query}'")

    db = next(get_db())
    try:
        # 1. Keyword Search (SQLAlchemy)
        keyword_results = db.query(Paper).filter(
            or_(
                func.lower(Paper.title).contains(func.lower(query)),
                func.lower(Paper.abstract).contains(func.lower(query))
            )
        ).all()
        
        print(f"Keyword search executed. Found {len(keyword_results)} results.")

        keyword_paper_ids = {paper.id for paper in keyword_results}
        print(f"Keyword search found {len(keyword_results)} hits with IDs: {keyword_paper_ids}")

        # 2. Semantic Search (FAISS)
        if not faiss_store:
            return "Vector store (FAISS) is not initialized. Please upload some papers first."

        print(f"Searching FAISS for internal papers with query: '{query}'")
        relevant_docs_with_scores = faiss_store.similarity_search_with_score(query, k=10)

        semantic_paper_ids = []
        semantic_results_dict = {}
        seen_ids = set()
        for doc, score in relevant_docs_with_scores:
            if doc.metadata and "paper_id" in doc.metadata:
                metadata_paper_id = doc.metadata["paper_id"]
                if metadata_paper_id is not None:
                    try:
                        paper_id = int(metadata_paper_id)
                        if paper_id not in seen_ids:
                            seen_ids.add(paper_id)
                            semantic_paper_ids.append(paper_id)
                            semantic_results_dict[paper_id] = (doc, score)
                    except ValueError:
                        print(f"Warning: Could not convert metadata paper_id '{metadata_paper_id}' to int. Skipping doc.")
                else:
                    print("Warning: Found 'paper_id' in metadata but its value is None. Skipping doc.")
            else:
                print("Warning: 'paper_id' not found in metadata for a document. Skipping doc.")

        print(f"Semantic search found {len(semantic_paper_ids)} unique papers with IDs: {semantic_paper_ids}")

        # 3. Combine and Rank Results
        combined_results = {}
        for paper in keyword_results:
            combined_results[paper.id] = {
                "paper": paper,
                "keyword_hit": True,
                "semantic_score": -1.0
            }

        for paper_id in semantic_paper_ids:
            doc, score = semantic_results_dict[paper_id]
            if paper_id in combined_results:
                combined_results[paper_id]["semantic_score"] = score
            else:
                paper = db.query(Paper).get(paper_id)
                if paper:
                    combined_results[paper_id] = {
                        "paper": paper,
                        "keyword_hit": False,
                        "semantic_score": score
                    }
                else:
                    print(f"Warning: Paper with ID {paper_id} found in FAISS but not in SQL DB.")

        # 4. Rank based on keyword_hit and semantic_score
        ranked_results = sorted(
            combined_results.items(),
            key=lambda item: (item[1]["keyword_hit"], item[1]["semantic_score"]),
            reverse=True
        )

        # 5. Store results in conversation_state for comparison
        retrieved_papers = []
        for paper_id, data in ranked_results[:5]:
            paper = data["paper"]
            retrieved_papers.append({
                "id": paper.id,
                "title": paper.title,
                "source": "internal_database",
                "abstract": paper.abstract or "No abstract available",
                "url": None  # No URL for internal papers
            })
        conversation_state.add_retrieved_papers(retrieved_papers, "internal_database")
        print(f"Saved internal papers to conversation state: {retrieved_papers}")

        # 6. Format and Return Top Results
        top_results = ranked_results[:5]
        formatted_results = []
        for paper_id, data in top_results:
            paper = data["paper"]
            score = data["semantic_score"] if data["semantic_score"] != -1.0 else "N/A"
            formatted_results.append(
                f"ID: {paper.id}, Title: {paper.title}\nAbstract: {paper.abstract[:200]}...\nSource: {paper.source}, Stored File: {paper.file_path}\nKeyword Hit: {data['keyword_hit']}, Semantic Score: {score}"
            )

        if formatted_results:
            return f"Top matching papers from internal database:\n" + "\n---\n".join(formatted_results)
        else:
            return "No matching papers found in the internal database."

    except Exception as e:
        print(f"Error in search_internal_papers: {e}")
        import traceback
        traceback.print_exc()
        return f"Error searching internal papers: {e}"
    finally:
        db.close()

search_internal_papers_tool = Tool(
    name="search_MY_LOCAL_LIBRARY_of_papers",
    func=search_internal_papers,
    return_direct=True,
    description="Use THIS tool ONLY when the user wants to search papers ALREADY UPLOADED AND STORED IN THIS SYSTEM (your local library). Keywords like 'my library', 'internal', 'uploaded papers' mean you should use this. Input is the search query string.",
)

# Optional: For Semantic Scholar API Key
# SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

@cache_search_result
def search_external_papers(search_string: str) -> str:
    """
    Searches external academic databases (arXiv or Semantic Scholar) for papers.
    Input should be a single string. You can specify source and max_results within the string, e.g.,
    '[the user's actual topic] source:arxiv max_results:5' or '[the user's actual topic] from semantic_scholar'.
    If not specified, defaults to arXiv and 3 results.
    Returns a formatted string of search results or an error message.
    """
    query = search_string
    source = "arxiv"
    max_results = 5

    # Regex to extract source (case-insensitive)
    source_match = re.search(r"source:(\w+)", query, re.IGNORECASE)
    if source_match:
        extracted_source = source_match.group(1).lower()
        if extracted_source in ["arxiv", "semantic_scholar", "s2"]:
            source = "semantic_scholar" if extracted_source == "s2" else extracted_source
        query = re.sub(r"source:\w+", "", query, flags=re.IGNORECASE).strip()

    # Regex to extract max_results (case-insensitive)
    max_results_match = re.search(r"max_results:(\d+)", query, re.IGNORECASE)
    if max_results_match:
        try:
            max_results = int(max_results_match.group(1))
            if max_results > 20: max_results = 20
            if max_results < 1: max_results = 1
            query = re.sub(r"max_results:\d+", "", query, flags=re.IGNORECASE).strip()
        except ValueError:
            print(f"Warning: Could not parse max_results value from '{max_results_match.group(1)}'. Using default {max_results}.")

    query = re.sub(r"\b(from|on)\s+(arxiv|semantic scholar|s2)\b", "", query, flags=re.IGNORECASE).strip()
    query = query.replace("'", "").replace('"',"").strip()

    if not query:
        return "Error: No search query provided after parsing."

    print(f"Executing external search - Query: '{query}', Source: '{source}', Max Results: {max_results}")

    results_output = []
    retrieved_papers = []
    try:
        if source == "arxiv":
            search_client = arxiv.Client()
            search_results = search_client.results(arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            ))
            
            arxiv_papers = list(search_results)
            if not arxiv_papers:
                return f"No results found on arXiv for '{query}'."
            
            for r in arxiv_papers:
                authors = ", ".join([a.name for a in r.authors])
                results_output.append(
                    f"Title: {r.title}\n"
                    f"Authors: {authors}\n"
                    f"Published: {r.published.strftime('%Y-%m-%d') if r.published else 'N/A'}\n"
                    f"Summary: {r.summary[:350].replace(chr(10), ' ') if r.summary else 'N/A'}...\n"
                    f"ArXiv ID: {r.get_short_id()}\n"
                    f"PDF Link: {r.pdf_url}\n"
                    f"Primary Category: {r.primary_category if hasattr(r, 'primary_category') else 'N/A'}"
                )
                paper_info = {
                    "url": r.pdf_url,
                    "title": r.title,
                    "id": r.get_short_id(),
                    "source": "arxiv",
                    "abstract": r.summary
                }
                retrieved_papers.append(paper_info)
            conversation_state.add_retrieved_papers(retrieved_papers, source)
            print(f"Saved papers are: {conversation_state.retrieved_papers}")

        elif source == "semantic_scholar":
            headers = {'User-Agent': 'ResearchAssistant/1.0'}
            if SEMANTIC_SCHOLAR_API_KEY:
                headers['x-api-key'] = SEMANTIC_SCHOLAR_API_KEY
            
            params = {
                'query': query,
                'limit': max_results,
                'fields': 'title,authors,year,abstract,tldr,url,venue,publicationDate,isOpenAccess,openAccessPdf,primaryDiscipline',
                'sort': 'publicationDate:desc'
            }
            response = requests.get(S2_API_URL, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            s2_data = response.json()

            if not s2_data.get("data") or len(s2_data["data"]) == 0:
                return f"No results found on Semantic Scholar for '{query}'."

            for item in s2_data["data"]:
                title_s2 = item.get("title", "N/A")
                authors_s2 = ", ".join([a['name'] for a in item.get("authors", [])]) if item.get("authors") else "N/A"
                year_s2 = item.get("year", "N/A")
                abstract_s2 = item.get("abstract", "")
                if not abstract_s2 and item.get("tldr") and isinstance(item.get("tldr"), dict) and item['tldr'].get('text'):
                    abstract_s2 = "TLDR: " + item['tldr']['text']
                
                pdf_url_s2 = "Not available"
                if item.get("isOpenAccess") and item.get("openAccessPdf") and isinstance(item.get("openAccessPdf"), dict) and item['openAccessPdf'].get('url'):
                    pdf_url_s2 = item['openAccessPdf']['url']
                elif item.get("url"): 
                    pdf_url_s2 = item.get("url")

                results_output.append(
                    f"Title: {title_s2}\n"
                    f"Authors: {authors_s2}\n"
                    f"Year: {year_s2}\n"
                    f"Venue: {item.get('venue', 'N/A')}\n"
                    f"Abstract: {abstract_s2[:350].replace(chr(10), ' ') if abstract_s2 else 'N/A'}...\n"
                    f"Semantic Scholar URL: {item.get('url', 'N/A')}\n"
                    f"OpenAccess PDF Link: {pdf_url_s2}"
                )
                paper_info = {
                    "url": item.get("url"),
                    "title": item.get("title", "Unknown"),
                    "id": item.get("paperId", ""),
                    "source": "semantic_scholar",
                    "abstract": item.get("abstract", "")
                }
                retrieved_papers.append(paper_info)
                
            conversation_state.add_retrieved_papers(retrieved_papers, source)
            print(f"Saved papers are: {conversation_state.retrieved_papers}")
        else:
            return f"Unsupported external source: '{source}'. Please use 'arxiv' or 'semantic_scholar'."

        if not results_output:
             return f"No results found on {source} for '{query}' after processing."
        return f"Found {len(results_output)} results from {source} for '{query}':\n\n" + "\n\n---\n\n".join(results_output)

    except requests.exceptions.Timeout:
        return f"API request to {source} timed out. Please try again later."
    except requests.exceptions.RequestException as e:
        return f"API request error for {source}: {e}"
    except arxiv.arxiv.ArxivError as e:
        return f"ArXiv API error: {e}"
    except Exception as e:
        print(f"Unexpected error in search_external_papers (source: {source}, query: '{query}'): {e}")
        import traceback
        traceback.print_exc()
        return f"An unexpected error occurred while searching {source}: {str(e)}"
    
search_external_papers_tool = Tool(
    name="search_ONLINE_ACADEMIC_DATABASES_for_new_papers",
    func=search_external_papers,
    return_direct=True,
    description="Use THIS tool to find NEW papers from ONLINE sources like arXiv or Semantic Scholar. Do NOT use this if the user asks to search their local library or papers already in the system. Input is a single string, e.g., '[the user's actual topic] source:arxiv'.",
)

def get_paper_content(paper_id_or_url):
    """
    Retrieves paper content from either the local database or by downloading from a URL.
    Returns a dict with paper metadata and content.
    """
    try:
        # Case 1: Integer or string integer (local database ID)
        if isinstance(paper_id_or_url, int) or (
            isinstance(paper_id_or_url, str) and paper_id_or_url.isdigit()
        ):
            db = next(get_db())
            paper = db.query(Paper).get(int(paper_id_or_url))
            db.close()
            
            if not paper:
                print(f"No paper found in database with ID: {paper_id_or_url}")
                return None
                
            file_path = os.path.join(UPLOAD_DIR, paper.file_path)
            paper_content = extract_pdf_content(file_path)
            
            return {
                "id": paper.id,
                "title": paper.title,
                "abstract": paper.abstract,
                "full_text": paper_content.get("full_text", ""),
                "source": "local_database"
            }
        
        # Case 2: Dictionary from conversation_state.retrieved_papers
        elif isinstance(paper_id_or_url, dict):
            paper_source = paper_id_or_url.get("source")
            paper_id = paper_id_or_url.get("id")
            paper_title = paper_id_or_url.get("title")
            abstract = paper_id_or_url.get("abstract")
            
            if paper_source == "internal_database":
                # Fetch from local database using paper_id
                db = next(get_db())
                paper = db.query(Paper).get(int(paper_id))
                db.close()
                
                if not paper:
                    print(f"No paper found in database with ID: {paper_id}")
                    return None
                    
                file_path = os.path.join(UPLOAD_DIR, paper.file_path)
                paper_content = extract_pdf_content(file_path)
                
                return {
                    "id": paper.id,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "full_text": paper_content.get("full_text", ""),
                    "source": "local_database"
                }
            
            elif paper_source in ["arxiv", "semantic_scholar"]:
                pdf_url = paper_id_or_url.get("url")
                if not pdf_url:
                    print(f"No PDF URL provided for {paper_source} paper: {paper_id}")
                    return None
                    
                # Download and extract content
                if paper_source == "arxiv":
                    temp_path, _ = download_arxiv_pdf(pdf_url, UPLOAD_DIR)
                else:
                    temp_path, _ = download_generic_pdf(pdf_url, UPLOAD_DIR)
                
                if not temp_path:
                    print(f"Failed to download PDF from {pdf_url}")
                    return None
                    
                paper_content = extract_pdf_content(temp_path)
                try:
                    os.remove(temp_path)
                    print(f"Cleaned up temporary file: {temp_path}")
                except Exception as e:
                    print(f"Error cleaning up temp file {temp_path}: {e}")
                
                return {
                    "url": pdf_url,
                    "title": paper_title or "Unknown Title",
                    "abstract": abstract or "No abstract available",
                    "full_text": paper_content.get("full_text", ""),
                    "source": paper_source,
                }
            
            else:
                print(f"Unsupported source in dict: {paper_source}")
                return None
        
        # Case 3: URL string (direct external paper)
        elif isinstance(paper_id_or_url, str) and (
            paper_id_or_url.startswith("http://") or paper_id_or_url.startswith("https://")
        ):
            if "arxiv.org" in paper_id_or_url.lower():
                temp_path, paper_id = download_arxiv_pdf(paper_id_or_url, UPLOAD_DIR)
            else:
                temp_path, paper_id = download_generic_pdf(paper_id_or_url, UPLOAD_DIR)
                
            if not temp_path:
                print(f"Failed to download PDF from {paper_id_or_url}")
                return None
                
            paper_content = extract_pdf_content(temp_path)
            try:
                os.remove(temp_path)
                print(f"Cleaned up temporary file: {temp_path}")
            except Exception as e:
                print(f"Error cleaning up temp file {temp_path}: {e}")
                
            return {
                "url": paper_id_or_url,
                "title": paper_content.get("title", "Unknown Title"),
                "abstract": paper_content.get("abstract", "No abstract available"),
                "full_text": paper_content.get("full_text", ""),
                "source": "web"
            }
        
        else:
            print(f"Invalid paper_id_or_url format: {paper_id_or_url}")
            return None

    except Exception as e:
        print(f"Error in get_paper_content: {e}")
        import traceback
        traceback.print_exc()
        return None
        
def download_generic_pdf(url, save_dir):
    """
    Downloads a PDF from a generic URL (non-arXiv).
    Returns the path to the downloaded file.
    """
    try:
        response = requests.get(url, stream=True, headers={'User-Agent': 'ResearchAssistant/1.0'}, timeout=20)
        response.raise_for_status()
        
        filename = f"temp_download_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return filepath, None
    except Exception as e:
        print(f"Error downloading from {url}: {e}")
        return None, None
        
def generate_comparison_report(paper1, paper2):
    """
    Uses the LLM to generate a structured comparison report between two papers.
    """
    comparison_prompt = f"""
    You are a research assistant tasked with comparing two academic papers.
    
    PAPER 1:
    Title: {paper1.get('title', 'Unknown')}
    Abstract: {paper1.get('abstract', 'N/A')}
    
    PAPER 2:
    Title: {paper2.get('title', 'Unknown')}
    Abstract: {paper2.get('abstract', 'N/A')}
    
    Please generate a 200-300 word structured comparison report with the following sections:
    1. Research Goals: Compare the main objectives of both papers
    2. Methodologies: Compare the approaches and techniques used
    3. Key Contributions: Identify the main contributions of each paper
    4. Strengths and Weaknesses: Analyze the strengths and limitations of each approach
    5. Similarities and Differences: Highlight the key similarities and differences
    
    Present this as a well-structured report with clear section headings.
    """
    
    response = llm(comparison_prompt)
    return response

def compare_papers(paper_specs: str) -> str:
    """
    Compares two research papers and generates a structured report.
    Input format: "paper1_id_or_url,paper2_id_or_url" or "last_uploaded,retrieval_index:N" (0-based index)
    Returns a structured comparison report.
    """
    print(f"Original paper_specs: {paper_specs}")
    try:
        # Clean the input
        paper_specs = paper_specs.strip()
        paper_specs = re.sub(r'(retrieval_index:\d+).*', r'\1', paper_specs, flags=re.DOTALL)
        paper_specs = re.sub(r'\s*\(.*?\)\s*', '', paper_specs)
        print(f"Cleaned paper_specs: {paper_specs}")

        if paper_specs.lower().startswith("last_uploaded"):
            parts = paper_specs.split(",")
            print(f"Parts: {parts}")
            
            if len(parts) != 2:
                return "Error: Invalid format. Use 'last_uploaded,retrieval_index:N'"
                
            latest_uploaded = conversation_state.get_latest_uploaded_paper()
            print(f"Latest uploaded: {latest_uploaded}")
            if not latest_uploaded:
                return "Error: No paper has been uploaded yet."
            paper1_id = latest_uploaded["id"]
            print(f"Paper1 ID: {paper1_id}")
            
            retrieval_part = parts[1].strip()
            print(f"Retrieval part: {retrieval_part}")
            if not retrieval_part.startswith("retrieval_index:"):
                return "Error: Invalid format for retrieval paper. Use 'retrieval_index:N'"
            
            try:
                index_match = re.match(r'retrieval_index:(\d+)', retrieval_part)
                print(f"Index match: {index_match}")
                if not index_match:
                    return "Error: Invalid retrieval index format - it should be 'retrieval_index:N'"
                retrieval_idx = int(index_match.group(1))
                retrieved_papers = conversation_state.get_retrieved_papers()
                print(f"Retrieved papers in compare_papers: {retrieved_papers}")
                if not retrieved_papers or retrieval_idx >= len(retrieved_papers):
                    return f"Error: No retrieved paper at index {retrieval_idx}. Available papers: {len(retrieved_papers)}"
                
                paper2_spec = retrieved_papers[retrieval_idx]
                print(f"Paper2 spec: {paper2_spec}")
            except (ValueError, AttributeError):
                return "Error: Invalid retrieval index format"
        else:
            parts = paper_specs.split(",")
            if len(parts) != 2:
                return "Error: Please provide exactly two paper IDs or URLs for comparison"
            paper1_id = parts[0].strip()
            paper2_spec = parts[1].strip()
            
        paper1_content = get_paper_content(paper1_id)
        paper2_content = get_paper_content(paper2_spec)
        
        if not paper1_content or not paper2_content:
            return "Error: Could not retrieve content for one or both papers"
            
        print(f"Comparing papers: {paper1_content['title']} vs {paper2_content['title']}")
        comparison_report = generate_comparison_report(paper1_content, paper2_content)
        return comparison_report
        
    except Exception as e:
        print(f"Error in compare_papers: {e}")
        import traceback
        traceback.print_exc()
        return f"An error occurred while comparing papers: {str(e)}"

compare_papers_tool = Tool(
    name="compare_papers_and_generate_report",
    func=compare_papers,
    return_direct=True,
    description="Compares two research papers and generates a structured comparison report. Input format: 'paper1_id_or_url,paper2_id_or_url' or 'last_uploaded,retrieval_index:1' (0-based index). Use this when asked to compare papers or generate a comparison report.",
)

tools = [upload_pdf_tool, search_internal_papers_tool, search_external_papers_tool, compare_papers_tool]

class SearchFeedback(Base):
    __tablename__ = "search_feedback"
    id = Column(Integer, primary_key=True)
    query = Column(Text, nullable=False)
    paper_id = Column(Integer, nullable=False)
    is_relevant = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(engine)

templates = Jinja2Templates(directory="templates")

app = FastAPI(
    title="Research Paper Assistant",
    description="A natural language interface for managing research papers",
    version="1.0.0"
)

UPLOAD_DIR = "uploaded_papers"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount(f"/{UPLOAD_DIR}", StaticFiles(directory=UPLOAD_DIR), name=UPLOAD_DIR)

@app.on_event("startup")
async def startup_event():
    init_db()
    try:
        chroma_client.get_collection(name="research_papers")
    except ValueError:
        chroma_client.create_collection(name="research_papers")
    print("ChromaDB collection 'research_papers' ensured.")
    
    global memory, faiss_store
    global conversation_state
    conversation_state = ConversationState()
    print("Conversation state initialized.")
    
    if os.path.exists(memory_index_path):
        try:
            print(f"Loading FAISS index from {memory_index_path}...")
            faiss_store = FAISS.load_local(memory_index_path, instructor_embeddings, allow_dangerous_deserialization=True)
            if faiss_store:
                memory = VectorStoreRetrieverMemory(retriever=faiss_store.as_retriever(search_kwargs={"k": 5}))
                print("FAISS index loaded and memory initialized successfully.")
            else:
                print("FAISS index loaded as None.")
        except Exception as e:
            print(f"Error loading FAISS index: {e}. Will start with an empty index.")
            faiss_store = None
            memory = None
    else:
        print("No FAISS index found. Will start with an empty index.")
        faiss_store = None
        memory = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    papers = await get_upload_history()
    return templates.TemplateResponse("index.html", {"request": request, "papers": papers})

@app.post("/process_input")
async def process_input(background_tasks: BackgroundTasks, user_input: str = Form(...)):
    global memory, tools, conversation_state
    
    if not hasattr(app.state, 'conversation_state'):
        app.state.conversation_state = ConversationState()
    
    try:
        print(f"Received user input: {user_input}")
        def upload_pdf_with_background(source_path_input: str) -> str:
            return upload_pdf(source_path_input, background_tasks=background_tasks)
        
        upload_tool = Tool(
            name="upload_paper_from_path_or_url",
            func=upload_pdf_with_background,
            return_direct=True,
            description="Uploads a PDF research paper from a local file path or URL..."
        )
        
        tools[:] = [upload_tool, search_internal_papers_tool, search_external_papers_tool, compare_papers_tool] 
        
        tool_summaries = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        
        system_prompt_for_agent = """
        You are a research assistant that helps with paper management, search tasks, and paper comparison.

        When deciding which tool to use:
        - For uploading papers: Use "upload_paper_from_path_or_url" when the user provides a URL or file path.
        - For searching existing papers: Use "search_MY_LOCAL_LIBRARY_of_papers" when the user mentions finding papers about a topic in their database, or uses keywords like 'my library', 'internal', 'uploaded papers', 'my papers', or 'previously uploaded papers'.
        - For finding new papers: Use "search_ONLINE_ACADEMIC_DATABASES_for_new_papers" ONLY when the user explicitly requests new or recent papers with keywords like 'recent papers', 'new papers', 'latest papers', 'arxiv', 'semantic scholar', 'web', 'online', or 'external search'. Append 'source:arxiv' to the query unless 'semantic scholar' or 's2' is specified.
        - For comparing papers: Use "compare_papers_and_generate_report" when the user asks to compare papers or generate a comparative analysis. If the user references 'the Nth paper you found', use the papers already stored from the most recent search (internal or external) in conversation_state.retrieved_papers.

        ACTION INPUT FORMATTING:
        - For upload: Provide ONLY the URL or file path.
        - For search: Provide ONLY the search query string.
        - For comparison: Use format "last_uploaded,retrieval_index:N" to compare the most recently uploaded paper with the Nth retrieved paper (0-based index) from the latest search.

        PAPER COMPARISON EXAMPLES:
        - "Compare the paper I just uploaded with the most recent paper on self-supervised learning you found" → Use "last_uploaded,retrieval_index:0" with papers from the prior search.
        - "Compare the uploaded paper with the second paper you found on diffusion models" → Use "last_uploaded,retrieval_index:1" with papers from the prior search.
        - "Compare the paper I uploaded with the first web paper you found" → Use "last_uploaded,retrieval_index:0" with papers from the prior external search.

        Respond IMMEDIATELY after getting tool results. Do not perform additional searches unless explicitly requested. Use existing search results for comparisons when referenced.
        
        Examples:
        User: "Upload this paper: https://arxiv.org/pdf/2303.12712.pdf"
        Action: upload_paper_from_path_or_url
        Action Input: https://arxiv.org/pdf/2303.12712.pdf
        
        User: "I want to upload my latest researach paper. Here is the file: /home/user/paper.pdf"
        action: upload_paper_from_path_or_url
        Action Input: /home/user/paper.pdf
        
        User: "Here is a paper I'd like to add to the database: https://arxiv.org/pdf/2303.12712.pdf"
        Action: upload_paper_from_path_or_url
        Action Input: https://arxiv.org/pdf/2303.12712.pdf

        User: "Find recent papers about large language models"
        Action: search_ONLINE_ACADEMIC_DATABASES_for_new_papers
        Action Input: large language models source:arxiv or source:semantic_scholar max_results:5
        
        User: "Can you find the latest papers from ICLR about diffusion models?"
        Action: search_ONLINE_ACADEMIC_DATABASES_for_new_papers
        Action Input: diffusion models source:arxiv or source:semantic_scholar max_results:5
        
        User: "Look for recent ACL papers about multilingual transformers."
        Action: search_ONLINE_ACADEMIC_DATABASES_for_new_papers
        Action Input: multilingual transformers source:arxiv or source:semantic_scholar max_results:5

        User: "Search my library for transformer papers"
        Action: search_MY_LOCAL_LIBRARY_of_papers
        Action Input: transformer
        
        User: "Find papers about contrastive learning for vision models."
        Action: search_MY_LOCAL_LIBRARY_of_papers
        Action Input: contrastive learning for vision models
        
        User: "Find papers on self-supervised learning for robotics."
        Action: search_MY_LOCAL_LIBRARY_of_papers
        Action Input: self-supervised learning for robotics
        """
        
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            handle_parsing_errors=True, 
            max_iterations=1,
            agent_kwargs={
                "prefix": system_prompt_for_agent,
            },
            early_stopping_method="generate"
        )

        try:
            async def run_agent():
                return agent.run(input=user_input)
            
            agent_response = await asyncio.wait_for(run_agent(), timeout=60.0)
            print(f"Agent response (before JSON): {agent_response} (type: {type(agent_response)})")
            return JSONResponse({"message": str(agent_response)})
        except asyncio.TimeoutError:
            return JSONResponse({
                "message": "The operation timed out. Please try a simpler query or try again later."
            }, status_code=504)
                
        except Exception as agent_error:
            print(f"Agent execution error: {agent_error}")
            return JSONResponse({
                "message": f"I encountered an error processing your request: {str(agent_error)}"
            }, status_code=500)

    except FileNotFoundError as e_fnf:
        print(f"Tool raised FileNotFoundError: {e_fnf}")
        return JSONResponse({"message": f"Tool Error: {str(e_fnf)}"}, status_code=400)
    except Exception as e:
        print(f"Error in process_input: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"message": f"General Error: {str(e)}"}, status_code=500)
    
@app.get("/upload_history")
async def get_upload_history():
    db = next(get_db())
    papers_db = db.query(Paper).order_by(Paper.created_at.desc()).all()
    print(f"Found {len(papers_db)} papers in DB for history.")
    result = []
    for p in papers_db:
        paper_item = {
            "id": p.id,
            "title": p.title or "Untitled Paper",
            "abstract": (p.abstract[:300] + "..." if p.abstract else "No abstract available."),
            "uploaded": p.created_at.strftime("%Y-%m-%d %H:%M") if p.created_at else "Date Unknown",
            "file_path": f"/{UPLOAD_DIR}/{quote(p.file_path)}" if p.file_path else "#",
            "source_type": p.source or "Unknown Source",
            "original_source": p.source_path or "N/A"
        }
        result.append(paper_item)
    db.close()
    return result

@app.post("/search_feedback")
async def add_search_feedback(
    query: str = Form(...),
    paper_id: int = Form(...),
    is_relevant: bool = Form(...)
):
    db = next(get_db())
    try:
        feedback = SearchFeedback(
            query=query,
            paper_id=paper_id,
            is_relevant=1 if is_relevant else 0
        )
        db.add(feedback)
        db.commit()
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        db.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        db.close()

if __name__ == "__main__":
    print("To run the app, use: uvicorn main:app --reload (replace main with your script file name)")
    pass