import os
import sqlite3
import requests
from bs4 import BeautifulSoup
from PIL import Image
import fitz  # PyMuPDF
from fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP("DataExtractorServer")

DB_PATH = os.path.abspath("extracted_data.db")
TEMP_IMAGE_DIR = os.path.abspath("temp_images")

# Helper to ensure directories exist
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

@mcp.tool()
def read_digital_pdf(pdf_path: str) -> str:
    """
    Extracts raw text content from a digital-native (searchable) PDF file.
    
    Args:
        pdf_path: The absolute or relative path to the PDF file.
        
    Returns:
        The extracted text with page number dividers, or an error message.
    """
    if not os.path.exists(pdf_path):
        return f"Error: File not found at {pdf_path}"
    
    try:
        doc = fitz.open(pdf_path)
        extracted_text = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            extracted_text.append(f"--- Page {page_num} ---\n{text}")
        return "\n\n".join(extracted_text)
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

@mcp.tool()
def render_pdf_to_images(pdf_path: str) -> list[str]:
    """
    Renders pages of a PDF document to PNG images. This is essential for scanned
    PDFs to allow multimodal LLM models to perform OCR and layout analysis.
    
    Args:
        pdf_path: The absolute or relative path to the PDF file.
        
    Returns:
        A list of absolute file paths to the generated PNG images.
    """
    if not os.path.exists(pdf_path):
        return [f"Error: File not found at {pdf_path}"]
        
    try:
        doc = fitz.open(pdf_path)
        saved_paths = []
        for page_num, page in enumerate(doc, 1):
            # Render page to a high-quality image (150 DPI)
            pix = page.get_pixmap(dpi=150)
            img_filename = f"page_{page_num}.png"
            img_path = os.path.join(TEMP_IMAGE_DIR, img_filename)
            pix.save(img_path)
            saved_paths.append(img_path)
        return saved_paths
    except Exception as e:
        return [f"Error rendering PDF pages: {str(e)}"]

@mcp.tool()
def fetch_web_content(url: str) -> str:
    """
    Fetches raw HTML content from a URL and extracts clean readable text.
    
    Args:
        url: The web page URL to fetch.
        
    Returns:
        A text representation of the web page with headers, links, and tables intact.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
            
        # Extract plain text with some spacing preserved
        text = soup.get_text(separator="\n")
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = "\n".join(chunk for chunk in chunks if chunk)
        
        return clean_text
    except Exception as e:
        return f"Error fetching web page: {str(e)}"

@mcp.tool()
def db_execute(query: str) -> str:
    """
    Executes a SQL command on the local SQLite database. Use this tool to
    create tables (DDL) and insert or update data (DML).
    
    Args:
        query: The raw SQL query to run.
        
    Returns:
        'Success' or an error message.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        conn.close()
        return "Success"
    except Exception as e:
        return f"Database error: {str(e)}"

@mcp.tool()
def db_query(query: str) -> list[dict]:
    """
    Queries the local SQLite database and returns the rows as a list of dictionaries.
    Use this for SELECT queries.
    
    Args:
        query: The SELECT query to execute.
        
    Returns:
        A list of dictionaries representing the query rows, or an empty list if no results.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        result = [dict(row) for row in rows]
        conn.close()
        return result
    except Exception as e:
        return [{"error": str(e)}]

@mcp.tool()
def db_get_schema(table_name: str) -> list[dict]:
    """
    Retrieves the column details (name, type, nullability) of a specific database table.
    
    Args:
        table_name: The table to inspect.
        
    Returns:
        A list of dictionaries containing column information.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        conn.close()
        
        # PRAGMA table_info returns (cid, name, type, notnull, dflt_value, pk)
        return [
            {
                "cid": col[0],
                "name": col[1],
                "type": col[2],
                "notnull": bool(col[3]),
                "default_value": col[4],
                "primary_key": bool(col[5])
            }
            for col in columns
        ]
    except Exception as e:
        return [{"error": str(e)}]

if __name__ == "__main__":
    mcp.run()
