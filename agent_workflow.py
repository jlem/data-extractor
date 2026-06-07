import os
import base64
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field, create_model
from dotenv import load_dotenv

# LangChain and LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# Import tools directly from our MCP server module
from mcp_server import (
    read_digital_pdf, 
    render_pdf_to_images, 
    fetch_web_content, 
    db_execute
)

# Load environment variables (such as GOOGLE_API_KEY)
load_dotenv()

# Check for API key
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# Define LangGraph State Schema
class AgentState(TypedDict):
    source_type: str            # "pdf_digital", "pdf_scanned", "image", "url"
    source_path: str            # File path or URL
    image_paths: List[str]      # List of rendered image paths (for vision processing)
    extracted_text: str        # Extracted text (for text processing)
    table_name: str             # Dynamically generated database table name
    columns: List[Dict[str, str]] # List of {"name": "...", "type": "...", "description": "..."}
    extracted_rows: List[Dict[str, Any]] # List of rows extracted from the source matching the schema
    log: List[str]              # Logs to display in the Streamlit UI

# Pydantic models for structured schema architecture output
class ColumnDefinition(BaseModel):
    name: str = Field(description="Column name in snake_case. Must be a valid SQLite column identifier (no spaces, special characters).")
    type: str = Field(description="SQLite database type. Must be one of: TEXT, INTEGER, REAL.")
    description: str = Field(description="Short explanation of what this column represents.")

class SchemaResponse(BaseModel):
    table_name: str = Field(description="Descriptive table name in snake_case summarizing the table's contents (e.g. invoice_items, user_directory).")
    columns: List[ColumnDefinition] = Field(description="Checklist of columns identified from the tabular data source.")

# Helper function to convert local image to base64 dict for Gemini Vision API
def get_image_message_content(image_path: str) -> dict:
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{encoded}"
        }
    }

# ----------------- GRAPH NODES -----------------

def ingest_node(state: AgentState) -> Dict[str, Any]:
    """
    Ingests raw source material: reads digital PDFs, fetches URL HTML content, 
    or prepares image references for vision workflows.
    """
    source_type = state["source_type"]
    source_path = state["source_path"]
    logs = list(state.get("log", []))
    logs.append(f"🔄 Starting ingestion for source: {source_path} ({source_type})")
    
    extracted_text = ""
    image_paths = []
    
    try:
        if source_type == "pdf_digital":
            logs.append("Reading digital PDF text...")
            extracted_text = read_digital_pdf(source_path)
            if extracted_text.startswith("Error"):
                raise ValueError(extracted_text)
            logs.append(f"Successfully extracted {len(extracted_text)} characters of text.")
            
        elif source_type == "url":
            logs.append(f"Scraping web page: {source_path}...")
            extracted_text = fetch_web_content(source_path)
            if extracted_text.startswith("Error"):
                raise ValueError(extracted_text)
            logs.append(f"Successfully extracted {len(extracted_text)} characters from webpage.")
            
        elif source_type == "pdf_scanned":
            logs.append("PDF appears to be scanned or contains images. Rendering pages to images...")
            paths = render_pdf_to_images(source_path)
            if paths and paths[0].startswith("Error"):
                raise ValueError(paths[0])
            image_paths = paths
            logs.append(f"Rendered {len(image_paths)} pages to temporary images for vision processing.")
            
        elif source_type == "image":
            logs.append(f"Image source identified: {source_path}")
            image_paths = [source_path]
            
        else:
            raise ValueError(f"Unknown source type: {source_type}")
            
    except Exception as e:
        logs.append(f"❌ Ingestion failed: {str(e)}")
        return {"log": logs, "extracted_text": "", "image_paths": []}
        
    return {
        "extracted_text": extracted_text,
        "image_paths": image_paths,
        "log": logs
    }

def schema_architect_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes the extracted text or image list, identifies properties, 
    and outputs a structured database schema.
    """
    logs = list(state.get("log", []))
    logs.append("🧠 Activating Schema Architect Agent...")
    
    if not api_key:
        logs.append("❌ API Key missing. Please set GEMINI_API_KEY in your environment.")
        return {"log": logs}
        
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.1)
    structured_llm = llm.with_structured_output(SchemaResponse)
    
    system_prompt = (
        "You are an expert DB Schema Architect. Your job is to analyze the data structure from "
        "the provided input (text or images) and define a structured database table schema to store "
        "every row of data found. In your response:\n"
        "1. Identify a logical table name in snake_case (e.g. employee_salaries, invoice_details).\n"
        "2. Break down the spreadsheet or table columns. Columns must be snake_case, unique, and valid SQL names.\n"
        "3. Map values to SQLite types (TEXT, INTEGER, or REAL).\n"
        "Ensure columns are atomic and reflect all the data columns in the document."
    )
    
    try:
        # Build prompt messages based on source modality
        messages = [HumanMessage(content=system_prompt)]
        
        if state["image_paths"]:
            # Multimodal inputs
            logs.append(f"Sending {len(state['image_paths'])} document image(s) to Gemini Vision...")
            user_content = [{"type": "text", "text": "Analyze the table layout in these document image(s) and define the schema:"}]
            for path in state["image_paths"]:
                if os.path.exists(path):
                    user_content.append(get_image_message_content(path))
            messages.append(HumanMessage(content=user_content))
        else:
            # Text inputs
            logs.append("Sending raw text content to Gemini Text model...")
            sample_text = state["extracted_text"][:20000] # Limit input text to avoid massive payload
            messages.append(HumanMessage(content=f"Analyze this raw extracted document text and define the schema:\n\n{sample_text}"))
            
        schema_result = structured_llm.invoke(messages)
        
        # Parse output
        table_name = schema_result.table_name
        columns = [{"name": c.name, "type": c.type, "description": c.description} for c in schema_result.columns]
        
        logs.append(f"✅ Schema Designed! Table: '{table_name}' with {len(columns)} columns:")
        for col in columns:
            logs.append(f"  - {col['name']} ({col['type']}): {col['description']}")
            
        return {
            "table_name": table_name,
            "columns": columns,
            "log": logs
        }
    except Exception as e:
        logs.append(f"❌ Schema extraction failed: {str(e)}")
        return {"log": logs}

def db_setup_node(state: AgentState) -> Dict[str, Any]:
    """
    Creates the table dynamically in the SQLite database based on the schema node's plan.
    """
    logs = list(state.get("log", []))
    table_name = state["table_name"]
    columns = state["columns"]
    
    if not table_name or not columns:
        logs.append("❌ Database creation skipped: schema is missing.")
        return {"log": logs}
        
    logs.append(f"💾 Creating database table '{table_name}'...")
    
    # Build CREATE TABLE command
    column_defs = []
    for col in columns:
        column_defs.append(f"{col['name']} {col['type']}")
        
    create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
    create_sql += "  id INTEGER PRIMARY KEY AUTOINCREMENT,\n  "
    create_sql += ",\n  ".join(column_defs)
    create_sql += "\n);"
    
    # Run creation tool
    result = db_execute(create_sql)
    
    if result == "Success":
        logs.append(f"✅ Table '{table_name}' created successfully in SQLite database.")
    else:
        logs.append(f"❌ Database error: {result}")
        
    return {"log": logs}

def data_extraction_node(state: AgentState) -> Dict[str, Any]:
    """
    Extracts all rows of data matching the schema structure from the text or image.
    Uses runtime-generated Pydantic models to guarantee matching shapes.
    """
    logs = list(state.get("log", []))
    table_name = state["table_name"]
    columns = state["columns"]
    
    if not table_name or not columns:
        logs.append("❌ Structured extraction skipped: no table schema defined.")
        return {"log": logs}
        
    logs.append("🕵️ Running Structured Data Extractor Agent...")
    
    if not api_key:
        logs.append("❌ API Key missing. Skipping data extraction.")
        return {"log": logs}
        
    # 1. Create dynamic Pydantic model at runtime for structured output
    fields = {}
    for col in columns:
        col_type = col["type"]
        py_type = str
        if col_type == "INTEGER":
            py_type = int
        elif col_type == "REAL":
            py_type = float
            
        # Define field with description, default to None (nullable SQLite fields)
        fields[col["name"]] = (Optional[py_type], Field(description=col["description"], default=None))
        
    # Dynamically build a Row representation model
    RowModel = create_model("RowModel", **fields)
    
    # Dynamically build a Table container model
    class ExtractedDataset(BaseModel):
        rows: List[RowModel] = Field(description="List of records extracted from the document.")
        
    # Initialize the LLM and bind schema
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.1)
    structured_extractor = llm.with_structured_output(ExtractedDataset)
    
    system_prompt = (
        "You are an advanced data extraction agent. Your job is to extract all raw data records "
        "matching the requested table schema structure. Look carefully at the document (text or images) "
        "and pull out all tabular rows. Ensure numbers are parsed as numbers, dates/strings as text. "
        "Do not leave any valid rows behind."
    )
    
    try:
        messages = [HumanMessage(content=system_prompt)]
        
        if state["image_paths"]:
            logs.append("Extracting structured rows from document images via Gemini Vision...")
            user_content = [{"type": "text", "text": "Extract all rows from the document layout matching the schema:"}]
            for path in state["image_paths"]:
                if os.path.exists(path):
                    user_content.append(get_image_message_content(path))
            messages.append(HumanMessage(content=user_content))
        else:
            logs.append("Extracting structured rows from raw text...")
            messages.append(HumanMessage(content=f"Extract all rows from the document text matching the schema:\n\n{state['extracted_text']}"))
            
        dataset_result = structured_extractor.invoke(messages)
        extracted_rows = [row.model_dump() for row in dataset_result.rows]
        
        logs.append(f"✅ Extraction complete! Identified {len(extracted_rows)} total records.")
        return {
            "extracted_rows": extracted_rows,
            "log": logs
        }
    except Exception as e:
        logs.append(f"❌ Structured extraction failed: {str(e)}")
        return {"log": logs}

def db_load_node(state: AgentState) -> Dict[str, Any]:
    """
    Saves the extracted structured rows into the dynamically created SQLite table.
    """
    logs = list(state.get("log", []))
    table_name = state["table_name"]
    rows = state["extracted_rows"]
    
    if not table_name or not rows:
        logs.append("❌ Load step skipped: no table name or rows found.")
        return {"log": logs}
        
    logs.append(f"📥 Loading {len(rows)} rows into database table '{table_name}'...")
    
    success_count = 0
    fail_count = 0
    
    # Loop over rows and insert them individually
    for row in rows:
        cols_str = []
        vals_str = []
        for col_name, val in row.items():
            cols_str.append(col_name)
            if val is None:
                vals_str.append("NULL")
            elif isinstance(val, (int, float)):
                vals_str.append(str(val))
            else:
                # Basic escaping for single quotes in SQLite
                escaped_val = str(val).replace("'", "''")
                vals_str.append(f"'{escaped_val}'")
                
        insert_sql = f"INSERT INTO {table_name} ({', '.join(cols_str)}) VALUES ({', '.join(vals_str)});"
        
        result = db_execute(insert_sql)
        if result == "Success":
            success_count += 1
        else:
            fail_count += 1
            
    logs.append(f"🏁 Load finished! {success_count} rows loaded successfully. {fail_count} errors.")
    return {"log": logs}

# ----------------- ASSEMBLE GRAPH -----------------

def create_workflow():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("schema_architect", schema_architect_node)
    workflow.add_node("db_setup", db_setup_node)
    workflow.add_node("data_extractor", data_extraction_node)
    workflow.add_node("db_load", db_load_node)
    
    # Establish Edges
    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "schema_architect")
    workflow.add_edge("schema_architect", "db_setup")
    workflow.add_edge("db_setup", "data_extractor")
    workflow.add_edge("data_extractor", "db_load")
    workflow.add_edge("db_load", END)
    
    return workflow.compile()
