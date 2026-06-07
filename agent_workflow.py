import os
import re
import time
import uuid
import json
import sqlite3
import asyncio
import base64
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field, create_model
from dotenv import load_dotenv

# LangChain and LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langchain_core.callbacks import BaseCallbackHandler

import fitz  # PyMuPDF

# Import tools directly from our MCP server module
from mcp_server import (
    read_digital_pdf, 
    render_pdf_to_images, 
    fetch_web_content, 
    db_execute,
    db_execute_many
)

# Load environment variables (such as GOOGLE_API_KEY)
load_dotenv()

# Check for API key
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# Define structures for Ingestion
class IngestedDoc(TypedDict):
    source_name: str
    source_type: str
    extracted_text: str          # Full combined text
    extracted_pages: List[str]    # Text split page-by-page
    image_paths: List[str]        # List of rendered page images

# Define Pydantic models for dynamic joining metadata
class DocumentRole(BaseModel):
    source_name: str = Field(description="The exact name of the document as shown in the ingested documents list.")
    role: str = Field(description="Role of this document: 'primary' (the main records are extracted from here) or 'lookup' (contains extra columns to be joined into the primary records).")
    join_key: Optional[str] = Field(description="For lookup documents, the column name in the primary table used to match/join. Leave empty for primary documents.")
    lookup_key: Optional[str] = Field(description="For lookup documents, the column name in this lookup document that matches the primary table's join_key. Leave empty for primary documents.")

# Define LangGraph State Schema supporting multiple sources, instructions, and telemetry
class AgentState(TypedDict):
    sources: List[Dict[str, str]]       # List of {"type": "...", "path": "..."}
    user_instructions: str              # Custom instructions
    custom_table_name: str              # Optional user-specified database table name
    ingested_docs: List[IngestedDoc]    # Content extracted per source document
    table_name: str                     # Dynamically generated database table name
    columns: List[Dict[str, str]]       # List of {"name": "...", "type": "...", "description": "...", "source_doc": "..."}
    document_roles: List[Dict[str, Any]] # Document roles and join mapping keys
    extracted_rows: List[Dict[str, Any]] # Combined list of rows extracted from all sources
    log: List[str]                      # Running console log history
    
    # TELEMETRY FIELDS
    job_id: str
    prompt_tokens: int
    completion_tokens: int
    ingest_duration: float
    schema_architect_duration: float
    db_setup_duration: float
    extraction_duration: float
    db_load_duration: float
    total_duration: float

# Pydantic models for structured schema architecture output
class ColumnDefinition(BaseModel):
    name: str = Field(description="Column name in snake_case. Must be unique and a valid SQLite column identifier.")
    type: str = Field(description="SQLite database type. Must be one of: TEXT, INTEGER, REAL.")
    description: str = Field(description="Short explanation of what this column represents.")
    source_doc: str = Field(description="The source_name of the document this column's data comes from.")

class SchemaResponse(BaseModel):
    table_name: str = Field(description="Descriptive table name in snake_case summarizing the table's contents.")
    columns: List[ColumnDefinition] = Field(description="Checklist of columns identified from the tabular data source.")
    document_roles: List[DocumentRole] = Field(description="Assign a role ('primary' or 'lookup') to each source document based on layout and user instructions.")

# Custom Callback to dynamically capture token count usage metrics from Gemini calls
class TokenUsageCallback(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def on_llm_end(self, response, **kwargs):
        # 1. Check direct response level token_usage
        if response.llm_output:
            usage = response.llm_output.get("token_usage") or {}
            if usage:
                self.prompt_tokens += usage.get("prompt_tokens") or usage.get("input_token_count") or 0
                self.completion_tokens += usage.get("completion_tokens") or usage.get("output_token_count") or 0
                return

        # 2. Check if gen message has usage_metadata (standard in LangChain for Gemini)
        for generations in response.generations:
            for gen in generations:
                message = getattr(gen, "message", None)
                if message is not None:
                    usage = getattr(message, "usage_metadata", None) or {}
                    if usage:
                        self.prompt_tokens += usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                        self.completion_tokens += usage.get("output_tokens") or usage.get("completion_tokens") or 0
                        continue

                # 3. Fall back to generation info metadata block
                info = gen.generation_info or {}
                usage = info.get("usage_metadata") or info.get("token_usage") or {}
                if usage:
                    self.prompt_tokens += usage.get("prompt_tokens") or usage.get("input_token_count") or usage.get("input_tokens") or 0
                    self.completion_tokens += usage.get("completion_tokens") or usage.get("output_token_count") or usage.get("output_tokens") or 0

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

# Helper to normalize SQL table/column identifiers
def clean_sql_identifier(name: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', name).strip('_').lower()
    if cleaned and cleaned[0].isdigit():
        cleaned = f"col_{cleaned}"
    return cleaned

# Helper to map a filename to the closest actual source name
def find_closest_source(doc_name: str, actual_sources: List[str]) -> str:
    if not actual_sources:
        return doc_name
    if doc_name in actual_sources:
        return doc_name
    for s in actual_sources:
        if s.lower() == doc_name.lower():
            return s
    for s in actual_sources:
        if s.lower() in doc_name.lower() or doc_name.lower() in s.lower():
            return s
    return actual_sources[0]

# Helper to normalize key values for robust string-matching in joins
def normalize_join_key(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    return str(val).strip().lower()

# Save performance execution run telemetry into separate SQLite database
def save_telemetry(state: AgentState, total_elapsed: float):
    db_path = os.path.abspath("pipeline_telemetry.db")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create metrics table if not exists
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS job_telemetry (
            job_id TEXT PRIMARY KEY,
            timestamp DATETIME,
            sources_list TEXT,
            records_count INTEGER,
            ingest_duration REAL,
            schema_architect_duration REAL,
            db_setup_duration REAL,
            extraction_duration REAL,
            db_load_duration REAL,
            total_duration REAL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER
        );
        """)
        
        sources_json = json.dumps(state.get("sources", []))
        prompt_t = state.get("prompt_tokens", 0)
        completion_t = state.get("completion_tokens", 0)
        total_t = prompt_t + completion_t
        
        cursor.execute("""
        INSERT INTO job_telemetry (
            job_id, timestamp, sources_list, records_count,
            ingest_duration, schema_architect_duration, db_setup_duration,
            extraction_duration, db_load_duration, total_duration,
            prompt_tokens, completion_tokens, total_tokens
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            state.get("job_id", str(uuid.uuid4())),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            sources_json,
            len(state.get("extracted_rows", [])),
            state.get("ingest_duration", 0.0),
            state.get("schema_architect_duration", 0.0),
            state.get("db_setup_duration", 0.0),
            state.get("extraction_duration", 0.0),
            state.get("db_load_duration", 0.0),
            total_elapsed,
            prompt_t,
            completion_t,
            total_t
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        # Fallback to standard print if logging database connection fails
        print(f"[WARNING] Telemetry database save failed: {str(e)}")

# ----------------- GRAPH NODES -----------------

def ingest_node(state: AgentState) -> Dict[str, Any]:
    """
    Iterates over all provided input sources and ingests their content.
    """
    start_time = time.perf_counter()
    
    sources = state["sources"]
    user_instructions = state.get("user_instructions", "").strip()
    logs = list(state.get("log", []))
    logs.append(f"[INGEST] Starting ingestion for {len(sources)} sources...")
    
    if user_instructions:
        logs.append(f"[INFO] Ingestion node is aware of global custom instructions: '{user_instructions[:40]}...'")
        
    ingested_docs = []
    
    for idx, src in enumerate(sources, 1):
        source_type = src["type"]
        source_path = src["path"]
        source_name = os.path.basename(source_path) if source_type != "url" else source_path
        
        logs.append(f"[FILE] Processing source {idx}/{len(sources)}: '{source_name}' ({source_type})")
        extracted_text = ""
        extracted_pages = []
        image_paths = []
        
        try:
            if source_type == "pdf_digital":
                if not os.path.exists(source_path):
                    raise FileNotFoundError(f"File not found at {source_path}")
                doc = fitz.open(source_path)
                for page_num, page in enumerate(doc, 1):
                    p_text = page.get_text()
                    extracted_pages.append(p_text)
                doc.close()
                extracted_text = "\n\n".join([f"--- Page {i} ---\n{t}" for i, t in enumerate(extracted_pages, 1)])
                logs.append(f"  Successfully extracted {len(extracted_pages)} pages ({len(extracted_text)} characters).")
                
            elif source_type == "url":
                extracted_text = fetch_web_content(source_path)
                if extracted_text.startswith("Error"):
                    raise ValueError(extracted_text)
                extracted_pages = [extracted_text]
                logs.append(f"  Successfully scraped {len(extracted_text)} characters.")
                
            elif source_type == "pdf_scanned":
                paths = render_pdf_to_images(source_path)
                if paths and paths[0].startswith("Error"):
                    raise ValueError(paths[0])
                image_paths = paths
                extracted_pages = [f"[Page Image {i}]" for i in range(1, len(image_paths) + 1)]
                logs.append(f"  Rendered {len(image_paths)} pages to images for vision processing.")
                
            elif source_type == "image":
                image_paths = [source_path]
                extracted_pages = ["[Image Source]"]
                logs.append(f"  Image file ready.")
                
            else:
                raise ValueError(f"Unknown source type: {source_type}")
                
            ingested_docs.append({
                "source_name": source_name,
                "source_type": source_type,
                "extracted_text": extracted_text,
                "extracted_pages": extracted_pages,
                "image_paths": image_paths
            })
            
        except Exception as e:
            logs.append(f"  [ERROR] Failed to ingest '{source_name}': {str(e)}")
            
    elapsed = time.perf_counter() - start_time
    
    return {
        "ingested_docs": ingested_docs,
        "log": logs,
        "ingest_duration": elapsed,
        "job_id": state.get("job_id") or str(uuid.uuid4())
    }

def schema_architect_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes all ingested materials together with the custom user instructions 
    to assign primary vs lookup roles and design a unified database table schema.
    """
    start_time = time.perf_counter()
    
    ingested_docs = state["ingested_docs"]
    user_instructions = state.get("user_instructions", "").strip()
    custom_table_name = state.get("custom_table_name", "").strip()
    logs = list(state.get("log", []))
    
    if not ingested_docs:
        logs.append("[ERROR] Ingestion yielded no documents. Skipping schema architect.")
        return {"log": logs}
        
    logs.append("[ARCHITECT] Activating Schema Architect Agent...")
    
    if not api_key:
        logs.append("[ERROR] API Key missing. Please set GEMINI_API_KEY in your environment.")
        return {"log": logs}
        
    # Build a combined text prompt describing all the source properties
    source_summaries = []
    all_image_paths = []
    actual_source_names = [doc["source_name"] for doc in ingested_docs]
    
    for doc in ingested_docs:
        doc_summary = f"Source Name: '{doc['source_name']}' (Type: {doc['source_type']})\n"
        if doc["extracted_text"]:
            doc_summary += f"Content Sample:\n{doc['extracted_text'][:2500]}\n"
        if doc["image_paths"]:
            doc_summary += f"Renders: {len(doc['image_paths'])} image page(s).\n"
            all_image_paths.extend(doc["image_paths"])
        source_summaries.append(doc_summary)
        
    combined_sources_desc = "\n---\n".join(source_summaries)
    
    # Setup LLM and link token usage callbacks
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.1, google_api_key=api_key)
    structured_llm = llm.with_structured_output(SchemaResponse)
    token_callback = TokenUsageCallback()
    
    system_prompt = (
        "You are an expert DB Schema Architect. Your job is to analyze the structures of the "
        "provided data source(s) and design a single, unified database schema table that can "
        "accommodate the rows of data from ALL sources.\n\n"
        "1. Classify each document into a role: 'primary' (holds the core list of records) "
        "or 'lookup' (contains extra columns to be joined into the primary records).\n"
        "2. You MUST prioritize and strictly follow any user instructions regarding which document is primary "
        "and which is a lookup, and which columns to match. Otherwise, autonomously infer them.\n"
        "3. For each column you define in the table, specify the 'source_doc' it should be extracted from.\n"
        "4. Table names and column names must be clean, snake_case, and valid SQL identifiers.\n"
        "5. A join_key and lookup_key must be specified for every lookup document so the join can be completed."
    )
    
    user_prompt = f"Here are the combined source descriptions:\n{combined_sources_desc}\n\n"
    if user_instructions:
        user_prompt += f"⚠️ USER CUSTOM SCHEMA INSTRUCTIONS:\n{user_instructions}\n\n"
    user_prompt += "Design a table name, document roles (primary vs lookup), and column list to represent the merged dataset."
    
    try:
        messages = [
            HumanMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # If there are images (scanned PDFs or direct screenshots), attach them for vision analysis
        if all_image_paths:
            logs.append(f"Attaching {len(all_image_paths)} images for multi-source vision analysis...")
            image_content = [{"type": "text", "text": "Inspect the visual table structures of the following images:"}]
            for path in all_image_paths:
                if os.path.exists(path):
                    image_content.append(get_image_message_content(path))
            messages.append(HumanMessage(content=image_content))
            
        schema_result = structured_llm.invoke(messages, config={"callbacks": [token_callback]})
        
        # Determine Table Name: user override or AI generated, sanitized
        raw_table_name = custom_table_name if custom_table_name else schema_result.table_name
        table_name = clean_sql_identifier(raw_table_name)
        if not table_name:
            table_name = "extracted_data_table"
            
        # Normalize columns and fix filenames if slightly modified by the LLM
        columns = []
        has_source_tracker = False
        
        for col in schema_result.columns:
            col_name = clean_sql_identifier(col.name)
            if not col_name or col_name == "id":
                continue
                
            if col_name == "source_document_name":
                has_source_tracker = True
                
            corrected_src_doc = find_closest_source(col.source_doc, actual_source_names)
            
            columns.append({
                "name": col_name,
                "type": col.type.upper(),
                "description": col.description,
                "source_doc": corrected_src_doc
            })
            
        if not has_source_tracker:
            columns.append({
                "name": "source_document_name",
                "type": "TEXT",
                "description": "Metadata column storing the filename or URL from which this row was extracted.",
                "source_doc": actual_source_names[0]
            })
            
        # Save Document Roles, correcting source names
        document_roles = []
        for doc_role in schema_result.document_roles:
            corrected_name = find_closest_source(doc_role.source_name, actual_source_names)
            document_roles.append({
                "source_name": corrected_name,
                "role": doc_role.role.lower(),
                "join_key": clean_sql_identifier(doc_role.join_key) if doc_role.join_key else None,
                "lookup_key": clean_sql_identifier(doc_role.lookup_key) if doc_role.lookup_key else None
            })
            
        logs.append(f"[SUCCESS] Schema Architect designed table '{table_name}' with {len(columns)} columns.")
        logs.append("Roles Assigned:")
        for role in document_roles:
            if role["role"] == "lookup":
                logs.append(f"  - '{role['source_name']}' is lookup (joins '{role['lookup_key']}' to primary '{role['join_key']}')")
            else:
                logs.append(f"  - '{role['source_name']}' is primary source")
                
        elapsed = time.perf_counter() - start_time
        
        return {
            "table_name": table_name,
            "columns": columns,
            "document_roles": document_roles,
            "log": logs,
            "schema_architect_duration": elapsed,
            "prompt_tokens": state.get("prompt_tokens", 0) + token_callback.prompt_tokens,
            "completion_tokens": state.get("completion_tokens", 0) + token_callback.completion_tokens
        }
        
    except Exception as e:
        logs.append(f"[ERROR] Schema design failed: {str(e)}")
        elapsed = time.perf_counter() - start_time
        return {"log": logs, "schema_architect_duration": elapsed}

def db_setup_node(state: AgentState) -> Dict[str, Any]:
    """
    Creates the table dynamically in the SQLite database based on the schema node's plan.
    """
    start_time = time.perf_counter()
    
    logs = list(state.get("log", []))
    table_name = state["table_name"]
    columns = state["columns"]
    
    if not table_name or not columns:
        logs.append("[ERROR] Database creation skipped: schema is missing.")
        elapsed = time.perf_counter() - start_time
        return {"log": logs, "db_setup_duration": elapsed}
        
    logs.append(f"[DB] Creating database table '{table_name}'...")
    
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
        logs.append(f"[SUCCESS] Table '{table_name}' created successfully in SQLite database.")
    else:
        logs.append(f"[ERROR] Database error: {result}")
        
    elapsed = time.perf_counter() - start_time
    return {"log": logs, "db_setup_duration": elapsed}

# Asynchronous helper functions to pull data concurrently
async def async_extract_page(semaphore, structured_extractor, system_prompt, user_prompt, page_text, image_path, is_scanned, token_callback, doc_name):
    async with semaphore:
        messages = [HumanMessage(content=system_prompt)]
        if is_scanned:
            if os.path.exists(image_path):
                messages.append(HumanMessage(content=[
                    {"type": "text", "text": f"{user_prompt}Extract rows from this layout page:"},
                    get_image_message_content(image_path)
                ]))
        else:
            messages.append(HumanMessage(content=f"{user_prompt}Extract rows from this page text:\n\n{page_text}"))
            
        result = await structured_extractor.ainvoke(messages, config={"callbacks": [token_callback]})
        return doc_name, [r.model_dump() for r in result.rows]

async def async_extract_single(semaphore, structured_extractor, system_prompt, user_prompt, full_text, image_path, is_scanned, token_callback, doc_name):
    async with semaphore:
        messages = [HumanMessage(content=system_prompt)]
        if is_scanned:
            messages.append(HumanMessage(content=[
                {"type": "text", "text": f"{user_prompt}Extract all rows:"},
                get_image_message_content(image_path)
            ]))
        else:
            messages.append(HumanMessage(content=f"{user_prompt}Extract all rows from this document:\n\n{full_text}"))
            
        result = await structured_extractor.ainvoke(messages, config={"callbacks": [token_callback]})
        return doc_name, [r.model_dump() for r in result.rows]

def data_extraction_node(state: AgentState) -> Dict[str, Any]:
    """
    Extracts all rows of data matching the schema structure from all ingested documents.
    Executes multiple page tasks asynchronously in parallel using a asyncio event loop.
    """
    start_time = time.perf_counter()
    
    logs = list(state.get("log", []))
    ingested_docs = state["ingested_docs"]
    table_name = state["table_name"]
    columns = state["columns"]
    document_roles = state["document_roles"]
    user_instructions = state.get("user_instructions", "").strip()
    
    if not table_name or not columns or not ingested_docs:
        logs.append("[ERROR] Extraction skipped: missing schema, roles, or ingested documents.")
        elapsed = time.perf_counter() - start_time
        return {"log": logs, "extraction_duration": elapsed}
        
    logs.append(f"[EXTRACTOR] Running Parallel Data Extractor Agent on {len(ingested_docs)} documents...")
    
    if not api_key:
        logs.append("[ERROR] API Key missing. Skipping data extraction.")
        elapsed = time.perf_counter() - start_time
        return {"log": logs, "extraction_duration": elapsed}
        
    # Build a lookup dictionary of roles by document name
    roles_map = {r["source_name"]: r for r in document_roles}
    
    # Initialize LLM and semaphore
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.1, google_api_key=api_key, thinking_budget=0)
    semaphore = asyncio.Semaphore(6)
    
    # Track LLM token usage using a single aggregated callback
    token_callback = TokenUsageCallback()
    
    # Accumulate async tasks
    async_tasks = []
    
    for doc in ingested_docs:
        source_name = doc["source_name"]
        doc_role_info = roles_map.get(source_name, {"role": "primary", "join_key": None, "lookup_key": None})
        role = doc_role_info["role"]
        
        # Filter unique target columns belonging to this document
        target_cols = []
        for col in columns:
            is_mapped = (col["source_doc"] == source_name)
            is_join_key = (role == "primary" and col["name"] == doc_role_info["join_key"])
            is_lookup_key = (role == "lookup" and col["name"] == doc_role_info["lookup_key"])
            if is_mapped or is_join_key or is_lookup_key:
                target_cols.append(col)
                
        seen_cols = set()
        unique_target_cols = []
        for col in target_cols:
            if col["name"] not in seen_cols:
                seen_cols.add(col["name"])
                unique_target_cols.append(col)
                
        if not unique_target_cols:
            continue
            
        # Build dynamic model for subset of columns
        fields = {}
        for col in unique_target_cols:
            col_name = col["name"]
            col_type = col["type"]
            py_type = str
            if col_type == "INTEGER":
                py_type = int
            elif col_type == "REAL":
                py_type = float
            fields[col_name] = (Optional[py_type], Field(description=col["description"], default=None))
            
        DocRowModel = create_model("DocRowModel", **fields)
        
        class DocDataset(BaseModel):
            rows: List[DocRowModel] = Field(description=f"List of records extracted from {source_name}.")
            
        structured_extractor = llm.with_structured_output(DocDataset)
        
        system_prompt = (
            "You are an advanced data extraction agent. Your job is to extract all raw data records "
            "matching the requested table schema structure. Look carefully at the document (text or images) "
            "and pull out all tabular rows.\n\n"
            "Crucially, you must obey the user's custom instructions to filter, normalize, format, or "
            "populate columns correctly."
        )
        
        is_scanned = doc["source_type"] in ["pdf_scanned", "image"]
        total_len = len(doc["extracted_text"]) if doc["extracted_text"] else 0
        use_chunking = (is_scanned and len(doc["image_paths"]) > 1) or (not is_scanned and total_len > 10000 and len(doc["extracted_pages"]) > 1)
        
        user_prompt = f"Target Schema Columns: {', '.join([c['name'] for c in unique_target_cols])}\n\n"
        if user_instructions:
            user_prompt += f"⚠️ USER CUSTOM EXTRACTION INSTRUCTIONS:\n{user_instructions}\n\n"
            
        if use_chunking:
            total_pages = len(doc["image_paths"]) if is_scanned else len(doc["extracted_pages"])
            logs.append(f"  [MODE] Adding {total_pages} page extraction tasks for '{source_name}' to concurrency pool...")
            for page_idx in range(total_pages):
                task = async_extract_page(
                    semaphore,
                    structured_extractor, system_prompt, user_prompt,
                    "" if is_scanned else doc["extracted_pages"][page_idx],
                    doc["image_paths"][page_idx] if is_scanned else "",
                    is_scanned, token_callback, source_name
                )
                async_tasks.append(task)
        else:
            logs.append(f"  [MODE] Adding single extraction task for '{source_name}' to concurrency pool...")
            task = async_extract_single(
                semaphore,
                structured_extractor, system_prompt, user_prompt,
                doc["extracted_text"],
                doc["image_paths"][0] if is_scanned else "",
                is_scanned, token_callback, source_name
            )
            async_tasks.append(task)

    # 5. Run async tasks in parallel using a dedicated event loop
    doc_extractions = {doc["source_name"]: [] for doc in ingested_docs}
    
    if async_tasks:
        logs.append(f"[INFO] Launching {len(async_tasks)} concurrent LLM tasks in parallel...")
        
        async def run_all_tasks():
            return await asyncio.gather(*async_tasks)
            
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            task_outputs = loop.run_until_complete(run_all_tasks())
            loop.close()
            
            # Map task outputs back to doc mapping lists
            for src_name, rows in task_outputs:
                doc_extractions[src_name].extend(rows)
                
            logs.append("  [OK] All concurrent extractions finished successfully!")
        except Exception as async_ex:
            logs.append(f"  [ERROR] Async parallel execution failed: {str(async_ex)}")
            elapsed = time.perf_counter() - start_time
            return {"log": logs, "extraction_duration": elapsed}
            
    # 6. PERFORM IN-MEMORY SQL-LIKE JOIN
    logs.append("[JOIN] Executing dynamic table joins...")
    
    primary_docs_extracted = [doc["source_name"] for doc in ingested_docs if roles_map.get(doc["source_name"], {"role": "primary"})["role"] == "primary"]
    
    if not primary_docs_extracted:
        logs.append("[ERROR] No primary sources detected. Cannot perform join.")
        elapsed = time.perf_counter() - start_time
        return {"log": logs, "extraction_duration": elapsed}
        
    base_rows = []
    for p_doc in primary_docs_extracted:
        p_rows = doc_extractions.get(p_doc, [])
        for r in p_rows:
            r["source_document_name"] = p_doc
            base_rows.append(r)
            
    logs.append(f"Initialized {len(base_rows)} base rows from primary documents.")
    
    for doc in ingested_docs:
        s_name = doc["source_name"]
        role_info = roles_map.get(s_name, {"role": "primary"})
        if role_info["role"] != "lookup":
            continue
            
        join_key = role_info["join_key"]
        lookup_key = role_info["lookup_key"]
        lookup_rows = doc_extractions.get(s_name, [])
        
        if not join_key or not lookup_key:
            continue
            
        logs.append(f"  Joining lookup source '{s_name}' on primary key '{join_key}' = lookup key '{lookup_key}'...")
        
        lookup_cols = [c["name"] for c in columns if c["source_doc"] == s_name and c["name"] != lookup_key]
        
        lookup_index = {}
        for r in lookup_rows:
            key_val = r.get(lookup_key)
            if key_val is not None:
                norm_key = normalize_join_key(key_val)
                lookup_index[norm_key] = {col: r.get(col) for col in lookup_cols}
                
        match_count = 0
        for r in base_rows:
            pk_val = r.get(join_key)
            norm_pk = normalize_join_key(pk_val)
            
            if norm_pk in lookup_index:
                r.update(lookup_index[norm_pk])
                match_count += 1
            else:
                for col in lookup_cols:
                    if col not in r:
                        r[col] = None
                        
        logs.append(f"[FINISH] Join operations complete! Total combined rows: {len(base_rows)}")
        
    elapsed = time.perf_counter() - start_time
    
    return {
        "extracted_rows": base_rows,
        "log": logs,
        "extraction_duration": elapsed,
        "prompt_tokens": state.get("prompt_tokens", 0) + token_callback.prompt_tokens,
        "completion_tokens": state.get("completion_tokens", 0) + token_callback.completion_tokens
    }

def db_load_node(state: AgentState) -> Dict[str, Any]:
    """
    Saves the extracted structured rows into the SQLite database in a single bulk transaction.
    Logs run duration statistics and token usage into a separate pipeline_telemetry.db file.
    """
    start_time = time.perf_counter()
    
    logs = list(state.get("log", []))
    table_name = state["table_name"]
    rows = state["extracted_rows"]
    columns = state["columns"]
    
    if not table_name or not rows:
        logs.append("[ERROR] Load step skipped: no table name or rows found.")
        elapsed = time.perf_counter() - start_time
        return {"log": logs, "db_load_duration": elapsed}
        
    logs.append(f"[LOAD] Batch loading {len(rows)} rows into SQLite database table '{table_name}'...")
    
    # 1. Map columns names list
    columns_names = [c["name"] for c in columns]
    placeholders = ", ".join(["?" for _ in columns_names])
    insert_sql = f"INSERT INTO {table_name} ({', '.join(columns_names)}) VALUES ({placeholders})"
    
    # 2. Build list of parameter tuples
    param_list = []
    for row in rows:
        row_vals = []
        for col in columns_names:
            row_vals.append(row.get(col))
        param_list.append(tuple(row_vals))
        
    # 3. Execute bulk insert inside a single database transaction
    result = db_execute_many(insert_sql, param_list)
    
    success_count = len(rows) if result == "Success" else 0
    fail_count = 0 if result == "Success" else len(rows)
    
    if result == "Success":
        logs.append(f"[FINISH] Load finished! {success_count} rows loaded successfully. {fail_count} errors.")
    else:
        logs.append(f"[ERROR] Bulk insert failed: {result}")
        
    elapsed = time.perf_counter() - start_time
    
    # 4. Construct complete telemetry state and save it
    # Calculate total duration by adding up all recorded step durations plus the load time
    ingest_d = state.get("ingest_duration", 0.0)
    schema_d = state.get("schema_architect_duration", 0.0)
    setup_d = state.get("db_setup_duration", 0.0)
    extract_d = state.get("extraction_duration", 0.0)
    total_elapsed = ingest_d + schema_d + setup_d + extract_d + elapsed
    
    # Setup state metrics to pass to logger helper
    final_state_metrics = {
        **state,
        "ingest_duration": ingest_d,
        "schema_architect_duration": schema_d,
        "db_setup_duration": setup_d,
        "extraction_duration": extract_d,
        "db_load_duration": elapsed,
        "extracted_rows": rows
    }
    
    # Log metrics to pipeline_telemetry.db
    save_telemetry(final_state_metrics, total_elapsed)
    
    return {
        "log": logs,
        "db_load_duration": elapsed,
        "total_duration": total_elapsed
    }

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
