import os
import re
import base64
import fitz
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

# Define structures for the updated multi-source Ingestion
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
    join_key: Optional[str] = Field(description="For lookup documents, the column name in the primary table used to match/join (e.g. 'arp_number'). Leave empty for primary documents.")
    lookup_key: Optional[str] = Field(description="For lookup documents, the column name in this lookup document that matches the primary table's join_key. Leave empty for primary documents.")

# Define LangGraph State Schema supporting multiple sources and instructions
class AgentState(TypedDict):
    sources: List[Dict[str, str]]       # List of {"type": "...", "path": "..."}
    user_instructions: str              # Custom instructions to guide the behavior of all agents
    custom_table_name: str              # Optional user-specified database table name
    ingested_docs: List[IngestedDoc]    # Content extracted per source document
    table_name: str                     # Dynamically generated database table name
    columns: List[Dict[str, str]]       # List of {"name": "...", "type": "...", "description": "...", "source_doc": "..."}
    document_roles: List[Dict[str, Any]] # Document roles and join mapping keys
    extracted_rows: List[Dict[str, Any]] # Combined list of rows extracted from all sources
    log: List[str]                      # Running console log history

# Pydantic models for structured schema architecture output
class ColumnDefinition(BaseModel):
    name: str = Field(description="Column name in snake_case. Must be unique and a valid SQLite column identifier.")
    type: str = Field(description="SQLite database type. Must be one of: TEXT, INTEGER, REAL.")
    description: str = Field(description="Short explanation of what this column represents.")
    source_doc: str = Field(description="The source_name of the document this column's data comes from (must exactly match one of the source names).")

class SchemaResponse(BaseModel):
    table_name: str = Field(description="Descriptive table name in snake_case summarizing the table's contents.")
    columns: List[ColumnDefinition] = Field(description="Checklist of columns identified from the tabular data source.")
    document_roles: List[DocumentRole] = Field(description="Assign a role ('primary' or 'lookup') to each source document based on layout and user instructions.")

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
    # If it's a float that represents an integer, convert to int (e.g. 1.0 -> 1)
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    # Strip spaces and cast to lowercase string
    return str(val).strip().lower()

# ----------------- GRAPH NODES -----------------

def ingest_node(state: AgentState) -> Dict[str, Any]:
    """
    Iterates over all provided input sources and ingests their content 
    (digital text extraction page-by-page, web scraping, or page-to-image rendering).
    """
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
                # Extract text page-by-page to keep page structures intact
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
            
    return {
        "ingested_docs": ingested_docs,
        "log": logs
    }

def schema_architect_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes all ingested materials together with the custom user instructions 
    to assign primary vs lookup roles and design a unified database table schema.
    """
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
    
    # Setup LLM
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.1)
    structured_llm = llm.with_structured_output(SchemaResponse)
    
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
        user_prompt += f"[WARNING] USER CUSTOM SCHEMA INSTRUCTIONS:\n{user_instructions}\n\n"
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
            
        schema_result = structured_llm.invoke(messages)
        
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
                
            # Clean source filename mapping
            corrected_src_doc = find_closest_source(col.source_doc, actual_source_names)
            
            columns.append({
                "name": col_name,
                "type": col.type.upper(),
                "description": col.description,
                "source_doc": corrected_src_doc
            })
            
        # Automatically insert the source document tracking metadata column if not present
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
                
        return {
            "table_name": table_name,
            "columns": columns,
            "document_roles": document_roles,
            "log": logs
        }
        
    except Exception as e:
        logs.append(f"[ERROR] Schema design failed: {str(e)}")
        return {"log": logs}

def db_setup_node(state: AgentState) -> Dict[str, Any]:
    """
    Creates the table dynamically in the SQLite database based on the schema node's plan.
    """
    logs = list(state.get("log", []))
    table_name = state["table_name"]
    columns = state["columns"]
    
    if not table_name or not columns:
        logs.append("[ERROR] Database creation skipped: schema is missing.")
        return {"log": logs}
        
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
        
    return {"log": logs}

def data_extraction_node(state: AgentState) -> Dict[str, Any]:
    """
    Extracts all rows of data matching the schema structure from all ingested documents.
    Applies page-by-page chunking to large files to prevent truncation, and
    performs an in-memory SQL-like join matching lookup files to primary lists.
    """
    logs = list(state.get("log", []))
    ingested_docs = state["ingested_docs"]
    table_name = state["table_name"]
    columns = state["columns"]
    document_roles = state["document_roles"]
    user_instructions = state.get("user_instructions", "").strip()
    
    if not table_name or not columns or not ingested_docs:
        logs.append("[ERROR] Extraction skipped: missing schema, roles, or ingested documents.")
        return {"log": logs}
        
    logs.append(f"[EXTRACTOR] Running Data Extractor Agent on {len(ingested_docs)} documents...")
    
    if not api_key:
        logs.append("[ERROR] API Key missing. Skipping data extraction.")
        return {"log": logs}
        
    # Build a lookup dictionary of roles by document name for quick reference
    roles_map = {r["source_name"]: r for r in document_roles}
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.1)
    
    # We will perform extraction document-by-document
    doc_extractions = {} # Maps source_name -> List[Dict]
    
    for idx, doc in enumerate(ingested_docs, 1):
        source_name = doc["source_name"]
        doc_role_info = roles_map.get(source_name, {"role": "primary", "join_key": None, "lookup_key": None})
        role = doc_role_info["role"]
        
        # 1. Identify which columns belong to this document
        # - Always include columns mapped to this document
        # - Plus, if lookup: include the lookup_key column
        # - Plus, if primary: include the join_key column
        target_cols = []
        for col in columns:
            is_mapped = (col["source_doc"] == source_name)
            is_join_key = (role == "primary" and col["name"] == doc_role_info["join_key"])
            is_lookup_key = (role == "lookup" and col["name"] == doc_role_info["lookup_key"])
            
            if is_mapped or is_join_key or is_lookup_key:
                target_cols.append(col)
                
        # Deduplicate target columns by name
        seen_cols = set()
        unique_target_cols = []
        for col in target_cols:
            if col["name"] not in seen_cols:
                seen_cols.add(col["name"])
                unique_target_cols.append(col)
                
        if not unique_target_cols:
            logs.append(f"[WARNING] Warning: No columns assigned to '{source_name}'. Skipping extraction.")
            continue
            
        logs.append(f"[FILE] [{role.upper()}] Extracting from '{source_name}' ({len(unique_target_cols)} columns)...")
        
        # 2. Build dynamic Pydantic model for this specific document's subset of columns
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
        
        # 3. Determine Extraction Mode (Heuristic: page-by-page chunking if text length > 10,000)
        # For visual formats (images or pdf_scanned), we check length of image_paths list
        is_scanned = doc["source_type"] in ["pdf_scanned", "image"]
        total_len = len(doc["extracted_text"]) if doc["extracted_text"] else 0
        use_chunking = (is_scanned and len(doc["image_paths"]) > 1) or (not is_scanned and total_len > 10000 and len(doc["extracted_pages"]) > 1)
        
        extracted_rows = []
        
        # 4. Perform extraction
        try:
            if use_chunking:
                # Page-by-page loop
                total_pages = len(doc["image_paths"]) if is_scanned else len(doc["extracted_pages"])
                logs.append(f"  [MODE] File size triggers chunked extraction mode ({total_pages} pages)...")
                
                for page_idx in range(total_pages):
                    logs.append(f"    Processing page {page_idx + 1}/{total_pages}...")
                    user_prompt = f"Target Schema Columns: {', '.join([c['name'] for c in unique_target_cols])}\n\n"
                    if user_instructions:
                        user_prompt += f"[WARNING] USER CUSTOM EXTRACTION INSTRUCTIONS:\n{user_instructions}\n\n"
                        
                    messages = [HumanMessage(content=system_prompt)]
                    
                    if is_scanned:
                        img_path = doc["image_paths"][page_idx]
                        if os.path.exists(img_path):
                            messages.append(HumanMessage(content=[
                                {"type": "text", "text": f"{user_prompt}Extract rows from this layout page:"},
                                get_image_message_content(img_path)
                            ]))
                    else:
                        page_text = doc["extracted_pages"][page_idx]
                        messages.append(HumanMessage(content=f"{user_prompt}Extract rows from this page text:\n\n{page_text}"))
                        
                    page_dataset = structured_extractor.invoke(messages)
                    extracted_rows.extend([r.model_dump() for r in page_dataset.rows])
            else:
                # Single-call extraction
                logs.append("  [MODE] Small file: extracting in a single pass to preserve layout context...")
                user_prompt = f"Target Schema Columns: {', '.join([c['name'] for c in unique_target_cols])}\n\n"
                if user_instructions:
                    user_prompt += f"[WARNING] USER CUSTOM EXTRACTION INSTRUCTIONS:\n{user_instructions}\n\n"
                    
                messages = [HumanMessage(content=system_prompt)]
                
                if is_scanned:
                    img_path = doc["image_paths"][0]
                    messages.append(HumanMessage(content=[
                        {"type": "text", "text": f"{user_prompt}Extract all rows:"},
                        get_image_message_content(img_path)
                    ]))
                else:
                    messages.append(HumanMessage(content=f"{user_prompt}Extract all rows from this document:\n\n{doc['extracted_text']}"))
                    
                dataset_result = structured_extractor.invoke(messages)
                extracted_rows.extend([r.model_dump() for r in dataset_result.rows])
                
            logs.append(f"  [OK] Successfully extracted {len(extracted_rows)} records from '{source_name}'.")
            doc_extractions[source_name] = extracted_rows
            
        except Exception as e:
            logs.append(f"  [ERROR] Extraction failed for '{source_name}': {str(e)}")
            doc_extractions[source_name] = []
            
    # 5. PERFORM IN-MEMORY SQL-LIKE JOIN
    logs.append("[JOIN] Executing dynamic table joins...")
    
    # Identify primary document rows (base rows)
    primary_docs_extracted = []
    for doc in ingested_docs:
        s_name = doc["source_name"]
        role_info = roles_map.get(s_name, {"role": "primary"})
        if role_info["role"] == "primary":
            primary_docs_extracted.append(s_name)
            
    if not primary_docs_extracted:
        logs.append("[ERROR] Error: No primary sources detected. Cannot perform join.")
        return {"log": logs}
        
    # Concatenate base records from all primary documents
    base_rows = []
    for p_doc in primary_docs_extracted:
        p_rows = doc_extractions.get(p_doc, [])
        for r in p_rows:
            # Inject source tracking metadata
            r["source_document_name"] = p_doc
            base_rows.append(r)
            
    logs.append(f"Initialized {len(base_rows)} base rows from primary documents.")
    
    # Process each lookup document and join its fields into base_rows
    for doc in ingested_docs:
        s_name = doc["source_name"]
        role_info = roles_map.get(s_name, {"role": "primary"})
        if role_info["role"] != "lookup":
            continue
            
        join_key = role_info["join_key"]
        lookup_key = role_info["lookup_key"]
        lookup_rows = doc_extractions.get(s_name, [])
        
        if not join_key or not lookup_key:
            logs.append(f"  [WARNING] Skipping join for lookup source '{s_name}': missing match key definitions.")
            continue
            
        logs.append(f"  Joining lookup source '{s_name}' on primary key '{join_key}' = lookup key '{lookup_key}'...")
        
        # Identify columns that belong to this lookup file (excluding the lookup_key itself)
        lookup_cols = [c["name"] for c in columns if c["source_doc"] == s_name and c["name"] != lookup_key]
        
        # Build index mapping: normalized_key_value -> dict_of_lookup_values
        lookup_index = {}
        for r in lookup_rows:
            key_val = r.get(lookup_key)
            if key_val is not None:
                norm_key = normalize_join_key(key_val)
                lookup_index[norm_key] = {col: r.get(col) for col in lookup_cols}
                
        # Merge values into base_rows
        match_count = 0
        for r in base_rows:
            pk_val = r.get(join_key)
            norm_pk = normalize_join_key(pk_val)
            
            if norm_pk in lookup_index:
                r.update(lookup_index[norm_pk])
                match_count += 1
            else:
                # Populate empty columns with None (NULL)
                for col in lookup_cols:
                    if col not in r:
                        r[col] = None
                        
        logs.append(f"  Joined {len(lookup_cols)} lookup columns into primary records. Matches found: {match_count}/{len(base_rows)}.")
        
    logs.append(f"[FINISH] Join operations complete! Total combined rows: {len(base_rows)}")
    return {
        "extracted_rows": base_rows,
        "log": logs
    }

def db_load_node(state: AgentState) -> Dict[str, Any]:
    """
    Saves the extracted structured rows into the dynamically created SQLite table.
    """
    logs = list(state.get("log", []))
    table_name = state["table_name"]
    rows = state["extracted_rows"]
    
    if not table_name or not rows:
        logs.append("[ERROR] Load step skipped: no table name or rows found.")
        return {"log": logs}
        
    logs.append(f"[LOAD] Loading {len(rows)} rows into database table '{table_name}'...")
    
    success_count = 0
    fail_count = 0
    
    # Loop over rows and insert them individually
    for row in rows:
        cols_str = []
        vals_str = []
        for col_name, val in row.items():
            # Skip metadata source trackers if not in the official column definition list, 
            # but usually it is added by the schema architect so it will match.
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
            logs.append(f"[ERROR] Insert failed: {result} | SQL: {insert_sql}")
            
    logs.append(f"[FINISH] Load finished! {success_count} rows loaded successfully. {fail_count} errors.")
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
