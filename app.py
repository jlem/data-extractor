import os
import sqlite3
import pandas as pd
import streamlit as st
from agent_workflow import create_workflow

# Page Configuration
st.set_page_config(
    page_title="Agentic Multi-Source Data Extractor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Sleek CSS for Modern Dark/Glassmorphic Aesthetics
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Title and Header Gradients */
    .title-container {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .title-container h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .title-container p {
        font-size: 1.1rem;
        font-weight: 300;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Cards and Columns styling */
    .card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    
    .dark .card {
        background-color: #1e293b;
        border-color: #334155;
    }
    
    /* Agent Console Box */
    .console-box {
        background-color: #0f172a;
        color: #38bdf8;
        font-family: 'Courier New', Courier, monospace;
        padding: 1.25rem;
        border-radius: 8px;
        border-left: 4px solid #4f46e5;
        max-height: 400px;
        overflow-y: auto;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-bottom: 1.5rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Custom button enhancements */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white !important;
        border: none;
        padding: 0.6rem 1.8rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 6px rgba(99, 102, 241, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(99, 102, 241, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Application Header Banner
st.markdown("""
<div class="title-container">
    <h1>📊 Agentic Multi-Source Data Extractor</h1>
    <p>Ingest multiple PDFs/images and websites, organize them using custom instructions, and build SQLite tables.</p>
</div>
""", unsafe_allow_html=True)

# Ensure folders exist
os.makedirs("temp_uploads", exist_ok=True)
DB_PATH = os.path.abspath("extracted_data.db")

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/database.png", width=80)
    st.header("⚙️ Configuration")
    
    # API Key Handling
    api_key_input = st.text_input(
        "Google Gemini API Key", 
        type="password",
        help="Input your Gemini API key. If empty, the app will try to read GEMINI_API_KEY from environment/dotenv."
    )
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input
        os.environ["GEMINI_API_KEY"] = api_key_input

    st.markdown("---")
    st.subheader("📁 Ingestion Category")
    source_category = st.radio(
        "Select Source Format:",
        options=["Upload Files (PDF, PNG, JPG)", "Web Page URLs"],
        index=0
    )
    
    ocr_pdf = False
    if source_category == "Upload Files (PDF, PNG, JPG)":
        ocr_pdf = st.checkbox("Force OCR for PDF files (use Vision)", value=False,
                              help="Check this if the PDFs are scanned images or drawings rather than searchable digital text.")

# Main Application Layout split into 2 Columns
left_col, right_col = st.columns([1, 1])

# Global execution state tracking
if "last_table_name" not in st.session_state:
    st.session_state["last_table_name"] = ""
if "console_logs" not in st.session_state:
    st.session_state["console_logs"] = []

# --- LEFT COLUMN: INPUTS AND RUN CONSOLE ---
with left_col:
    st.subheader("📥 Source Inputs & Instructions")
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    sources = []
    
    if source_category == "Upload Files (PDF, PNG, JPG)":
        uploaded_files = st.file_uploader(
            "Upload files (Multiple supported)", 
            type=["pdf", "png", "jpg", "jpeg"], 
            accept_multiple_files=True
        )
        if uploaded_files:
            for f in uploaded_files:
                path = os.path.join("temp_uploads", f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                
                ext = os.path.splitext(f.name)[1].lower()
                if ext == ".pdf":
                    t = "pdf_scanned" if ocr_pdf else "pdf_digital"
                else:
                    t = "image"
                    
                sources.append({"type": t, "path": path})
                
    elif source_category == "Web Page URLs":
        urls_input = st.text_area(
            "Enter URLs (One per line)", 
            placeholder="https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)\nhttps://en.wikipedia.org/wiki/List_of_countries_by_population_(United_Nations)"
        )
        if urls_input:
            urls = [u.strip() for u in urls_input.split("\n") if u.strip()]
            for u in urls:
                sources.append({"type": "url", "path": u})
                
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Custom Instructions Card
    st.subheader("✍️ Organization Guidelines")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    user_instructions = st.text_area(
        "Enter custom instructions for how the data should be organized:",
        placeholder="e.g. Standardize date columns to YYYY-MM-DD. Convert names to uppercase. Filter out rows where total cost is 0. Ensure currency columns are REAL.",
        height=100
    )
    
    custom_table_name = st.text_input(
        "Destination Database Table Name (Optional)",
        placeholder="e.g. combined_invoices_data"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Button to Trigger Pipeline
    run_btn = st.button("🚀 Run Agentic Pipeline", disabled=len(sources) == 0)
    
    st.subheader("🕵️ Agent Active Console Logs")
    console_placeholder = st.empty()
    
    # Render Console Logs from state initially
    if st.session_state["console_logs"]:
        log_html = "".join([f"&gt; {line}<br/>" for line in st.session_state["console_logs"]])
        console_placeholder.markdown(f'<div class="console-box">{log_html}</div>', unsafe_allow_html=True)
    else:
        console_placeholder.markdown('<div class="console-box">&gt; System idle. Waiting for source input...</div>', unsafe_allow_html=True)

    # Execution Action Loop
    if run_btn:
        st.session_state["console_logs"] = ["Initializing Multi-Source Agent Workflow..."]
        console_placeholder.markdown(f'<div class="console-box">&gt; {st.session_state["console_logs"][0]}</div>', unsafe_allow_html=True)
        
        # Verify API key exists
        test_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not test_key:
            st.error("❌ Gemini API Key not found. Please input it in the sidebar or set the environment variable.")
        else:
            with st.spinner("Agents are processing. See console for logs..."):
                try:
                    # 1. Compile LangGraph workflow
                    workflow = create_workflow()
                    
                    # 2. Setup initial state
                    initial_state = {
                        "sources": sources,
                        "user_instructions": user_instructions,
                        "custom_table_name": custom_table_name,
                        "ingested_docs": [],
                        "table_name": "",
                        "columns": [],
                        "document_roles": [],
                        "extracted_rows": [],
                        "log": [],
                        "job_id": "",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "ingest_duration": 0.0,
                        "schema_architect_duration": 0.0,
                        "db_setup_duration": 0.0,
                        "extraction_duration": 0.0,
                        "db_load_duration": 0.0,
                        "total_duration": 0.0
                    }
                    
                    # 3. Stream graph steps to update UI logs dynamically
                    final_state = {}
                    for output in workflow.stream(initial_state):
                        node_name = list(output.keys())[0]
                        node_state = output[node_name]
                        
                        final_state.update(node_state)
                        
                        # Grab updated logs from the node state changes
                        node_logs = node_state.get("log", [])
                        if node_logs:
                            st.session_state["console_logs"] = node_logs
                            log_html = "".join([f"&gt; {line}<br/>" for line in node_logs])
                            console_placeholder.markdown(f'<div class="console-box">{log_html}</div>', unsafe_allow_html=True)
                            
                        # Store properties in session state when they become available
                        if "table_name" in node_state and node_state["table_name"]:
                            st.session_state["last_table_name"] = node_state["table_name"]
                            
                    st.success("🎉 Agent workflow completed successfully!")
                    
                    # 4. Display performance metrics summary card in UI
                    if "total_duration" in final_state:
                        prompt_t = final_state.get("prompt_tokens", 0)
                        comp_t = final_state.get("completion_tokens", 0)
                        tot_t = prompt_t + comp_t
                        
                        st.markdown(f"""
                        <div class="card" style="border-left: 5px solid #22c55e; padding: 1.25rem;">
                            <h3 style="margin-top:0; color:#22c55e; font-size:1.15rem;">⏱️ Job Performance Metrics</h3>
                            <table style="width:100%; border:none; font-size:0.85rem; line-height: 1.6;">
                                <tr><td>Ingestion Duration:</td><td style="text-align:right;"><b>{final_state.get('ingest_duration', 0.0):.3f}s</b></td></tr>
                                <tr><td>Schema Design Duration:</td><td style="text-align:right;"><b>{final_state.get('schema_architect_duration', 0.0):.3f}s</b></td></tr>
                                <tr><td>Database Setup Duration:</td><td style="text-align:right;"><b>{final_state.get('db_setup_duration', 0.0):.3f}s</b></td></tr>
                                <tr><td>Data Extraction Duration:</td><td style="text-align:right;"><b>{final_state.get('extraction_duration', 0.0):.3f}s</b></td></tr>
                                <tr><td>Database Loading Duration:</td><td style="text-align:right;"><b>{final_state.get('db_load_duration', 0.0):.3f}s</b></td></tr>
                                <tr style="border-top:1px solid #e2e8f0; height: 10px;"><td></td><td></td></tr>
                                <tr>
                                    <td><b>Total Processing Time:</b></td>
                                    <td style="text-align:right;"><span style="background-color:#22c55e; color:white; padding: 2px 6px; border-radius: 4px; font-weight:700;"><b>{final_state.get('total_duration', 0.0):.3f}s</b></span></td>
                                </tr>
                                <tr>
                                    <td><b>Gemini Token Usage:</b></td>
                                    <td style="text-align:right;">Prompt: <b>{prompt_t}</b> | Comp: <b>{comp_t}</b> | Total: <b>{tot_t}</b></td>
                                </tr>
                            </table>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as ex:
                    st.error(f"Execution Error: {str(ex)}")
                    st.session_state["console_logs"].append(f"❌ Error: {str(ex)}")
                    log_html = "".join([f"&gt; {line}<br/>" for line in st.session_state["console_logs"]])
                    console_placeholder.markdown(f'<div class="console-box">{log_html}</div>', unsafe_allow_html=True)

# --- RIGHT COLUMN: EXTRACTED SCHEMA & DATA VIEW ---
with right_col:
    st.subheader("📋 Dynamic Data Viewer")
    
    table_to_query = st.session_state["last_table_name"]
    
    all_tables = []
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            all_tables = [row[0] for row in cursor.fetchall()]
            conn.close()
        except Exception as db_err:
            st.warning(f"Could not load tables: {str(db_err)}")
            
    if all_tables:
        select_index = 0
        if table_to_query in all_tables:
            select_index = all_tables.index(table_to_query)
            
        selected_table = st.selectbox("Select SQLite Table to View:", options=all_tables, index=select_index)
    else:
        st.info("No database tables extracted yet. Run a pipeline first.")
        selected_table = None

    if selected_table:
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
            
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({selected_table})")
            columns_info = cursor.fetchall()
            conn.close()
            
            st.markdown(f"**Total Records:** `{len(df)}` | **SQLite Table:** `{selected_table}`")
            
            st.dataframe(df, use_container_width=True)
            
            with st.expander("🛠️ Inferred Schema Specifications"):
                schema_df = pd.DataFrame([
                    {"Column ID": col[0], "Name": col[1], "Data Type": col[2], "Primary Key": bool(col[5])}
                    for col in columns_info
                ])
                st.table(schema_df)
                
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"{selected_table}.csv",
                mime="text/csv",
            )
            
        except Exception as render_ex:
            st.error(f"Error loading table content: {str(render_ex)}")
