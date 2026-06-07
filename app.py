import os
import sqlite3
import pandas as pd
import streamlit as st
from agent_workflow import create_workflow

# Page Configuration
st.set_page_config(
    page_title="Agentic Data Ingest & Schema Builder",
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        border-left: 4px solid #0284c7;
        max-height: 400px;
        overflow-y: auto;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-bottom: 1.5rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Custom button enhancements */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        color: white !important;
        border: none;
        padding: 0.6rem 1.8rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
        margin-right: 0.5rem;
    }
    
    .badge-primary {
        background-color: #dbeafe;
        color: #1e40af;
    }
</style>
""", unsafe_allow_html=True)

# Application Header Banner
st.markdown("""
<div class="title-container">
    <h1>📊 Agentic Data Extractor & Schema Architect</h1>
    <p>Upload files (digital, scanned PDFs or images) or paste web links. Watch AI agents dynamically architect SQL tables and extract records.</p>
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
    st.subheader("📁 Ingestion Source")
    source_type = st.radio(
        "Select Data Source Type:",
        options=["Digital-native PDF", "Scanned PDF (OCR)", "Image", "Web Page URL"],
        index=0
    )
    
    st.info("💡 **Digital-native PDF** reads embedded text objects directly. **Scanned PDF** uses multimodal visual recognition on generated page layouts.")

# Main Application Layout split into 2 Columns
left_col, right_col = st.columns([1, 1])

# Global execution state tracking
if "last_table_name" not in st.session_state:
    st.session_state["last_table_name"] = ""
if "console_logs" not in st.session_state:
    st.session_state["console_logs"] = []

# --- LEFT COLUMN: INPUTS AND RUN CONSOLE ---
with left_col:
    st.subheader("📥 Source Input")
    
    # Input container Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    source_path = ""
    source_type_code = ""
    
    if source_type == "Digital-native PDF":
        source_type_code = "pdf_digital"
        uploaded_file = st.file_uploader("Upload Digital PDF", type=["pdf"])
        if uploaded_file:
            source_path = os.path.join("temp_uploads", uploaded_file.name)
            with open(source_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
    elif source_type == "Scanned PDF (OCR)":
        source_type_code = "pdf_scanned"
        uploaded_file = st.file_uploader("Upload Scanned PDF", type=["pdf"])
        if uploaded_file:
            source_path = os.path.join("temp_uploads", uploaded_file.name)
            with open(source_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
    elif source_type == "Image":
        source_type_code = "image"
        uploaded_file = st.file_uploader("Upload Table Screenshot/Image", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            source_path = os.path.join("temp_uploads", uploaded_file.name)
            with open(source_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
    elif source_type == "Web Page URL":
        source_type_code = "url"
        source_path = st.text_input("Enter Webpage URL", placeholder="https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Button to Trigger Pipeline
    run_btn = st.button("🚀 Run Agentic Pipeline", disabled=not source_path)
    
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
        st.session_state["console_logs"] = ["Initializing Agent Workflow..."]
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
                        "source_type": source_type_code,
                        "source_path": source_path,
                        "image_paths": [],
                        "extracted_text": "",
                        "table_name": "",
                        "columns": [],
                        "extracted_rows": [],
                        "log": []
                    }
                    
                    # 3. Stream graph steps to update UI logs dynamically
                    for output in workflow.stream(initial_state):
                        # The stream yields outputs from each node as {"node_name": {state_changes}}
                        node_name = list(output.keys())[0]
                        node_state = output[node_name]
                        
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
                except Exception as ex:
                    st.error(f"Execution Error: {str(ex)}")
                    st.session_state["console_logs"].append(f"❌ Error: {str(ex)}")
                    log_html = "".join([f"&gt; {line}<br/>" for line in st.session_state["console_logs"]])
                    console_placeholder.markdown(f'<div class="console-box">{log_html}</div>', unsafe_allow_html=True)

# --- RIGHT COLUMN: EXTRACTED SCHEMA & DATA VIEW ---
with right_col:
    st.subheader("📋 Dynamic Data Viewer")
    
    # Grab the last table generated, or allow querying historically
    table_to_query = st.session_state["last_table_name"]
    
    # Check what dynamic tables exist in our SQLite DB to let the user select
    all_tables = []
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            all_tables = [row[0] for row in cursor.fetchall()]
            conn.close()
        except Exception as db_err:
            st.warning(f"Could not load historical tables: {str(db_err)}")
            
    if all_tables:
        # If a new table was just created, prioritize selecting it in the dropdown
        select_index = 0
        if table_to_query in all_tables:
            select_index = all_tables.index(table_to_query)
            
        selected_table = st.selectbox("Select SQLite Table to View:", options=all_tables, index=select_index)
    else:
        st.info("No databases tables extracted yet. Run a pipeline on a PDF/image first.")
        selected_table = None

    if selected_table:
        try:
            # Query the table
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
            
            # Fetch Schema columns to show column definitions
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({selected_table})")
            columns_info = cursor.fetchall()
            conn.close()
            
            # Display stats
            st.markdown(f"**Total Records:** `{len(df)}` | **SQLite Table:** `{selected_table}`")
            
            # Render interactive data table (sortable, filterable, scrollable)
            st.dataframe(df, use_container_width=True)
            
            # Display Table Schema details in an expander
            with st.expander("🛠️ Inferred Schema Specifications"):
                schema_df = pd.DataFrame([
                    {"Column ID": col[0], "Name": col[1], "Data Type": col[2], "Primary Key": bool(col[5])}
                    for col in columns_info
                ])
                st.table(schema_df)
                
            # Download Data button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"{selected_table}.csv",
                mime="text/csv",
            )
            
        except Exception as render_ex:
            st.error(f"Error loading table content: {str(render_ex)}")
