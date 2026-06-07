import os
import sys
from dotenv import load_dotenv
from agent_workflow import create_workflow

# Load keys
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: No API key found!")
    sys.exit(1)

# Compile workflow
workflow = create_workflow()

# Setup input files in local directory
sources = [
    {"type": "pdf_digital", "path": "Northern-Arp-by-Arp.pdf"},
    {"type": "pdf_digital", "path": "Arp-by-Constellation.pdf"}
]

initial_state = {
    "sources": sources,
    "user_instructions": "Use Northern-Arp-by-Arp.pdf as the primary list of galaxies. Use Arp-by-Constellation.pdf to look up and include the constellation for each Arp number.",
    "custom_table_name": "arp_galaxies_debug_run", # Use a unique table name to avoid existing schema conflicts!
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

print("Running pipeline with full logging...")
try:
    final_state = workflow.invoke(initial_state)
    print("\n" + "=" * 50)
    print(" PIPELINE COMPLETED")
    print("=" * 50)
    
    # Print all logs
    for log_line in final_state.get("log", []):
        print(f"Log: {log_line}")
        
    print(f"\nFinal Total Duration: {final_state.get('total_duration', 0.0):.3f}s")
    print(f"Prompt Tokens: {final_state.get('prompt_tokens', 0)}")
    print(f"Completion Tokens: {final_state.get('completion_tokens', 0)}")
    print(f"Row count: {len(final_state.get('extracted_rows', []))}")
except Exception as e:
    print(f"\n❌ Pipeline failed: {str(e)}")
