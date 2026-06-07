import os
import argparse
import json
import sqlite3
from dotenv import load_dotenv
from agent_workflow import create_workflow

# Load API keys
load_dotenv()

def print_banner(text):
    print("\n" + "=" * 50)
    print(f" {text}")
    print("=" * 50)

def verify_database(table_name):
    print_banner(f"Inspecting SQLite Table: '{table_name}'")
    db_path = os.path.abspath("extracted_data.db")
    if not os.path.exists(db_path):
        print("[ERROR] SQLite Database not found!")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check structure
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        print("\nColumns Inferred:")
        for col in columns:
            print(f"  - ID {col[0]}: {col[1]} ({col[2]})")
            
        # Check count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cursor.fetchone()[0]
        print(f"\nTotal Rows Inserted: {row_count}")
        
        # Sample records
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
        rows = cursor.fetchall()
        print("\nFirst 5 Records:")
        for r in rows:
            print(f"  {r}")
            
        conn.close()
    except Exception as e:
        print(f"[ERROR] Database verification error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Test Multi-Source Agentic extractor workflow.")
    parser.add_argument(
        "--types", 
        required=True, 
        help="Comma-separated list of types (e.g. url,pdf_digital)"
    )
    parser.add_argument(
        "--paths", 
        required=True, 
        help="Comma-separated list of local file paths or website URLs matching the types"
    )
    parser.add_argument(
        "--instructions", 
        default="", 
        help="Custom organization or data cleaning instructions"
    )
    parser.add_argument(
        "--table", 
        default="", 
        help="Optional custom table name"
    )
    
    args = parser.parse_args()
    
    # Parse comma separated variables
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    
    if len(types) != len(paths):
        print(f"[ERROR] Number of types ({len(types)}) does not match number of paths ({len(paths)}).")
        return
        
    sources = [{"type": t, "path": p} for t, p in zip(types, paths)]
    
    # Check for api key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[WARNING] GEMINI_API_KEY or GOOGLE_API_KEY env variables not found.")
        print("Please set your API key in a '.env' file or your shell before running.")
        return

    print_banner("Starting Multi-Source Ingest Pipeline")
    print(f"Number of sources: {len(sources)}")
    print(f"Instructions: '{args.instructions}'")
    if args.table:
        print(f"Custom Target Table: '{args.table}'")
    
    try:
        # Compile workflow
        workflow = create_workflow()
        
        # Set up initial state
        initial_state = {
            "sources": sources,
            "user_instructions": args.instructions,
            "custom_table_name": args.table,
            "ingested_docs": [],
            "table_name": "",
            "columns": [],
            "extracted_rows": [],
            "log": []
        }
        
        # Execute workflow
        print("\nRunning workflow graph...")
        final_state = {}
        for step in workflow.stream(initial_state):
            node_name = list(step.keys())[0]
            print(f"[OK] Finished Node: [{node_name}]")
            
            # Merge step state changes
            final_state.update(step[node_name])
            
            # Print latest logs
            if "log" in final_state and final_state["log"]:
                print(f"   Log: {final_state['log'][-1]}")
        
        # Final output summary
        print_banner("Workflow Finished Successfully!")
        table_name = final_state.get("table_name")
        extracted_rows = final_state.get("extracted_rows", [])
        
        if table_name:
            print(f"Inferred Table Name: {table_name}")
            print(f"Number of columns: {len(final_state.get('columns', []))}")
            print(f"Number of extracted rows: {len(extracted_rows)}")
            
            # Verify the database state
            verify_database(table_name)
        else:
            print("[ERROR] No table name was inferred by the Schema Architect.")
            
    except Exception as e:
        print(f"\n[ERROR] Pipeline runtime execution failed: {str(e)}")

if __name__ == "__main__":
    main()
