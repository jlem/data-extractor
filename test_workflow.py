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
        print("❌ SQLite Database not found!")
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
        print(f"❌ Database verification error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Test Agentic PDF / URL extractor workflow.")
    parser.add_argument(
        "--type", 
        choices=["pdf_digital", "pdf_scanned", "image", "url"], 
        required=True, 
        help="Type of input source (pdf_digital, pdf_scanned, image, url)"
    )
    parser.add_argument(
        "--path", 
        required=True, 
        help="Local file path or website URL"
    )
    
    args = parser.parse_args()
    
    # Check for api key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️ Warning: GEMINI_API_KEY or GOOGLE_API_KEY env variables not found.")
        print("Please set your API key in a '.env' file or your shell before running.")
        return

    print_banner("Starting LangGraph Extraction Pipeline")
    print(f"Input Type: {args.type}")
    print(f"Input Path: {args.path}")
    
    try:
        # Compile workflow
        workflow = create_workflow()
        
        # Set up initial state
        initial_state = {
            "source_type": args.type,
            "source_path": args.path,
            "image_paths": [],
            "extracted_text": "",
            "table_name": "",
            "columns": [],
            "extracted_rows": [],
            "log": []
        }
        
        # Execute workflow
        print("\nRunning workflow graph...")
        final_state = {}
        for step in workflow.stream(initial_state):
            # Print node completion details
            node_name = list(step.keys())[0]
            print(f"✔️ Finished Node: [{node_name}]")
            
            # Merge the step's changes into our accumulated state
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
            print("❌ Error: No table name was inferred by the Schema Architect.")
            
    except Exception as e:
        print(f"\n❌ Pipeline runtime execution failed: {str(e)}")

if __name__ == "__main__":
    main()
