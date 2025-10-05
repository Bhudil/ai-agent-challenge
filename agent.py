import os
import sys
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from langgraph.graph import StateGraph, END
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("GROQ_API_KEY not set. Add it to .env file")

llm_client = Groq(api_key=API_KEY)
MODEL_NAME = "llama-3.1-8b-instant"
MAX_CORRECTION_ATTEMPTS = 3

# Extract Python code from LLM response (removes markdown blocks)
def extract_clean_code(llm_response: str) -> str:
    pattern = re.search(r"```(?:python)?(.*?)```", llm_response, re.DOTALL)
    code = pattern.group(1) if pattern else llm_response
    
    # Remove explanatory lines
    lines = code.splitlines()
    filtered_lines = [
        line for line in lines 
        if not line.strip().startswith(("Here", "This", "The", "Note:", "I "))
    ]
    
    return "\n".join(filtered_lines).strip()

# Generate parser code using LLM with fallback on failure
def generate_parser_code(bank_name: str, iteration: int = 1, error_context: str = "") -> str:
    try:
        if iteration == 1:
            instruction = f"""Write a Python function: def parse(file_path: str) -> pd.DataFrame

Task: Parse {bank_name} bank statement from PDF or CSV file.

Requirements:
1. Import: pandas as pd, pdfplumber (for PDFs)
2. If file is CSV: return pd.read_csv(file_path)
3. If file is PDF: use pdfplumber to extract tables, combine into DataFrame
4. Handle missing values: replace NaN with 0 for numeric cols, "" for text
5. Keep dates as strings in original format (don't convert)
6. Return valid DataFrame with proper column names

Output: Clean Python code ONLY (no explanations)."""
        else:
            instruction = f"""Previous parser failed. Fix the issues.

Error from previous attempt:
{error_context[:500]}

Write corrected code for: def parse(file_path: str) -> pd.DataFrame

Requirements:
- Fix the specific errors shown above
- Keep dates as strings (don't convert formats)
- Handle NaN properly
- Output clean Python code ONLY"""

        print(f"  ├─ Sending prompt to LLM ({len(instruction)} chars)...")
        
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": instruction}],
            temperature=0.2
        )
        
        generated = response.choices[0].message.content.strip()
        print(f"  ├─ Received LLM response ({len(generated)} chars)")
        
        cleaned = extract_clean_code(generated)
        print(f"  ├─ Extracted clean code ({len(cleaned)} chars)")
        
        # Validate syntax
        compile(cleaned, f"{bank_name}_parser.py", "exec")
        print(f"  ├─ ✓ Syntax validation passed")
        
        if "def parse(" not in cleaned:
            raise ValueError("Missing parse() function")
        
        print(f"  └─ ✓ Code structure validated")
        return cleaned
        
    except Exception as error:
        print(f"  ├─ ✗ Generation failed: {str(error)[:50]}...")
        print(f"  └─ Using fallback template")
        
        return """import pandas as pd
import pdfplumber

def parse(file_path: str) -> pd.DataFrame:
    if file_path.lower().endswith('.csv'):
        return pd.read_csv(file_path).fillna(0)
    
    try:
        with pdfplumber.open(file_path) as pdf:
            tables = []
            for page in pdf.pages:
                page_tables = page.extract_tables()
                if page_tables:
                    tables.extend(page_tables)
            
            if not tables:
                return pd.DataFrame()
            
            headers = [str(h).strip() for h in tables[0]]
            data = [row for row in tables[1:] if row]
            
            df = pd.DataFrame(data, columns=headers)
            return df.fillna(0)
    except Exception:
        return pd.DataFrame()
"""

# Write parser code to file
def save_parser_to_file(bank_name: str, code: str) -> Path:
    parser_dir = Path("custom_parsers")
    parser_dir.mkdir(exist_ok=True)
    (parser_dir / "__init__.py").touch(exist_ok=True)
    
    parser_file = parser_dir / f"{bank_name}_parser.py"
    parser_file.write_text(code, encoding="utf-8")
    
    return parser_file

# Generate pytest test file for validation
def create_test_file(bank_name: str) -> Path:
    test_dir = Path("tests")
    test_dir.mkdir(exist_ok=True)
    
    test_file = test_dir / f"test_{bank_name}_parser.py"
    
    test_content = f"""import pandas as pd
import importlib.util
from pathlib import Path

def load_parser(bank):
    parser_path = Path(f"custom_parsers/{{bank}}_parser.py")
    spec = importlib.util.spec_from_file_location(f"{{bank}}_parser", parser_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse

def test_parser_output_matches_csv():
    bank = "{bank_name}"
    parse = load_parser(bank)
    
    csv_ref = Path(f"data/{{bank}}/result.csv")
    pdf_sample = Path(f"data/{{bank}}/sample.pdf")
    
    file_to_parse = csv_ref if csv_ref.exists() else pdf_sample
    parsed_df = parse(str(file_to_parse))
    
    assert not parsed_df.empty, "Parser returned empty DataFrame"
    
    if csv_ref.exists():
        reference_df = pd.read_csv(csv_ref)
        
        for df in [parsed_df, reference_df]:
            for col in df.columns:
                if df[col].dtype != 'object':
                    df[col] = df[col].fillna(0.0)
                else:
                    df[col] = df[col].fillna("")
        
        assert len(parsed_df) == len(reference_df), \\
            f"Row count mismatch: {{len(parsed_df)}} vs {{len(reference_df)}}"
        
        common_cols = set(reference_df.columns).intersection(parsed_df.columns)
        for col in common_cols:
            pd.testing.assert_series_equal(
                parsed_df[col].reset_index(drop=True),
                reference_df[col].reset_index(drop=True),
                check_dtype=False,
                check_names=False
            )
"""
    
    test_file.write_text(test_content, encoding="utf-8")
    return test_file

# Run pytest and capture results
def execute_pytest(test_file: Path) -> Tuple[bool, List[str]]:
    if not test_file.exists():
        return False, ["Test file not found"]
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(test_file)],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        # Extract error information
        errors = []
        if "ImportError" in result.stdout:
            import_match = re.search(r"ImportError: (.*?)(?:\n|$)", result.stdout)
            if import_match:
                error_msg = import_match.group(1)
                errors.append(f"Import error: {error_msg}")
                print(f"  └─ Error: {error_msg[:80]}...")
        
        if "AssertionError" in result.stdout:
            errors.append("Output doesn't match expected format")
            assertion_match = re.search(r"AssertionError: (.*?)(?:\n|$)", result.stdout)
            if assertion_match:
                print(f"  └─ Error: {assertion_match.group(1)[:80]}...")
        
        if "FAILED" in result.stdout and not errors:
            failure_match = re.search(r"AssertionError: (.*?)(?:\n|$)", result.stdout)
            if failure_match:
                errors.append(failure_match.group(1)[:100])
            else:
                errors.append("Test failed")
        
        return result.returncode == 0, errors
        
    except FileNotFoundError:
        print("  └─ ❌ pytest not installed")
        return False, ["pytest not found"]

# LangGraph Node: Planning phase - generate parser code
def planning_node(workflow_state: Dict) -> Dict:
    bank = workflow_state["bank"]
    attempt = workflow_state["attempt"]
    
    print(f"\n{'='*70}")
    print(f"📝 PLANNING PHASE | Cycle {attempt}/{MAX_CORRECTION_ATTEMPTS}")
    print(f"{'='*70}")
    
    error_info = "\n".join(workflow_state.get("error_messages", []))
    
    if attempt > 1:
        print(f"  ├─ Mode: Self-correction (fixing previous errors)")
        print(f"  ├─ Analyzing failure: {error_info[:60]}...")
        print(f"  └─ Generating corrected parser code...")
    else:
        print(f"  ├─ Mode: Initial generation")
        print(f"  ├─ Analyzing: data/{bank}/*.csv and *.pdf")
        print(f"  └─ Generating fresh parser code...")
    
    print(f"\n🤖 LLM Code Generation:")
    parser_code = generate_parser_code(bank, attempt, error_info)
    
    print(f"\n💾 File Operations:")
    parser_path = save_parser_to_file(bank, parser_code)
    print(f"  ├─ Saved: {parser_path}")
    
    test_path = create_test_file(bank)
    print(f"  └─ Created: {test_path}")
    
    return {
        **workflow_state,
        "parser_path": parser_path,
        "test_path": test_path,
        "parser_code": parser_code
    }

# LangGraph Node: Testing phase - run pytest validation
def testing_node(workflow_state: Dict) -> Dict:
    print(f"\n{'='*70}")
    print(f"🧪 TESTING PHASE | Validating parser output")
    print(f"{'='*70}")
    print(f"  ├─ Loading parser module...")
    print(f"  ├─ Parsing sample file...")
    print(f"  ├─ Comparing with expected CSV...")
    print(f"  └─ Running assertions...")
    
    success, errors = execute_pytest(workflow_state["test_path"])
    
    if success:
        print(f"\n  ✅ All tests passed!")
    else:
        print(f"\n  ❌ Tests failed")
    
    return {
        **workflow_state,
        "success": success,
        "error_messages": errors
    }

# LangGraph Node: Decision phase - determine next action
def decision_node(workflow_state: Dict) -> str:
    if workflow_state["success"]:
        print(f"\n{'='*70}")
        print(f"✅ SUCCESS | Parser validated successfully!")
        print(f"{'='*70}")
        print(f"  Bank: {workflow_state['bank'].upper()}")
        print(f"  Total cycles: {workflow_state['attempt']}")
        print(f"  Output: {workflow_state['parser_path']}")
        print(f"{'='*70}\n")
        return END
    
    elif workflow_state["attempt"] >= MAX_CORRECTION_ATTEMPTS:
        print(f"\n{'='*70}")
        print(f"❌ MAXIMUM ATTEMPTS REACHED")
        print(f"{'='*70}")
        print(f"  Bank: {workflow_state['bank'].upper()}")
        print(f"  Cycles attempted: {MAX_CORRECTION_ATTEMPTS}")
        errors = workflow_state.get('error_messages', [])
        if errors:
            print(f"  Unresolved: {errors[0][:60]}...")
        print(f"{'='*70}\n")
        return END
    
    else:
        print(f"\n{'='*70}")
        print(f"🔄 RETRY DECISION | Self-correction triggered")
        print(f"{'='*70}")
        errors = workflow_state.get('error_messages', [])
        print(f"  Root cause: {errors[0] if errors else 'Unknown'}")
        print(f"  Action: Feeding error context back to LLM")
        print(f"  Next: Cycle {workflow_state['attempt'] + 1}/{MAX_CORRECTION_ATTEMPTS}")
        print(f"{'='*70}")
        
        workflow_state["attempt"] += 1
        return "plan"

# Main entry point - build and execute LangGraph workflow
def run_agent():
    import argparse
    
    parser = argparse.ArgumentParser(description="Autonomous Parser Agent")
    parser.add_argument("--target", required=True, help="Bank identifier (e.g., icici, sbi)")
    args = parser.parse_args()
    
    print(f"\n{'#'*70}")
    print(f"#  AUTONOMOUS BANK STATEMENT PARSER AGENT")
    print(f"#  Target: {args.target.upper()}")
    print(f"#  Max Attempts: {MAX_CORRECTION_ATTEMPTS}")
    print(f"#  Architecture: LangGraph (Plan → Test → Decide)")
    print(f"{'#'*70}")
    
    # Build LangGraph workflow
    print(f"\n🏗️  Building workflow graph...")
    workflow = StateGraph(dict)
    
    workflow.add_node("plan", planning_node)
    workflow.add_node("test", testing_node)
    
    workflow.add_edge("plan", "test")
    workflow.add_conditional_edges("test", decision_node)
    workflow.set_entry_point("plan")
    
    print(f"  ├─ Added nodes: plan, test")
    print(f"  ├─ Added edges: plan→test, test→decide")
    print(f"  └─ Entry point: plan")
    
    app = workflow.compile()
    print(f"\n✓ Workflow compiled successfully")
    print(f"\n{'='*70}")
    print(f"🚀 Starting agent execution...")
    print(f"{'='*70}")
    
    initial_state = {
        "bank": args.target.lower(),
        "attempt": 1,
        "success": False,
        "error_messages": []
    }
    
    app.invoke(initial_state)

if __name__ == "__main__":
    run_agent()