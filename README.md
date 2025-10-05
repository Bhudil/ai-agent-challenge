# ai-agent-challenge
Coding agent challenge which write custom parsers for Bank statement PDF.

## Overview

This agent implements a self-correcting workflow that:
1. Analyzes bank statement PDFs and expected CSV outputs
2. Generates Python parser code using LLM
3. Validates output against ground truth
4. Self-corrects through iterative refinement (up to 3 attempts)

## Architecture

The agent follows a LangGraph-based workflow with three main phases:

```
┌─────────────────────────────────────────────────────────────────┐
│                      AGENT WORKFLOW                              │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │   START      │
    └──────┬───────┘
           │
           v
    ┌──────────────────────────────────────────────────────────┐
    │  PLANNING PHASE (Node: plan)                             │
    │  - Read data/{bank}/*.pdf and *.csv files                │
    │  - Generate parser code via LLM (Groq)                   │
    │  - Validate syntax and structure                         │
    │  - Write to custom_parsers/{bank}_parser.py              │
    │  - Create pytest test file                               │
    └──────────────┬───────────────────────────────────────────┘
                   │
                   v
    ┌──────────────────────────────────────────────────────────┐
    │  TESTING PHASE (Node: test)                              │
    │  - Import generated parser                               │
    │  - Parse PDF and compare with expected CSV               │
    │  - Execute pytest validation                             │
    │  - Capture errors and output                             │
    └──────────────┬───────────────────────────────────────────┘
                   │
                   v
    ┌──────────────────────────────────────────────────────────┐
    │  DECISION PHASE (Node: decide)                           │
    │  - Evaluate test results                                 │
    │  - If success → END                                      │
    │  - If failed and attempts < 3 → loop back to PLANNING    │
    │    with error context for self-correction                │
    │  - If max attempts reached → END with failure            │
    └──────────────┬───────────────────────────────────────────┘
                   │
                   v
            ┌──────────┐
            │   END    │
            └──────────┘

Key Features:
- Self-debugging loop with error feedback
- Automatic fallback to baseline parser on syntax errors
- Flexible file detection (works with various naming conventions)
- DataFrame.equals validation as per specification
```

## Requirements

### System Requirements
- Python 3.8+
- pip package manager

### Python Dependencies
```
langgraph
groq
pandas
pdfplumber
python-dotenv
pytest
openpyxl  # for Excel file support
```

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-repo/ai-agent-challenge.git
cd ai-agent-challenge
```

### Step 2: Install Dependencies
```bash
pip install langgraph groq pandas pdfplumber python-dotenv pytest openpyxl
```

### Step 3: Configure API Key
Create a `.env` file in the project root:
```bash
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

To obtain a free Groq API key:
- Visit https://console.groq.com
- Sign up for an account
- Navigate to API Keys section
- Generate a new key

### Step 4: Prepare Data
Organize your bank statement data:
```
data/
├── icici/
│   ├── icici sample.pdf  (or any .pdf file)
│   └── result.csv        (or any .csv/.xlsx file)
└── sbi/
    ├── statement.pdf
    └── expected.csv
```

The agent automatically detects PDF and CSV/Excel files in each bank's data directory.

### Step 5: Run the Agent
```bash
python agent.py --target icici
```

Replace `icici` with your target bank identifier (must match the folder name in `data/`).

## Usage Examples

### Generate Parser for ICICI Bank
```bash
python agent.py --target icici
```

### Generate Parser for SBI Bank
```bash
python agent.py --target sbi
```

### Run Tests Manually
```bash
pytest tests/test_icici_parser.py -v
```

## Project Structure

```
ai-agent-challenge/
├── agent.py                      # Main agent implementation
├── .env                          # API configuration (create this)
├── data/
│   └── {bank}/
│       ├── *.pdf                 # Bank statement PDF
│       └── *.csv or *.xlsx       # Expected output
├── custom_parsers/               # Generated parsers (auto-created)
│   ├── __init__.py
│   └── {bank}_parser.py
├── tests/                        # Generated tests (auto-created)
│   └── test_{bank}_parser.py
└── README.md
```

- Architecture diagram included above

## How It Works

1. **Planning Phase**: Agent analyzes the data directory, sends a prompt to Groq LLM requesting parser code, validates syntax, and writes the file.

2. **Testing Phase**: Agent dynamically imports the generated parser, runs it against the PDF, and compares output with expected CSV using DataFrame.equals.

3. **Decision Phase**: If tests pass, agent terminates successfully. If tests fail and attempts remain, agent extracts error information and loops back to planning with error context for self-correction.



## Contributing

This is a challenge submission. For evaluation purposes only.
