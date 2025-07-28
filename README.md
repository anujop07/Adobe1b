# Challenge 1b: Multi-Collection PDF Analysis

## Overview
Advanced PDF analysis solution that processes multiple document collections and extracts relevant content based on specific personas and use cases.

## Project Structure
```
Challenge_1b/
├── Collection 1/                    # Travel Planning
│   ├── PDFs/                        # South of France guides
│   ├── challenge1b_input.json       # Input configuration
│   └── challenge1b_output.json      # Analysis results
├── Collection 2/                    # Adobe Acrobat Learning
│   ├── PDFs/                        # Acrobat tutorials
│   ├── challenge1b_input.json       # Input configuration
│   └── challenge1b_output.json      # Analysis results
├── Collection 3/                    # Recipe Collection
│   ├── PDFs/                        # Cooking guides
│   ├── challenge1b_input.json       # Input configuration
│   └── challenge1b_output.json      # Analysis results
├── src/
│   ├── main.py                      # Main processing script
│   ├── chunking.py                  # PDF chunking utilities
├── requirements.txt
├── Dockerfile
└── README.md
```

## Key Features
- Persona-based content analysis
- Importance ranking of extracted sections
- Multi-collection document processing
- Structured JSON output with metadata

## Usage

### 1. Build the Docker image
```sh
docker build -t challenge1b-analyzer .
```

### 2. Run the container
```sh
docker run --rm -v "$PWD/Collection\ 1":/app/Collection1 -v "$PWD/Collection\ 2":/app/Collection2 -v "$PWD/Collection\ 3":/app/Collection3 challenge1b-analyzer
```

- Adjust the `-v` flags to mount your collections if your script reads/writes to those folders.

### 3. (Optional) Run with custom input/output
If your script takes arguments for input/output files:
```sh
docker run --rm -v "$PWD/Collection\ 1":/app/Collection1 challenge1b-analyzer python src/main.py --input /app/Collection1/challenge1b_input.json --output /app/Collection1/challenge1b_output.json
``` 