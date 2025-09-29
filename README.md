# ECE30861-Team19
## Team
Sanya Dod
Spoorthi Koppula
Romita Pakrasi
Suhani Mathur

## Overview
The system is a comprehensive machine learning model auditing platform that evaluates the quality and trustworthiness of ML models hosted on platforms like Hugging Face through eight distinct metrics. The system accepts input files containing triplets of URLs (code repository, dataset, model) and processes each model asynchronously to compute weighted quality scores across multiple dimensions including ramp-up time (ease of getting started), bus factor (contributor diversity), performance claims (documented benchmarks), license compatibility, deployment size feasibility, dataset-code linkage quality, dataset documentation quality, and code structure quality. Each metric is computed in parallel using a configurable scoring system with weights defined in a YAML configuration file, and the results are outputted as NDJSON format for easy integration with other tools. The platform includes robust error handling, comprehensive logging, environment validation (including GitHub token verification), and a complete test suite with coverage requirements, making it suitable for automated ML model evaluation pipelines in research and production environments.

## Features
- CLI interface with ./run supporting:
    - ./run install – installs dependencies.

    - ./run URL_FILE – evaluates models/datasets/code URLs and outputs NDJSON with required fields.

    - ./run test – runs the test suite.

- Outputs 8 different metrics.
    - bus factor
    - code quality
    - dataset and code quality
    - dataset quality
    - license 
    - performance
    - ramp up
    - size

- Parallel metric calculation for better latency.

## Structure
```bash
├── src/                # main code: metrics, utilities
├── tests/              # unit tests and integration tests
├── requirements.txt    # Python dependencies
├── run                 # executable wrapper / entrypoint
├── sample_urls.txt     # example model URLs for testing
└── README.md           
```

## Installation
Install requirements given.
```bash
pip install -r requirements.txt
```

## Testing
Create virtual environment.
```bash
python3 -m venv venv
```

Open virtual environment (macOS and Linux).
```bash
source venv/bin/activate
```

Run test with sample urls.
```bash
./run test sample_urls.txt
```


