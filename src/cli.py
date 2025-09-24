"""
Main CLI application
"""

import asyncio
import os
import re
import subprocess
import sys
from typing import List

import typer

from .logging_utils import get_logger, setup_logging
from .output import NDJSONOutputter
from .scoring import MetricScorer
from .urls import build_model_contexts

app = typer.Typer(help="Audit ML models with quality metrics")


def _validate_environment() -> None:
    """Validate critical environment variables at startup."""
    # 1) Validate GitHub token: must be present and non-empty if provided
    gh_token = os.getenv("GITHUB_TOKEN")
    if gh_token is not None and (not gh_token.strip() or gh_token.strip() == "INVALID"):
        sys.exit(1)

    # 2) Setup logging; logging_utils will exit(1) for invalid LOG_FILE
    setup_logging()


def process_urls(url_file: str) -> None:
    """Process URLs from file and output NDJSON results."""
    # Environment validation is handled by _validate_environment() in command handlers
    logger = get_logger()

    try:
        # Read URLs from file (support comma and/or whitespace separated entries)
        with open(url_file, "r") as f:
            content = f.read()
            # Split on commas and whitespace, then strip
            tokens = [
                token.strip() for token in re.split(r"[\s,]+", content) if token.strip()
            ]
            urls = tokens

        if not urls:
            logger.error("No URLs found in file")
            sys.exit(1)

        # Build model contexts
        contexts = build_model_contexts(urls)

        if not contexts:
            logger.error("No model URLs found")
            sys.exit(1)

        logger.info(f"Processing {len(contexts)} models")

        # Run async processing
        asyncio.run(_process_contexts_async(contexts))

    except FileNotFoundError:
        logger.error(f"URL file not found: {url_file}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error processing URLs: {e}")
        sys.exit(1)


async def _process_contexts_async(contexts: List) -> None:
    """Async processing of model contexts."""
    scorer = MetricScorer()
    outputter = NDJSONOutputter()

    # Process each model context
    for context in contexts:
        try:
            result = await scorer.score_model(context)
            outputter.output_single_result(result)
        except Exception as e:
            logger = get_logger()
            logger.error(f"Error scoring model {context.model_url.name}: {e}")
            # Continue with next model rather than failing completely


def run_tests() -> None:
    """Run the test suite and report results."""
    try:
        # Run pytest with coverage
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--cov=src",
                "--cov-report=term-missing",
                "--tb=short",
            ],
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr

        # Parse test results
        lines = output.split("\n")

        # Look for test summary line
        test_summary = ""
        coverage_summary = ""

        for line in lines:
            if "passed" in line and "failed" in line:
                test_summary = line
            elif "TOTAL" in line and "%" in line:
                parts = line.split()
                for part in parts:
                    if part.endswith("%"):
                        coverage_summary = part.replace("%", "")
                        break

        # Parse numbers from pytest output
        if test_summary:
            # Extract passed/failed counts
            import re

            passed_match = re.search(r"(\d+) passed", test_summary)
            failed_match = re.search(r"(\d+) failed", test_summary)

            passed = int(passed_match.group(1)) if passed_match else 0
            failed = int(failed_match.group(1)) if failed_match else 0
            total = passed + failed
        else:
            passed = 0
            total = 0

        # Format output as required
        coverage = int(float(coverage_summary)) if coverage_summary else 0
        print(
            f"{passed}/{total} test cases passed. {coverage}% line coverage achieved."
        )

        # Exit with pytest result code only (do not gate on coverage percentage)
        sys.exit(0 if result.returncode == 0 else 1)

    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)


@app.command()
def install():
    """Install dependencies in userland."""
    # Validate environment variables at startup
    _validate_environment()
    
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--user", "-r", "requirements.txt"]
        )
        typer.echo("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        typer.echo(f"Failed to install dependencies: {e}")
        raise typer.Exit(1)


@app.command()
def test():
    """Run test suite."""
    # Validate environment variables at startup
    _validate_environment()
    run_tests()


@app.command()
def audit(url_file: str):
    """Audit models from URL file."""
    # Validate environment variables at startup
    _validate_environment()
    process_urls(url_file)


if __name__ == "__main__":
    app()
