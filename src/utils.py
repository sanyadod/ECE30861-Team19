"""
Utility functions
"""

import re
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional


@contextmanager
def measure_time():
    """Context manager to measure execution time."""
    start_time = time.perf_counter()
    try:
        yield lambda: int((time.perf_counter() - start_time) * 1000)
    finally:
        pass


def extract_model_size_from_text(text: str) -> Optional[float]:
    """
    Extract model size in GB from text using various patterns.

    Returns size in GB, or None if not found.
    """
    if not text:
        return None

    # Patterns to match size indicators
    size_patterns = [
        r"(\d+(?:\.\d+)?)\s*([MGT]?B)\b",  # e.g., "7B", "13.5GB", "270M"
        r"(\d+(?:\.\d+)?)\s*billion",  # e.g., "7 billion parameters"
        r"(\d+(?:\.\d+)?)\s*million",  # e.g., "270 million parameters"
        r"(\d+(?:\.\d+)?)\s*([MGT])\b",  # e.g., "270M", "13B" without B suffix
    ]

    text_lower = text.lower()

    for pattern in size_patterns:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            try:
                size_str = match.group(1)
                size_value = float(size_str)

                if len(match.groups()) > 1 and match.group(2):
                    unit = match.group(2).upper()
                else:
                    # Check context for unit hints
                    context = text_lower[max(0, match.start() - 20):match.end() + 20]
                    if "billion" in context or "billion" in match.group(0).lower():
                        unit = "B"
                    elif "million" in context or "million" in match.group(0).lower():
                        unit = "M"
                    else:
                        unit = ""

                # Convert to GB (rough parameter count to size estimation)
                if unit == "B":  # Billion parameters
                    return (
                        size_value * 2.0
                    )  # ~2GB per billion parameters (rough estimate)
                elif unit == "M":  # Million parameters
                    return size_value * 0.002  # ~2MB per million parameters
                elif unit == "GB":
                    return size_value
                elif unit == "MB":
                    return size_value / 1024.0
                elif unit == "TB":
                    return size_value * 1024.0

            except (ValueError, IndexError):
                continue

    return None


def parse_license_from_readme(readme_content: str) -> Optional[str]:
    """Extract license information from README content."""
    if not readme_content:
        return None

    # Look for license section
    license_patterns = [
        r"##?\s*License\s*\n\s*(.+?)(?:\n##|\n\n|\Z)",
        r"License:\s*(.+?)(?:\n|\Z)",
        r"\*\*License\*\*:?\s*(.+?)(?:\n|\Z)",
    ]

    for pattern in license_patterns:
        match = re.search(pattern, readme_content, re.IGNORECASE | re.DOTALL)
        if match:
            license_text = match.group(1).strip()
            # Clean up common license identifiers
            license_text = re.sub(
                r"\[([^\]]+)\]\([^\)]+\)", r"\1", license_text
            )  # Remove markdown links
            return license_text[:200]  # Limit length

    return None


def check_readme_sections(
    readme_content: str, required_sections: List[str]
) -> Dict[str, bool]:
    """
    Check for presence of required sections in README content.

    Args:
        readme_content: The README text content
        required_sections: List of section names to look for

    Returns:
        Dict mapping section names to boolean presence
    """
    if not readme_content:
        return {section: False for section in required_sections}

    readme_lower = readme_content.lower()
    results = {}

    for section in required_sections:
        section_lower = section.lower()
        # Look for section headers
        patterns = [
            rf"##?\s*{re.escape(section_lower)}\s*\n",  # Markdown header
            rf"\*\*{re.escape(section_lower)}\*\*",  # Bold text
            rf"{re.escape(section_lower)}:",  # Colon format
        ]

        found = any(re.search(pattern, readme_lower) for pattern in patterns)
        results[section] = found

    return results


def extract_performance_claims(
    readme_content: str, benchmark_keywords: List[str]
) -> Dict[str, Any]:
    """
    Extract performance claims and benchmark information from README.

    Returns:
        Dict with 'benchmarks_mentioned', 'numeric_results', 'citations'
    """
    if not readme_content:
        return {
            "benchmarks_mentioned": [],
            "numeric_results": False,
            "citations": False,
        }

    readme_lower = readme_content.lower()

    # Check for benchmark mentions
    benchmarks_found = []
    for benchmark in benchmark_keywords:
        if benchmark.lower() in readme_lower:
            benchmarks_found.append(benchmark)

    # Check for numeric results (patterns like "82.3%", "0.85", "82.3 accuracy")
    numeric_patterns = [
        r"\d+\.\d+%",  # Percentage
        r"\d+%",  # Percentage
        r"accuracy:\s*\d+",  # Accuracy score
        r"f1:\s*\d+\.\d+",  # F1 score
        r"score:\s*\d+\.\d+",  # Generic score
    ]

    has_numeric = any(re.search(pattern, readme_lower) for pattern in numeric_patterns)

    # Check for citations or links (markdown links, DOIs, arxiv)
    citation_patterns = [
        r"\[([^\]]+)\]\([^\)]+\)",  # Markdown links
        r"doi:\s*10\.\d+",  # DOI
        r"arxiv:\d+\.\d+",  # ArXiv
        r"https?://[^\s]+",  # General URLs
    ]

    has_citations = any(
        re.search(pattern, readme_lower) for pattern in citation_patterns
    )

    return {
        "benchmarks_mentioned": benchmarks_found,
        "numeric_results": has_numeric,
        "citations": has_citations,
    }
