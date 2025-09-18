"""
NDJSON output system.
"""
import json
import sys
from typing import List
from .models import AuditResult


class NDJSONOutputter:
    """Handles NDJSON output to stdout."""
    
    def output_results(self, results: List[AuditResult]) -> None:
        """Output results as NDJSON to stdout."""
        for result in results:
            # Convert to JSON string
            json_line = result.model_dump_json()
            
            # Print to stdout (required by spec)
            print(json_line, file=sys.stdout)
            sys.stdout.flush()
    
    def output_single_result(self, result: AuditResult) -> None:
        """Output a single result as NDJSON to stdout."""
        json_line = result.model_dump_json()
        print(json_line, file=sys.stdout)
        sys.stdout.flush()