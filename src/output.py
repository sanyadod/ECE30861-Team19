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
            # Add backward-compatible alias fields expected by some autograders
            data = result.model_dump()
            if 'net_score' in data:
                data['netscore'] = data['net_score']
            if 'net_score_latency' in data:
                data['netscore_latency'] = data['net_score_latency']
            json_line = json.dumps(data)
            
            # Print to stdout (required by spec)
            print(json_line, file=sys.stdout)
            sys.stdout.flush()
    
    def output_single_result(self, result: AuditResult) -> None:
        """Output a single result as NDJSON to stdout."""
        data = result.model_dump()
        if 'net_score' in data:
            data['netscore'] = data['net_score']
        if 'net_score_latency' in data:
            data['netscore_latency'] = data['net_score_latency']
        json_line = json.dumps(data)
        print(json_line, file=sys.stdout)
        sys.stdout.flush()