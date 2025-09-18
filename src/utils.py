#!/usr/bin/env python3
"""
Executable entry point 
"""
import sys
import os
import subprocess
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main entry point for the CLI."""
    if len(sys.argv) < 2:
        print("Usage: ./run <command>")
        print("Commands:")
        print("  install        - Install dependencies")
        print("  test          - Run test suite")
        print("  <URL_FILE>    - Process URLs from file")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "install":
        # Install dependencies in userland
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--user", 
                "-r", str(project_root / "requirements.txt")
            ])
            print("Dependencies installed successfully.")
            sys.exit(0)
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies: {e}")
            sys.exit(1)
    
    elif command == "test":
        # Run test suite
        try:
            from src.cli import run_tests
            run_tests()
        except ImportError as e:
            print(f"Failed to import test module: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Test execution failed: {e}")
            sys.exit(1)
    
    else:
        # Treat as URL file path
        url_file = Path(command)
        if not url_file.exists():
            print(f"Error: URL file '{url_file}' does not exist.")
            sys.exit(1)
        
        try:
            from src.cli import process_urls
            process_urls(str(url_file.absolute()))
        except ImportError as e:
            print(f"Failed to import CLI module: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"URL processing failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()