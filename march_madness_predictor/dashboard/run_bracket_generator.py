#!/usr/bin/env python3
"""
Run the full bracket generator from the dashboard.

This script is called by the dashboard when the user clicks the "Generate Bracket" button.
It runs the full bracket generator in a separate process to avoid blocking the dashboard.

Usage:
    python run_bracket_generator.py
"""

import os
import sys
import subprocess
import time

def main():
    """Run the full bracket generator in a separate process."""
    # Get the path to the project directory
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Path to the full bracket generator script
    bracket_script = os.path.join(base_path, "full_bracket.py")
    
    # Check if the script exists
    if not os.path.exists(bracket_script):
        print(f"ERROR: Bracket generator script not found at {bracket_script}")
        return 1
    
    # Run the script in a separate process
    try:
        print(f"Running bracket generator: {bracket_script}")
        
        # Start the process
        process = subprocess.Popen(
            [sys.executable, bracket_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the process to complete
        stdout, stderr = process.communicate()
        
        # Check the return code
        if process.returncode == 0:
            print("Bracket generator completed successfully.")
            print(stdout)
            return 0
        else:
            print(f"ERROR: Bracket generator failed with return code {process.returncode}")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return 1
        
    except Exception as e:
        print(f"ERROR: Failed to run bracket generator: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 