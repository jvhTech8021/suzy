#!/usr/bin/env python3
"""
Run the full bracket generator to create a complete NCAA tournament bracket.

This script runs the full bracket generator, which creates a simulated NCAA tournament
bracket based on the combined results from the Champion Profile and Exit Round prediction models.

Usage:
    python run_full_bracket.py

Output:
    - Full bracket text file
    - Tournament results CSV file
    - JSON representation of the bracket
"""

import os
import sys
import time

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Run the full bracket generator."""
    print("=" * 80)
    print("NCAA TOURNAMENT BRACKET GENERATOR")
    print("=" * 80)
    print("\nThis script will generate a complete NCAA tournament bracket simulation")
    print("based on the Champion Profile and Exit Round prediction models.\n")
    
    # Check if the required model outputs exist
    base_path = os.path.dirname(os.path.abspath(__file__))
    champion_file = os.path.join(base_path, "models/champion_profile/model/all_teams_champion_profile.csv")
    exit_round_file = os.path.join(base_path, "models/exit_round/model/tournament_teams_predictions.csv")
    
    missing_files = []
    if not os.path.exists(champion_file):
        missing_files.append("Champion Profile predictions (run the Champion Profile model first)")
    if not os.path.exists(exit_round_file):
        missing_files.append("Exit Round predictions (run the Exit Round model first)")
    
    if missing_files:
        print("ERROR: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease run the required models first and try again.")
        return 1
    
    print("All required model outputs found. Generating bracket...\n")
    
    try:
        # Import and run the full bracket generator
        from full_bracket import main as generate_bracket
        
        # Start timer
        start_time = time.time()
        
        # Run the generator
        generate_bracket()
        
        # End timer
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\nBracket generation completed in {elapsed_time:.2f} seconds.")
        
        # Check if the output files were created
        output_dir = os.path.join(base_path, "models/full_bracket/model")
        bracket_file = os.path.join(output_dir, "full_bracket.txt")
        results_file = os.path.join(output_dir, "tournament_results.csv")
        
        if os.path.exists(bracket_file) and os.path.exists(results_file):
            print("\nOutput files created successfully:")
            print(f"  - Full bracket: {bracket_file}")
            print(f"  - Tournament results: {results_file}")
            print("\nYou can now view the bracket in the dashboard at http://127.0.0.1:8050/full-bracket")
            return 0
        else:
            print("\nERROR: Output files were not created. Check the logs for errors.")
            return 1
        
    except Exception as e:
        print(f"\nERROR: An error occurred while generating the bracket: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 