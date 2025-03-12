#!/usr/bin/env python3
"""
Runner script for the full NCAA tournament bracket generator.
This script generates a complete 64-team bracket using data from both prediction models.
"""

import os
import sys
import traceback

# Add the project directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_dir)

# Import the bracket generator
from march_madness_predictor.models.full_bracket_generator import FullBracketGenerator

def main():
    """
    Main function to run the full bracket generator
    """
    try:
        print("Running Full NCAA Tournament Bracket Generator...")
        
        # Initialize the generator
        generator = FullBracketGenerator()
        
        # Generate the bracket
        generator.generate_bracket()
        
        print("\nFull NCAA Tournament Bracket Generator completed successfully!")
        print(f"Results saved to: {os.path.join(generator.output_dir)}")
        
    except Exception as e:
        print(f"Error generating full bracket: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 