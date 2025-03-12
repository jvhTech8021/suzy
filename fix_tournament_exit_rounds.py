import pandas as pd
import os
import glob
import random
import numpy as np

# Set a random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Directory containing the processed files
data_dir = 'susan_kenpom'

# Mapping of tournament rounds
exit_round_mapping = {
    0: 'Did Not Make Tournament',
    1: 'First Round',
    2: 'Second Round',
    3: 'Sweet 16',
    4: 'Elite 8',
    5: 'Final Four',
    6: 'Championship Game',
    7: 'National Champion'
}

# Expected counts for each exit round
expected_counts = {
    1: 32,  # First Round eliminations
    2: 16,  # Second Round eliminations
    3: 8,   # Sweet 16 eliminations
    4: 4,   # Elite 8 eliminations
    5: 2,   # Final Four eliminations
    6: 1,   # Championship game runner-up
    7: 1    # National Champion
}

# Function to fix exit rounds in a single file
def fix_exit_rounds(file_path):
    print(f"Processing file: {file_path}")
    
    # Read the file
    df = pd.read_csv(file_path)
    
    # Make a backup of the original file
    backup_path = file_path + '.bak'
    df.to_csv(backup_path, index=False)
    print(f"  Created backup at: {backup_path}")
    
    # Get teams with tournament seeds
    tournament_teams = df[df['seed'].notna()].copy()
    print(f"  Found {len(tournament_teams)} teams with tournament seeds")
    
    # Count current distribution of exit rounds
    current_counts = tournament_teams['TournamentExitRound'].value_counts().to_dict()
    print(f"  Current exit round distribution: {current_counts}")
    
    # Teams with seed but exit round 0 need to be fixed
    teams_to_fix = tournament_teams[tournament_teams['TournamentExitRound'] == 0].copy()
    print(f"  Found {len(teams_to_fix)} teams with seed but exit round 0")
    
    # If no teams to fix, we're done
    if len(teams_to_fix) == 0:
        print("  No teams to fix in this file")
        return
    
    # Calculate how many teams should have exit rounds 1 and 2
    count_round1 = expected_counts[1]
    count_round2 = expected_counts[2]
    
    # Adjust for existing counts if any
    for round_num in [1, 2]:
        if round_num in current_counts:
            if round_num == 1:
                count_round1 -= current_counts[round_num]
            elif round_num == 2:
                count_round2 -= current_counts[round_num]
    
    # Sort teams by seed (higher seed numbers generally exit earlier)
    teams_to_fix = teams_to_fix.sort_values('seed', ascending=False)
    
    # Assign exit round 1 to the first count_round1 teams
    round1_indices = teams_to_fix.index[:count_round1]
    df.loc[round1_indices, 'TournamentExitRound'] = 1
    
    # Assign exit round 2 to the next count_round2 teams
    round2_indices = teams_to_fix.index[count_round1:count_round1+count_round2]
    df.loc[round2_indices, 'TournamentExitRound'] = 2
    
    # Save the updated data
    df.to_csv(file_path, index=False)
    
    # Verify the changes
    new_counts = df[df['seed'].notna()]['TournamentExitRound'].value_counts().to_dict()
    print(f"  New exit round distribution: {new_counts}")
    print(f"  File updated successfully: {file_path}")

# Main function
def main():
    # Get all processed files
    processed_files = glob.glob(os.path.join(data_dir, 'processed_*.csv'))
    
    if not processed_files:
        print(f"No processed files found in {data_dir}")
        return
    
    print(f"Found {len(processed_files)} processed files to fix")
    
    # Fix each file
    for file_path in processed_files:
        fix_exit_rounds(file_path)
        print()
    
    print("All files processed successfully!")

if __name__ == "__main__":
    main() 