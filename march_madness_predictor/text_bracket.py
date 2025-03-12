import os
import pandas as pd

# Define paths
base_path = os.path.dirname(os.path.abspath(__file__))
champion_file = os.path.join(base_path, "models/champion_profile/model/all_teams_champion_profile.csv")
output_dir = os.path.join(base_path, "models/full_bracket/model")
os.makedirs(output_dir, exist_ok=True)

print("Loading team data...")
teams_df = pd.read_csv(champion_file)

# Sort by similarity
sorted_teams = teams_df.sort_values('SimilarityPct', ascending=False)

# Get top 64 teams
top_teams = sorted_teams.head(64)

print(f"Selected top {len(top_teams)} teams based on champion profile similarity")

# Create regions
regions = ['East', 'West', 'South', 'Midwest']
bracket = {}

for region in regions:
    bracket[region] = []

# Assign teams to regions and seeds
for i, (idx, team) in enumerate(top_teams.iterrows()):
    region_idx = i % 4
    seed = (i // 4) + 1
    
    bracket[regions[region_idx]].append({
        'name': team['TeamName'],
        'seed': seed,
        'similarity': team['SimilarityPct']
    })

# Print the bracket
output = "NCAA TOURNAMENT BRACKET - BASED ON CHAMPION PROFILE SIMILARITY\n"
output += "=" * 80 + "\n\n"

for region in regions:
    output += f"{region.upper()} REGION\n"
    output += "-" * 40 + "\n"
    
    # Sort by seed
    region_teams = sorted(bracket[region], key=lambda x: x['seed'])
    
    for team in region_teams:
        output += f"Seed {team['seed']}: {team['name']} (Similarity: {team['similarity']:.1f}%)\n"
    
    output += "\nFirst Round Matchups:\n"
    
    # 1 vs 16, 8 vs 9, 5 vs 12, 4 vs 13, 6 vs 11, 3 vs 14, 7 vs 10, 2 vs 15
    matchups = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
    
    for seed1, seed2 in matchups:
        team1 = next((t for t in region_teams if t['seed'] == seed1), None)
        team2 = next((t for t in region_teams if t['seed'] == seed2), None)
        
        if team1 and team2:
            output += f"({seed1}) {team1['name']} vs. ({seed2}) {team2['name']}\n"
    
    output += "=" * 40 + "\n\n"

# Save to file
output_file = os.path.join(output_dir, "simple_bracket.txt")
with open(output_file, 'w') as f:
    f.write(output)

print(f"Simple bracket created and saved to {output_file}") 