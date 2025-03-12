import pandas as pd
import numpy as np

print("Generating Top 30 Teams Most Similar to Champion Profile (2025)")
print("=" * 80)

# Historical champion profile (2009-2024 average)
champ_profile = {
    'AdjEM': 28.72,
    'RankAdjEM': 5.4,
    'AdjOE': 120.6,
    'AdjDE': 91.8
}

# Function to calculate similarity to champion profile
def calculate_similarity(team_stats, champ_profile):
    # Calculate normalized differences for each metric
    em_diff = (team_stats['AdjEM'] - champ_profile['AdjEM'])**2
    rank_diff = (team_stats['RankAdjEM'] - champ_profile['RankAdjEM'])**2
    oe_diff = (team_stats['AdjOE'] - champ_profile['AdjOE'])**2
    de_diff = (team_stats['AdjDE'] - champ_profile['AdjDE'])**2
    
    # Weight the differences (normalizing by typical range of each metric)
    weighted_diff = np.sqrt(em_diff/100 + rank_diff + oe_diff/100 + de_diff/100)
    
    # Calculate similarity score (0-100 scale)
    similarity = max(0, 100 - (weighted_diff * 10))
    
    # Calculate percentage match for each component
    em_match = 100 - min(100, abs(team_stats['AdjEM'] - champ_profile['AdjEM']) / champ_profile['AdjEM'] * 100)
    rank_match = 100 - min(100, abs(team_stats['RankAdjEM'] - champ_profile['RankAdjEM']) / 20 * 100)  # Assume top 20 is the scale
    oe_match = 100 - min(100, abs(team_stats['AdjOE'] - champ_profile['AdjOE']) / champ_profile['AdjOE'] * 100)
    de_match = 100 - min(100, abs(team_stats['AdjDE'] - champ_profile['AdjDE']) / champ_profile['AdjDE'] * 100)
    
    return similarity, weighted_diff, em_match, rank_match, oe_match, de_match

# Load 2025 data
df_2025 = pd.read_csv('susan_kenpom/summary25.csv')

# Clean data
for col in df_2025.columns:
    if df_2025[col].dtype == 'object' and col != 'TeamName':
        df_2025[col] = df_2025[col].str.replace('"', '').astype(float)
    elif col == 'TeamName':
        df_2025[col] = df_2025[col].str.replace('"', '')

# Calculate similarity for all teams
all_teams = []
for _, row in df_2025.iterrows():
    team_stats = {
        'AdjEM': row['AdjEM'],
        'RankAdjEM': row['RankAdjEM'],
        'AdjOE': row['AdjOE'],
        'AdjDE': row['AdjDE']
    }
    
    similarity, diff, em_match, rank_match, oe_match, de_match = calculate_similarity(team_stats, champ_profile)
    
    all_teams.append({
        'TeamName': row['TeamName'],
        'Similarity': similarity, 
        'AdjEM': row['AdjEM'],
        'RankAdjEM': row['RankAdjEM'],
        'AdjOE': row['AdjOE'], 
        'AdjDE': row['AdjDE'],
        'EM_Match': em_match,
        'Rank_Match': rank_match,
        'OE_Match': oe_match,
        'DE_Match': de_match
    })

# Sort by similarity (higher is better)
all_teams.sort(key=lambda x: x['Similarity'], reverse=True)

# Create DataFrame with top 30 teams
top30_df = pd.DataFrame(all_teams[:30])

# Add champion profile difference columns
top30_df['EM_Diff'] = top30_df['AdjEM'] - champ_profile['AdjEM']
top30_df['Rank_Diff'] = top30_df['RankAdjEM'] - champ_profile['RankAdjEM']
top30_df['OE_Diff'] = top30_df['AdjOE'] - champ_profile['AdjOE']
top30_df['DE_Diff'] = top30_df['AdjDE'] - champ_profile['AdjDE']

# Add historical success probabilities based on similarity rank
# These are approximate values derived from historical analysis
champion_probs = [14.3, 7.1, 7.1, 0.0, 14.3, 0.0, 21.4, 0.0, 21.4, 0.0]  # First 10 ranks
champion_probs.extend([2.0] * 20)  # Ranks 11-30 (approximate)

# Format values for output
top30_df['ChampionPct'] = [champion_probs[i] if i < len(champion_probs) else 1.0 for i in range(len(top30_df))]
top30_df['SimilarityPct'] = top30_df['Similarity']
top30_df['EM_MatchPct'] = top30_df['EM_Match']
top30_df['Rank_MatchPct'] = top30_df['Rank_Match']
top30_df['OE_MatchPct'] = top30_df['OE_Match']
top30_df['DE_MatchPct'] = top30_df['DE_Match']

# Select and reorder columns for output
output_columns = [
    'TeamName', 'SimilarityPct', 'ChampionPct',
    'AdjEM', 'EM_Diff', 'EM_MatchPct',
    'RankAdjEM', 'Rank_Diff', 'Rank_MatchPct',
    'AdjOE', 'OE_Diff', 'OE_MatchPct',
    'AdjDE', 'DE_Diff', 'DE_MatchPct'
]

final_df = top30_df[output_columns]

# Save to CSV
csv_filename = 'top30_champion_resemblers.csv'
final_df.to_csv(csv_filename, index=False)

# Display preview of the results
print(f"\nTop 30 Teams Most Similar to Champion Profile (Saved to {csv_filename}):")
print('=' * 100)
print(f"{'Rank':<5}{'Team':<30}{'Similarity':<10}{'AdjEM':<10}{'Natl Rank':<10}{'Off Eff':<10}{'Def Eff':<10}")
print('-' * 100)

for i, team in enumerate(all_teams[:30], 1):
    print(f"{i:<5}{team['TeamName']:<30}{team['Similarity']:<10.1f}{team['AdjEM']:<10.2f}{team['RankAdjEM']:<10.0f}{team['AdjOE']:<10.1f}{team['AdjDE']:<10.1f}")

print(f"\nHistorical Champion Profile: AdjEM: {champ_profile['AdjEM']}, Rank: {champ_profile['RankAdjEM']}, Off Eff: {champ_profile['AdjOE']}, Def Eff: {champ_profile['AdjDE']}")
print(f"\nResults successfully saved to {csv_filename}") 