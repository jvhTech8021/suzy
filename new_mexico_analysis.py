import pandas as pd
import numpy as np
import os

# Historical champion profile
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
    
    return similarity, weighted_diff

# Load 2025 data
df_2025 = pd.read_csv('susan_kenpom/summary25.csv')

# Clean data
for col in df_2025.columns:
    if df_2025[col].dtype == 'object' and col != 'TeamName':
        df_2025[col] = df_2025[col].str.replace('"', '').astype(float)
    elif col == 'TeamName':
        df_2025[col] = df_2025[col].str.replace('"', '')

# Calculate similarity for all teams
all_similarities = []
for _, row in df_2025.iterrows():
    team_stats = {
        'AdjEM': row['AdjEM'],
        'RankAdjEM': row['RankAdjEM'],
        'AdjOE': row['AdjOE'],
        'AdjDE': row['AdjDE']
    }
    
    similarity, diff = calculate_similarity(team_stats, champ_profile)
    
    all_similarities.append({
        'Team': row['TeamName'],
        'Similarity': similarity,
        'AdjEM': row['AdjEM'],
        'RankAdjEM': row['RankAdjEM'],
        'AdjOE': row['AdjOE'],
        'AdjDE': row['AdjDE'],
        'Diff': diff
    })

# Sort by similarity
all_similarities.sort(key=lambda x: x['Similarity'], reverse=True)

# Print the top 20 teams for 2025
print("\nTop 20 Teams by Similarity to Champion Profile for 2025:")
print('=' * 95)
print(f"{'Rank':<5}{'Team':<25}{'Similarity':<10}{'AdjEM':<10}{'Natl Rank':<10}{'Off Eff':<10}{'Def Eff':<10}")
print('-' * 95)

for i, team in enumerate(all_similarities[:20], 1):
    print(f"{i:<5}{team['Team']:<25}{team['Similarity']:<10.1f}{team['AdjEM']:<10.2f}{team['RankAdjEM']:<10.0f}{team['AdjOE']:<10.1f}{team['AdjDE']:<10.1f}")

# New Mexico analysis
if 'New Mexico' in df_2025['TeamName'].values:
    # Find New Mexico in our similarity list
    nm_rank = next((i+1 for i, team in enumerate(all_similarities) if team['Team'] == 'New Mexico'), None)
    nm_data = next((team for team in all_similarities if team['Team'] == 'New Mexico'), None)
    
    if nm_data:
        nm_similarity = nm_data['Similarity']
        
        print("\nNew Mexico 2025 Analysis:")
        print("=" * 80)
        print(f"Similarity to Champion Profile: {nm_similarity:.1f}%")
        print(f"Similarity Rank: {nm_rank} out of {len(all_similarities)} teams")
        
        # Compare to champion profile
        print("\nNew Mexico's Stats vs Champion Profile:")
        print(f"Adjusted Efficiency Margin: {nm_data['AdjEM']:.2f} (Champion Profile: {champ_profile['AdjEM']})")
        print(f"National Ranking: {nm_data['RankAdjEM']:.0f} (Champion Profile: {champ_profile['RankAdjEM']})")
        print(f"Offensive Efficiency: {nm_data['AdjOE']:.1f} (Champion Profile: {champ_profile['AdjOE']})")
        print(f"Defensive Efficiency: {nm_data['AdjDE']:.1f} (Champion Profile: {champ_profile['AdjDE']})")
        
        # Analysis
        print("\nAnalysis:")
        if nm_rank <= 20:
            print("New Mexico has a statistical profile that resembles past champions.")
        elif nm_rank <= 50:
            print("New Mexico has some similarities to champion profiles but significant differences exist.")
        else:
            print("New Mexico's statistical profile differs substantially from historical champions.")
        
        # Specific strengths/weaknesses
        print("\nKey Differences from Champion Profile:")
        if abs(nm_data['AdjEM'] - champ_profile['AdjEM']) > 10:
            if nm_data['AdjEM'] < champ_profile['AdjEM']:
                print(f"- Overall efficiency ({nm_data['AdjEM']:.2f}) is significantly lower than champion average ({champ_profile['AdjEM']})")
            else:
                print(f"- Overall efficiency ({nm_data['AdjEM']:.2f}) is significantly higher than champion average ({champ_profile['AdjEM']})")
                
        if abs(nm_data['RankAdjEM'] - champ_profile['RankAdjEM']) > 20:
            print(f"- National ranking ({nm_data['RankAdjEM']:.0f}) is much lower than champion average ({champ_profile['RankAdjEM']})")
                
        if abs(nm_data['AdjOE'] - champ_profile['AdjOE']) > 10:
            if nm_data['AdjOE'] < champ_profile['AdjOE']:
                print(f"- Offensive efficiency ({nm_data['AdjOE']:.1f}) is weaker than champion average ({champ_profile['AdjOE']})")
            else:
                print(f"- Offensive efficiency ({nm_data['AdjOE']:.1f}) is stronger than champion average ({champ_profile['AdjOE']})")
                
        if abs(nm_data['AdjDE'] - champ_profile['AdjDE']) > 10:
            if nm_data['AdjDE'] > champ_profile['AdjDE']:
                print(f"- Defensive efficiency ({nm_data['AdjDE']:.1f}) is weaker than champion average ({champ_profile['AdjDE']})")
            else:
                print(f"- Defensive efficiency ({nm_data['AdjDE']:.1f}) is stronger than champion average ({champ_profile['AdjDE']})")
                
        # Historical context based on New Mexico's 2023-24 performance
        print("\nHistorical Context:")
        print("New Mexico made the NCAA tournament in 2023-24, winning the Mountain West Conference tournament.")
        print("They entered as an 11 seed but lost in the first round to Clemson.")
        
        # 2024-25 season so far
        print("\n2024-25 Season Performance:")
        print("Current record: 24-6 (16-3 in Mountain West Conference)")
        print("New Mexico is currently receiving votes in AP polls but not ranked in the Top 25.")
else:
    print("\nNew Mexico data not found in the dataset.") 