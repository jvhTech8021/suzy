import pandas as pd
import numpy as np

# Load 2025 data
df_2025 = pd.read_csv('susan_kenpom/summary25.csv')

# Clean data
for col in df_2025.columns:
    if df_2025[col].dtype == 'object' and col != 'TeamName':
        df_2025[col] = df_2025[col].str.replace('"', '').astype(float)
    elif col == 'TeamName':
        df_2025[col] = df_2025[col].str.replace('"', '')

# Historical champion profile
champ_profile = {
    'AdjEM': 28.72,
    'RankAdjEM': 5.4,
    'AdjOE': 120.6,
    'RankAdjOE': 4.8,  # Average rank of champion offensive efficiency
    'AdjDE': 91.8,
    'RankAdjDE': 11.2  # Average rank of champion defensive efficiency
}

# Calculate distances for each team
teams = []
for _, row in df_2025.iterrows():
    # Calculate differences for absolute values
    em_diff = (row['AdjEM'] - champ_profile['AdjEM'])**2
    oe_diff = (row['AdjOE'] - champ_profile['AdjOE'])**2
    de_diff = (row['AdjDE'] - champ_profile['AdjDE'])**2
    
    # Calculate differences for rankings (with higher weight)
    em_rank_diff = (row['RankAdjEM'] - champ_profile['RankAdjEM'])**2
    
    # Get rankings for offensive and defensive efficiency if available
    oe_rank_diff = 0
    de_rank_diff = 0
    
    if 'RankAdjOE' in row:
        oe_rank_diff = (row['RankAdjOE'] - champ_profile['RankAdjOE'])**2
    else:
        # Use the regular OE rank if available
        oe_rank_diff = (row['RankAdjOE'] - champ_profile['RankAdjOE'])**2
    
    if 'RankAdjDE' in row:
        de_rank_diff = (row['RankAdjDE'] - champ_profile['RankAdjDE'])**2
    else:
        # Use the regular DE rank if available
        de_rank_diff = (row['RankAdjDE'] - champ_profile['RankAdjDE'])**2
    
    # Weight the differences - increased weight on rankings
    # Values: em_diff/100 + oe_diff/100 + de_diff/100 = normalized value differences (lower weight)
    # Rankings: em_rank_diff*2 + oe_rank_diff*1.5 + de_rank_diff*1.5 = ranking differences (higher weight)
    weighted_diff = np.sqrt(
        # Value differences (30% weight)
        (em_diff/100 + oe_diff/100 + de_diff/100) * 0.3 +
        # Ranking differences (70% weight)
        (em_rank_diff*2 + oe_rank_diff*1.5 + de_rank_diff*1.5) * 0.7
    )
    
    # Calculate similarity score (0-100 scale)
    similarity = max(0, 100 - (weighted_diff * 8))
    
    # Calculate percentage match for each component
    # Value matches (absolute values)
    em_match = 100 - min(100, abs(row['AdjEM'] - champ_profile['AdjEM']) / champ_profile['AdjEM'] * 100)
    oe_match = 100 - min(100, abs(row['AdjOE'] - champ_profile['AdjOE']) / champ_profile['AdjOE'] * 100)
    de_match = 100 - min(100, abs(row['AdjDE'] - champ_profile['AdjDE']) / champ_profile['AdjDE'] * 100)
    
    # Rank matches
    rank_match = 100 - min(100, abs(row['RankAdjEM'] - champ_profile['RankAdjEM']) / 20 * 100)
    
    # Add rank matches for OE and DE if available
    oe_rank_match = 100 - min(100, abs(row['RankAdjOE'] - champ_profile['RankAdjOE']) / 20 * 100)
    de_rank_match = 100 - min(100, abs(row['RankAdjDE'] - champ_profile['RankAdjDE']) / 20 * 100)
    
    # Overall rank match is the average of all rank matches
    overall_rank_match = (rank_match + oe_rank_match + de_rank_match) / 3
    
    teams.append({
        'Team': row['TeamName'],
        'Diff': weighted_diff,
        'Similarity': similarity,
        'AdjEM': row['AdjEM'],
        'Rank': row['RankAdjEM'],
        'AdjOE': row['AdjOE'],
        'RankOE': row['RankAdjOE'],
        'AdjDE': row['AdjDE'],
        'RankDE': row['RankAdjDE'],
        'EM_Match': em_match,
        'Rank_Match': overall_rank_match,
        'OE_Match': oe_match,
        'DE_Match': de_match,
        'OE_Rank_Match': oe_rank_match,
        'DE_Rank_Match': de_rank_match
    })

# Sort by similarity (higher is better) instead of difference
teams.sort(key=lambda x: x['Similarity'], reverse=True)

# Print the top 10 teams closest to the champion profile
print('\nTop 10 Teams Most Closely Resembling the Historical Champion Profile:')
print('=' * 120)
print(f"{'Rank':<5}{'Team':<20}{'Similarity':<10}{'AdjEM':<8}{'Natl Rank':<10}{'Off Eff':<8}{'Off Rank':<10}{'Def Eff':<8}{'Def Rank':<10}{'Rank Match':<12}")
print('-' * 120)

for i, team in enumerate(teams[:10], 1):
    print(f"{i:<5}{team['Team']:<20}{team['Similarity']:.1f}%    {team['AdjEM']:<8.2f}{team['Rank']:<10.0f}{team['AdjOE']:<8.1f}{team['RankOE']:<10.0f}{team['AdjDE']:<8.1f}{team['RankDE']:<10.0f}{team['Rank_Match']:.1f}%")

# Print historical champion profile for reference
print('\nHistorical Champion Profile:')
print(f"AdjEM: {champ_profile['AdjEM']}, Natl Rank: {champ_profile['RankAdjEM']}, Off Eff: {champ_profile['AdjOE']} (Rank: {champ_profile['RankAdjOE']}), Def Eff: {champ_profile['AdjDE']} (Rank: {champ_profile['RankAdjDE']})")

# Print detailed analysis of the top 3 teams
print('\nDetailed Analysis of Top 3 Teams:')
for i, team in enumerate(teams[:3], 1):
    print(f"\n{i}. {team['Team']} (Overall Match: {team['Similarity']:.1f}%)")
    print(f"   Adjusted Efficiency Margin: {team['AdjEM']:.2f} (Champion Profile: {champ_profile['AdjEM']}) - {team['EM_Match']:.1f}% match")
    print(f"   National Ranking: {team['Rank']:.0f} (Champion Profile: {champ_profile['RankAdjEM']}) - {team['Rank_Match']:.1f}% match")
    print(f"   Offensive Efficiency: {team['AdjOE']:.1f} (Rank: {team['RankOE']:.0f}) (Champion Profile: {champ_profile['AdjOE']}, Rank: {champ_profile['RankAdjOE']:.1f}) - {team['OE_Match']:.1f}% match")
    print(f"   Defensive Efficiency: {team['AdjDE']:.1f} (Rank: {team['RankDE']:.0f}) (Champion Profile: {champ_profile['AdjDE']}, Rank: {champ_profile['RankAdjDE']:.1f}) - {team['DE_Match']:.1f}% match")
    
    # Assess strengths and weaknesses compared to champion profile
    print("   Analysis:")
    
    if team['AdjEM'] > champ_profile['AdjEM']:
        print(f"   - Overall efficiency is {team['AdjEM'] - champ_profile['AdjEM']:.2f} points higher than typical champions")
    else:
        print(f"   - Overall efficiency is {champ_profile['AdjEM'] - team['AdjEM']:.2f} points lower than typical champions")
        
    if team['Rank'] < champ_profile['RankAdjEM']:
        print(f"   - National ranking ({team['Rank']:.0f}) is better than typical champions ({champ_profile['RankAdjEM']:.1f})")
    else:
        print(f"   - National ranking ({team['Rank']:.0f}) is worse than typical champions ({champ_profile['RankAdjEM']:.1f})")
        
    if team['AdjOE'] > champ_profile['AdjOE']:
        print(f"   - Offense is {team['AdjOE'] - champ_profile['AdjOE']:.1f} points better than typical champions")
    else:
        print(f"   - Offense is {champ_profile['AdjOE'] - team['AdjOE']:.1f} points worse than typical champions")
        
    if team['RankOE'] < champ_profile['RankAdjOE']:
        print(f"   - Offensive ranking ({team['RankOE']:.0f}) is better than typical champions ({champ_profile['RankAdjOE']:.1f})")
    else:
        print(f"   - Offensive ranking ({team['RankOE']:.0f}) is worse than typical champions ({champ_profile['RankAdjOE']:.1f})")
        
    if team['AdjDE'] < champ_profile['AdjDE']:  # Lower is better for defense
        print(f"   - Defense is {champ_profile['AdjDE'] - team['AdjDE']:.1f} points better than typical champions")
    else:
        print(f"   - Defense is {team['AdjDE'] - champ_profile['AdjDE']:.1f} points worse than typical champions")
        
    if team['RankDE'] < champ_profile['RankAdjDE']:  # Lower is better for ranks
        print(f"   - Defensive ranking ({team['RankDE']:.0f}) is better than typical champions ({champ_profile['RankAdjDE']:.1f})")
    else:
        print(f"   - Defensive ranking ({team['RankDE']:.0f}) is worse than typical champions ({champ_profile['RankAdjDE']:.1f})") 