import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print(f"NCAA Tournament Champion Profile Analysis (Generated: {datetime.now().strftime('%Y-%m-%d')})")
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

# Function to assess strengths/weaknesses vs champion profile
def assess_team(team_stats, champ_profile):
    assessment = []
    
    # Assess overall efficiency (AdjEM)
    if abs(team_stats['AdjEM'] - champ_profile['AdjEM']) < 3:
        assessment.append("Nearly perfect efficiency margin")
    elif team_stats['AdjEM'] > champ_profile['AdjEM'] + 5:
        assessment.append("Significantly stronger overall efficiency than typical champions")
    elif team_stats['AdjEM'] > champ_profile['AdjEM']:
        assessment.append("Stronger overall efficiency than typical champions")
    elif team_stats['AdjEM'] < champ_profile['AdjEM'] - 5:
        assessment.append("Significantly weaker overall efficiency than typical champions")
    else:
        assessment.append("Slightly weaker overall efficiency than typical champions")
    
    # Assess national ranking
    if abs(team_stats['RankAdjEM'] - champ_profile['RankAdjEM']) <= 2:
        assessment.append("Ideal national ranking (top 5-7)")
    elif team_stats['RankAdjEM'] <= 10:
        assessment.append("Strong national ranking (top 10)")
    elif team_stats['RankAdjEM'] <= 15:
        assessment.append("Good national ranking (top 15)")
    else:
        assessment.append("Ranking too low for typical champion")
    
    # Assess offensive efficiency
    if abs(team_stats['AdjOE'] - champ_profile['AdjOE']) < 5:
        assessment.append("Typical champion-level offense")
    elif team_stats['AdjOE'] > champ_profile['AdjOE'] + 5:
        assessment.append("Elite offense (better than typical champions)")
    elif team_stats['AdjOE'] < champ_profile['AdjOE'] - 5:
        assessment.append("Offense below champion standards")
    
    # Assess defensive efficiency
    if abs(team_stats['AdjDE'] - champ_profile['AdjDE']) < 3:
        assessment.append("Typical champion-level defense")
    elif team_stats['AdjDE'] < champ_profile['AdjDE'] - 3:
        assessment.append("Elite defense (better than typical champions)")
    elif team_stats['AdjDE'] > champ_profile['AdjDE'] + 5:
        assessment.append("Defense significantly below champion standards")
    else:
        assessment.append("Defense slightly below champion standards")
    
    return "; ".join(assessment)

# Load 2025 data
df_2025 = pd.read_csv('susan_kenpom/summary25.csv')

# Clean data
for col in df_2025.columns:
    if df_2025[col].dtype == 'object' and col != 'TeamName':
        df_2025[col] = df_2025[col].str.replace('"', '').astype(float)
    elif col == 'TeamName':
        df_2025[col] = df_2025[col].str.replace('"', '')

print(f"Loaded data for {len(df_2025)} teams from 2025 season")

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
    assessment = assess_team(team_stats, champ_profile)
    
    # Historical tournament success probabilities based on similarity rank
    # These will be assigned later after sorting
    
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
        'DE_Match': de_match,
        'Assessment': assessment
    })

# Sort by similarity (higher is better)
all_teams.sort(key=lambda x: x['Similarity'], reverse=True)

# Create DataFrame with all teams
teams_df = pd.DataFrame(all_teams)

# Add ranked position
teams_df['SimilarityRank'] = range(1, len(teams_df) + 1)

# Historical success probabilities based on similarity rank from 2009-2024 data
champion_probs = {
    1: 14.3, 2: 7.1, 3: 7.1, 4: 0.0, 5: 14.3,
    6: 0.0, 7: 21.4, 8: 0.0, 9: 21.4, 10: 0.0
}

# Assign probabilities for ranks 11-30 (using historical data or approximations)
for i in range(11, 31):
    champion_probs[i] = 2.0  # 2% chance for teams ranked 11-30

# Final Four probabilities
final_four_probs = {
    1: 14.3, 2: 14.3, 3: 35.7, 4: 0.0, 5: 28.6,
    6: 7.1, 7: 28.6, 8: 28.6, 9: 42.9, 10: 14.3
}

# Assign probabilities for ranks 11-30
for i in range(11, 31):
    final_four_probs[i] = 5.0  # 5% chance for teams ranked 11-30

# Add champion and Final Four probabilities based on similarity rank
teams_df['ChampionPct'] = teams_df['SimilarityRank'].apply(
    lambda x: champion_probs.get(x, 1.0) if x <= 30 else 0.5
)
teams_df['FinalFourPct'] = teams_df['SimilarityRank'].apply(
    lambda x: final_four_probs.get(x, 5.0) if x <= 30 else 2.0
)

# Add champion profile difference columns
teams_df['EM_Diff'] = teams_df['AdjEM'] - champ_profile['AdjEM']
teams_df['Rank_Diff'] = teams_df['RankAdjEM'] - champ_profile['RankAdjEM']
teams_df['OE_Diff'] = teams_df['AdjOE'] - champ_profile['AdjOE']
teams_df['DE_Diff'] = teams_df['AdjDE'] - champ_profile['AdjDE']

# Create a formula that combines various metrics for tournament success potential
# This is a composite score based on historical champion metrics and tournament performance
teams_df['TournamentPotential'] = (
    teams_df['Similarity'] * 0.5 +         # Similarity to champion profile
    (100 - teams_df['RankAdjEM']) * 0.3 +  # National ranking (higher rank = lower number = better)
    teams_df['AdjEM'] * 1.0                # Adjusted efficiency margin
) / 50  # Scale to a more readable number

# Format percentage columns
teams_df['SimilarityPct'] = teams_df['Similarity']
teams_df['EM_MatchPct'] = teams_df['EM_Match']
teams_df['Rank_MatchPct'] = teams_df['Rank_Match']
teams_df['OE_MatchPct'] = teams_df['OE_Match']
teams_df['DE_MatchPct'] = teams_df['DE_Match']

# Select and reorder columns for output
output_columns = [
    'SimilarityRank', 'TeamName', 'SimilarityPct', 'ChampionPct', 'FinalFourPct', 'TournamentPotential',
    'AdjEM', 'EM_Diff', 'EM_MatchPct',
    'RankAdjEM', 'Rank_Diff', 'Rank_MatchPct',
    'AdjOE', 'OE_Diff', 'OE_MatchPct',
    'AdjDE', 'DE_Diff', 'DE_MatchPct',
    'Assessment'
]

# Extract top 30 teams for our primary CSV
top30_df = teams_df.sort_values('SimilarityPct', ascending=False).head(30)[output_columns]

# Save to CSV files
top30_filename = 'top30_champion_resemblers.csv'
all_teams_filename = 'all_teams_champion_profile.csv'

top30_df.to_csv(top30_filename, index=False)
teams_df[output_columns].to_csv(all_teams_filename, index=False)

# Display preview of the results
print(f"\nTop 30 Teams Most Similar to Champion Profile (Saved to {top30_filename}):")
print('=' * 100)
print(f"{'Rank':<5}{'Team':<30}{'Similarity':<10}{'Champion%':<10}{'Final Four%':<12}{'AdjEM':<10}{'Off Eff':<10}{'Def Eff':<10}")
print('-' * 100)

for i, team in enumerate(all_teams[:30], 1):
    champion_pct = champion_probs.get(i, 1.0)
    final_four_pct = final_four_probs.get(i, 5.0)
    print(f"{i:<5}{team['TeamName']:<30}{team['Similarity']:<10.1f}{champion_pct:<10.1f}{final_four_pct:<12.1f}{team['AdjEM']:<10.2f}{team['AdjOE']:<10.1f}{team['AdjDE']:<10.1f}")

print(f"\nHistorical Champion Profile: AdjEM: {champ_profile['AdjEM']}, Rank: {champ_profile['RankAdjEM']}, Off Eff: {champ_profile['AdjOE']}, Def Eff: {champ_profile['AdjDE']}")
print(f"\nResults saved to:")
print(f"- Top 30 teams: {top30_filename}")
print(f"- All teams: {all_teams_filename}")

# Create a "tournament bracket" of the top 16 teams by similarity
print("\nMock Tournament Bracket Based on Champion Profile Similarity:")
print("=" * 80)
print("EAST                      WEST")
print(f"1. {all_teams[0]['TeamName']:<20} 1. {all_teams[1]['TeamName']}")
print(f"8. {all_teams[7]['TeamName']:<20} 8. {all_teams[6]['TeamName']}")
print(f"4. {all_teams[3]['TeamName']:<20} 4. {all_teams[2]['TeamName']}")
print(f"5. {all_teams[4]['TeamName']:<20} 5. {all_teams[5]['TeamName']}")
print("")
print("SOUTH                     MIDWEST")
print(f"1. {all_teams[8]['TeamName']:<20} 1. {all_teams[9]['TeamName']}")
print(f"8. {all_teams[15]['TeamName']:<20} 8. {all_teams[14]['TeamName']}")
print(f"4. {all_teams[11]['TeamName']:<20} 4. {all_teams[10]['TeamName']}")
print(f"5. {all_teams[12]['TeamName']:<20} 5. {all_teams[13]['TeamName']}")

# Generate a scatter plot comparing teams to the champion profile
try:
    # Create a figure
    plt.figure(figsize=(12, 8))
    
    # Plot all teams
    plt.scatter(teams_df['AdjOE'], teams_df['AdjDE'], 
                s=teams_df['AdjEM']*3, alpha=0.3, c='gray')
    
    # Highlight top 30 teams
    top30 = teams_df.iloc[:30]
    plt.scatter(top30['AdjOE'], top30['AdjDE'], 
                s=top30['AdjEM']*3, alpha=0.8, c='blue')
    
    # Highlight top 10 teams with labels
    for i, row in teams_df.iloc[:10].iterrows():
        plt.annotate(row['TeamName'], 
                    (row['AdjOE'], row['AdjDE']),
                    xytext=(5, 5), textcoords='offset points')
    
    # Plot champion profile
    plt.scatter([champ_profile['AdjOE']], [champ_profile['AdjDE']], 
                s=champ_profile['AdjEM']*3, c='gold', edgecolors='black', marker='*')
    plt.annotate('Champion Profile', 
                (champ_profile['AdjOE'], champ_profile['AdjDE']),
                xytext=(10, -10), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
    
    # Configure the plot
    plt.axhline(y=champ_profile['AdjDE'], color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=champ_profile['AdjOE'], color='r', linestyle='--', alpha=0.3)
    plt.title('Team Comparison to Champion Profile (2025)')
    plt.xlabel('Offensive Efficiency')
    plt.ylabel('Defensive Efficiency (Lower is Better)')
    plt.grid(True, alpha=0.3)
    
    # Reverse Y-axis since lower defensive numbers are better
    plt.gca().invert_yaxis()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('champion_profile_comparison.png', dpi=300)
    print("\nVisualization saved as 'champion_profile_comparison.png'")
except Exception as e:
    print(f"\nCouldn't generate visualization: {e}") 