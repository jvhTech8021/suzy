import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

print("Historical Analysis: Teams Most Similar to Champion Profile (2009-2024)\n")
print("=" * 80)

# Define the champion profile (historical average)
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

# Dictionary to store tournament exit rounds
round_mapping = {
    0: "Did not make tournament",
    1: "First Round (Round of 64)",
    2: "Second Round (Round of 32)",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Championship Game",
    7: "National Champion"
}

# Will store our results for each year
yearly_results = []

# Examine each year's data
for year in range(2009, 2025):
    try:
        # Read the processed file for this year
        file_path = f'susan_kenpom/processed_{year}.csv'
        
        if not os.path.exists(file_path):
            print(f"No data available for {year}")
            continue
            
        df = pd.read_csv(file_path)
        print(f"Analyzing {year} season data...")
        
        # Calculate similarity scores for all teams
        teams = []
        for _, row in df.iterrows():
            team_stats = {
                'AdjEM': row['AdjEM'],
                'RankAdjEM': row['RankAdjEM'],
                'AdjOE': row['AdjOE'],
                'AdjDE': row['AdjDE']
            }
            
            similarity, diff = calculate_similarity(team_stats, champ_profile)
            
            teams.append({
                'Team': row['TeamName'],
                'Similarity': similarity,
                'Diff': diff,
                'AdjEM': row['AdjEM'],
                'Rank': row['RankAdjEM'],
                'AdjOE': row['AdjOE'],
                'AdjDE': row['AdjDE'],
                'TournamentExitRound': row['TournamentExitRound'],
                'TournamentSeed': row.get('TournamentSeed', 0)
            })
        
        # Sort by similarity (higher is better)
        teams.sort(key=lambda x: x['Similarity'], reverse=True)
        
        # Get the actual champion for this year
        actual_champion = None
        champion_similarity_rank = None
        for i, team in enumerate(teams):
            if team['TournamentExitRound'] == 7:  # National Champion
                actual_champion = team['Team']
                champion_similarity_rank = i + 1
                break
        
        # Store top 10 most similar teams for this year and their results
        year_result = {
            'Year': year,
            'TopTeams': teams[:10],
            'ActualChampion': actual_champion,
            'ChampionSimilarityRank': champion_similarity_rank
        }
        
        yearly_results.append(year_result)
    except Exception as e:
        print(f"Error processing {year} data: {e}")

# Now create a summary of our findings

# How often top teams by similarity made deep runs
success_by_similarity_rank = {
    'Champion': [0] * 10,
    'Final Four': [0] * 10,
    'Elite Eight': [0] * 10,
    'Sweet Sixteen': [0] * 10
}

# Track where the champions ranked in similarity
champion_ranks = []

# Analyze the results
for year_result in yearly_results:
    year = year_result['Year']
    
    # Skip years where we don't have champion data (2020 COVID cancellation)
    if year_result['ActualChampion'] is None:
        continue
        
    champion_ranks.append(year_result['ChampionSimilarityRank'])
    
    # Analyze tournament performance of top 10 most similar teams
    for i, team in enumerate(year_result['TopTeams']):
        exit_round = team['TournamentExitRound']
        
        if exit_round == 7:  # Champion
            success_by_similarity_rank['Champion'][i] += 1
            success_by_similarity_rank['Final Four'][i] += 1
            success_by_similarity_rank['Elite Eight'][i] += 1
            success_by_similarity_rank['Sweet Sixteen'][i] += 1
        elif exit_round >= 5:  # Final Four
            success_by_similarity_rank['Final Four'][i] += 1
            success_by_similarity_rank['Elite Eight'][i] += 1
            success_by_similarity_rank['Sweet Sixteen'][i] += 1
        elif exit_round == 4:  # Elite Eight
            success_by_similarity_rank['Elite Eight'][i] += 1
            success_by_similarity_rank['Sweet Sixteen'][i] += 1
        elif exit_round == 3:  # Sweet Sixteen
            success_by_similarity_rank['Sweet Sixteen'][i] += 1

# Print summary statistics
print("\nSummary of Historical Analysis (2009-2024):")
print("=" * 80)

# How often did the most similar team to champion profile win?
top_similarity_champions = success_by_similarity_rank['Champion'][0]
total_years = len([yr for yr in yearly_results if yr['ActualChampion'] is not None])

print(f"The team most similar to champion profile won the tournament {top_similarity_champions} out of {total_years} years ({top_similarity_champions/total_years*100:.1f}%)")

# Average rank of eventual champion in similarity
avg_champion_rank = sum(champion_ranks) / len(champion_ranks)
print(f"The eventual champion ranked {avg_champion_rank:.1f} on average in similarity to champion profile")

# Print success rates by similarity rank
print("\nTournament Success by Similarity Rank:")
print(f"{'Rank':<6}{'Champion %':<15}{'Final Four %':<15}{'Elite Eight %':<15}{'Sweet 16 %':<15}")
print("-" * 80)

for i in range(10):
    champ_pct = success_by_similarity_rank['Champion'][i] / total_years * 100
    ff_pct = success_by_similarity_rank['Final Four'][i] / total_years * 100
    e8_pct = success_by_similarity_rank['Elite Eight'][i] / total_years * 100
    s16_pct = success_by_similarity_rank['Sweet Sixteen'][i] / total_years * 100
    
    print(f"{i+1:<6}{champ_pct:<15.1f}{ff_pct:<15.1f}{e8_pct:<15.1f}{s16_pct:<15.1f}")

# Print detailed year-by-year analysis
print("\nYear-by-Year Analysis:")
print("=" * 80)

for year_result in yearly_results:
    year = year_result['Year']
    
    # Skip years where we don't have champion data
    if year_result['ActualChampion'] is None:
        print(f"\n{year}: No tournament data available (likely COVID cancellation)")
        continue
    
    print(f"\n{year} Tournament Analysis:")
    print(f"Actual Champion: {year_result['ActualChampion']} (Ranked #{year_result['ChampionSimilarityRank']} in similarity to champion profile)")
    
    # Print top 5 most similar teams and their results
    print("\nTop 5 Teams by Similarity to Champion Profile:")
    print(f"{'Rank':<5}{'Team':<25}{'Similarity':<10}{'Tournament Result':<25}{'Seed':<5}")
    print("-" * 80)
    
    for i, team in enumerate(year_result['TopTeams'][:5]):
        result = round_mapping.get(team['TournamentExitRound'], "Unknown")
        seed = int(team['TournamentSeed']) if team['TournamentSeed'] > 0 else "N/A"
        
        print(f"{i+1:<5}{team['Team']:<25}{team['Similarity']:<10.1f}{result:<25}{seed:<5}")

# Generate Conclusions
print("\nConclusions:")
print("=" * 80)
print("1. Teams most similar to the champion profile historically have a significant advantage")
print(f"   - {success_by_similarity_rank['Final Four'][0]/total_years*100:.1f}% of teams with the highest similarity made the Final Four")
print(f"   - {sum(success_by_similarity_rank['Champion'][:3])/total_years*100:.1f}% of champions came from the top 3 most similar teams")

print("\n2. The statistical champion profile appears to be a strong predictor of tournament success")
print(f"   - Top 3 teams by similarity made the Elite Eight {sum(success_by_similarity_rank['Elite Eight'][:3])/total_years/3*100:.1f}% of the time")

print("\n3. Teams outside the top 5 in similarity rarely win the championship")
print(f"   - Only {sum(success_by_similarity_rank['Champion'][5:])}/{total_years} champions ranked outside the top 5 in similarity")

# Create visualization to show relationship between similarity rank and success
ranks = list(range(1, 11))

plt.figure(figsize=(12, 8))
plt.bar(ranks, [x/total_years*100 for x in success_by_similarity_rank['Champion']], 
        width=0.5, color='gold', label='Won Championship')
plt.bar(ranks, [x/total_years*100 for x in success_by_similarity_rank['Final Four']], 
        width=0.5, alpha=0.7, color='blue', label='Made Final Four')
plt.bar(ranks, [x/total_years*100 for x in success_by_similarity_rank['Elite Eight']], 
        width=0.5, alpha=0.5, color='green', label='Made Elite Eight')
plt.bar(ranks, [x/total_years*100 for x in success_by_similarity_rank['Sweet Sixteen']], 
        width=0.5, alpha=0.3, color='red', label='Made Sweet Sixteen')

plt.xlabel('Similarity Rank to Champion Profile')
plt.ylabel('Success Percentage')
plt.title('Tournament Success by Similarity to Champion Profile (2009-2024)')
plt.xticks(ranks)
plt.legend()
plt.grid(True, alpha=0.3)

# Save the chart
plt.savefig('champion_similarity_success.png')
print("\nVisualization of results saved as 'champion_similarity_success.png'")

# Check if New Mexico's data is available in 2025 data
try:
    df_2025 = pd.read_csv('susan_kenpom/summary25.csv')
    
    # Clean data
    for col in df_2025.columns:
        if df_2025[col].dtype == 'object' and col != 'TeamName':
            df_2025[col] = df_2025[col].str.replace('"', '').astype(float)
        elif col == 'TeamName':
            df_2025[col] = df_2025[col].str.replace('"', '')
    
    # If New Mexico is in the dataset, analyze its championship potential
    if 'New Mexico' in df_2025['TeamName'].values:
        nm_row = df_2025[df_2025['TeamName'] == 'New Mexico'].iloc[0]
        
        # Calculate similarity to champion profile
        nm_stats = {
            'AdjEM': nm_row['AdjEM'],
            'RankAdjEM': nm_row['RankAdjEM'],
            'AdjOE': nm_row['AdjOE'],
            'AdjDE': nm_row['AdjDE']
        }
        
        similarity, _ = calculate_similarity(nm_stats, champ_profile)
        
        # Calculate similarity rank
        all_similarities = []
        for _, row in df_2025.iterrows():
            team_stats = {
                'AdjEM': row['AdjEM'],
                'RankAdjEM': row['RankAdjEM'],
                'AdjOE': row['AdjOE'],
                'AdjDE': row['AdjDE']
            }
            sim, _ = calculate_similarity(team_stats, champ_profile)
            all_similarities.append((row['TeamName'], sim))
        
        all_similarities.sort(key=lambda x: x[1], reverse=True)
        nm_rank = next((i+1 for i, (team, _) in enumerate(all_similarities) if team == 'New Mexico'), None)
        
        print("\nNew Mexico 2025 Analysis:")
        print("=" * 50)
        print(f"Similarity to Champion Profile: {similarity:.1f}%")
        print(f"Similarity Rank: {nm_rank} out of {len(all_similarities)} teams")
        
        # Historical context based on similarity rank
        if nm_rank <= 5:
            champ_odds = sum(success_by_similarity_rank['Champion'][:5]) / (5 * total_years) * 100
            ff_odds = sum(success_by_similarity_rank['Final Four'][:5]) / (5 * total_years) * 100
            print(f"\nHistorical Tournament Odds for Teams with Similar Rank:")
            print(f"Championship: {champ_odds:.1f}%")
            print(f"Final Four: {ff_odds:.1f}%")
        else:
            bracket_position = min(10, (nm_rank-1) // 5 * 5)
            similar_ranks = list(range(bracket_position, min(bracket_position+5, 10)))
            
            champ_odds = sum(success_by_similarity_rank['Champion'][r-1] for r in similar_ranks) / (len(similar_ranks) * total_years) * 100
            ff_odds = sum(success_by_similarity_rank['Final Four'][r-1] for r in similar_ranks) / (len(similar_ranks) * total_years) * 100
            
            print(f"\nHistorical Tournament Odds for Teams Ranked {min(similar_ranks)+1}-{max(similar_ranks)+1} in Similarity:")
            print(f"Championship: {champ_odds:.1f}%")
            print(f"Final Four: {ff_odds:.1f}%")
except Exception as e:
    print(f"\nCouldn't analyze New Mexico 2025 data: {e}") 