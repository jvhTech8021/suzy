import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load the 2025 season data
df_2025 = pd.read_csv('susan_kenpom/summary25.csv')

# Clean data by removing quotes if present
for col in df_2025.columns:
    if df_2025[col].dtype == 'object' and col != 'TeamName':
        df_2025[col] = df_2025[col].str.replace('"', '').astype(float)
    elif col == 'TeamName':
        df_2025[col] = df_2025[col].str.replace('"', '')

# Extract data for Idaho and Montana
team1 = df_2025[df_2025['TeamName'] == "Idaho"].iloc[0]
team2 = df_2025[df_2025['TeamName'] == "Montana"].iloc[0]

# Display team metrics
print("\nTeam Statistics:")
print("-" * 70)
print(f"{'Metric':<25} {'Idaho':<15} {'Montana':<15} {'Difference':<15}")
print("-" * 70)

# Define the key metrics to compare
key_metrics = [
    'AdjOE', 'RankAdjOE', 'AdjDE', 'RankAdjDE', 
    'AdjEM', 'RankAdjEM', 'AdjTempo', 'RankAdjTempo'
]

for metric in key_metrics:
    team1_val = team1[metric]
    team2_val = team2[metric]
    diff = team1_val - team2_val
    
    # Format based on whether it's a rank (lower is better) or a raw metric (higher is better)
    if 'Rank' in metric:
        better_team = "Idaho" if diff < 0 else "Montana"
        diff_formatted = f"{abs(diff):.1f} ({better_team})"
    else:
        better_team = "Idaho" if diff > 0 else "Montana"
        diff_formatted = f"{abs(diff):.1f} ({better_team})"
    
    print(f"{metric:<25} {team1_val:<15.1f} {team2_val:<15.1f} {diff_formatted}")

print("-" * 70)

# Simple prediction model using only the current season data
print("\nBuilding a simple prediction model based on current season metrics:")

# Our features will be the four main KenPom metrics
features = ['AdjOE', 'AdjDE', 'AdjTempo', 'AdjEM']

# 1. Simple point-based approach
team1_points = 0
team2_points = 0

print("\nApproach 1: Simple point-based comparison")
print("-" * 45)
for feature in features:
    if feature == 'AdjDE':  # For defense, lower is better
        if team1[feature] < team2[feature]:
            team1_points += 1
            winner = "Idaho"
        else:
            team2_points += 1
            winner = "Montana"
    else:  # For other metrics, higher is better
        if team1[feature] > team2[feature]:
            team1_points += 1
            winner = "Idaho"
        else:
            team2_points += 1
            winner = "Montana"
    
    print(f"{feature}: {winner} scores a point")

# Weight AdjEM more heavily as it's the overall efficiency metric
em_winner = "Idaho" if team1['AdjEM'] > team2['AdjEM'] else "Montana"
if em_winner == "Idaho":
    team1_points += 2
else:
    team2_points += 2
print(f"AdjEM (weighted x2): {em_winner} scores 2 points")

print(f"\nPoints for Idaho: {team1_points}")
print(f"Points for Montana: {team2_points}")

predicted_winner_1 = "Idaho" if team1_points > team2_points else "Montana"
print(f"Predicted winner (Approach 1): {predicted_winner_1}")

# Calculate win probability based on points difference
total_points = team1_points + team2_points
team1_win_prob_1 = team1_points / total_points
team2_win_prob_1 = team2_points / total_points

print(f"Idaho win probability: {team1_win_prob_1:.1%}")
print(f"Montana win probability: {team2_win_prob_1:.1%}")

# 2. Efficiency margin and home court approach
print("\nApproach 2: Efficiency Margin (AdjEM) with adjustments")
print("-" * 55)

# Get the efficiency margins
team1_em = team1['AdjEM']
team2_em = team2['AdjEM']
em_diff = abs(team1_em - team2_em)

print(f"Idaho AdjEM: {team1_em:.1f}")
print(f"Montana AdjEM: {team2_em:.1f}")
print(f"Raw difference: {em_diff:.1f}")

# Determine if a home court advantage should be applied
# Let's assume Montana has home court advantage (3.5 points)
home_court_advantage = -3.5  # Negative because we're assuming Montana is home
court_description = "Montana home court"

team1_adjusted_em = team1_em + home_court_advantage
print(f"Adjusted difference (including {court_description}): {abs(team1_adjusted_em - team2_em):.1f}")

# Predict winner based on adjusted efficiency margin
predicted_winner_2 = "Idaho" if team1_adjusted_em > team2_em else "Montana"
print(f"Predicted winner (Approach 2): {predicted_winner_2}")

# Calculate win probability using a sigmoid function on the efficiency margin difference
adjusted_diff = team1_adjusted_em - team2_em
team1_win_prob_2 = 1 / (1 + np.exp(-adjusted_diff/5))  # Divide by 5 to scale appropriately
team2_win_prob_2 = 1 - team1_win_prob_2

print(f"Idaho win probability: {team1_win_prob_2:.1%}")
print(f"Montana win probability: {team2_win_prob_2:.1%}")

# 3. Tempo-adjusted approach
print("\nApproach 3: Tempo and efficiency interaction")
print("-" * 45)

# Consider how teams perform at different tempos
team1_tempo = team1['AdjTempo']
team2_tempo = team2['AdjTempo']
print(f"Idaho tempo: {team1_tempo:.1f} (Rank: {team1['RankAdjTempo']:.0f})")
print(f"Montana tempo: {team2_tempo:.1f} (Rank: {team2['RankAdjTempo']:.0f})")

# Calculate expected pace of the game
expected_tempo = (team1_tempo + team2_tempo) / 2
print(f"Expected game tempo: {expected_tempo:.1f}")

# Analyze which team benefits from this tempo
tempo_advantage = "Neither team"
if abs(expected_tempo - team1_tempo) < abs(expected_tempo - team2_tempo):
    tempo_advantage = "Idaho"
else:
    tempo_advantage = "Montana"

print(f"Tempo advantage: {tempo_advantage}")

# Factor tempo into our prediction
tempo_factor = 0.05  # 5% weight for tempo advantage
base_prob = team1_win_prob_2  # Using our adjusted EM probability as the base

# Adjust probability based on tempo advantage
if tempo_advantage == "Idaho":
    team1_win_prob_3 = base_prob * (1 + tempo_factor)
elif tempo_advantage == "Montana":
    team1_win_prob_3 = base_prob * (1 - tempo_factor)
else:
    team1_win_prob_3 = base_prob

team1_win_prob_3 = min(max(team1_win_prob_3, 0.01), 0.99)  # Keep probability between 1% and 99%
team2_win_prob_3 = 1 - team1_win_prob_3

predicted_winner_3 = "Idaho" if team1_win_prob_3 > 0.5 else "Montana"
print(f"Predicted winner (Approach 3): {predicted_winner_3}")
print(f"Idaho win probability: {team1_win_prob_3:.1%}")
print(f"Montana win probability: {team2_win_prob_3:.1%}")

# 4. Defensive/Offensive strength approach
print("\nApproach 4: Offensive vs Defensive strength matchup")
print("-" * 55)

# Compare Montana's offense vs Idaho's defense
team2_off = team2['AdjOE']
team1_def = team1['AdjDE']
team2_offense_advantage = team2_off - team1_def
print(f"Montana offense ({team2_off:.1f}) vs Idaho defense ({team1_def:.1f})")
print(f"Montana's offensive advantage: {team2_offense_advantage:.1f}")

# Compare Idaho's offense vs Montana's defense
team1_off = team1['AdjOE']
team2_def = team2['AdjDE']
team1_offense_advantage = team1_off - team2_def
print(f"Idaho offense ({team1_off:.1f}) vs Montana defense ({team2_def:.1f})")
print(f"Idaho's offensive advantage: {team1_offense_advantage:.1f}")

# Predict winner based on offensive advantage difference
offense_advantage_diff = team1_offense_advantage - team2_offense_advantage
predicted_winner_4 = "Idaho" if offense_advantage_diff > 0 else "Montana"

print(f"Net offensive advantage: {'Idaho' if offense_advantage_diff > 0 else 'Montana'} by {abs(offense_advantage_diff):.1f}")
print(f"Predicted winner (Approach 4): {predicted_winner_4}")

# Calculate win probability
team1_win_prob_4 = 1 / (1 + np.exp(-offense_advantage_diff/10))  # Scale by 10 to moderate the effect
team2_win_prob_4 = 1 - team1_win_prob_4

print(f"Idaho win probability: {team1_win_prob_4:.1%}")
print(f"Montana win probability: {team2_win_prob_4:.1%}")

# Final consensus model
print("\n" + "=" * 50)
print("FINAL PREDICTION (Consensus of all approaches)")
print("=" * 50)

# Average the win probabilities from all approaches
team1_win_prob_final = (team1_win_prob_1 + team1_win_prob_2 + team1_win_prob_3 + team1_win_prob_4) / 4
team2_win_prob_final = 1 - team1_win_prob_final

final_winner = "Idaho" if team1_win_prob_final > 0.5 else "Montana"
print(f"\nConsensus winner: {final_winner}")
print(f"Idaho win probability: {team1_win_prob_final:.1%}")
print(f"Montana win probability: {team2_win_prob_final:.1%}")

# How many points we expect
expected_total_points = 140  # Typical college basketball total

# Calculate point spread based on efficiency margin difference
# In college basketball, a common rule is 1 point of efficiency ~ 1 point in spread
efficiency_diff = team2['AdjEM'] - team1['AdjEM']

# Temper the spread a bit for higher-ranked teams
if efficiency_diff > 0:
    point_spread = min(efficiency_diff * 1.5, 15)  # Cap at 15 points
else:
    point_spread = max(efficiency_diff * 1.5, -15)  # Cap at -15 points

# Calculate expected scores
team2_expected_points = (expected_total_points / 2) + (point_spread / 2)
team1_expected_points = (expected_total_points / 2) - (point_spread / 2)

# Round to whole numbers for a more realistic basketball score
team2_expected_points = round(team2_expected_points)
team1_expected_points = round(team1_expected_points)

print(f"\nExpected score:")
print(f"Idaho: {team1_expected_points}")
print(f"Montana: {team2_expected_points}")
print(f"Spread: {final_winner} by {abs(team1_expected_points - team2_expected_points)}")

print("\nKey metrics used in prediction:")
print("1. AdjOE - Adjusted Offensive Efficiency")
print("2. AdjDE - Adjusted Defensive Efficiency")
print("3. AdjEM - Adjusted Efficiency Margin") 
print("4. AdjTempo - Adjusted Tempo")
print("\nNote: This prediction includes a home court advantage for Montana.")
print("      Factors like injuries, recent form, and specific matchup history")
print("      are not fully accounted for.") 