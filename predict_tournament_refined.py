import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("NCAA March Madness Tournament Prediction for 2025 (Refined Model)\n")
print("=" * 70)

# Load the 2025 season data (current season)
df_2025 = pd.read_csv('susan_kenpom/summary25.csv')

# Clean data by removing quotes if present
for col in df_2025.columns:
    if df_2025[col].dtype == 'object' and col != 'TeamName':
        df_2025[col] = df_2025[col].str.replace('"', '').astype(float)
    elif col == 'TeamName':
        df_2025[col] = df_2025[col].str.replace('"', '')

# Load historical data
historical_data = []
available_years = []

for year in range(2009, 2025):
    try:
        # Read and clean the data
        file_path = f'susan_kenpom/processed_{year}.csv'
        df = pd.read_csv(file_path)
        
        # Add this file to our historical dataset
        historical_data.append(df)
        available_years.append(year)
        
        print(f"Loaded data for {year} tournament")
    except:
        print(f"Could not load data for {year}")

if not historical_data:
    print("No historical data found. Cannot make predictions.")
    exit()

# Combine all historical data
all_data = pd.concat(historical_data)

# Map the tournament exit rounds to their actual meanings
round_mapping = {
    0: "Did not make tournament",
    1: "First Round (Round of 64)",
    2: "Second Round (Round of 32)",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four (lost semifinal)",
    6: "Championship Game (runner-up)",
    7: "National Champion"
}

# Check distribution of tournament rounds
print("\nDistribution of tournament results in historical data:")
round_counts = all_data[all_data['TournamentExitRound'] > 0]['TournamentExitRound'].value_counts().sort_index()
for round_num, count in round_counts.items():
    print(f"{round_mapping[round_num]}: {count} teams")

# Analyze past champions
print("\nPast NCAA Champions (in our dataset):")
print("=" * 60)
print(f"{'Year':<10}{'Champion':<25}{'Adj EM':<10}{'Natl Rank':<10}{'Off Eff':<10}{'Def Eff':<10}")
print("-" * 60)

champions_stats = []
for year in available_years:
    year_data = all_data[all_data['Season'] == year]
    champion = year_data[year_data['TournamentExitRound'] == 7]
    
    if not champion.empty:
        champion_row = champion.iloc[0]
        print(f"{year:<10}{champion_row['TeamName']:<25}{champion_row['AdjEM']:<10.2f}{champion_row['RankAdjEM']:<10.0f}{champion_row['AdjOE']:<10.1f}{champion_row['AdjDE']:<10.1f}")
        champions_stats.append({
            'AdjEM': champion_row['AdjEM'],
            'RankAdjEM': champion_row['RankAdjEM'],
            'AdjOE': champion_row['AdjOE'],
            'AdjDE': champion_row['AdjDE']
        })

# Calculate average champion stats
avg_champion = {
    'AdjEM': np.mean([c['AdjEM'] for c in champions_stats]),
    'RankAdjEM': np.mean([c['RankAdjEM'] for c in champions_stats]),
    'AdjOE': np.mean([c['AdjOE'] for c in champions_stats]),
    'AdjDE': np.mean([c['AdjDE'] for c in champions_stats])
}

print("\nAverage Champion Profile:")
print(f"Adjusted Efficiency Margin (AdjEM): {avg_champion['AdjEM']:.2f}")
print(f"National Ranking: {avg_champion['RankAdjEM']:.1f}")
print(f"Offensive Efficiency: {avg_champion['AdjOE']:.1f}")
print(f"Defensive Efficiency: {avg_champion['AdjDE']:.1f}")

# Prepare the training data for prediction
# Feature selection: Use the key KenPom metrics with emphasis on AdjEM
features = ['AdjOE', 'RankAdjOE', 'AdjDE', 'RankAdjDE', 'AdjEM', 'RankAdjEM', 'AdjTempo', 'RankAdjTempo']

# Add engineered features - SQRT of AdjEM to reduce impact of extremely high values
all_data['AdjEM_SQRT'] = np.sqrt(np.abs(all_data['AdjEM'])) * np.sign(all_data['AdjEM'])
# Add AdjEM^2 to increase the weight of this important metric
all_data['AdjEM_SQ'] = all_data['AdjEM'] ** 2

# Add these engineered features to our feature list
features += ['AdjEM_SQRT', 'AdjEM_SQ']

# Feature engineering for current season
df_2025['AdjEM_SQRT'] = np.sqrt(np.abs(df_2025['AdjEM'])) * np.sign(df_2025['AdjEM'])
df_2025['AdjEM_SQ'] = df_2025['AdjEM'] ** 2

# Training data - only use teams that made the tournament
X_train = all_data[all_data['TournamentExitRound'] > 0][features]
y_train = all_data[all_data['TournamentExitRound'] > 0]['TournamentExitRound']

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train a random forest model to predict how far teams will go
print("\nTraining tournament performance prediction model...")
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# Feature importance
feature_importances = model.feature_importances_
print("\nFeature Importance for Tournament Success:")
for feature, importance in zip(features, feature_importances):
    print(f"{feature}: {importance:.4f}")

# Make predictions for 2025 teams
X_predict = df_2025[features]
X_predict_scaled = scaler.transform(X_predict)
predictions = model.predict_proba(X_predict_scaled)

# Create a prediction dataframe
pred_df = pd.DataFrame(index=df_2025['TeamName'])

# Add predicted probabilities for each round
for i in range(1, 8):
    if i in model.classes_:
        idx = np.where(model.classes_ == i)[0][0]
        pred_df[f'Prob_Round_{i}'] = predictions[:, idx]
    else:
        pred_df[f'Prob_Round_{i}'] = 0

# Calculate expected tournament exit round
pred_df['Expected_Round'] = 0
for i in range(1, 8):
    pred_df['Expected_Round'] += i * pred_df[f'Prob_Round_{i}']

# Manual adjustment based on historical champion profile
# Create a champion profile similarity score
for idx, team in enumerate(df_2025['TeamName']):
    team_em = df_2025.loc[df_2025['TeamName'] == team, 'AdjEM'].values[0]
    team_rank = df_2025.loc[df_2025['TeamName'] == team, 'RankAdjEM'].values[0]
    team_oe = df_2025.loc[df_2025['TeamName'] == team, 'AdjOE'].values[0]
    team_de = df_2025.loc[df_2025['TeamName'] == team, 'AdjDE'].values[0]
    
    # Calculate z-scores compared to champion averages (how many std devs from champion mean)
    em_z = (team_em - avg_champion['AdjEM']) / np.std([c['AdjEM'] for c in champions_stats])
    rank_z = (team_rank - avg_champion['RankAdjEM']) / np.std([c['RankAdjEM'] for c in champions_stats])
    oe_z = (team_oe - avg_champion['AdjOE']) / np.std([c['AdjOE'] for c in champions_stats])
    de_z = (team_de - avg_champion['AdjDE']) / np.std([c['AdjDE'] for c in champions_stats])
    
    # For rankings, lower is better so invert the z-score
    rank_z = -rank_z
    
    # Overall similarity to champion profile (negative is worse, positive is better)
    # Note: DE is better when lower, so we invert that z-score
    champion_similarity = (1.5 * em_z + 0.5 * rank_z + 0.7 * oe_z + 0.7 * -de_z) / 3.4
    
    # Adjust champion probability based on similarity score
    # Scale similarity to a reasonable adjustment factor 
    adjustment = np.clip(champion_similarity * 0.1, -0.05, 0.15)
    
    # Apply adjustment
    if 'Prob_Round_7' in pred_df.columns:
        pred_df.loc[team, 'Prob_Round_7'] = np.clip(pred_df.loc[team, 'Prob_Round_7'] + adjustment, 0, 1)
    
    # Store similarity score
    pred_df.loc[team, 'Champion_Similarity'] = champion_similarity

# Calculate championship probability score
# This weights deeper runs more heavily
pred_df['Championship_Score'] = (
    pred_df['Prob_Round_4'] * 0.05 +  # Elite 8
    pred_df['Prob_Round_5'] * 0.15 +  # Final Four
    pred_df['Prob_Round_6'] * 0.3 +   # Runner-up
    pred_df['Prob_Round_7'] * 0.5     # Champion
)

# Add key metrics for display
for metric in ['AdjOE', 'AdjDE', 'AdjEM', 'RankAdjEM']:
    pred_df[metric] = df_2025[metric].values

# Low ranked teams with negative AdjEM are unlikely to make deep tournament runs
# Apply a logical filter to reduce unrealistic predictions
for team in pred_df.index:
    team_em = pred_df.loc[team, 'AdjEM']
    team_rank = pred_df.loc[team, 'RankAdjEM']
    
    # If a team has a negative efficiency margin or rank > 50, reduce Final Four and beyond chances
    if team_em < 0 or team_rank > 50:
        reduction_factor = 0.1 if team_em < 0 else 0.5
        
        # Progressively reduce chances for deeper runs
        for round_num in range(5, 8):
            if f'Prob_Round_{round_num}' in pred_df.columns:
                pred_df.loc[team, f'Prob_Round_{round_num}'] *= reduction_factor
                
        # Update championship score
        pred_df.loc[team, 'Championship_Score'] = (
            pred_df.loc[team, 'Prob_Round_4'] * 0.05 +
            pred_df.loc[team, 'Prob_Round_5'] * 0.15 +
            pred_df.loc[team, 'Prob_Round_6'] * 0.3 +
            pred_df.loc[team, 'Prob_Round_7'] * 0.5
        )

# Sort by championship probability score
pred_df = pred_df.sort_values('Championship_Score', ascending=False)

# Display the top 25 teams most likely to make a tournament run
print("\nTop 25 Teams Most Likely to Make a Tournament Run in 2025:")
print("=" * 100)
print(f"{'Rank':<5}{'Team':<25}{'Adj EM':<10}{'Natl Rank':<10}{'Off Eff':<10}{'Def Eff':<10}{'Champion %':<15}{'Final Four %':<15}")
print("-" * 100)

for i, (team, row) in enumerate(pred_df.head(25).iterrows(), 1):
    champion_pct = row.get('Prob_Round_7', 0) * 100
    final_four_pct = (row.get('Prob_Round_5', 0) + row.get('Prob_Round_6', 0) + row.get('Prob_Round_7', 0)) * 100
    
    print(f"{i:<5}{team:<25}{row['AdjEM']:<10.2f}{row['RankAdjEM']:<10.0f}{row['AdjOE']:<10.1f}{row['AdjDE']:<10.1f}{champion_pct:<15.1f}{final_four_pct:<15.1f}")

# Display top 10 teams likely to be national champions
print("\nTop 10 Predicted National Champions for 2025:")
print("=" * 60)
champion_df = pred_df.sort_values('Prob_Round_7', ascending=False).head(10)
for i, (team, row) in enumerate(champion_df.iterrows(), 1):
    print(f"{i:<3}{team:<25}{row['Prob_Round_7']*100:<10.1f}% chance")

# Display top 10 teams likely to make the Final Four
print("\nTop 10 Predicted Final Four Teams for 2025:")
print("=" * 60)
final_four_probs = pred_df['Prob_Round_5'] + pred_df['Prob_Round_6'] + pred_df['Prob_Round_7']
final_four_df = pd.DataFrame({'Team': pred_df.index, 'Final_Four_Prob': final_four_probs})
final_four_df = final_four_df.sort_values('Final_Four_Prob', ascending=False).head(10)
for i, row in enumerate(final_four_df.itertuples(), 1):
    print(f"{i:<3}{row.Team:<25}{row.Final_Four_Prob*100:<10.1f}% chance")

# Cinderella candidates - teams outside the top 25 that have a solid chance
print("\nPotential 'Cinderella' Teams (Ranked 25-75 with solid tournament chances):")
print("=" * 80)
cinderella_df = df_2025[(df_2025['RankAdjEM'] > 25) & (df_2025['RankAdjEM'] <= 75)].copy()
cinderella_teams = []

for team in cinderella_df['TeamName']:
    team_row = pred_df.loc[team]
    sweet16_prob = team_row.get('Prob_Round_3', 0)
    elite8_prob = team_row.get('Prob_Round_4', 0)
    final4_prob = team_row.get('Prob_Round_5', 0) + team_row.get('Prob_Round_6', 0) + team_row.get('Prob_Round_7', 0)
    
    # Compute a "cinderella score"
    cinderella_score = sweet16_prob * 0.2 + elite8_prob * 0.3 + final4_prob * 0.5
    
    if cinderella_score > 0.1:  # Only include teams with some meaningful chance
        cinderella_teams.append({
            'Team': team,
            'AdjEM': cinderella_df.loc[cinderella_df['TeamName'] == team, 'AdjEM'].values[0],
            'Rank': cinderella_df.loc[cinderella_df['TeamName'] == team, 'RankAdjEM'].values[0],
            'Sweet16': sweet16_prob * 100,
            'Elite8': elite8_prob * 100,
            'FinalFour': final4_prob * 100,
            'Score': cinderella_score
        })

# Sort cinderella teams by score
cinderella_teams = sorted(cinderella_teams, key=lambda x: x['Score'], reverse=True)

# Display top 10 cinderella teams
print(f"{'Rank':<5}{'Team':<25}{'Adj EM':<10}{'Natl Rank':<10}{'Sweet 16%':<12}{'Elite 8%':<12}{'Final Four%':<12}")
print("-" * 80)
for i, team in enumerate(cinderella_teams[:10], 1):
    print(f"{i:<5}{team['Team']:<25}{team['AdjEM']:<10.2f}{team['Rank']:<10.0f}{team['Sweet16']:<12.1f}{team['Elite8']:<12.1f}{team['FinalFour']:<12.1f}")

print("\nImportant Notes:")
print("1. This refined model weights Adjusted Efficiency Margin (AdjEM) heavily.")
print("2. Historical champion profiles were analyzed to create more realistic predictions.")
print("3. Teams with metrics far from typical champion profiles were penalized.")
print("4. Final Four and championship predictions especially favor top-ranked teams.")
print("5. Remember that tournament success involves many factors beyond statistics.")
print("\nMethod: Random Forest with engineered features + champion profile similarity adjustment") 