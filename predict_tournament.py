import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("NCAA March Madness Tournament Prediction for 2025\n")
print("=" * 60)

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

# Prepare the training data for prediction
# Feature selection: Use the key KenPom metrics
features = ['AdjOE', 'RankAdjOE', 'AdjDE', 'RankAdjDE', 'AdjEM', 'RankAdjEM', 'AdjTempo', 'RankAdjTempo']

# Training data
X_train = all_data[all_data['TournamentExitRound'] > 0][features]
y_train = all_data[all_data['TournamentExitRound'] > 0]['TournamentExitRound']

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train a random forest model to predict how far teams will go
print("\nTraining tournament performance prediction model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
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

# Sort by championship probability
pred_df = pred_df.sort_values('Championship_Score', ascending=False)

# Display the top 25 teams most likely to make a tournament run
print("\nTop 25 Teams Most Likely to Make a Tournament Run in 2025:")
print("=" * 100)
print(f"{'Rank':<5}{'Team':<25}{'Adj EM':<10}{'Natl Rank':<10}{'Off Eff':<10}{'Def Eff':<10}{'Champion %':<15}{'Final Four %':<15}")
print("-" * 100)

for i, (team, row) in enumerate(pred_df.head(25).iterrows(), 1):
    champion_pct = row.get('Prob_Round_7', 0) * 100
    final_four_pct = row.get('Prob_Round_5', 0) * 100 + row.get('Prob_Round_6', 0) * 100 + row.get('Prob_Round_7', 0) * 100
    
    print(f"{i:<5}{team:<25}{row['AdjEM']:<10.2f}{row['RankAdjEM']:<10.0f}{row['AdjOE']:<10.1f}{row['AdjDE']:<10.1f}{champion_pct:<15.1f}{final_four_pct:<15.1f}")

# Show predicted bracket performance for past champions
print("\nRecent Champions and Their Predicted Performance:")
print("=" * 80)

recent_champions = []
for year in range(2021, 2025):
    try:
        champion_df = pd.read_csv(f'susan_kenpom/processed_{year}.csv')
        champion = champion_df[champion_df['TournamentExitRound'] == 7]['TeamName'].values[0]
        metrics = champion_df[champion_df['TournamentExitRound'] == 7][features].values[0]
        
        # Scale the champion's metrics using the same scaler
        scaled_metrics = scaler.transform([metrics])
        champion_pred = model.predict_proba(scaled_metrics)
        
        # Calculate probabilities for champion
        champ_probs = {}
        for i in range(1, 8):
            if i in model.classes_:
                idx = np.where(model.classes_ == i)[0][0]
                champ_probs[f'Prob_Round_{i}'] = champion_pred[0, idx]
            else:
                champ_probs[f'Prob_Round_{i}'] = 0
        
        champion_pct = champ_probs.get('Prob_Round_7', 0) * 100
        final_four_pct = (champ_probs.get('Prob_Round_5', 0) + champ_probs.get('Prob_Round_6', 0) + 
                         champ_probs.get('Prob_Round_7', 0)) * 100
                         
        recent_champions.append({
            'Year': year,
            'Champion': champion,
            'AdjEM': metrics[4],
            'RankAdjEM': metrics[5],
            'Champion_Probability': champion_pct,
            'Final_Four_Probability': final_four_pct
        })
        
    except Exception as e:
        pass

# Display recent champions
if recent_champions:
    print(f"{'Year':<10}{'Champion':<25}{'Adj EM':<10}{'Natl Rank':<10}{'Champion %':<15}{'Final Four %':<15}")
    print("-" * 80)
    for champ in recent_champions:
        print(f"{champ['Year']:<10}{champ['Champion']:<25}{champ['AdjEM']:<10.2f}{champ['RankAdjEM']:<10.0f}{champ['Champion_Probability']:<15.1f}{champ['Final_Four_Probability']:<15.1f}")

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

print("\nImportant Notes:")
print("1. These predictions are based on historical KenPom data and tournament results.")
print("2. Tournament success involves many factors not captured by statistics (matchups, injuries, etc.)")
print("3. The NCAA tournament is known for upsets and unpredictability.")
print("\nMethod: Random Forest classifier trained on historical KenPom metrics and tournament performance") 