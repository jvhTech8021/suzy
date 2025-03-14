import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path so we can import our modules
sys.path.append(os.getcwd())

from march_madness_predictor.models.game_predictor import GamePredictor

class TestPredictor(GamePredictor):
    """A test subclass of GamePredictor with mocked data for testing"""
    
    def __init__(self):
        super().__init__()
        
        # Mock the tournament data
        self.tournament_data = pd.DataFrame({
            'TeamName': ['Alabama', 'Duke'],
            'Seed': [1, 2],
            'ChampionshipPct': [12.5, 7.8],
            'FinalFourPct': [35.2, 22.1],
            'EliteEightPct': [55.0, 42.0],
            'SweetSixteenPct': [75.0, 65.0],
            'Round32Pct': [90.0, 85.0],
            'PredictedExit': ['Championship', 'Elite 8']
        })
        
        # Mock team data
        self.team_data = pd.DataFrame({
            'TeamName': ['Alabama', 'Duke'],
            'AdjEM': [30.5, 28.2],
            'AdjO': [118.5, 116.2],
            'AdjD': [88.0, 88.0],
            'AdjT': [70.1, 67.5],
            'Luck': [0.025, 0.015],
            'Strength': [11.2, 10.8],
            'AdjOE': [118.5, 116.2],
            'AdjDE': [88.0, 88.0],
            'AdjTempo': [70.1, 67.5],
            'FThrows': [75.2, 72.1]
        })
        
    def _get_team_data(self, team_name):
        """Override to return mock data"""
        return self.team_data[self.team_data['TeamName'] == team_name].iloc[0]
    
    def _get_tournament_prediction_data(self, team_name):
        """Override to return mock tournament data"""
        row = self.tournament_data[self.tournament_data['TeamName'] == team_name].iloc[0]
        return {
            "seed": row['Seed'],
            "championship_pct": row['ChampionshipPct'],
            "final_four_pct": row['FinalFourPct'],
            "elite_eight_pct": row['EliteEightPct'],
            "sweet_sixteen_pct": row['SweetSixteenPct'],
            "round_32_pct": row['Round32Pct'],
            "predicted_exit": row['PredictedExit']
        }
    
    def _get_historical_matchups(self, team1_name, team2_name):
        """Override to return mock historical data"""
        return {"total_matchups": 0}

# Initialize the predictor
predictor = TestPredictor()

# Run prediction with test teams
result = predictor.predict_game('Alabama', 'Duke')

# Print the relevant details
print("\n--- TEAM SCORES ---")
print(f"Alabama score: {result['team1']['predicted_score']:.2f}")
print(f"Duke score: {result['team2']['predicted_score']:.2f}")

print("\n--- TOURNAMENT ADJUSTMENTS ---")
print(f"Alabama total adjustment: {result['tournament_adjustment_team1']:.2f}")
print(f"Duke total adjustment: {result['tournament_adjustment_team2']:.2f}")
print(f"Net tournament adjustment: {result['tournament_adjustment']:.2f}")

# Check the tournament adjustment detail structure for UI
print("\n--- TOURNAMENT ADJUSTMENT STRUCTURE CHECK ---")
print("team1['tournament_adjustment_detail'] exists:", 'tournament_adjustment_detail' in result['team1'])
print("team2['tournament_adjustment_detail'] exists:", 'tournament_adjustment_detail' in result['team2'])

# Display detailed tournament adjustment breakdown for Alabama
print("\n--- DETAILED ADJUSTMENT BREAKDOWN (ALABAMA) ---")
if 'tournament_adjustment_detail' in result['team1']:
    details = result['team1']['tournament_adjustment_detail']
    round_names = {
        "championship_pct": "Championship",
        "final_four_pct": "Final Four",
        "elite_eight_pct": "Elite Eight",
        "sweet_sixteen_pct": "Sweet Sixteen",
        "round_32_pct": "Round of 32"
    }
    
    for round_key, round_name in round_names.items():
        if round_key in details:
            detail = details[round_key]
            print(f"{round_name}: {detail['percentage']:.1f}% × {detail['factor']:.2f} = {detail['points']:.2f} points")
else:
    print("No detailed adjustment data available")

# Display detailed tournament adjustment breakdown for Duke
print("\n--- DETAILED ADJUSTMENT BREAKDOWN (DUKE) ---")
if 'tournament_adjustment_detail' in result['team2']:
    details = result['team2']['tournament_adjustment_detail']
    for round_key, round_name in round_names.items():
        if round_key in details:
            detail = details[round_key]
            print(f"{round_name}: {detail['percentage']:.1f}% × {detail['factor']:.2f} = {detail['points']:.2f} points")
else:
    print("No detailed adjustment data available")

print("\n--- SPREAD CALCULATION ---")
print(f"Spread: {result['spread']:.2f} points")
print(f"Predicted winner: {'Alabama' if result['spread'] > 0 else 'Duke'}") 