import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path so we can import our modules
sys.path.append(os.getcwd())

from march_madness_predictor.models.game_predictor import GamePredictor

class TestScalingPredictor(GamePredictor):
    """A test class to check tournament adjustments using class properties"""
    
    def __init__(self):
        super().__init__()
        
        # Mock the tournament data
        self.tournament_data = pd.DataFrame({
            'TeamName': ['Alabama', 'Duke'],
            'Seed': [1, 2],
            'ChampionshipPct': [20.0, 10.0],  # Simple values for easy checking
            'FinalFourPct': [40.0, 30.0],
            'EliteEightPct': [60.0, 50.0],
            'SweetSixteenPct': [80.0, 70.0],
            'Round32Pct': [90.0, 85.0],
            'PredictedExit': ['Championship', 'Elite 8']
        })
        
        # Mock team data with identical offensive/defensive efficiencies
        # This ensures any score difference is purely from tournament adjustments
        self.team_data = pd.DataFrame({
            'TeamName': ['Alabama', 'Duke'],
            'AdjEM': [20.0, 20.0],
            'AdjO': [110.0, 110.0],
            'AdjD': [90.0, 90.0],
            'AdjT': [70.0, 70.0],
            'AdjOE': [110.0, 110.0],
            'AdjDE': [90.0, 90.0],
            'AdjTempo': [70.0, 70.0]
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
            "predicted_exit": row['PredictedExit'],
            "has_exit_round_data": True,
            "has_champion_profile_data": True
        }
    
    def _get_historical_matchups(self, team1_name, team2_name):
        """Override to return empty historical data"""
        return {"total_matchups": 0, "team1_wins": 0, "team2_wins": 0, "avg_margin": 0}

# Run tests with different scaling factors
print("\n--- TEST WITH DEFAULT SCALING FACTORS ---")
default_predictor = TestScalingPredictor()
default_result = default_predictor.predict_game('Alabama', 'Duke')
print(f"Team1 score: {default_result['team1']['predicted_score']:.2f}")
print(f"Team2 score: {default_result['team2']['predicted_score']:.2f}")
print(f"Tournament adjustment team1: {default_result['tournament_adjustment_team1']:.2f}")
print(f"Tournament adjustment team2: {default_result['tournament_adjustment_team2']:.2f}")
print(f"Net tournament adjustment: {default_result['tournament_adjustment']:.2f}")
print(f"Spread: {default_result['spread']:.2f}")

print("\n--- TEST WITH MODIFIED SCALING FACTORS ---")
modified_predictor = TestScalingPredictor()
# Double the championship scaling factor
modified_predictor.TOURNAMENT_SCALING_FACTORS = {
    "championship_pct": 0.15,   # Doubled from 0.075
    "final_four_pct": 0.05,
    "elite_eight_pct": 0.04,
    "sweet_sixteen_pct": 0.025,
    "round_32_pct": 0.0125
}
modified_result = modified_predictor.predict_game('Alabama', 'Duke')
print(f"Team1 score: {modified_result['team1']['predicted_score']:.2f}")
print(f"Team2 score: {modified_result['team2']['predicted_score']:.2f}")
print(f"Tournament adjustment team1: {modified_result['tournament_adjustment_team1']:.2f}")
print(f"Tournament adjustment team2: {modified_result['tournament_adjustment_team2']:.2f}")
print(f"Net tournament adjustment: {modified_result['tournament_adjustment']:.2f}")
print(f"Spread: {modified_result['spread']:.2f}")

print("\n--- COMPARISON ---")
print(f"Default spread: {default_result['spread']:.2f}")
print(f"Modified spread: {modified_result['spread']:.2f}")
print(f"Difference: {modified_result['spread'] - default_result['spread']:.2f}")

# Also check if the adjustment is visible in the UI display
print("\n--- UI REPRESENTATION ---")
# Calculate what would be shown in the UI
if default_result['team1']['predicted_score'] > default_result['team2']['predicted_score']:
    default_ui_spread = abs(round(default_result['spread'], 1))
    default_ui_text = f"Alabama by {default_ui_spread}"
else:
    default_ui_spread = abs(round(default_result['spread'], 1))
    default_ui_text = f"Duke by {default_ui_spread}"

if modified_result['team1']['predicted_score'] > modified_result['team2']['predicted_score']:
    modified_ui_spread = abs(round(modified_result['spread'], 1))
    modified_ui_text = f"Alabama by {modified_ui_spread}"
else:
    modified_ui_spread = abs(round(modified_result['spread'], 1))
    modified_ui_text = f"Duke by {modified_ui_spread}"

print(f"Default UI display: Predicted Spread: {default_ui_text}")
print(f"Modified UI display: Predicted Spread: {modified_ui_text}") 