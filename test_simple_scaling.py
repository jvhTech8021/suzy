import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path so we can import our modules
sys.path.append(os.getcwd())

from march_madness_predictor.models.game_predictor import GamePredictor

class SimpleTestPredictor(GamePredictor):
    """A simple test class to check tournament adjustments"""
    
    def __init__(self):
        super().__init__()
        
        # Mock the tournament data
        self.tournament_data = pd.DataFrame({
            'TeamName': ['Alabama', 'Duke'],
            'Seed': [1, 2],
            'ChampionshipPct': [20.0, 10.0],  # Keep simple values for easy checking
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

def test_prediction_with_scaling_factor(scaling_factor):
    """Run a test prediction with a specified championship scaling factor"""
    # Create a new predictor instance
    predictor = SimpleTestPredictor()
    
    # Manually override the scaling factor in the predict_game method
    original_predict_game = predictor.predict_game
    
    def modified_predict_game(team1_name, team2_name, location='neutral'):
        # Get tournament data for the teams
        team1_tournament_data = predictor._get_tournament_prediction_data(team1_name)
        team2_tournament_data = predictor._get_tournament_prediction_data(team2_name)
        
        # Use custom scaling factors
        custom_scaling_factors = {
            "championship_pct": scaling_factor,  # Modified value
            "final_four_pct": 0.05,
            "elite_eight_pct": 0.04,
            "sweet_sixteen_pct": 0.025,
            "round_32_pct": 0.0125
        }
        
        # Calculate the expected adjustments based on our data and scaling
        expected_team1_adj = (
            team1_tournament_data["championship_pct"] * custom_scaling_factors["championship_pct"] +
            team1_tournament_data["final_four_pct"] * custom_scaling_factors["final_four_pct"] +
            team1_tournament_data["elite_eight_pct"] * custom_scaling_factors["elite_eight_pct"] +
            team1_tournament_data["sweet_sixteen_pct"] * custom_scaling_factors["sweet_sixteen_pct"] +
            team1_tournament_data["round_32_pct"] * custom_scaling_factors["round_32_pct"]
        )
        
        expected_team2_adj = (
            team2_tournament_data["championship_pct"] * custom_scaling_factors["championship_pct"] +
            team2_tournament_data["final_four_pct"] * custom_scaling_factors["final_four_pct"] +
            team2_tournament_data["elite_eight_pct"] * custom_scaling_factors["elite_eight_pct"] +
            team2_tournament_data["sweet_sixteen_pct"] * custom_scaling_factors["sweet_sixteen_pct"] +
            team2_tournament_data["round_32_pct"] * custom_scaling_factors["round_32_pct"]
        )
        
        # Call original predict_game
        result = original_predict_game(team1_name, team2_name, location)
        
        # Print diagnostics
        print(f"Expected Team1 Adjustment: {expected_team1_adj:.2f}")
        print(f"Actual Team1 Adjustment: {result['tournament_adjustment_team1']:.2f}")
        print(f"Expected Team2 Adjustment: {expected_team2_adj:.2f}")
        print(f"Actual Team2 Adjustment: {result['tournament_adjustment_team2']:.2f}")
        print(f"Team1 Score: {result['team1']['predicted_score']:.2f}")
        print(f"Team2 Score: {result['team2']['predicted_score']:.2f}")
        print(f"Spread: {result['spread']:.2f}")
        
        return result
    
    # Temporarily replace predict_game with our modified version
    predictor.predict_game = modified_predict_game
    
    # Run the prediction
    result = predictor.predict_game('Alabama', 'Duke')
    
    # Restore original predict_game
    predictor.predict_game = original_predict_game
    
    return result

# Run tests with different scaling factors
print("\n----- TEST WITH DEFAULT SCALING FACTOR (0.075) -----")
default_result = test_prediction_with_scaling_factor(0.075)

print("\n----- TEST WITH DOUBLED SCALING FACTOR (0.15) -----")
doubled_result = test_prediction_with_scaling_factor(0.15)

print("\n----- COMPARISON -----")
default_spread = default_result['spread']
doubled_spread = doubled_result['spread']
print(f"Default Factor Spread: {default_spread:.2f}")
print(f"Doubled Factor Spread: {doubled_spread:.2f}")
print(f"Difference: {doubled_spread - default_spread:.2f}")

# Also check if the adjustment is visible in the UI display
print("\n----- UI REPRESENTATION -----")
# Calculate what would be shown in the UI
if default_result['team1']['predicted_score'] > default_result['team2']['predicted_score']:
    default_ui_spread = abs(round(default_result['spread'], 1))
    default_ui_text = f"Alabama by {default_ui_spread}"
else:
    default_ui_spread = abs(round(default_result['spread'], 1))
    default_ui_text = f"Duke by {default_ui_spread}"

if doubled_result['team1']['predicted_score'] > doubled_result['team2']['predicted_score']:
    doubled_ui_spread = abs(round(doubled_result['spread'], 1))
    doubled_ui_text = f"Alabama by {doubled_ui_spread}"
else:
    doubled_ui_spread = abs(round(doubled_result['spread'], 1))
    doubled_ui_text = f"Duke by {doubled_ui_spread}"

print(f"Default UI display: Predicted Spread: {default_ui_text}")
print(f"Doubled UI display: Predicted Spread: {doubled_ui_text}") 