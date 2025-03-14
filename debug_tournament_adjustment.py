import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path so we can import our modules
sys.path.append(os.getcwd())

from march_madness_predictor.models.game_predictor import GamePredictor

class DebugPredictor(GamePredictor):
    """A debug version of the predictor that logs details about tournament adjustments"""
    
    def predict_game(self, team1_name, team2_name, location='neutral'):
        # Get tournament data for the teams
        team1_tournament_data = self._get_tournament_prediction_data(team1_name)
        team2_tournament_data = self._get_tournament_prediction_data(team2_name)
        
        # Print tournament data
        print(f"\n--- Tournament Data for {team1_name} ---")
        for key, value in team1_tournament_data.items():
            if key in self.TOURNAMENT_SCALING_FACTORS or key == "seed" or key == "predicted_exit":
                print(f"{key}: {value}")
        
        print(f"\n--- Tournament Data for {team2_name} ---")
        for key, value in team2_tournament_data.items():
            if key in self.TOURNAMENT_SCALING_FACTORS or key == "seed" or key == "predicted_exit":
                print(f"{key}: {value}")
        
        # Print scaling factors
        print("\n--- Tournament Scaling Factors ---")
        for key, value in self.TOURNAMENT_SCALING_FACTORS.items():
            print(f"{key}: {value}")
        
        # Call the parent predict_game method
        result = super().predict_game(team1_name, team2_name, location)
        
        # Print result details
        print(f"\n--- Prediction Results ---")
        print(f"{team1_name} Expected Score: {result['team1']['predicted_score']:.2f}")
        print(f"{team2_name} Expected Score: {result['team2']['predicted_score']:.2f}")
        print(f"Spread ({team1_name} - {team2_name}): {result['spread']:.2f}")
        print(f"{team1_name} Tournament Adjustment: {result['tournament_adjustment_team1']:.2f}")
        print(f"{team2_name} Tournament Adjustment: {result['tournament_adjustment_team2']:.2f}")
        print(f"Net Tournament Adjustment: {result['tournament_adjustment']:.2f}")
        
        # Print score components
        team1_base = result['team1']['predicted_score'] - result['tournament_adjustment_team1']
        team2_base = result['team2']['predicted_score'] - result['tournament_adjustment_team2']
        print(f"\n--- Score Components ---")
        print(f"{team1_name} Base Score: {team1_base:.2f}")
        print(f"{team1_name} Tournament Adjustment: +{result['tournament_adjustment_team1']:.2f}")
        print(f"{team1_name} Final Score: {result['team1']['predicted_score']:.2f}")
        print(f"{team2_name} Base Score: {team2_base:.2f}")
        print(f"{team2_name} Tournament Adjustment: +{result['tournament_adjustment_team2']:.2f}")
        print(f"{team2_name} Final Score: {result['team2']['predicted_score']:.2f}")
        
        # Check detailed tournament adjustment
        if "tournament_adjustment_detail" in result["team1"]:
            print(f"\n--- Detailed {team1_name} Tournament Adjustment ---")
            team1_details = result["team1"]["tournament_adjustment_detail"]
            for round_key, detail in team1_details.items():
                print(f"{round_key}: {detail['percentage']:.1f}% × {detail['factor']:.3f} = {detail['points']:.2f} points")
        
        if "tournament_adjustment_detail" in result["team2"]:
            print(f"\n--- Detailed {team2_name} Tournament Adjustment ---")
            team2_details = result["team2"]["tournament_adjustment_detail"]
            for round_key, detail in team2_details.items():
                print(f"{round_key}: {detail['percentage']:.1f}% × {detail['factor']:.3f} = {detail['points']:.2f} points")
        
        return result

# Create the debug predictor and run prediction
debug_predictor = DebugPredictor()
result = debug_predictor.predict_game('Duke', 'Georgia Tech')

# Now let's manually calculate what we'd expect for Duke based on tournament percentages
print("\n--- Manual Tournament Adjustment Calculation for Duke ---")
duke_data = debug_predictor._get_tournament_prediction_data('Duke')
factors = debug_predictor.TOURNAMENT_SCALING_FACTORS

total_adjustment = 0
for key, factor in factors.items():
    if key in duke_data and duke_data[key] is not None:
        adjustment = duke_data[key] * factor
        total_adjustment += adjustment
        print(f"{key}: {duke_data[key]:.1f}% × {factor:.3f} = {adjustment:.2f} points")

print(f"Total Expected Adjustment: {total_adjustment:.2f} points")
print(f"Actual Adjustment in Model: {result['tournament_adjustment_team1']:.2f} points")
print(f"Difference: {total_adjustment - result['tournament_adjustment_team1']:.2f} points") 