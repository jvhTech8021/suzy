import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path so we can import our modules
sys.path.append(os.getcwd())

from march_madness_predictor.models.game_predictor import GamePredictor

class ManualDataPredictor(GamePredictor):
    """A predictor that allows manually setting tournament data"""
    
    def __init__(self):
        super().__init__()
        
        # Create tournament data with the values from the UI
        self.manual_tournament_data = pd.DataFrame({
            'TeamName': ['Duke', 'Georgia Tech'],
            'Seed': [1, 16],
            'ChampionshipPct': [15.0, 0.0],
            'FinalFourPct': [40.0, 0.0],
            'EliteEightPct': [72.0, 0.0],
            'SweetSixteenPct': [100.0, 0.0],
            'Round32Pct': [100.0, 0.0],
            'PredictedExit': ['Final Four', 'Round of 64']
        })
        
        # Set this as our exit_round_data
        self.exit_round_data = self.manual_tournament_data
    
    def _get_tournament_prediction_data(self, team_name):
        """Override to use our manual data"""
        result = super()._get_tournament_prediction_data(team_name)
        
        # If using our manual data for the teams we care about
        if team_name in ['Duke', 'Georgia Tech']:
            team_data = self.manual_tournament_data[self.manual_tournament_data['TeamName'] == team_name]
            if len(team_data) > 0:
                result["has_exit_round_data"] = True
                result["championship_pct"] = team_data.iloc[0].get('ChampionshipPct', None)
                result["final_four_pct"] = team_data.iloc[0].get('FinalFourPct', None)
                result["elite_eight_pct"] = team_data.iloc[0].get('EliteEightPct', None)
                result["sweet_sixteen_pct"] = team_data.iloc[0].get('SweetSixteenPct', None)
                result["round_32_pct"] = team_data.iloc[0].get('Round32Pct', None)
                result["predicted_exit"] = team_data.iloc[0].get('PredictedExit', None)
                result["seed"] = team_data.iloc[0].get('Seed', None)
        
        return result

# Create predictor with manual data
predictor = ManualDataPredictor()
result = predictor.predict_game('Duke', 'Georgia Tech')

# The UI would get these values from the prediction result
duke_score = result['team1']['predicted_score']  # Full score with adjustment
duke_base_score = duke_score - result['tournament_adjustment_team1']  # Base score without adjustment
gt_score = result['team2']['predicted_score']

# Test the hypothesis: Is the UI displaying rounded base scores instead of rounded final scores?
print("--- ACTUAL SCORE VALUES ---")
print(f"Duke full score (with tournament adjustment): {duke_score:.2f}")
print(f"Duke base score (no adjustment): {duke_base_score:.2f}")
print(f"Georgia Tech score: {gt_score:.2f}")
print(f"Spread (Duke - GT): {result['spread']:.2f}")

print("\n--- UI DISPLAY SIMULATION ---")
print("What should be displayed (correctly showing adjusted scores)")
print(f"Duke {round(duke_score)} - {round(gt_score)} Georgia Tech")
print(f"Predicted Spread: Duke by {abs(round(result['spread'], 1))}")

print("\nWhat might be displayed (incorrectly showing base scores)")
print(f"Duke {round(duke_base_score)} - {round(gt_score)} Georgia Tech")
print(f"Predicted Spread: Duke by {abs(round(result['spread'], 1))}")

# Check if this matches what's in the screenshot
print("\n--- SCREENSHOT VALUES ---")
print("Duke 81 - 68 Georgia Tech")
print("Predicted Spread: Duke by 12.3")

# Check if the rounded base score matches the screenshot value
if round(duke_base_score) == 81 and round(gt_score) == 68:
    print("\n✓ CONFIRMED: The UI is displaying the base score (no tournament adjustment)")
    print("  But the spread calculation IS using the tournament adjustment correctly.")
else:
    print("\n✗ Values don't match screenshot. There might be another issue.") 