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

# Show the current scaling factors
print("Current tournament scaling factors:")
for key, value in predictor.TOURNAMENT_SCALING_FACTORS.items():
    print(f"  {key}: {value}")

# Run prediction with manual data
result = predictor.predict_game('Duke', 'Georgia Tech')

# Calculate expected tournament adjustment for Duke manually
print("\n--- Manual Tournament Adjustment Calculation for Duke ---")
duke_data = predictor._get_tournament_prediction_data('Duke')
factors = predictor.TOURNAMENT_SCALING_FACTORS

duke_total_adjustment = 0
for key, factor in factors.items():
    if key in duke_data and duke_data[key] is not None:
        adjustment = duke_data[key] * factor
        duke_total_adjustment += adjustment
        print(f"{key}: {duke_data[key]:.1f}% × {factor:.3f} = {adjustment:.2f} points")

print(f"Total Expected Duke Adjustment: {duke_total_adjustment:.2f} points")
print(f"Actual Duke Adjustment in Model: {result['tournament_adjustment_team1']:.2f} points")

# Calculate expected tournament adjustment for Georgia Tech manually
print("\n--- Manual Tournament Adjustment Calculation for Georgia Tech ---")
gt_data = predictor._get_tournament_prediction_data('Georgia Tech')
gt_total_adjustment = 0
for key, factor in factors.items():
    if key in gt_data and gt_data[key] is not None:
        adjustment = gt_data[key] * factor
        gt_total_adjustment += adjustment
        print(f"{key}: {gt_data[key]:.1f}% × {factor:.3f} = {adjustment:.2f} points")

print(f"Total Expected Georgia Tech Adjustment: {gt_total_adjustment:.2f} points")
print(f"Actual Georgia Tech Adjustment in Model: {result['tournament_adjustment_team2']:.2f} points")

# Print prediction summary
print("\n--- Prediction Summary ---")
print(f"Duke Base Score (no adjustments): {result['team1']['predicted_score'] - result['tournament_adjustment_team1']:.2f}")
print(f"Duke Tournament Adjustment: +{result['tournament_adjustment_team1']:.2f}")
print(f"Duke Final Score: {result['team1']['predicted_score']:.2f}")
print(f"Georgia Tech Base Score (no adjustments): {result['team2']['predicted_score'] - result['tournament_adjustment_team2']:.2f}")
print(f"Georgia Tech Tournament Adjustment: +{result['tournament_adjustment_team2']:.2f}")
print(f"Georgia Tech Final Score: {result['team2']['predicted_score']:.2f}")
print(f"Spread (Duke - Georgia Tech): {result['spread']:.2f}")
print(f"Net Tournament Adjustment: {result['tournament_adjustment']:.2f}")

# Expected spread calculation
expected_base_spread = (result['team1']['predicted_score'] - result['tournament_adjustment_team1']) - (result['team2']['predicted_score'] - result['tournament_adjustment_team2'])
expected_adjusted_spread = expected_base_spread + duke_total_adjustment - gt_total_adjustment
print(f"\nExpected Base Spread (no adjustments): {expected_base_spread:.2f}")
print(f"Expected Adjusted Spread (with tournament adjustments): {expected_adjusted_spread:.2f}")
print(f"Actual Spread in Model: {result['spread']:.2f}")

# Check if tournament adjustment is included in spread
if abs(expected_adjusted_spread - result['spread']) < 0.01:
    print("\n✓ CORRECT: Tournament adjustments ARE being included in the spread!")
else:
    print("\n✗ ERROR: Tournament adjustments are NOT being correctly included in the spread!")
    print(f"Discrepancy: {expected_adjusted_spread - result['spread']:.2f} points") 