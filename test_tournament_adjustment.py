from march_madness_predictor.models.game_predictor import GamePredictor
import pandas as pd

# Create a predictor with mocked tournament data
class TestPredictor(GamePredictor):
    def _get_tournament_prediction_data(self, team_name):
        if team_name == 'Alabama':
            return {
                "has_exit_round_data": True,
                "has_champion_profile_data": True,
                "championship_pct": 5.0,  # 5% championship probability
                "final_four_pct": 20.0,
                "predicted_exit": 4,
                "similarity_pct": 70.0,
                "seed": 2
            }
        else:  # Duke
            return {
                "has_exit_round_data": True,
                "has_champion_profile_data": True,
                "championship_pct": 3.0,  # 3% championship probability
                "final_four_pct": 15.0,
                "predicted_exit": 3,
                "similarity_pct": 65.0,
                "seed": 4
            }

# Initialize the predictor
predictor = TestPredictor()

# Run prediction with real teams but mocked tournament data
result = predictor.predict_game('Alabama', 'Duke')

# Print the relevant details
print("\n--- TEAM SCORES ---")
print(f"Alabama score: {result['team1']['predicted_score']:.2f}")
print(f"Duke score: {result['team2']['predicted_score']:.2f}")

print("\n--- TOURNAMENT ADJUSTMENTS ---")
print(f"Alabama adjustment: {result['tournament_adjustment_team1']:.2f}")
print(f"Duke adjustment: {result['tournament_adjustment_team2']:.2f}")
print(f"Net tournament adjustment: {result['tournament_adjustment']:.2f}")

print("\n--- SPREAD CALCULATION ---")
print(f"Alabama - Duke spread: {result['spread']:.2f}")

# Calculate what the spread would be without tournament adjustments
adjusted_team1_score = result['team1']['predicted_score'] - result['tournament_adjustment_team1']
adjusted_team2_score = result['team2']['predicted_score'] - result['tournament_adjustment_team2']
base_spread = adjusted_team1_score - adjusted_team2_score

print("\n--- VERIFICATION ---")
print(f"Spread without tournament adjustments: {base_spread:.2f}")
print(f"Expected impact of tournament adjustments on spread: {result['tournament_adjustment']:.2f}")
print(f"Actual impact of tournament adjustments on spread: {result['spread'] - base_spread:.2f}")
print(f"Do they match? {'Yes' if abs((result['spread'] - base_spread) - result['tournament_adjustment']) < 0.01 else 'No'}")

# Let's also check how the win probability is affected
print("\n--- WIN PROBABILITY ---")
print(f"Alabama win probability: {result['team1']['win_probability']:.4f}")
print(f"Duke win probability: {result['team2']['win_probability']:.4f}") 