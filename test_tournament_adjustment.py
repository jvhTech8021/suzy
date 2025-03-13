from march_madness_predictor.models.game_predictor import GamePredictor
import pandas as pd

# Create a predictor with mocked tournament data
class TestPredictor(GamePredictor):
    def _get_tournament_prediction_data(self, team_name):
        if team_name == 'Alabama':
            return {
                "has_exit_round_data": True,
                "has_champion_profile_data": True,
                "championship_pct": 5.0,            # 5% championship probability
                "final_four_pct": 20.0,             # 20% Final Four probability
                "elite_eight_pct": 35.0,            # 35% Elite Eight probability
                "sweet_sixteen_pct": 55.0,          # 55% Sweet Sixteen probability
                "round_32_pct": 75.0,               # 75% Round of 32 probability
                "predicted_exit": 4,
                "similarity_pct": 70.0,
                "seed": 2
            }
        else:  # Duke
            return {
                "has_exit_round_data": True,
                "has_champion_profile_data": True,
                "championship_pct": 3.0,            # 3% championship probability
                "final_four_pct": 15.0,             # 15% Final Four probability
                "elite_eight_pct": 25.0,            # 25% Elite Eight probability
                "sweet_sixteen_pct": 45.0,          # 45% Sweet Sixteen probability
                "round_32_pct": 70.0,               # 70% Round of 32 probability
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
print(f"Alabama total adjustment: {result['tournament_adjustment_team1']:.2f}")
print(f"Duke total adjustment: {result['tournament_adjustment_team2']:.2f}")
print(f"Net tournament adjustment: {result['tournament_adjustment']:.2f}")

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