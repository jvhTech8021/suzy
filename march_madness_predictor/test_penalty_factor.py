import sys
import os

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from march_madness_predictor.models.game_predictor import GamePredictor
from march_madness_predictor.utils.data_loader import DataLoader

def test_penalty_factors(team1_name, team2_name):
    """
    Test different penalty factors and display their effect on predictions
    """
    print(f"\n===== TESTING PENALTY FACTORS FOR {team1_name} vs {team2_name} =====")
    
    # Load data
    data_loader = DataLoader()
    
    # Test with different penalty factors
    penalty_factors = [0.0, 1.0, 5.0, 10.0]
    
    for factor in penalty_factors:
        print(f"\n----- Testing with penalty factor: {factor} -----")
        
        # Create a new predictor with the current penalty factor
        predictor = GamePredictor(data_loader)
        predictor.TOURNAMENT_PENALTY_FACTOR = factor
        
        # Make a prediction
        prediction = predictor.predict_game(team1_name, team2_name)
        
        # Extract relevant results
        team1_score = prediction["team1"]["predicted_score"]
        team2_score = prediction["team2"]["predicted_score"]
        spread = prediction["spread"]
        team1_adjustment = prediction["tournament_adjustment_team1"]
        team2_adjustment = prediction["tournament_adjustment_team2"]
        net_adjustment = prediction["tournament_adjustment"]
        
        # Print formatted results
        print(f"Team1 ({team1_name}) Score: {team1_score:.2f}")
        print(f"Team2 ({team2_name}) Score: {team2_score:.2f}")
        print(f"Spread: {spread:.2f}")
        print(f"Team1 Tournament Adjustment: {team1_adjustment:.2f}")
        print(f"Team2 Tournament Adjustment: {team2_adjustment:.2f}")
        print(f"Net Tournament Adjustment: {net_adjustment:.2f}")
    
    print("\n===== TEST COMPLETE =====")

if __name__ == "__main__":
    # Default test teams
    team1 = "VCU"
    team2 = "St. Bonaventure"
    
    # Use command line args if provided
    if len(sys.argv) > 2:
        team1 = sys.argv[1]
        team2 = sys.argv[2]
    
    test_penalty_factors(team1, team2) 