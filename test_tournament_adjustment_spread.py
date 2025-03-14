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
    
    def run_test_with_different_factors(self, champ_factor, ff_factor):
        """Run a prediction with custom scaling factors"""
        # Save the original scaling factors
        orig_factors = None
        if hasattr(self, 'custom_scaling_factors'):
            orig_factors = self.custom_scaling_factors
            
        # Set custom scaling factors
        self.custom_scaling_factors = {
            "championship_pct": champ_factor,
            "final_four_pct": ff_factor,
            "elite_eight_pct": 0.04,
            "sweet_sixteen_pct": 0.025,
            "round_32_pct": 0.0125
        }
        
        # Run prediction with test teams
        result = self.predict_game('Alabama', 'Duke')
        
        # Reset to original factors
        if orig_factors:
            self.custom_scaling_factors = orig_factors
        else:
            delattr(self, 'custom_scaling_factors')
            
        return result
    
    def predict_game(self, team1_name, team2_name, location='neutral'):
        # Override the scaling factors before prediction
        if hasattr(self, 'custom_scaling_factors'):
            # Temporarily save the original scaling factors
            original_factors = {}
            if team1_tournament_data := self._get_tournament_prediction_data(team1_name):
                if team2_tournament_data := self._get_tournament_prediction_data(team2_name):
                    if team1_tournament_data["championship_pct"] is not None and team2_tournament_data["championship_pct"] is not None:
                        # Save custom factors to the class for this prediction only
                        original_factors = self.custom_scaling_factors.copy()
                        
        # Call the parent method to get the prediction
        result = super().predict_game(team1_name, team2_name, location)
        
        return result

# Initialize the predictor
predictor = TestPredictor()

# Test with different championship scaling factors
print("\n--- TEST WITH DIFFERENT SCALING FACTORS ---")
# Base test with default factors
base_result = predictor.predict_game('Alabama', 'Duke')
print(f"Base Test - Scaling factors unchanged")
print(f"Team 1 score: {base_result['team1']['predicted_score']:.2f}")
print(f"Team 2 score: {base_result['team2']['predicted_score']:.2f}")
print(f"Spread: {base_result['spread']:.2f}")
print(f"Net tournament adjustment: {base_result['tournament_adjustment']:.2f}")

# Modified predictor with override capability
class ScalingFactorTestPredictor(GamePredictor):
    """Test predictor that allows modifying scaling factors at runtime"""
    
    def __init__(self):
        super().__init__()
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
        return self.team_data[self.team_data['TeamName'] == team_name].iloc[0]
    
    def _get_tournament_prediction_data(self, team_name):
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
        return {"total_matchups": 0, "team1_wins": 0, "team2_wins": 0, "avg_margin": 0}
    
    def predict_game_with_factors(self, team1_name, team2_name, scaling_factors):
        """Run a prediction with custom scaling factors"""
        # Save the original predict_game method
        original_predict_game = self.predict_game
        
        # Override the predict_game method to use custom scaling factors
        def custom_predict_game(t1, t2, location='neutral'):
            result = original_predict_game(t1, t2, location)
            # Print debug info for verification
            print(f"Team1 score: {result['team1']['predicted_score']:.2f}")
            print(f"Team2 score: {result['team2']['predicted_score']:.2f}")
            print(f"Tournament adj team1: {result['tournament_adjustment_team1']:.2f}")
            print(f"Tournament adj team2: {result['tournament_adjustment_team2']:.2f}")
            print(f"Net tournament adj: {result['tournament_adjustment']:.2f}")
            print(f"Spread: {result['spread']:.2f}")
            return result
        
        # Temporarily replace the predict_game method
        self.predict_game = custom_predict_game
        
        # Run the prediction and restore the original method
        try:
            result = self.predict_game(team1_name, team2_name)
        finally:
            self.predict_game = original_predict_game
            
        return result

# Run a test with a simple direct approach
test_predictor = ScalingFactorTestPredictor()

# Standard scaling factors
print("\nSTANDARD FACTORS TEST")
standard_result = test_predictor.predict_game('Alabama', 'Duke', 'neutral')
print(f"Spread with standard factors: {standard_result['spread']:.2f}")

# Now try with higher championship factor
print("\nDOUBLED CHAMPIONSHIP FACTOR TEST")
# We'll directly modify the scaling factors in the class
original_champ_factor = GamePredictor.predict_game.__globals__['scaling_factors']['championship_pct']
GamePredictor.predict_game.__globals__['scaling_factors']['championship_pct'] = original_champ_factor * 2
doubled_result = test_predictor.predict_game('Alabama', 'Duke', 'neutral')
print(f"Spread with doubled championship factor: {doubled_result['spread']:.2f}")

# Restore original factors
GamePredictor.predict_game.__globals__['scaling_factors']['championship_pct'] = original_champ_factor

print("\nSUMMARY")
print(f"Spread difference: {doubled_result['spread'] - standard_result['spread']:.2f}") 