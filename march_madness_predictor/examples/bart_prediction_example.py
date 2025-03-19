#!/usr/bin/env python3
"""
This example demonstrates how to use the enhanced GamePredictor with BART data
to make more accurate tournament predictions.

The script will:
1. Load the GamePredictor with BART data
2. Compare predictions with and without historical BART data
3. Show how to analyze team trends using the BART historical model
4. Demonstrate prediction of a full tournament matchup
"""

import os
import sys
import pandas as pd
import numpy as np
import json

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from march_madness_predictor.models.game_predictor import GamePredictor
    from march_madness_predictor.models.bart_historical_model import BartHistoricalModel
except ImportError:
    from models.game_predictor import GamePredictor
    from models.bart_historical_model import BartHistoricalModel

def print_prediction_comparison(team1, team2, location='neutral'):
    """
    Compare predictions with and without BART historical data
    
    Parameters:
    -----------
    team1 : str
        Name of the first team
    team2 : str
        Name of the second team
    location : str
        Game location: 'home_1' (team1 at home), 'home_2' (team2 at home), or 'neutral'
    """
    print(f"\n{'='*80}")
    print(f"MATCHUP: {team1} vs {team2} ({location} site)")
    print(f"{'='*80}")
    
    # Create a game predictor
    predictor = GamePredictor()
    
    # Get base prediction without historical data
    base_prediction = predictor.predict_game(team1, team2, location)
    
    # Get enhanced prediction with historical data
    enhanced_prediction = predictor.predict_game_with_history(team1, team2, location)
    
    # Extract key metrics for comparison
    base_score1 = base_prediction['team1']['predicted_score']
    base_score2 = base_prediction['team2']['predicted_score']
    base_spread = base_prediction['spread']
    base_wp = base_prediction['team1']['win_probability'] * 100
    
    enhanced_score1 = enhanced_prediction['team1']['predicted_score']
    enhanced_score2 = enhanced_prediction['team2']['predicted_score']
    enhanced_spread = enhanced_prediction['spread']
    enhanced_wp = enhanced_prediction['team1']['win_probability'] * 100
    
    # Print the comparison
    print(f"\n{'Base Prediction':^40} | {'Enhanced Prediction':^40}")
    print(f"{'-'*40} | {'-'*40}")
    print(f"{team1+' Score':25}: {base_score1:6.1f}        | {team1+' Score':25}: {enhanced_score1:6.1f}")
    print(f"{team2+' Score':25}: {base_score2:6.1f}        | {team2+' Score':25}: {enhanced_score2:6.1f}")
    print(f"{'Spread ('+team1+')':25}: {base_spread:6.1f}        | {'Spread ('+team1+')':25}: {enhanced_spread:6.1f}")
    print(f"{team1+' Win Probability':25}: {base_wp:5.1f}%        | {team1+' Win Probability':25}: {enhanced_wp:5.1f}%")
    
    # Print the adjustments from historical data if available
    if 'historical_model_available' in enhanced_prediction and enhanced_prediction['historical_model_available']:
        if 'bart_historical' in enhanced_prediction:
            hist_data = enhanced_prediction['bart_historical']
            print(f"\nHistorical Adjustments:")
            print(f"  Trend Adjustment: {hist_data['trend_adjustment']:+.2f} points")
            print(f"  Tournament Experience Adjustment: {hist_data['tournament_adjustment']:+.2f} points")
            print(f"  Total Historical Adjustment: {hist_data['trend_adjustment'] + hist_data['tournament_adjustment']:+.2f} points")
    
    # Print key factors
    print("\nKey Factors:")
    for factor in base_prediction['key_factors'][:3]:  # Just show top 3
        print(f"  {factor['factor']}: Advantage {factor['advantage']} ({factor['description']})")
    
    # Print BART data if available
    if 'bart_data' in base_prediction['team1'] and base_prediction['team1']['bart_data']:
        print(f"\nBART Metrics for {team1}:")
        for metric, value in base_prediction['team1']['bart_data'].items():
            print(f"  {metric}: {value}")
    
    if 'bart_data' in base_prediction['team2'] and base_prediction['team2']['bart_data']:
        print(f"\nBART Metrics for {team2}:")
        for metric, value in base_prediction['team2']['bart_data'].items():
            print(f"  {metric}: {value}")
    
    # Print historical matchup data if available
    if 'historical_matchups' in base_prediction and base_prediction['historical_matchups']['total_matchups'] > 0:
        matchups = base_prediction['historical_matchups']
        print(f"\nHistorical Matchups:")
        print(f"  {team1} leads {matchups['team1_wins']}-{matchups['team2_wins']} (avg margin: {matchups['avg_margin']:+.1f})")
        
        # Show up to 3 recent matchups
        if matchups['matchups']:
            print(f"  Recent matchups:")
            for matchup in sorted(matchups['matchups'], key=lambda x: x['year'], reverse=True)[:3]:
                print(f"    {matchup['year']}: {matchup['winner']} would have won by {abs(matchup['diff']):.1f}")

def print_team_historical_analysis(team_name):
    """
    Print historical analysis for a specific team
    
    Parameters:
    -----------
    team_name : str
        Name of the team to analyze
    """
    print(f"\n{'='*80}")
    print(f"HISTORICAL ANALYSIS: {team_name}")
    print(f"{'='*80}")
    
    # Create the BART historical model
    model = BartHistoricalModel()
    
    # Get team trend data
    trend = model.get_team_trend(team_name)
    
    if trend is None:
        print(f"No historical data available for {team_name}")
        return
    
    # Print basic information
    print(f"\nYears of data: {len(trend['years_available'])} ({min(trend['years_available'])}-{max(trend['years_available'])})")
    
    # Print barthag trend
    if trend['barthag_trend']:
        print(f"\nBarthag Trend:")
        barthag_trend = trend['barthag_trend']
        
        # Determine trend direction
        if barthag_trend['trend'] > 0.01:
            direction = "IMPROVING"
        elif barthag_trend['trend'] < -0.01:
            direction = "DECLINING"
        else:
            direction = "STABLE"
        
        print(f"  Direction: {direction} ({barthag_trend['trend']:+.4f} per year)")
        print(f"  Recent Average: {barthag_trend['recent_avg']:.4f}")
        print(f"  All-time Average: {barthag_trend['all_time_avg']:.4f}")
    
    # Print tournament history
    print(f"\nTournament History:")
    tournament_history = trend['tournament_history']
    
    if not tournament_history:
        print(f"  No tournament appearances found in the data")
    else:
        print(f"  {len(tournament_history)} tournament appearances")
        print(f"  Average Seed: {np.mean([th['seed'] for th in tournament_history]):.1f}")
        
        # Print recent appearances
        recent_years = sorted([th['year'] for th in tournament_history], reverse=True)[:5]
        print(f"  Recent Appearances: {', '.join(map(str, recent_years))}")
    
    # Print consistency metrics
    if 'consistency' in trend:
        print(f"\nConsistency Metrics:")
        for metric, value in trend['consistency'].items():
            print(f"  {metric}: {value:.4f}")

def predict_tournament_game(team1, team2, team1_seed=None, team2_seed=None):
    """
    Predict a tournament game with enhanced BART data
    
    Parameters:
    -----------
    team1 : str
        Name of the first team
    team2 : str
        Name of the second team
    team1_seed : int
        Seed of the first team (optional)
    team2_seed : int
        Seed of the second team (optional)
    
    Returns:
    --------
    dict
        Enhanced prediction with historical adjustments
    """
    # Create a game predictor
    predictor = GamePredictor()
    
    # Get enhanced prediction with historical data
    prediction = predictor.predict_game_with_history(team1, team2, 'neutral')
    
    # If seed data is provided, add it to the prediction
    if team1_seed is not None and team2_seed is not None:
        # Add a simple seed-based adjustment (1.5 points per seed difference)
        seed_diff = team2_seed - team1_seed
        seed_adjustment = seed_diff * 0.15  # 0.15 points per seed difference
        
        prediction['team1']['predicted_score'] += seed_adjustment
        prediction['spread'] = prediction['team1']['predicted_score'] - prediction['team2']['predicted_score']
        
        # Apply to win probability (approximate conversion)
        wp_adjustment = seed_adjustment * 0.04  # ~4% win probability per point
        prediction['team1']['win_probability'] = min(0.99, max(0.01, prediction['team1']['win_probability'] + wp_adjustment))
        prediction['team2']['win_probability'] = 1 - prediction['team1']['win_probability']
        
        # Store the seed data in the prediction
        prediction['team1']['seed'] = team1_seed
        prediction['team2']['seed'] = team2_seed
        prediction['seed_adjustment'] = seed_adjustment
    
    return prediction

def simulate_tournament_game(team1, team2, team1_seed=None, team2_seed=None):
    """
    Simulate a tournament game and print the results
    
    Parameters:
    -----------
    team1 : str
        Name of the first team
    team2 : str
        Name of the second team
    team1_seed : int
        Seed of the first team (optional)
    team2_seed : int
        Seed of the second team (optional)
    
    Returns:
    --------
    str
        Name of the predicted winner
    """
    # Get prediction
    prediction = predict_tournament_game(team1, team2, team1_seed, team2_seed)
    
    # Determine winner
    if prediction['team1']['win_probability'] > 0.5:
        winner = team1
        win_prob = prediction['team1']['win_probability'] * 100
    else:
        winner = team2
        win_prob = prediction['team2']['win_probability'] * 100
    
    # Format matchup with seeds
    if team1_seed is not None and team2_seed is not None:
        matchup = f"({team1_seed}) {team1} vs ({team2_seed}) {team2}"
    else:
        matchup = f"{team1} vs {team2}"
    
    # Print prediction
    print(f"{matchup:50} => {winner} wins ({win_prob:.1f}%)")
    
    return winner

def simulate_tournament_region(region_name, teams):
    """
    Simulate a tournament region (e.g., East, West, South, Midwest)
    
    Parameters:
    -----------
    region_name : str
        Name of the region
    teams : list
        List of (team_name, seed) tuples in bracket order
    
    Returns:
    --------
    str
        Name of the region winner
    """
    print(f"\n{'='*80}")
    print(f"{region_name.upper()} REGION SIMULATION")
    print(f"{'='*80}\n")
    
    # First round
    print("First Round:")
    winners_r1 = []
    for i in range(0, len(teams), 2):
        team1, seed1 = teams[i]
        team2, seed2 = teams[i+1]
        winner = simulate_tournament_game(team1, team2, seed1, seed2)
        winners_r1.append((winner, seed1 if winner == team1 else seed2))
    
    # Second round
    print("\nSecond Round:")
    winners_r2 = []
    for i in range(0, len(winners_r1), 2):
        team1, seed1 = winners_r1[i]
        team2, seed2 = winners_r1[i+1]
        winner = simulate_tournament_game(team1, team2, seed1, seed2)
        winners_r2.append((winner, seed1 if winner == team1 else seed2))
    
    # Sweet 16
    print("\nSweet 16:")
    winners_s16 = []
    for i in range(0, len(winners_r2), 2):
        team1, seed1 = winners_r2[i]
        team2, seed2 = winners_r2[i+1]
        winner = simulate_tournament_game(team1, team2, seed1, seed2)
        winners_s16.append((winner, seed1 if winner == team1 else seed2))
    
    # Elite 8
    print("\nElite 8:")
    team1, seed1 = winners_s16[0]
    team2, seed2 = winners_s16[1]
    region_winner = simulate_tournament_game(team1, team2, seed1, seed2)
    region_seed = seed1 if region_winner == team1 else seed2
    
    print(f"\n{region_name} Region Winner: ({region_seed}) {region_winner}")
    
    return region_winner, region_seed

def main():
    """Main function to run the examples"""
    print("\nBART Historical Prediction Example")
    print("================================\n")
    
    # Sample matchups to compare
    matchups = [
        ("Gonzaga", "Duke", "neutral"),
        ("Houston", "Purdue", "neutral"),
        ("Auburn", "Alabama", "neutral"),
    ]
    
    # Compare predictions for each matchup
    for team1, team2, location in matchups:
        print_prediction_comparison(team1, team2, location)
    
    # Team historical analysis
    teams_to_analyze = ["Gonzaga", "Duke", "Houston", "Kentucky"]
    for team in teams_to_analyze:
        print_team_historical_analysis(team)
    
    # Simulate a tournament region
    east_region = [
        ("Duke", 1), ("Norfolk State", 16),
        ("Tennessee", 8), ("Saint Joseph's", 9),
        ("Michigan St.", 5), ("Saint Mary's", 12),
        ("Virginia", 4), ("McNeese", 13),
        ("Creighton", 6), ("Florida Atlantic", 11),
        ("Mississippi", 3), ("Louisville", 14),
        ("Northwestern", 7), ("Seton Hall", 10),
        ("UCLA", 2), ("Long Beach St.", 15)
    ]
    
    # Simulate the region
    east_winner, east_seed = simulate_tournament_region("East", east_region)
    
    # Now let's demo the Final Four
    south_winner, south_seed = ("Auburn", 1)  # Simulated elsewhere
    midwest_winner, midwest_seed = ("Houston", 1)  # Simulated elsewhere
    west_winner, west_seed = ("Florida", 1)  # Simulated elsewhere
    
    print(f"\n{'='*80}")
    print(f"FINAL FOUR SIMULATION")
    print(f"{'='*80}\n")
    
    # Semifinal 1
    print("Semifinal 1:")
    finalist1 = simulate_tournament_game(east_winner, south_winner, east_seed, south_seed)
    
    # Semifinal 2
    print("\nSemifinal 2:")
    finalist2 = simulate_tournament_game(midwest_winner, west_winner, midwest_seed, west_seed)
    
    # Championship
    print("\nNational Championship:")
    champion = simulate_tournament_game(finalist1, finalist2)
    
    print(f"\n{'-'*80}")
    print(f"NCAA TOURNAMENT CHAMPION: {champion}")
    print(f"{'-'*80}")

if __name__ == "__main__":
    main() 