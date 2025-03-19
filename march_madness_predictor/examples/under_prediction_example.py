#!/usr/bin/env python3
"""
This example demonstrates how to use the enhanced GamePredictor with improved
capabilities for identifying potential "under" betting opportunities.

The script will:
1. Load the GamePredictor with the enhancements
2. Analyze matchups for potential under opportunities
3. Display factors that contribute to under predictions
4. Compare with historical under performance
"""

import os
import sys
import pandas as pd
import numpy as np
from pprint import pprint

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from march_madness_predictor.models.game_predictor import GamePredictor
except ImportError:
    from models.game_predictor import GamePredictor

def analyze_under_opportunity(team1, team2, location='neutral'):
    """
    Analyze a matchup for potential under betting opportunities
    
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
    print(f"UNDER ANALYSIS: {team1} vs {team2} ({location} site)")
    print(f"{'='*80}")
    
    # Create a game predictor
    predictor = GamePredictor()
    
    # Get prediction with enhanced under analysis
    prediction = predictor.predict_game(team1, team2, location)
    
    # Extract key metrics
    total = prediction['total']
    under_analysis = prediction['total_analysis']
    under_probability = under_analysis['under_probability']
    recommendation = under_analysis['recommendation']
    controlling_team = prediction['tempo_control']['controlling_team']
    control_factor = prediction['tempo_control']['control_factor']
    
    # Print the analysis
    print(f"\nPredicted Total: {total:.1f}")
    print(f"Under Probability: {under_probability*100:.1f}%")
    print(f"Recommendation: {recommendation}")
    
    # Print tempo control information
    print(f"\nTempo Control Analysis:")
    print(f"  Controlling Team: {controlling_team} (control factor: {control_factor:.2f})")
    
    # Print the factors contributing to under potential
    if under_analysis['factors']:
        print(f"\nFactors contributing to under potential:")
        for i, (factor, weight) in enumerate(zip(under_analysis['factors'], under_analysis['factor_weights'])):
            print(f"  {i+1}. {factor} ({weight*100:.1f}%)")
    else:
        print(f"\nNo significant factors indicating an under opportunity")
    
    # Print defensive matchup score
    defensive_score = prediction['defensive_matchup']['score']
    print(f"\nDefensive Matchup Score: {defensive_score:.1f}/100")
    if defensive_score >= 70:
        print("  This is an exceptionally strong defensive matchup")
    elif defensive_score >= 60:
        print("  This is a strong defensive matchup")
    elif defensive_score >= 50:
        print("  This is an above-average defensive matchup")
    else:
        print("  This is not a particularly strong defensive matchup")
    
    # Print historical matchup data if available
    if prediction['historical_matchups']['total_matchups'] > 0:
        hist = prediction['historical_matchups']
        print(f"\nHistorical Matchup Data:")
        print(f"  Total Matchups: {hist['total_matchups']}")
        if 'avg_total' in hist and hist['avg_total']:
            print(f"  Historical Average Total: {hist['avg_total']:.1f}")
            print(f"  Current Total vs Historical: {total - hist['avg_total']:+.1f}")
    
    # Print team defensive percentiles if available
    if hasattr(predictor, 'defensive_percentiles') and predictor.defensive_percentiles:
        team1_def_pct = predictor.defensive_percentiles.get(team1, "N/A")
        team2_def_pct = predictor.defensive_percentiles.get(team2, "N/A")
        
        print(f"\nDefensive Percentiles:")
        print(f"  {team1}: {team1_def_pct if team1_def_pct == 'N/A' else f'{team1_def_pct:.1f}%'}")
        print(f"  {team2}: {team2_def_pct if team2_def_pct == 'N/A' else f'{team2_def_pct:.1f}%'}")
    
    # Print team tempo percentiles if available
    if hasattr(predictor, 'tempo_percentiles') and predictor.tempo_percentiles:
        team1_tempo_pct = predictor.tempo_percentiles.get(team1, "N/A")
        team2_tempo_pct = predictor.tempo_percentiles.get(team2, "N/A")
        
        print(f"\nTempo Percentiles (higher = faster):")
        print(f"  {team1}: {team1_tempo_pct if team1_tempo_pct == 'N/A' else f'{team1_tempo_pct:.1f}%'}")
        print(f"  {team2}: {team2_tempo_pct if team2_tempo_pct == 'N/A' else f'{team2_tempo_pct:.1f}%'}")

def batch_analyze_matchups(matchups):
    """
    Analyze multiple matchups and sort by under probability
    
    Parameters:
    -----------
    matchups : list
        List of (team1, team2, location) tuples
    """
    print(f"\n{'='*80}")
    print(f"BATCH UNDER ANALYSIS FOR {len(matchups)} MATCHUPS")
    print(f"{'='*80}")
    
    # Create a game predictor
    predictor = GamePredictor()
    
    # Analyze each matchup
    results = []
    for team1, team2, location in matchups:
        prediction = predictor.predict_game(team1, team2, location)
        
        results.append({
            'team1': team1,
            'team2': team2,
            'location': location,
            'total': prediction['total'],
            'under_probability': prediction['total_analysis']['under_probability'],
            'recommendation': prediction['total_analysis']['recommendation'],
            'defensive_score': prediction['defensive_matchup']['score'],
            'factors': len(prediction['total_analysis']['factors'])
        })
    
    # Sort by under probability (highest to lowest)
    results.sort(key=lambda x: x['under_probability'], reverse=True)
    
    # Print sorted results
    print(f"\n{'Team 1':<20} {'Team 2':<20} {'Total':<8} {'Under %':<10} {'Recommendation'}")
    print(f"{'-'*20} {'-'*20} {'-'*8} {'-'*10} {'-'*30}")
    
    for r in results:
        print(f"{r['team1']:<20} {r['team2']:<20} {r['total']:<8.1f} {r['under_probability']*100:<10.1f}% {r['recommendation']}")

def identify_best_unders_from_top_teams():
    """
    Identify the best potential under opportunities from games involving top teams
    """
    print(f"\n{'='*80}")
    print(f"IDENTIFYING BEST UNDER OPPORTUNITIES FROM TOP TEAMS")
    print(f"{'='*80}")
    
    # Create a game predictor
    predictor = GamePredictor()
    
    # Get a list of all teams
    all_teams = predictor.get_available_teams()
    
    # Analyze top defensive teams (if percentiles are available)
    if hasattr(predictor, 'defensive_percentiles') and predictor.defensive_percentiles:
        # Sort teams by defensive percentile
        def_percentiles = [(team, predictor.defensive_percentiles.get(team, 0)) 
                         for team in all_teams]
        def_percentiles.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 10 defensive teams
        top_defensive_teams = [team for team, pct in def_percentiles[:10]]
        
        print(f"\nTop 10 Defensive Teams:")
        for i, (team, pct) in enumerate(def_percentiles[:10]):
            print(f"  {i+1}. {team} ({pct:.1f}%)")
        
        # Get slow-paced teams (if percentiles are available)
        if hasattr(predictor, 'tempo_percentiles') and predictor.tempo_percentiles:
            # Sort teams by tempo percentile (low to high for slowest)
            tempo_percentiles = [(team, predictor.tempo_percentiles.get(team, 50)) 
                               for team in all_teams]
            tempo_percentiles.sort(key=lambda x: x[1])
            
            # Get top 10 slowest teams
            slowest_teams = [team for team, pct in tempo_percentiles[:10]]
            
            print(f"\nTop 10 Slowest-Paced Teams:")
            for i, (team, pct) in enumerate(tempo_percentiles[:10]):
                print(f"  {i+1}. {team} ({pct:.1f}%)")
            
            # Generate matchups between top defensive teams
            matchups = []
            for i, team1 in enumerate(top_defensive_teams):
                for team2 in top_defensive_teams[i+1:]:
                    matchups.append((team1, team2, 'neutral'))
            
            # Generate matchups between top defensive teams and slowest teams
            for team1 in top_defensive_teams:
                for team2 in slowest_teams:
                    if team1 != team2:  # Avoid matching team against itself
                        matchups.append((team1, team2, 'neutral'))
            
            # Analyze these matchups
            if matchups:
                print(f"\nAnalyzing {len(matchups)} potential high-quality under matchups...")
                batch_analyze_matchups(matchups)

def main():
    """Main function to run the examples"""
    print("\nNCAA Basketball Under Prediction Example")
    print("========================================\n")
    
    # Sample matchups to analyze
    matchups = [
        ("Virginia", "Wisconsin", "neutral"),
        ("Tennessee", "Michigan St.", "neutral"),
        ("Baylor", "Texas", "neutral"),
        ("Gonzaga", "Duke", "neutral"),
        ("Kentucky", "Auburn", "neutral"),
    ]
    
    # Analyze individual matchups
    for team1, team2, location in matchups:
        analyze_under_opportunity(team1, team2, location)
    
    # Batch analyze all matchups
    batch_analyze_matchups(matchups)
    
    # Find the best under opportunities from top teams
    identify_best_unders_from_top_teams()

if __name__ == "__main__":
    main() 