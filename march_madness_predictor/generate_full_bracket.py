import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from utils.data_loader import DataLoader

def generate_full_bracket():
    """
    Generate a full 64-team NCAA tournament bracket using data from the
    champion profile and exit round prediction models.
    """
    print("=" * 80)
    print(f"FULL NCAA TOURNAMENT BRACKET GENERATOR - Run Date: {datetime.now()}")
    print("=" * 80)
    
    # Initialize the data loader
    data_loader = DataLoader()
    
    # Create output directory
    output_dir = os.path.join(script_dir, "models/full_bracket/model")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load necessary data
    try:
        print("Loading prediction data...")
        
        # Load combined predictions if available
        combined_data = data_loader.get_combined_predictions()
        if combined_data is None:
            raise ValueError("Combined predictions not available")
            
        # Load champion profile predictions
        champion_data = data_loader.get_champion_profile_predictions()
        
        # Load exit round predictions
        exit_data = data_loader.get_exit_round_predictions()
        
        # Get current season data
        current_data = data_loader.get_current_season_data()
        
        print("Data loaded successfully!")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create the bracket structure
    print("Creating bracket structure...")
    bracket = create_bracket_structure(combined_data)
    
    # Simulate the tournament
    print("Simulating tournament games...")
    results = simulate_tournament(bracket)
    
    # Save the results
    print("Saving tournament results...")
    save_results(results, output_dir)
    
    print("=" * 80)
    print("Full bracket generation completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)

def create_bracket_structure(data):
    """
    Create the initial bracket structure with 64 teams
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing team data and predictions
        
    Returns:
    --------
    dict
        Dictionary containing the bracket structure
    """
    # Sort teams by combined score or championship probability
    if 'CombinedScore' in data.columns:
        sorted_teams = data.sort_values('CombinedScore', ascending=False)
    else:
        sorted_teams = data.sort_values('ChampionshipPct_Combined', ascending=False)
        
    # Get top 64 teams
    tournament_teams = sorted_teams.head(64).copy()
    
    # Assign seeds based on ranking
    regions = ['East', 'West', 'South', 'Midwest']
    tournament_teams['Region'] = ''
    tournament_teams['Seed'] = 0
    
    # Pre-assign regions and seeds to all teams
    for i in range(len(tournament_teams)):
        region_idx = i % 4
        seed_within_region = (i // 4) + 1
        
        # Get the index at position i
        idx = tournament_teams.index[i]
        
        # Assign region and seed
        tournament_teams.at[idx, 'Region'] = regions[region_idx]
        tournament_teams.at[idx, 'Seed'] = seed_within_region
    
    # Create the bracket structure
    bracket = {
        'regions': {},
        'final_four': {
            'teams': [],
            'winner': None
        },
        'championship': {
            'teams': [],
            'winner': None
        }
    }
    
    # Initialize the regions
    for region in regions:
        bracket['regions'][region] = {
            'teams': [],
            'rounds': {
                'first_round': [],
                'second_round': [],
                'sweet_16': [],
                'elite_8': [],
                'region_winner': None
            }
        }
    
    # Fill in the teams for each region
    for region in regions:
        region_teams = tournament_teams[tournament_teams['Region'] == region].sort_values('Seed')
        
        for _, team in region_teams.iterrows():
            team_dict = {
                'name': team['TeamName'],
                'seed': int(team['Seed']),
                'championship_pct': float(team['ChampionshipPct_Combined']),
                'similarity_pct': float(team['SimilarityPct']) if 'SimilarityPct' in team else 0,
                'predicted_exit_round': float(team['PredictedExitRound']) if 'PredictedExitRound' in team else 0
            }
            bracket['regions'][region]['teams'].append(team_dict)
            
        # Create the first round matchups
        teams = bracket['regions'][region]['teams']
        first_round = []
        
        # Standard NCAA tournament seeding pairs: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
        pairings = [(0, 15), (7, 8), (4, 11), (3, 12), (5, 10), (2, 13), (6, 9), (1, 14)]
        
        for pair in pairings:
            if len(teams) > max(pair[0], pair[1]):
                matchup = {
                    'team1': teams[pair[0]],
                    'team2': teams[pair[1]],
                    'winner': None
                }
                first_round.append(matchup)
        
        bracket['regions'][region]['rounds']['first_round'] = first_round
        
    return bracket

def simulate_tournament(bracket):
    """
    Simulate the tournament from the first round to the championship
    
    Parameters:
    -----------
    bracket : dict
        Dictionary containing the bracket structure
        
    Returns:
    --------
    dict
        Dictionary containing the tournament results
    """
    # Simulate each region's games
    for region_name, region in bracket['regions'].items():
        print(f"Simulating {region_name} region...")
        
        # First round
        second_round = []
        for matchup in region['rounds']['first_round']:
            winner = predict_winner(matchup['team1'], matchup['team2'])
            matchup['winner'] = winner
            second_round.append(winner)
        
        # Create second round matchups
        sweet_16 = []
        for i in range(0, len(second_round), 2):
            if i+1 < len(second_round):
                matchup = {
                    'team1': second_round[i],
                    'team2': second_round[i+1],
                    'winner': None
                }
                winner = predict_winner(matchup['team1'], matchup['team2'])
                matchup['winner'] = winner
                sweet_16.append(winner)
                region['rounds']['second_round'].append(matchup)
        
        # Create Sweet 16 matchups
        elite_8 = []
        for i in range(0, len(sweet_16), 2):
            if i+1 < len(sweet_16):
                matchup = {
                    'team1': sweet_16[i],
                    'team2': sweet_16[i+1],
                    'winner': None
                }
                winner = predict_winner(matchup['team1'], matchup['team2'])
                matchup['winner'] = winner
                elite_8.append(winner)
                region['rounds']['sweet_16'].append(matchup)
        
        # Create Elite 8 matchup
        if len(elite_8) >= 2:
            matchup = {
                'team1': elite_8[0],
                'team2': elite_8[1],
                'winner': None
            }
            winner = predict_winner(matchup['team1'], matchup['team2'])
            matchup['winner'] = winner
            region['rounds']['elite_8'].append(matchup)
            region['rounds']['region_winner'] = winner
            bracket['final_four']['teams'].append(winner)
    
    # Simulate Final Four
    if len(bracket['final_four']['teams']) >= 4:
        # Create championship matchups
        championship = []
        
        # First Final Four matchup (East vs West)
        matchup1 = {
            'team1': bracket['final_four']['teams'][0],  # East winner
            'team2': bracket['final_four']['teams'][1],  # West winner
            'winner': None
        }
        winner1 = predict_winner(matchup1['team1'], matchup1['team2'])
        matchup1['winner'] = winner1
        championship.append(winner1)
        
        # Second Final Four matchup (South vs Midwest)
        matchup2 = {
            'team1': bracket['final_four']['teams'][2],  # South winner
            'team2': bracket['final_four']['teams'][3],  # Midwest winner
            'winner': None
        }
        winner2 = predict_winner(matchup2['team1'], matchup2['team2'])
        matchup2['winner'] = winner2
        championship.append(winner2)
        
        bracket['final_four']['matchups'] = [matchup1, matchup2]
        
        # Championship game
        championship_matchup = {
            'team1': championship[0],
            'team2': championship[1],
            'winner': None
        }
        champion = predict_winner(championship_matchup['team1'], championship_matchup['team2'])
        championship_matchup['winner'] = champion
        
        bracket['championship']['teams'] = championship
        bracket['championship']['matchup'] = championship_matchup
        bracket['championship']['winner'] = champion
    
    return bracket

def predict_winner(team1, team2):
    """
    Predict the winner of a matchup between two teams
    
    Parameters:
    -----------
    team1 : dict
        Dictionary containing team1 information
    team2 : dict
        Dictionary containing team2 information
        
    Returns:
    --------
    dict
        Dictionary containing the winning team information
    """
    # Use a combination of championship percentage, similarity percentage, and predicted exit round
    # to determine the winner
    
    # Calculate a score for each team
    team1_score = (0.4 * team1['championship_pct'] + 
                   0.3 * team1['similarity_pct'] + 
                   0.3 * (team1['predicted_exit_round'] / 7 * 100))
    
    team2_score = (0.4 * team2['championship_pct'] + 
                   0.3 * team2['similarity_pct'] + 
                   0.3 * (team2['predicted_exit_round'] / 7 * 100))
    
    # Adjust by seed difference (favor lower seeds slightly)
    seed_diff = team2['seed'] - team1['seed']
    seed_advantage = seed_diff * 2  # 2 points per seed difference
    
    team1_score += seed_advantage
    
    # Determine the winner
    if team1_score > team2_score:
        return team1
    else:
        return team2

def save_results(bracket, output_dir):
    """
    Save the tournament results to files
    
    Parameters:
    -----------
    bracket : dict
        Dictionary containing the bracket structure
    output_dir : str
        Directory to save the results to
    """
    # Save the full bracket as JSON
    bracket_file = os.path.join(output_dir, "full_bracket.json")
    with open(bracket_file, 'w') as f:
        json.dump(bracket, f, indent=4)
    
    # Create a text representation of the bracket for easy viewing
    bracket_text = create_bracket_text(bracket)
    text_file = os.path.join(output_dir, "full_bracket.txt")
    with open(text_file, 'w') as f:
        f.write(bracket_text)
    
    # Save a CSV with all tournament teams and their predicted finish
    save_teams_csv(bracket, output_dir)
    
    print(f"Results saved to {output_dir}")

def create_bracket_text(bracket):
    """
    Create a text representation of the bracket
    
    Parameters:
    -----------
    bracket : dict
        Dictionary containing the bracket structure
        
    Returns:
    --------
    str
        Text representation of the bracket
    """
    text = "NCAA TOURNAMENT BRACKET PREDICTION\n"
    text += "=" * 80 + "\n\n"
    
    # Add information for each region
    for region_name, region in bracket['regions'].items():
        text += f"{region_name.upper()} REGION\n"
        text += "-" * 40 + "\n"
        
        # First round
        text += "First Round:\n"
        for matchup in region['rounds']['first_round']:
            team1 = matchup['team1']
            team2 = matchup['team2']
            winner = matchup['winner']
            if winner:
                text += f"({team1['seed']}) {team1['name']} vs ({team2['seed']}) {team2['name']} → "
                text += f"Winner: ({winner['seed']}) {winner['name']}\n"
            else:
                text += f"({team1['seed']}) {team1['name']} vs ({team2['seed']}) {team2['name']} → No winner determined\n"
        
        text += "\nSecond Round:\n"
        if not region['rounds']['second_round']:
            text += "No games simulated\n"
        else:
            for matchup in region['rounds']['second_round']:
                team1 = matchup['team1']
                team2 = matchup['team2']
                winner = matchup['winner']
                if winner:
                    text += f"({team1['seed']}) {team1['name']} vs ({team2['seed']}) {team2['name']} → "
                    text += f"Winner: ({winner['seed']}) {winner['name']}\n"
                else:
                    text += f"({team1['seed']}) {team1['name']} vs ({team2['seed']}) {team2['name']} → No winner determined\n"
        
        text += "\nSweet 16:\n"
        if not region['rounds']['sweet_16']:
            text += "No games simulated\n"
        else:
            for matchup in region['rounds']['sweet_16']:
                team1 = matchup['team1']
                team2 = matchup['team2']
                winner = matchup['winner']
                if winner:
                    text += f"({team1['seed']}) {team1['name']} vs ({team2['seed']}) {team2['name']} → "
                    text += f"Winner: ({winner['seed']}) {winner['name']}\n"
                else:
                    text += f"({team1['seed']}) {team1['name']} vs ({team2['seed']}) {team2['name']} → No winner determined\n"
        
        text += "\nElite 8:\n"
        if not region['rounds']['elite_8']:
            text += "No games simulated\n"
        else:
            for matchup in region['rounds']['elite_8']:
                team1 = matchup['team1']
                team2 = matchup['team2']
                winner = matchup['winner']
                if winner:
                    text += f"({team1['seed']}) {team1['name']} vs ({team2['seed']}) {team2['name']} → "
                    text += f"Winner: ({winner['seed']}) {winner['name']}\n"
                else:
                    text += f"({team1['seed']}) {team1['name']} vs ({team2['seed']}) {team2['name']} → No winner determined\n"
        
        if region['rounds']['region_winner']:
            text += f"\nRegion Winner: ({region['rounds']['region_winner']['seed']}) {region['rounds']['region_winner']['name']}\n"
        else:
            text += "\nRegion Winner: Not determined\n"
        text += "=" * 40 + "\n\n"
    
    # Final Four
    if 'matchups' in bracket['final_four']:
        text += "FINAL FOUR\n"
        text += "-" * 40 + "\n"
        
        matchup1 = bracket['final_four']['matchups'][0]
        matchup2 = bracket['final_four']['matchups'][1]
        
        team1 = matchup1['team1']
        team2 = matchup1['team2']
        winner1 = matchup1['winner']
        
        team3 = matchup2['team1']
        team4 = matchup2['team2']
        winner2 = matchup2['winner']
        
        if winner1:
            text += f"({team1['seed']}) {team1['name']} vs ({team2['seed']}) {team2['name']} → "
            text += f"Winner: ({winner1['seed']}) {winner1['name']}\n"
        else:
            text += f"({team1['seed']}) {team1['name']} vs ({team2['seed']}) {team2['name']} → No winner determined\n"
        
        if winner2:
            text += f"({team3['seed']}) {team3['name']} vs ({team4['seed']}) {team4['name']} → "
            text += f"Winner: ({winner2['seed']}) {winner2['name']}\n"
        else:
            text += f"({team3['seed']}) {team3['name']} vs ({team4['seed']}) {team4['name']} → No winner determined\n"
        
        text += "=" * 40 + "\n\n"
    
    # Championship
    if 'matchup' in bracket['championship']:
        text += "NATIONAL CHAMPIONSHIP\n"
        text += "-" * 40 + "\n"
        
        matchup = bracket['championship']['matchup']
        team1 = matchup['team1']
        team2 = matchup['team2']
        winner = matchup['winner']
        
        if winner:
            text += f"({team1['seed']}) {team1['name']} vs ({team2['seed']}) {team2['name']} → "
            text += f"Winner: ({winner['seed']}) {winner['name']}\n"
            
            text += "=" * 40 + "\n\n"
            
            text += f"NATIONAL CHAMPION: ({winner['seed']}) {winner['name']}\n"
        else:
            text += f"({team1['seed']}) {team1['name']} vs ({team2['seed']}) {team2['name']} → No winner determined\n"
            
            text += "=" * 40 + "\n\n"
            
            text += "NATIONAL CHAMPION: Not determined\n"
        text += "=" * 40 + "\n"
    
    return text

def save_teams_csv(bracket, output_dir):
    """
    Save a CSV with all tournament teams and their predicted finish
    
    Parameters:
    -----------
    bracket : dict
        Dictionary containing the bracket structure
    output_dir : str
        Directory to save the results to
    """
    # Collect all teams and their predicted finish
    teams = []
    
    # For each region, collect the teams and mark how far they made it
    for region_name, region in bracket['regions'].items():
        for team in region['rounds']['first_round']:
            team1 = team['team1']
            team2 = team['team2']
            
            # Create entry for team1
            team1_entry = {
                'TeamName': team1['name'],
                'Seed': team1['seed'],
                'Region': region_name,
                'ChampionshipPct': team1['championship_pct'],
                'SimilarityPct': team1['similarity_pct'],
                'PredictedExitRound': team1['predicted_exit_round']
            }
            
            # Create entry for team2
            team2_entry = {
                'TeamName': team2['name'],
                'Seed': team2['seed'],
                'Region': region_name,
                'ChampionshipPct': team2['championship_pct'],
                'SimilarityPct': team2['similarity_pct'],
                'PredictedExitRound': team2['predicted_exit_round']
            }
            
            # Set actual tournament results
            if team['winner']['name'] == team1['name']:
                team1_entry['ActualFinish'] = 'Second Round or Better'
                team2_entry['ActualFinish'] = 'First Round'
            else:
                team1_entry['ActualFinish'] = 'First Round'
                team2_entry['ActualFinish'] = 'Second Round or Better'
            
            teams.append(team1_entry)
            teams.append(team2_entry)
    
    # Update teams that made it to the second round
    for region_name, region in bracket['regions'].items():
        for team in region['rounds']['second_round']:
            if team['winner']['name'] == team['team1']['name']:
                # Update team1 to Sweet 16
                for t in teams:
                    if t['TeamName'] == team['team1']['name']:
                        t['ActualFinish'] = 'Sweet 16 or Better'
                # Update team2 to Second Round
                for t in teams:
                    if t['TeamName'] == team['team2']['name']:
                        t['ActualFinish'] = 'Second Round'
            else:
                # Update team2 to Sweet 16
                for t in teams:
                    if t['TeamName'] == team['team2']['name']:
                        t['ActualFinish'] = 'Sweet 16 or Better'
                # Update team1 to Second Round
                for t in teams:
                    if t['TeamName'] == team['team1']['name']:
                        t['ActualFinish'] = 'Second Round'
    
    # Update teams that made it to the Sweet 16
    for region_name, region in bracket['regions'].items():
        for team in region['rounds']['sweet_16']:
            if team['winner']['name'] == team['team1']['name']:
                # Update team1 to Elite 8
                for t in teams:
                    if t['TeamName'] == team['team1']['name']:
                        t['ActualFinish'] = 'Elite 8 or Better'
                # Update team2 to Sweet 16
                for t in teams:
                    if t['TeamName'] == team['team2']['name']:
                        t['ActualFinish'] = 'Sweet 16'
            else:
                # Update team2 to Elite 8
                for t in teams:
                    if t['TeamName'] == team['team2']['name']:
                        t['ActualFinish'] = 'Elite 8 or Better'
                # Update team1 to Sweet 16
                for t in teams:
                    if t['TeamName'] == team['team1']['name']:
                        t['ActualFinish'] = 'Sweet 16'
    
    # Update teams that made it to the Elite 8
    for region_name, region in bracket['regions'].items():
        for team in region['rounds']['elite_8']:
            if team['winner']['name'] == team['team1']['name']:
                # Update team1 to Final Four
                for t in teams:
                    if t['TeamName'] == team['team1']['name']:
                        t['ActualFinish'] = 'Final Four or Better'
                # Update team2 to Elite 8
                for t in teams:
                    if t['TeamName'] == team['team2']['name']:
                        t['ActualFinish'] = 'Elite 8'
            else:
                # Update team2 to Final Four
                for t in teams:
                    if t['TeamName'] == team['team2']['name']:
                        t['ActualFinish'] = 'Final Four or Better'
                # Update team1 to Elite 8
                for t in teams:
                    if t['TeamName'] == team['team1']['name']:
                        t['ActualFinish'] = 'Elite 8'
    
    # Update teams that made it to the Final Four
    if 'matchups' in bracket['final_four']:
        for matchup in bracket['final_four']['matchups']:
            if matchup['winner']['name'] == matchup['team1']['name']:
                # Update team1 to Championship
                for t in teams:
                    if t['TeamName'] == matchup['team1']['name']:
                        t['ActualFinish'] = 'Championship Game or Better'
                # Update team2 to Final Four
                for t in teams:
                    if t['TeamName'] == matchup['team2']['name']:
                        t['ActualFinish'] = 'Final Four'
            else:
                # Update team2 to Championship
                for t in teams:
                    if t['TeamName'] == matchup['team2']['name']:
                        t['ActualFinish'] = 'Championship Game or Better'
                # Update team1 to Final Four
                for t in teams:
                    if t['TeamName'] == matchup['team1']['name']:
                        t['ActualFinish'] = 'Final Four'
    
    # Update championship teams
    if 'matchup' in bracket['championship']:
        matchup = bracket['championship']['matchup']
        if matchup['winner']['name'] == matchup['team1']['name']:
            # Update team1 to Champion
            for t in teams:
                if t['TeamName'] == matchup['team1']['name']:
                    t['ActualFinish'] = 'National Champion'
            # Update team2 to Championship
            for t in teams:
                if t['TeamName'] == matchup['team2']['name']:
                    t['ActualFinish'] = 'Championship Game'
        else:
            # Update team2 to Champion
            for t in teams:
                if t['TeamName'] == matchup['team2']['name']:
                    t['ActualFinish'] = 'National Champion'
            # Update team1 to Championship
            for t in teams:
                if t['TeamName'] == matchup['team1']['name']:
                    t['ActualFinish'] = 'Championship Game'
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(teams)
    csv_file = os.path.join(output_dir, "tournament_teams_full_bracket.csv")
    df.to_csv(csv_file, index=False)

if __name__ == "__main__":
    generate_full_bracket() 