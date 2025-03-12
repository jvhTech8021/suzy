import os
import pandas as pd
import numpy as np
import json

def main():
    """
    Generate a simplified full bracket based on our model data
    """
    print("Generating simplified full NCAA tournament bracket...")
    
    # Setup paths
    base_path = os.path.dirname(os.path.abspath(__file__))
    champion_dir = os.path.join(base_path, "models/champion_profile/model")
    exit_dir = os.path.join(base_path, "models/exit_round/model")
    output_dir = os.path.join(base_path, "models/full_bracket/model")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    try:
        champion_file = os.path.join(champion_dir, "all_teams_champion_profile.csv")
        exit_file = os.path.join(exit_dir, "tournament_teams_predictions.csv")
        
        champion_data = pd.read_csv(champion_file)
        exit_data = pd.read_csv(exit_file)
        
        print(f"Loaded {len(champion_data)} teams from champion profile model")
        print(f"Loaded {len(exit_data)} teams from exit round model")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create a combined dataset
    try:
        # Merge on team name
        combined = pd.merge(
            champion_data,
            exit_data,
            on='TeamName',
            how='outer',
            suffixes=('_ChampProfile', '_ExitRound')
        )
        
        # Create a combined score
        combined['CombinedScore'] = (
            0.5 * combined['SimilarityPct'] + 
            0.5 * (combined['PredictedExitRound'] / 7 * 100)  # Normalize to 0-100 scale
        )
        
        print(f"Created combined dataset with {len(combined)} teams")
    except Exception as e:
        print(f"Error creating combined dataset: {e}")
        return
    
    # Select top 64 teams
    top_teams = combined.sort_values('CombinedScore', ascending=False).head(64)
    
    # Create the tournament structure
    tournament = {
        'regions': ['East', 'West', 'South', 'Midwest'],
        'teams': []
    }
    
    # Assign seeds and regions
    for i, (idx, team) in enumerate(top_teams.iterrows()):
        region_idx = i % 4
        seed = (i // 4) + 1
        
        tournament['teams'].append({
            'name': team['TeamName'],
            'region': tournament['regions'][region_idx],
            'seed': seed,
            'similarity': team['SimilarityPct'],
            'exit_round': team['PredictedExitRound'],
            'combined_score': team['CombinedScore']
        })
    
    # Create the matchups for each round
    matchups = create_matchups(tournament)
    
    # Save the results
    save_results(tournament, matchups, output_dir)
    
    print("Simplified bracket generation completed!")

def create_matchups(tournament):
    """
    Create matchups for each round based on seeds
    """
    regions = tournament['regions']
    matchups = {region: {} for region in regions}
    
    # Create matchups for each region
    for region in regions:
        region_teams = [t for t in tournament['teams'] if t['region'] == region]
        region_teams.sort(key=lambda x: x['seed'])
        
        # First round: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
        first_round = []
        pairings = [(0, 15), (7, 8), (4, 11), (3, 12), (5, 10), (2, 13), (6, 9), (1, 14)]
        
        for pair in pairings:
            if len(region_teams) > max(pair[0], pair[1]):
                matchup = {
                    'team1': region_teams[pair[0]],
                    'team2': region_teams[pair[1]],
                    'winner': predict_winner(region_teams[pair[0]], region_teams[pair[1]])
                }
                first_round.append(matchup)
        
        matchups[region]['first_round'] = first_round
        
        # Second round
        winners = [m['winner'] for m in first_round]
        second_round = []
        
        for i in range(0, len(winners), 2):
            if i+1 < len(winners):
                matchup = {
                    'team1': winners[i],
                    'team2': winners[i+1],
                    'winner': predict_winner(winners[i], winners[i+1])
                }
                second_round.append(matchup)
        
        matchups[region]['second_round'] = second_round
        
        # Sweet 16
        winners = [m['winner'] for m in second_round]
        sweet_16 = []
        
        for i in range(0, len(winners), 2):
            if i+1 < len(winners):
                matchup = {
                    'team1': winners[i],
                    'team2': winners[i+1],
                    'winner': predict_winner(winners[i], winners[i+1])
                }
                sweet_16.append(matchup)
        
        matchups[region]['sweet_16'] = sweet_16
        
        # Elite 8
        winners = [m['winner'] for m in sweet_16]
        elite_8 = []
        
        if len(winners) >= 2:
            matchup = {
                'team1': winners[0],
                'team2': winners[1],
                'winner': predict_winner(winners[0], winners[1])
            }
            elite_8.append(matchup)
        
        matchups[region]['elite_8'] = elite_8
        
        # Region winner
        if elite_8:
            matchups[region]['winner'] = elite_8[0]['winner']
        else:
            matchups[region]['winner'] = None
    
    # Final Four
    final_four = []
    final_four_teams = [matchups[region]['winner'] for region in regions if matchups[region]['winner']]
    
    if len(final_four_teams) >= 4:
        # East vs West
        matchup1 = {
            'team1': final_four_teams[0],  # East
            'team2': final_four_teams[1],  # West
            'winner': predict_winner(final_four_teams[0], final_four_teams[1])
        }
        final_four.append(matchup1)
        
        # South vs Midwest
        matchup2 = {
            'team1': final_four_teams[2],  # South
            'team2': final_four_teams[3],  # Midwest
            'winner': predict_winner(final_four_teams[2], final_four_teams[3])
        }
        final_four.append(matchup2)
    
    matchups['final_four'] = final_four
    
    # Championship
    championship = []
    championship_teams = [m['winner'] for m in final_four]
    
    if len(championship_teams) >= 2:
        matchup = {
            'team1': championship_teams[0],
            'team2': championship_teams[1],
            'winner': predict_winner(championship_teams[0], championship_teams[1])
        }
        championship.append(matchup)
    
    matchups['championship'] = championship
    
    return matchups

def predict_winner(team1, team2):
    """
    Simple function to predict the winner between two teams
    """
    # Use combined score to determine winner
    if team1['combined_score'] > team2['combined_score']:
        return team1
    else:
        return team2

def save_results(tournament, matchups, output_dir):
    """
    Save the results to files
    """
    # Create a text representation of the bracket
    bracket_text = create_bracket_text(tournament, matchups)
    
    # Save the text file
    text_file = os.path.join(output_dir, "full_bracket.txt")
    with open(text_file, 'w') as f:
        f.write(bracket_text)
    
    # Save the JSON file
    json_file = os.path.join(output_dir, "full_bracket.json")
    with open(json_file, 'w') as f:
        json.dump({
            'tournament': tournament,
            'matchups': matchups
        }, f, default=lambda x: str(x) if isinstance(x, (np.integer, np.floating)) else x, indent=4)
    
    # Save the teams CSV
    teams_data = []
    for team in tournament['teams']:
        team_data = team.copy()
        
        # Check how far the team went
        for region in matchups['regions']:
            if team['region'] == region:
                # First round losers
                for m in matchups[region]['first_round']:
                    if (m['team1']['name'] == team['name'] or m['team2']['name'] == team['name']) and m['winner']['name'] != team['name']:
                        team_data['finish'] = 'First Round'
                
                # Second round losers
                for m in matchups[region]['second_round']:
                    if (m['team1']['name'] == team['name'] or m['team2']['name'] == team['name']) and m['winner']['name'] != team['name']:
                        team_data['finish'] = 'Second Round'
                
                # Sweet 16 losers
                for m in matchups[region]['sweet_16']:
                    if (m['team1']['name'] == team['name'] or m['team2']['name'] == team['name']) and m['winner']['name'] != team['name']:
                        team_data['finish'] = 'Sweet 16'
                
                # Elite 8 losers
                for m in matchups[region]['elite_8']:
                    if (m['team1']['name'] == team['name'] or m['team2']['name'] == team['name']) and m['winner']['name'] != team['name']:
                        team_data['finish'] = 'Elite 8'
                
                # Final Four losers
                for m in matchups['final_four']:
                    if (m['team1']['name'] == team['name'] or m['team2']['name'] == team['name']) and m['winner']['name'] != team['name']:
                        team_data['finish'] = 'Final Four'
                
                # Championship loser
                for m in matchups['championship']:
                    if (m['team1']['name'] == team['name'] or m['team2']['name'] == team['name']) and m['winner']['name'] != team['name']:
                        team_data['finish'] = 'Championship Game'
                
                # Champion
                for m in matchups['championship']:
                    if m['winner']['name'] == team['name']:
                        team_data['finish'] = 'National Champion'
        
        teams_data.append(team_data)
    
    # Convert to DataFrame and save
    teams_df = pd.DataFrame(teams_data)
    csv_file = os.path.join(output_dir, "tournament_teams_full_bracket.csv")
    teams_df.to_csv(csv_file, index=False)
    
    print(f"Results saved to {output_dir}")

def create_bracket_text(tournament, matchups):
    """
    Create a text representation of the bracket
    """
    text = "NCAA TOURNAMENT BRACKET PREDICTION\n"
    text += "=" * 80 + "\n\n"
    
    # Print each region
    for region in tournament['regions']:
        text += f"{region.upper()} REGION\n"
        text += "-" * 40 + "\n"
        
        # First round
        text += "First Round:\n"
        for matchup in matchups[region]['first_round']:
            team1 = matchup['team1']
            team2 = matchup['team2']
            winner = matchup['winner']
            text += f"({team1['seed']}) {team1['name']} vs ({team2['seed']}) {team2['name']} → "
            text += f"Winner: ({winner['seed']}) {winner['name']}\n"
        
        # Second round
        text += "\nSecond Round:\n"
        for matchup in matchups[region]['second_round']:
            team1 = matchup['team1']
            team2 = matchup['team2']
            winner = matchup['winner']
            text += f"({team1['seed']}) {team1['name']} vs ({team2['seed']}) {team2['name']} → "
            text += f"Winner: ({winner['seed']}) {winner['name']}\n"
        
        # Sweet 16
        text += "\nSweet 16:\n"
        for matchup in matchups[region]['sweet_16']:
            team1 = matchup['team1']
            team2 = matchup['team2']
            winner = matchup['winner']
            text += f"({team1['seed']}) {team1['name']} vs ({team2['seed']}) {team2['name']} → "
            text += f"Winner: ({winner['seed']}) {winner['name']}\n"
        
        # Elite 8
        text += "\nElite 8:\n"
        for matchup in matchups[region]['elite_8']:
            team1 = matchup['team1']
            team2 = matchup['team2']
            winner = matchup['winner']
            text += f"({team1['seed']}) {team1['name']} vs ({team2['seed']}) {team2['name']} → "
            text += f"Winner: ({winner['seed']}) {winner['name']}\n"
        
        # Region winner
        text += f"\nRegion Winner: ({matchups[region]['winner']['seed']}) {matchups[region]['winner']['name']}\n"
        text += "=" * 40 + "\n\n"
    
    # Final Four
    text += "FINAL FOUR\n"
    text += "-" * 40 + "\n"
    
    for matchup in matchups['final_four']:
        team1 = matchup['team1']
        team2 = matchup['team2']
        winner = matchup['winner']
        text += f"({team1['seed']}) {team1['name']} ({team1['region']}) vs ({team2['seed']}) {team2['name']} ({team2['region']}) → "
        text += f"Winner: ({winner['seed']}) {winner['name']} ({winner['region']})\n"
    
    text += "=" * 40 + "\n\n"
    
    # Championship
    text += "NATIONAL CHAMPIONSHIP\n"
    text += "-" * 40 + "\n"
    
    for matchup in matchups['championship']:
        team1 = matchup['team1']
        team2 = matchup['team2']
        winner = matchup['winner']
        text += f"({team1['seed']}) {team1['name']} ({team1['region']}) vs ({team2['seed']}) {team2['name']} ({team2['region']}) → "
        text += f"Winner: ({winner['seed']}) {winner['name']} ({winner['region']})\n"
    
    text += "=" * 40 + "\n\n"
    
    if matchups['championship']:
        champion = matchups['championship'][0]['winner']
        text += f"NATIONAL CHAMPION: ({champion['seed']}) {champion['name']} ({champion['region']})\n"
    else:
        text += "NATIONAL CHAMPION: Not determined\n"
    
    text += "=" * 40 + "\n"
    
    return text

if __name__ == "__main__":
    main() 