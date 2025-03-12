import os
import pandas as pd
import numpy as np
import json
from datetime import datetime

def main():
    """Generate a full NCAA tournament bracket based on model data."""
    print("Starting full bracket generation...")
    
    # Define paths
    base_path = os.path.dirname(os.path.abspath(__file__))
    champion_file = os.path.join(base_path, "models/champion_profile/model/all_teams_champion_profile.csv")
    exit_round_file = os.path.join(base_path, "models/exit_round/model/tournament_teams_predictions.csv")
    output_dir = os.path.join(base_path, "models/full_bracket/model")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    try:
        print(f"Loading champion profile data from {champion_file}")
        champion_df = pd.read_csv(champion_file)
        print(f"Loaded {len(champion_df)} teams with champion profile data")
        
        print(f"Loading exit round predictions from {exit_round_file}")
        exit_round_df = pd.read_csv(exit_round_file)
        print(f"Loaded {len(exit_round_df)} teams with exit round predictions")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Merge datasets on team name
    try:
        # Ensure team names are consistent
        champion_df['TeamName'] = champion_df['TeamName'].str.strip()
        exit_round_df['TeamName'] = exit_round_df['TeamName'].str.strip()
        
        # Merge datasets
        combined_df = pd.merge(
            champion_df, 
            exit_round_df, 
            on='TeamName', 
            how='outer',
            suffixes=('_Champion', '_Exit')
        )
        
        print(f"Combined dataset has {len(combined_df)} teams")
        
        # Fill NaN values with 0 for numerical columns
        numerical_cols = combined_df.select_dtypes(include=[np.number]).columns
        combined_df[numerical_cols] = combined_df[numerical_cols].fillna(0)
        
        # Calculate combined score
        combined_df['CombinedScore'] = (
            combined_df['SimilarityPct'] * 0.4 + 
            combined_df['ChampionshipPct'] * 0.6
        )
        
        # Sort by combined score
        combined_df = combined_df.sort_values('CombinedScore', ascending=False)
        
        # Get top 64 teams
        top_teams = combined_df.head(64)
        print(f"Selected top {len(top_teams)} teams based on combined score")
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return
    
    # Create bracket
    try:
        bracket = create_bracket(top_teams)
        results = simulate_tournament(bracket, top_teams)
        save_results(results, output_dir)
        create_bracket_text(results, output_dir)
        print(f"Bracket generation complete. Results saved to {output_dir}")
    except Exception as e:
        print(f"Error generating bracket: {e}")
        return

def create_bracket(teams_df):
    """Create the initial bracket structure."""
    # Define regions
    regions = ['East', 'West', 'South', 'Midwest']
    bracket = {region: {} for region in regions}
    
    # Assign teams to regions and seeds
    for i, (idx, team) in enumerate(teams_df.iterrows()):
        region_idx = i % 4
        seed = (i // 4) + 1
        region = regions[region_idx]
        
        if seed not in bracket[region]:
            bracket[region][seed] = []
            
        bracket[region][seed].append({
            'name': team['TeamName'],
            'seed': seed,
            'similarity': team['SimilarityPct'],
            'championship_pct': team['ChampionshipPct'],
            'combined_score': team['CombinedScore'],
            'predicted_exit_round': team['PredictedExitRound']
        })
    
    # Create matchups for first round
    for region in regions:
        bracket[region]['matchups'] = {
            'round_64': [],
            'round_32': [],
            'sweet_16': [],
            'elite_8': [],
            'region_winner': None
        }
        
        # Standard NCAA tournament seeding pairs
        matchups = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
        
        for seed1, seed2 in matchups:
            if seed1 in bracket[region] and seed2 in bracket[region]:
                team1 = bracket[region][seed1][0]
                team2 = bracket[region][seed2][0]
                
                bracket[region]['matchups']['round_64'].append({
                    'team1': team1,
                    'team2': team2,
                    'winner': None
                })
    
    # Create Final Four and Championship structure
    bracket['final_four'] = {
        'matchups': [],
        'championship': {
            'team1': None,
            'team2': None,
            'winner': None
        }
    }
    
    return bracket

def simulate_tournament(bracket, teams_df):
    """Simulate the tournament and determine winners for each round."""
    regions = ['East', 'West', 'South', 'Midwest']
    
    # Simulate each region
    for region in regions:
        # Round of 64
        winners_64 = []
        for matchup in bracket[region]['matchups']['round_64']:
            winner = predict_winner(matchup['team1'], matchup['team2'])
            matchup['winner'] = winner
            winners_64.append(winner)
        
        # Round of 32
        for i in range(0, len(winners_64), 2):
            if i + 1 < len(winners_64):
                matchup = {
                    'team1': winners_64[i],
                    'team2': winners_64[i + 1],
                    'winner': None
                }
                winner = predict_winner(matchup['team1'], matchup['team2'])
                matchup['winner'] = winner
                bracket[region]['matchups']['round_32'].append(matchup)
        
        # Sweet 16
        winners_32 = [m['winner'] for m in bracket[region]['matchups']['round_32']]
        for i in range(0, len(winners_32), 2):
            if i + 1 < len(winners_32):
                matchup = {
                    'team1': winners_32[i],
                    'team2': winners_32[i + 1],
                    'winner': None
                }
                winner = predict_winner(matchup['team1'], matchup['team2'])
                matchup['winner'] = winner
                bracket[region]['matchups']['sweet_16'].append(matchup)
        
        # Elite 8
        winners_sweet16 = [m['winner'] for m in bracket[region]['matchups']['sweet_16']]
        if len(winners_sweet16) >= 2:
            matchup = {
                'team1': winners_sweet16[0],
                'team2': winners_sweet16[1],
                'winner': None
            }
            winner = predict_winner(matchup['team1'], matchup['team2'])
            matchup['winner'] = winner
            bracket[region]['matchups']['elite_8'].append(matchup)
            bracket[region]['matchups']['region_winner'] = winner
    
    # Final Four
    region_winners = []
    for region in regions:
        if bracket[region]['matchups']['region_winner']:
            region_winners.append(bracket[region]['matchups']['region_winner'])
    
    # Create Final Four matchups
    if len(region_winners) >= 4:
        # Semifinal 1
        semifinal1 = {
            'team1': region_winners[0],
            'team2': region_winners[1],
            'winner': None
        }
        semifinal1['winner'] = predict_winner(semifinal1['team1'], semifinal1['team2'])
        
        # Semifinal 2
        semifinal2 = {
            'team1': region_winners[2],
            'team2': region_winners[3],
            'winner': None
        }
        semifinal2['winner'] = predict_winner(semifinal2['team1'], semifinal2['team2'])
        
        bracket['final_four']['matchups'] = [semifinal1, semifinal2]
        
        # Championship
        if semifinal1['winner'] and semifinal2['winner']:
            bracket['final_four']['championship'] = {
                'team1': semifinal1['winner'],
                'team2': semifinal2['winner'],
                'winner': None
            }
            
            bracket['final_four']['championship']['winner'] = predict_winner(
                bracket['final_four']['championship']['team1'],
                bracket['final_four']['championship']['team2']
            )
    
    return bracket

def predict_winner(team1, team2):
    """Predict the winner of a matchup based on team metrics."""
    if not team1 or not team2:
        return None
    
    # Calculate scores for each team
    team1_score = (
        team1['combined_score'] * 0.5 +
        (7 - team1['predicted_exit_round']) * 0.3 +
        (1 / team1['seed']) * 0.2
    )
    
    team2_score = (
        team2['combined_score'] * 0.5 +
        (7 - team2['predicted_exit_round']) * 0.3 +
        (1 / team2['seed']) * 0.2
    )
    
    # Add some randomness
    team1_score += np.random.normal(0, 0.05)
    team2_score += np.random.normal(0, 0.05)
    
    # Return the winner
    return team1 if team1_score > team2_score else team2

def save_results(bracket, output_dir):
    """Save the tournament results to files."""
    # Save full bracket as JSON
    json_file = os.path.join(output_dir, "full_bracket.json")
    
    # Convert bracket to serializable format
    serializable_bracket = {}
    
    for key, value in bracket.items():
        if isinstance(value, dict):
            serializable_bracket[key] = {}
            for k, v in value.items():
                if isinstance(v, dict):
                    serializable_bracket[key][k] = {}
                    for k2, v2 in v.items():
                        if isinstance(v2, list):
                            serializable_bracket[key][k][k2] = []
                            for item in v2:
                                if isinstance(item, dict):
                                    serializable_item = {}
                                    for k3, v3 in item.items():
                                        if k3 == 'team1' or k3 == 'team2' or k3 == 'winner':
                                            if v3:
                                                serializable_item[k3] = v3['name']
                                            else:
                                                serializable_item[k3] = None
                                        else:
                                            serializable_item[k3] = v3
                                    serializable_bracket[key][k][k2].append(serializable_item)
                                else:
                                    serializable_bracket[key][k][k2].append(str(item))
                        elif k2 == 'region_winner' and v2:
                            serializable_bracket[key][k][k2] = v2['name']
                        else:
                            serializable_bracket[key][k][k2] = str(v2)
                elif isinstance(v, list):
                    serializable_bracket[key][k] = []
                    for item in v:
                        if isinstance(item, dict):
                            serializable_item = {}
                            for k2, v2 in item.items():
                                if k2 == 'team1' or k2 == 'team2' or k2 == 'winner':
                                    if v2:
                                        serializable_item[k2] = v2['name']
                                    else:
                                        serializable_item[k2] = None
                                else:
                                    serializable_item[k2] = v2
                            serializable_bracket[key][k].append(serializable_item)
                        else:
                            serializable_bracket[key][k].append(str(item))
                else:
                    serializable_bracket[key][k] = str(v)
        else:
            serializable_bracket[key] = str(value)
    
    with open(json_file, 'w') as f:
        json.dump(serializable_bracket, f, indent=2)
    
    # Save champion and Final Four teams to CSV
    results_file = os.path.join(output_dir, "tournament_results.csv")
    
    results = []
    
    # Champion
    if bracket['final_four']['championship']['winner']:
        champion = bracket['final_four']['championship']['winner']
        results.append({
            'TeamName': champion['name'],
            'Seed': champion['seed'],
            'Region': 'Champion',
            'SimilarityPct': champion['similarity'],
            'ChampionshipPct': champion['championship_pct'],
            'CombinedScore': champion['combined_score']
        })
    
    # Final Four
    for matchup in bracket['final_four']['matchups']:
        if matchup['team1']:
            team = matchup['team1']
            results.append({
                'TeamName': team['name'],
                'Seed': team['seed'],
                'Region': 'Final Four',
                'SimilarityPct': team['similarity'],
                'ChampionshipPct': team['championship_pct'],
                'CombinedScore': team['combined_score']
            })
        
        if matchup['team2']:
            team = matchup['team2']
            results.append({
                'TeamName': team['name'],
                'Seed': team['seed'],
                'Region': 'Final Four',
                'SimilarityPct': team['similarity'],
                'ChampionshipPct': team['championship_pct'],
                'CombinedScore': team['combined_score']
            })
    
    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)

def create_bracket_text(bracket, output_dir):
    """Create a text representation of the tournament bracket."""
    regions = ['East', 'West', 'South', 'Midwest']
    
    output = "NCAA TOURNAMENT BRACKET - FULL SIMULATION\n"
    output += "=" * 80 + "\n\n"
    output += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Regional brackets
    for region in regions:
        output += f"{region.upper()} REGION\n"
        output += "-" * 40 + "\n\n"
        
        # Round of 64
        output += "FIRST ROUND:\n"
        for i, matchup in enumerate(bracket[region]['matchups']['round_64']):
            team1 = matchup['team1']['name'] if matchup['team1'] else "TBD"
            seed1 = matchup['team1']['seed'] if matchup['team1'] else "?"
            team2 = matchup['team2']['name'] if matchup['team2'] else "TBD"
            seed2 = matchup['team2']['seed'] if matchup['team2'] else "?"
            winner = matchup['winner']['name'] if matchup['winner'] else "TBD"
            
            output += f"({seed1}) {team1} vs. ({seed2}) {team2} --> {winner}\n"
        
        output += "\nSECOND ROUND:\n"
        for i, matchup in enumerate(bracket[region]['matchups']['round_32']):
            team1 = matchup['team1']['name'] if matchup['team1'] else "TBD"
            seed1 = matchup['team1']['seed'] if matchup['team1'] else "?"
            team2 = matchup['team2']['name'] if matchup['team2'] else "TBD"
            seed2 = matchup['team2']['seed'] if matchup['team2'] else "?"
            winner = matchup['winner']['name'] if matchup['winner'] else "TBD"
            
            output += f"({seed1}) {team1} vs. ({seed2}) {team2} --> {winner}\n"
        
        output += "\nSWEET 16:\n"
        for i, matchup in enumerate(bracket[region]['matchups']['sweet_16']):
            team1 = matchup['team1']['name'] if matchup['team1'] else "TBD"
            seed1 = matchup['team1']['seed'] if matchup['team1'] else "?"
            team2 = matchup['team2']['name'] if matchup['team2'] else "TBD"
            seed2 = matchup['team2']['seed'] if matchup['team2'] else "?"
            winner = matchup['winner']['name'] if matchup['winner'] else "TBD"
            
            output += f"({seed1}) {team1} vs. ({seed2}) {team2} --> {winner}\n"
        
        output += "\nELITE 8:\n"
        for i, matchup in enumerate(bracket[region]['matchups']['elite_8']):
            team1 = matchup['team1']['name'] if matchup['team1'] else "TBD"
            seed1 = matchup['team1']['seed'] if matchup['team1'] else "?"
            team2 = matchup['team2']['name'] if matchup['team2'] else "TBD"
            seed2 = matchup['team2']['seed'] if matchup['team2'] else "?"
            winner = matchup['winner']['name'] if matchup['winner'] else "TBD"
            
            output += f"({seed1}) {team1} vs. ({seed2}) {team2} --> {winner}\n"
        
        region_winner = bracket[region]['matchups']['region_winner']['name'] if bracket[region]['matchups']['region_winner'] else "TBD"
        output += f"\nREGION WINNER: {region_winner}\n"
        
        output += "=" * 40 + "\n\n"
    
    # Final Four
    output += "FINAL FOUR\n"
    output += "-" * 40 + "\n\n"
    
    if len(bracket['final_four']['matchups']) >= 2:
        # Semifinal 1
        semifinal1 = bracket['final_four']['matchups'][0]
        team1 = semifinal1['team1']['name'] if semifinal1['team1'] else "TBD"
        seed1 = semifinal1['team1']['seed'] if semifinal1['team1'] else "?"
        team2 = semifinal1['team2']['name'] if semifinal1['team2'] else "TBD"
        seed2 = semifinal1['team2']['seed'] if semifinal1['team2'] else "?"
        winner1 = semifinal1['winner']['name'] if semifinal1['winner'] else "TBD"
        
        output += f"SEMIFINAL 1: ({seed1}) {team1} vs. ({seed2}) {team2} --> {winner1}\n\n"
        
        # Semifinal 2
        semifinal2 = bracket['final_four']['matchups'][1]
        team3 = semifinal2['team1']['name'] if semifinal2['team1'] else "TBD"
        seed3 = semifinal2['team1']['seed'] if semifinal2['team1'] else "?"
        team4 = semifinal2['team2']['name'] if semifinal2['team2'] else "TBD"
        seed4 = semifinal2['team2']['seed'] if semifinal2['team2'] else "?"
        winner2 = semifinal2['winner']['name'] if semifinal2['winner'] else "TBD"
        
        output += f"SEMIFINAL 2: ({seed3}) {team3} vs. ({seed4}) {team4} --> {winner2}\n\n"
        
        # Championship
        championship = bracket['final_four']['championship']
        champ_team1 = championship['team1']['name'] if championship['team1'] else "TBD"
        champ_seed1 = championship['team1']['seed'] if championship['team1'] else "?"
        champ_team2 = championship['team2']['name'] if championship['team2'] else "TBD"
        champ_seed2 = championship['team2']['seed'] if championship['team2'] else "?"
        champion = championship['winner']['name'] if championship['winner'] else "TBD"
        
        output += f"CHAMPIONSHIP: ({champ_seed1}) {champ_team1} vs. ({champ_seed2}) {champ_team2} --> {champion}\n\n"
        
        output += "=" * 40 + "\n\n"
        output += f"NCAA TOURNAMENT CHAMPION: {champion}\n"
    
    # Save to file
    output_file = os.path.join(output_dir, "full_bracket.txt")
    with open(output_file, 'w') as f:
        f.write(output)
    
    print(f"Bracket text representation saved to {output_file}")

if __name__ == "__main__":
    main() 