import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import seaborn as sns

def generate_printable_bracket():
    """
    Generate a printable NCAA tournament bracket based on simulation results.
    """
    # Load simulation results
    results_file = 'tournament_simulations/simulation_results.csv'
    if not os.path.exists(results_file):
        print(f"Error: Simulation results file not found at {results_file}")
        return False
    
    results_df = pd.read_csv(results_file)
    
    # Sort teams by seed and championship probability
    results_df = results_df.sort_values(['Seed', 'ChampionshipProbability'], ascending=[True, False])
    
    # Define regions
    regions = ['East', 'West', 'South', 'Midwest']
    
    # Assign teams to regions
    teams_by_seed = {}
    for seed in range(1, 17):
        seed_teams = results_df[results_df['Seed'] == seed].copy()
        teams_by_seed[seed] = seed_teams
    
    # Create bracket structure
    bracket = {
        'First Four': [],
        'Regions': {region: [] for region in regions},
        'Final Four': [],
        'Championship': [],
        'Champion': None
    }
    
    # Assign First Four teams (4 teams: 2 with seed 16, 2 with seed 11)
    if 16 in teams_by_seed and len(teams_by_seed[16]) >= 2:
        first_four_16 = teams_by_seed[16].iloc[-2:].copy()
        teams_by_seed[16] = teams_by_seed[16].iloc[:-2]
        
        bracket['First Four'].append({
            'teams': [first_four_16.iloc[0], first_four_16.iloc[1]],
            'seed': 16,
            'region': regions[0]
        })
    
    if 11 in teams_by_seed and len(teams_by_seed[11]) >= 2:
        first_four_11 = teams_by_seed[11].iloc[-2:].copy()
        teams_by_seed[11] = teams_by_seed[11].iloc[:-2]
        
        bracket['First Four'].append({
            'teams': [first_four_11.iloc[0], first_four_11.iloc[1]],
            'seed': 11,
            'region': regions[1]
        })
    
    # Assign teams to regions
    for region_idx, region in enumerate(regions):
        region_teams = []
        
        # Assign seeds 1-16 to each region
        for seed in range(1, 17):
            if seed in teams_by_seed and len(teams_by_seed[seed]) > 0:
                # For seeds 16 and 11, check if we need a First Four placeholder
                if seed == 16 and region == regions[0]:
                    # This region gets the First Four winner for seed 16
                    region_teams.append({
                        'team': pd.Series({'TeamName': 'First Four Winner', 'Seed': 16}),
                        'seed': 16
                    })
                    continue
                elif seed == 11 and region == regions[1]:
                    # This region gets the First Four winner for seed 11
                    region_teams.append({
                        'team': pd.Series({'TeamName': 'First Four Winner', 'Seed': 11}),
                        'seed': 11
                    })
                    continue
                
                # Take the next team for this seed
                team = teams_by_seed[seed].iloc[0]
                teams_by_seed[seed] = teams_by_seed[seed].iloc[1:]
                
                # Add to region
                region_teams.append({'team': team, 'seed': seed})
        
        # Sort by seed
        region_teams = sorted(region_teams, key=lambda x: x['seed'])
        bracket['Regions'][region] = region_teams
    
    # Predict Final Four teams based on highest Final Four probabilities in each region
    for region, teams in bracket['Regions'].items():
        # Find team with highest Final Four probability
        best_team = max(teams, key=lambda x: x['team']['FinalFourProbability'] if 'FinalFourProbability' in x['team'] else 0)
        bracket['Final Four'].append(best_team['team'])
    
    # Predict Championship teams (top 2 from Final Four by championship probability)
    championship_teams = sorted(bracket['Final Four'], 
                               key=lambda x: x['ChampionshipProbability'] if 'ChampionshipProbability' in x else 0, 
                               reverse=True)[:2]
    bracket['Championship'] = championship_teams
    
    # Predict Champion (highest championship probability)
    if len(championship_teams) > 0:
        champion = max(championship_teams, 
                      key=lambda x: x['ChampionshipProbability'] if 'ChampionshipProbability' in x else 0)
        bracket['Champion'] = champion
    
    # Generate bracket visualization
    generate_bracket_image(bracket)
    
    # Generate text bracket
    generate_text_bracket(bracket)
    
    return True

def generate_bracket_image(bracket):
    """
    Generate a visual representation of the tournament bracket.
    
    Parameters:
    -----------
    bracket : dict
        Dictionary containing the tournament bracket structure
    """
    # Set up the figure
    plt.figure(figsize=(20, 15))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Set background color
    ax = plt.gca()
    ax.set_facecolor('#f8f8f8')
    
    # Title
    plt.title('NCAA Tournament Bracket Prediction', fontsize=24, pad=20)
    
    # Define regions and their positions
    regions = ['East', 'West', 'South', 'Midwest']
    region_positions = {
        'East': (0.05, 0.05, 0.4, 0.4),
        'West': (0.55, 0.05, 0.4, 0.4),
        'South': (0.05, 0.55, 0.4, 0.4),
        'Midwest': (0.55, 0.55, 0.4, 0.4)
    }
    
    # Draw regions
    for region, pos in region_positions.items():
        x, y, width, height = pos
        rect = Rectangle((x, y), width, height, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Region title
        plt.text(x + width/2, y + height - 0.02, region, fontsize=18, 
                ha='center', va='top', fontweight='bold')
        
        # Draw teams
        teams = bracket['Regions'][region]
        num_teams = len(teams)
        
        for i, team_data in enumerate(teams):
            team = team_data['team']
            seed = team_data['seed']
            
            # Team position
            team_x = x + 0.05
            team_y = y + height - 0.05 - (i+1) * (height-0.1) / (num_teams+1)
            
            # Team text
            if isinstance(team, pd.Series):
                team_name = team['TeamName']
                team_text = f"{seed}. {team_name}"
                
                # Add probability if available
                if 'FinalFourProbability' in team:
                    ff_prob = team['FinalFourProbability']
                    team_text += f" ({ff_prob:.1f}%)"
            else:
                team_text = f"{seed}. {team}"
            
            plt.text(team_x, team_y, team_text, fontsize=10, ha='left', va='center')
    
    # Draw Final Four
    ff_x, ff_y = 0.5, 0.5
    ff_radius = 0.1
    circle = plt.Circle((ff_x, ff_y), ff_radius, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    plt.text(ff_x, ff_y + ff_radius + 0.02, 'Final Four', fontsize=18, 
            ha='center', va='bottom', fontweight='bold')
    
    # Draw Final Four teams
    for i, team in enumerate(bracket['Final Four']):
        angle = i * np.pi/2 + np.pi/4
        team_x = ff_x + np.cos(angle) * ff_radius * 0.7
        team_y = ff_y + np.sin(angle) * ff_radius * 0.7
        
        if isinstance(team, pd.Series):
            team_name = team['TeamName']
            seed = int(team['Seed'])
            team_text = f"{seed}. {team_name}"
        else:
            team_text = team
            
        plt.text(team_x, team_y, team_text, fontsize=12, ha='center', va='center', 
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Draw Champion
    if bracket['Champion'] is not None:
        champion = bracket['Champion']
        
        if isinstance(champion, pd.Series):
            champion_name = champion['TeamName']
            seed = int(champion['Seed'])
            champion_text = f"Champion: {seed}. {champion_name}"
            
            if 'ChampionshipProbability' in champion:
                champ_prob = champion['ChampionshipProbability']
                champion_text += f" ({champ_prob:.1f}%)"
        else:
            champion_text = f"Champion: {champion}"
            
        plt.text(ff_x, ff_y, champion_text, fontsize=14, ha='center', va='center', 
                fontweight='bold', bbox=dict(facecolor='gold', alpha=0.7, boxstyle='round'))
    
    # Remove axes
    plt.axis('off')
    
    # Save the figure
    plt.savefig('tournament_simulations/bracket_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Bracket visualization saved to tournament_simulations/bracket_prediction.png")

def generate_text_bracket(bracket):
    """
    Generate a text representation of the tournament bracket.
    
    Parameters:
    -----------
    bracket : dict
        Dictionary containing the tournament bracket structure
    """
    # Create output file
    with open('tournament_simulations/bracket_prediction.txt', 'w') as f:
        f.write("NCAA TOURNAMENT BRACKET PREDICTION\n")
        f.write("==================================\n\n")
        
        # First Four
        f.write("FIRST FOUR\n")
        f.write("----------\n")
        for matchup in bracket['First Four']:
            seed = matchup['seed']
            region = matchup['region']
            teams = matchup['teams']
            
            team1 = teams[0]['TeamName'] if isinstance(teams[0], pd.Series) else teams[0]
            team2 = teams[1]['TeamName'] if isinstance(teams[1], pd.Series) else teams[1]
            
            f.write(f"({seed}) {team1} vs. ({seed}) {team2} → {region} Region\n")
        f.write("\n")
        
        # Regions
        for region, teams in bracket['Regions'].items():
            f.write(f"{region.upper()} REGION\n")
            f.write("-" * len(f"{region} REGION") + "\n")
            
            # First round matchups
            f.write("First Round:\n")
            for i in range(0, len(teams), 2):
                if i+1 < len(teams):
                    team1 = teams[i]['team']
                    team2 = teams[i+1]['team']
                    
                    seed1 = teams[i]['seed']
                    seed2 = teams[i+1]['seed']
                    
                    team1_name = team1['TeamName'] if isinstance(team1, pd.Series) else team1
                    team2_name = team2['TeamName'] if isinstance(team2, pd.Series) else team2
                    
                    f.write(f"({seed1}) {team1_name} vs. ({seed2}) {team2_name}\n")
            f.write("\n")
            
            # Regional winner
            for team in bracket['Final Four']:
                if isinstance(team, pd.Series) and team['TeamName'] in [t['team']['TeamName'] for t in teams if isinstance(t['team'], pd.Series)]:
                    seed = int(team['Seed'])
                    f.write(f"Regional Winner: ({seed}) {team['TeamName']}\n")
                    if 'FinalFourProbability' in team:
                        f.write(f"Final Four Probability: {team['FinalFourProbability']:.1f}%\n")
            f.write("\n")
        
        # Final Four
        f.write("FINAL FOUR\n")
        f.write("----------\n")
        if len(bracket['Final Four']) >= 4:
            f.write(f"({int(bracket['Final Four'][0]['Seed'])}) {bracket['Final Four'][0]['TeamName']} vs. ({int(bracket['Final Four'][1]['Seed'])}) {bracket['Final Four'][1]['TeamName']}\n")
            f.write(f"({int(bracket['Final Four'][2]['Seed'])}) {bracket['Final Four'][2]['TeamName']} vs. ({int(bracket['Final Four'][3]['Seed'])}) {bracket['Final Four'][3]['TeamName']}\n")
        f.write("\n")
        
        # Championship
        f.write("CHAMPIONSHIP GAME\n")
        f.write("----------------\n")
        if len(bracket['Championship']) >= 2:
            f.write(f"({int(bracket['Championship'][0]['Seed'])}) {bracket['Championship'][0]['TeamName']} vs. ({int(bracket['Championship'][1]['Seed'])}) {bracket['Championship'][1]['TeamName']}\n")
        f.write("\n")
        
        # Champion
        f.write("CHAMPION\n")
        f.write("--------\n")
        if bracket['Champion'] is not None:
            champion = bracket['Champion']
            seed = int(champion['Seed'])
            f.write(f"({seed}) {champion['TeamName']}\n")
            if 'ChampionshipProbability' in champion:
                f.write(f"Championship Probability: {champion['ChampionshipProbability']:.1f}%\n")
    
    print("Text bracket saved to tournament_simulations/bracket_prediction.txt")

if __name__ == "__main__":
    generate_printable_bracket() 