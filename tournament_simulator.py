import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class MarchMadnessSimulator:
    """
    NCAA Tournament Simulator that uses team strength predictions
    to simulate tournament outcomes while respecting the bracket structure.
    """
    
    def __init__(self, predictions_file='deep_learning_model/deep_learning_predictions.csv', 
                 output_dir='tournament_simulations', num_simulations=1000):
        """
        Initialize the tournament simulator.
        
        Parameters:
        -----------
        predictions_file : str
            Path to the CSV file containing team predictions from the deep learning model
        output_dir : str
            Directory to save simulation results
        num_simulations : int
            Number of tournament simulations to run
        """
        self.predictions_file = predictions_file
        self.output_dir = output_dir
        self.num_simulations = num_simulations
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Tournament structure constants
        self.regions = ['East', 'West', 'South', 'Midwest']
        self.seeds_per_region = 16
        
        # Exit round mapping
        self.exit_round_mapping = {
            0: 'Did Not Make Tournament',
            1: 'First Round',
            2: 'Second Round',
            3: 'Sweet 16',
            4: 'Elite 8',
            5: 'Final Four',
            6: 'Championship Game',
            7: 'National Champion'
        }
        
        # Load team predictions
        self.load_predictions()
        
    def load_predictions(self):
        """Load team predictions from the deep learning model"""
        print(f"Loading team predictions from {self.predictions_file}...")
        
        self.teams_df = pd.read_csv(self.predictions_file)
        
        # Ensure we have the necessary columns
        required_cols = ['TeamName', ' "seed"', 'AdjEM', 'PredictedExitRound']
        missing_cols = [col for col in required_cols if col not in self.teams_df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns in predictions file: {missing_cols}")
            
        # Sort teams by seed and AdjEM for consistent assignment
        self.teams_df = self.teams_df.sort_values(by=[' "seed"', 'AdjEM'], ascending=[True, False])
        
        # Take only the top 68 teams if we have more
        if len(self.teams_df) > 68:
            self.teams_df = self.teams_df.iloc[:68]
            
        print(f"Loaded {len(self.teams_df)} teams for tournament simulation")
        
    def create_bracket(self):
        """
        Create the initial NCAA tournament bracket with 68 teams.
        
        Returns:
        --------
        dict
            Dictionary containing the tournament bracket structure
        """
        # Create empty bracket
        bracket = {
            'First Four': [],
            'Regions': {region: [] for region in self.regions},
            'Final Four': [],
            'Championship': [],
            'Champion': None
        }
        
        # Get the teams
        teams = self.teams_df.copy()
        
        # Assign First Four teams (4 teams: 2 with seed 16, 2 with seed 11)
        first_four_16 = teams[teams[' "seed"'] == 16].iloc[-2:].copy()
        first_four_11 = teams[teams[' "seed"'] == 11].iloc[-2:].copy()
        
        # Remove these teams from the main pool
        teams = teams[~teams.index.isin(first_four_16.index)]
        teams = teams[~teams.index.isin(first_four_11.index)]
        
        # Add First Four matchups
        bracket['First Four'] = [
            {'teams': [first_four_16.iloc[0], first_four_16.iloc[1]], 'seed': 16, 'region': self.regions[0]},
            {'teams': [first_four_11.iloc[0], first_four_11.iloc[1]], 'seed': 11, 'region': self.regions[1]}
        ]
        
        # Distribute remaining teams to regions
        teams_by_seed = {}
        for seed in range(1, 17):
            seed_teams = teams[teams[' "seed"'] == seed].copy()
            teams_by_seed[seed] = seed_teams
        
        # Assign teams to regions
        for region_idx, region in enumerate(self.regions):
            region_teams = []
            
            # Assign seeds 1-16 to each region
            for seed in range(1, 17):
                if seed in teams_by_seed and len(teams_by_seed[seed]) > 0:
                    # For seeds 16 and 11, check if we need a First Four winner
                    if seed == 16 and region == self.regions[0]:
                        # This region gets the First Four winner for seed 16
                        continue
                    elif seed == 11 and region == self.regions[1]:
                        # This region gets the First Four winner for seed 11
                        continue
                    
                    # Take the next team for this seed
                    team = teams_by_seed[seed].iloc[0]
                    teams_by_seed[seed] = teams_by_seed[seed].iloc[1:]
                    
                    # Add to region
                    region_teams.append({'team': team, 'seed': seed})
            
            # Sort by seed
            region_teams = sorted(region_teams, key=lambda x: x['seed'])
            bracket['Regions'][region] = region_teams
            
        return bracket
    
    def simulate_game(self, team1, team2, round_num):
        """
        Simulate a single game between two teams.
        
        Parameters:
        -----------
        team1, team2 : pd.Series
            Team data including strength metrics
        round_num : int
            Current tournament round (1-6)
            
        Returns:
        --------
        pd.Series
            The winning team
        """
        # Extract team data
        team1_name = team1['TeamName']
        team2_name = team2['TeamName']
        team1_seed = int(team1[' "seed"'])
        team2_seed = int(team2[' "seed"'])
        team1_adjEM = team1['AdjEM']
        team2_adjEM = team2['AdjEM']
        team1_pred_exit = team1['PredictedExitRound']
        team2_pred_exit = team2['PredictedExitRound']
        
        # Calculate win probability based on adjusted efficiency margin and predicted exit round
        # Higher AdjEM and predicted exit round = stronger team
        
        # Base probability from AdjEM difference
        adjEM_diff = team1_adjEM - team2_adjEM
        base_prob = 1 / (1 + np.exp(-adjEM_diff/10))  # Logistic function
        
        # Adjust based on predicted exit round
        exit_round_diff = team1_pred_exit - team2_pred_exit
        exit_round_factor = 0.05 * exit_round_diff
        
        # Seed upset factor - lower seeds have a slight advantage in close games
        seed_diff = team2_seed - team1_seed
        seed_factor = 0.01 * seed_diff
        
        # Tournament round factor - higher seeds perform better in later rounds
        round_factor = 0.02 * round_num * (team2_seed - team1_seed) / 16
        
        # Calculate final probability
        win_prob = base_prob + exit_round_factor + seed_factor + round_factor
        
        # Clip probability to valid range
        win_prob = max(0.05, min(0.95, win_prob))
        
        # Simulate the game
        if random.random() < win_prob:
            return team1
        else:
            return team2
    
    def simulate_first_four(self, bracket):
        """Simulate the First Four games"""
        first_four_winners = []
        
        for matchup in bracket['First Four']:
            teams = matchup['teams']
            winner = self.simulate_game(teams[0], teams[1], 0)
            first_four_winners.append({
                'team': winner,
                'seed': matchup['seed'],
                'region': matchup['region']
            })
        
        # Add First Four winners to their respective regions
        for winner in first_four_winners:
            region = winner['region']
            seed = winner['seed']
            
            # Find the right position in the region
            region_teams = bracket['Regions'][region]
            
            # Add the First Four winner to the region
            region_teams.append({'team': winner['team'], 'seed': seed})
            
            # Sort by seed
            bracket['Regions'][region] = sorted(region_teams, key=lambda x: x['seed'])
        
        return bracket
    
    def simulate_round(self, teams, round_num):
        """
        Simulate a single round of the tournament.
        
        Parameters:
        -----------
        teams : list
            List of teams in the current round
        round_num : int
            Current tournament round (1-6)
            
        Returns:
        --------
        list
            List of winning teams
        """
        winners = []
        num_games = len(teams) // 2
        
        for i in range(num_games):
            team1 = teams[i*2]
            team2 = teams[i*2 + 1]
            winner = self.simulate_game(team1, team2, round_num)
            winners.append(winner)
            
        return winners
    
    def simulate_region(self, region_teams):
        """
        Simulate all rounds within a region.
        
        Parameters:
        -----------
        region_teams : list
            List of teams in the region
            
        Returns:
        --------
        pd.Series
            The region winner (Final Four team)
        """
        # First round - 1 vs 16, 8 vs 9, 5 vs 12, 4 vs 13, 6 vs 11, 3 vs 14, 7 vs 10, 2 vs 15
        matchups = [
            (0, 15), (7, 8), (4, 11), (3, 12), (5, 10), (2, 13), (6, 9), (1, 14)
        ]
        
        # Extract teams from region_teams
        teams = [item['team'] for item in region_teams]
        
        # Create first round matchups
        first_round = []
        for seed1, seed2 in matchups:
            first_round.append(teams[seed1])
            first_round.append(teams[seed2])
        
        # Simulate rounds
        round2_teams = self.simulate_round(first_round, 1)  # First round
        sweet16_teams = self.simulate_round(round2_teams, 2)  # Second round
        elite8_teams = self.simulate_round(sweet16_teams, 3)  # Sweet 16
        final4_team = self.simulate_round(elite8_teams, 4)[0]  # Elite 8
        
        return final4_team
    
    def simulate_tournament(self):
        """
        Simulate a complete NCAA tournament.
        
        Returns:
        --------
        dict
            Dictionary containing simulation results
        """
        # Create the bracket
        bracket = self.create_bracket()
        
        # Simulate First Four
        bracket = self.simulate_first_four(bracket)
        
        # Simulate each region to get Final Four teams
        final_four = []
        for region in self.regions:
            region_teams = bracket['Regions'][region]
            final_four_team = self.simulate_region(region_teams)
            final_four.append(final_four_team)
            
        # Simulate Final Four (national semifinals)
        championship_teams = self.simulate_round(final_four, 5)
        
        # Simulate Championship game
        champion = self.simulate_round(championship_teams, 6)[0]
        
        # Update bracket with results
        bracket['Final Four'] = final_four
        bracket['Championship'] = championship_teams
        bracket['Champion'] = champion
        
        return bracket
    
    def run_simulations(self):
        """
        Run multiple tournament simulations and aggregate results.
        
        Returns:
        --------
        dict
            Dictionary containing aggregated simulation results
        """
        print(f"Running {self.num_simulations} tournament simulations...")
        
        # Initialize counters
        results = {
            'champion_counts': defaultdict(int),
            'final_four_counts': defaultdict(int),
            'elite_eight_counts': defaultdict(int),
            'sweet_sixteen_counts': defaultdict(int),
            'round_of_32_counts': defaultdict(int),
            'exit_rounds': defaultdict(list)
        }
        
        # Run simulations
        for i in range(self.num_simulations):
            if (i+1) % 100 == 0:
                print(f"  Completed {i+1} simulations...")
                
            # Simulate tournament
            bracket = self.simulate_tournament()
            
            # Track champion
            champion_name = bracket['Champion']['TeamName']
            results['champion_counts'][champion_name] += 1
            
            # Track Final Four teams
            for team in bracket['Final Four']:
                team_name = team['TeamName']
                results['final_four_counts'][team_name] += 1
            
            # Track Championship teams
            for team in bracket['Championship']:
                team_name = team['TeamName']
                results['elite_eight_counts'][team_name] += 1
                
            # Track all teams' exit rounds in this simulation
            for region in self.regions:
                region_teams = bracket['Regions'][region]
                for team_data in region_teams:
                    team = team_data['team']
                    team_name = team['TeamName']
                    
                    # Determine exit round
                    if team_name == champion_name:
                        exit_round = 7  # Champion
                    elif team_name in [t['TeamName'] for t in bracket['Championship']]:
                        exit_round = 6  # Runner-up
                    elif team_name in [t['TeamName'] for t in bracket['Final Four']]:
                        exit_round = 5  # Final Four
                    else:
                        # Need to determine if team reached Elite 8, Sweet 16, etc.
                        # This would require tracking all rounds in the simulation
                        # For simplicity, we'll use a random value based on team strength
                        team_strength = team['AdjEM']
                        seed = int(team[' "seed"'])
                        
                        # Higher probability of deeper runs for stronger teams
                        p_sweet16 = max(0.1, min(0.9, (17 - seed) / 16))
                        p_elite8 = max(0.05, min(0.7, (17 - seed) / 20))
                        
                        if random.random() < p_elite8:
                            exit_round = 4  # Elite 8
                        elif random.random() < p_sweet16:
                            exit_round = 3  # Sweet 16
                        else:
                            exit_round = random.choices([1, 2], weights=[0.4, 0.6])[0]  # First or Second round
                    
                    results['exit_rounds'][team_name].append(exit_round)
        
        # Calculate average exit rounds
        avg_exit_rounds = {}
        for team_name, exit_rounds in results['exit_rounds'].items():
            avg_exit_rounds[team_name] = sum(exit_rounds) / len(exit_rounds)
        
        # Calculate probabilities
        champion_probs = {team: count/self.num_simulations*100 for team, count in results['champion_counts'].items()}
        final_four_probs = {team: count/self.num_simulations*100 for team, count in results['final_four_counts'].items()}
        
        # Sort by probability
        champion_probs = dict(sorted(champion_probs.items(), key=lambda x: x[1], reverse=True))
        final_four_probs = dict(sorted(final_four_probs.items(), key=lambda x: x[1], reverse=True))
        
        # Combine results
        simulation_results = {
            'champion_probs': champion_probs,
            'final_four_probs': final_four_probs,
            'avg_exit_rounds': avg_exit_rounds,
            'num_simulations': self.num_simulations
        }
        
        return simulation_results
    
    def save_results(self, results):
        """
        Save simulation results to CSV files and generate visualizations.
        
        Parameters:
        -----------
        results : dict
            Dictionary containing simulation results
        """
        # Create DataFrames
        champion_df = pd.DataFrame({
            'Team': list(results['champion_probs'].keys()),
            'Championship Probability': list(results['champion_probs'].values())
        })
        
        final_four_df = pd.DataFrame({
            'Team': list(results['final_four_probs'].keys()),
            'Final Four Probability': list(results['final_four_probs'].values())
        })
        
        # Merge with original team data
        team_data = self.teams_df[['TeamName', ' "seed"', 'AdjEM', 'PredictedExitRound']].copy()
        team_data.rename(columns={' "seed"': 'Seed'}, inplace=True)
        
        # Add average exit round
        team_data['SimulatedExitRound'] = team_data['TeamName'].map(results['avg_exit_rounds'])
        
        # Add championship and Final Four probabilities
        team_data['ChampionshipProbability'] = team_data['TeamName'].map(results['champion_probs']).fillna(0)
        team_data['FinalFourProbability'] = team_data['TeamName'].map(results['final_four_probs']).fillna(0)
        
        # Map exit rounds to names
        team_data['PredictedExit'] = team_data['PredictedExitRound'].map(self.exit_round_mapping)
        team_data['SimulatedExit'] = team_data['SimulatedExitRound'].apply(
            lambda x: self.exit_round_mapping.get(int(round(x)), f"Unknown ({x})") if pd.notna(x) else "Unknown"
        )
        
        # Sort by championship probability
        team_data = team_data.sort_values('ChampionshipProbability', ascending=False)
        
        # Save to CSV
        team_data.to_csv(os.path.join(self.output_dir, 'simulation_results.csv'), index=False)
        champion_df.to_csv(os.path.join(self.output_dir, 'championship_probabilities.csv'), index=False)
        final_four_df.to_csv(os.path.join(self.output_dir, 'final_four_probabilities.csv'), index=False)
        
        # Create visualizations
        self.create_visualizations(team_data)
        
        print(f"Simulation results saved to {self.output_dir}")
        
    def create_visualizations(self, team_data):
        """
        Create visualizations of simulation results.
        
        Parameters:
        -----------
        team_data : pd.DataFrame
            DataFrame containing team data and simulation results
        """
        # Set plot style
        plt.style.use('ggplot')
        
        # 1. Championship Probability Bar Chart (Top 10)
        plt.figure(figsize=(12, 6))
        top_champions = team_data.head(10)
        sns.barplot(x='ChampionshipProbability', y='TeamName', data=top_champions, 
                   hue='Seed', palette='viridis', dodge=False)
        plt.title('Top 10 Championship Contenders', fontsize=16)
        plt.xlabel('Championship Probability (%)', fontsize=12)
        plt.ylabel('Team', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'championship_probabilities.png'))
        
        # 2. Final Four Probability Bar Chart (Top 16)
        plt.figure(figsize=(12, 8))
        top_final_four = team_data.sort_values('FinalFourProbability', ascending=False).head(16)
        sns.barplot(x='FinalFourProbability', y='TeamName', data=top_final_four, 
                   hue='Seed', palette='viridis', dodge=False)
        plt.title('Top 16 Final Four Contenders', fontsize=16)
        plt.xlabel('Final Four Probability (%)', fontsize=12)
        plt.ylabel('Team', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'final_four_probabilities.png'))
        
        # 3. Predicted vs Simulated Exit Round Scatter Plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='PredictedExitRound', y='SimulatedExitRound', data=team_data, 
                       hue='Seed', size='AdjEM', sizes=(20, 200), palette='viridis')
        plt.title('Predicted vs Simulated Tournament Exit Rounds', fontsize=16)
        plt.xlabel('Predicted Exit Round', fontsize=12)
        plt.ylabel('Simulated Exit Round', fontsize=12)
        
        # Add team labels
        for _, row in team_data.iterrows():
            plt.annotate(row['TeamName'], 
                        (row['PredictedExitRound'], row['SimulatedExitRound']),
                        fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'exit_round_comparison.png'))
        
    def print_summary(self, results):
        """
        Print a summary of simulation results.
        
        Parameters:
        -----------
        results : dict
            Dictionary containing simulation results
        """
        print("\n" + "=" * 80)
        print(f"NCAA Tournament Simulation Results ({results['num_simulations']} simulations)")
        print("=" * 80)
        
        # Championship probabilities
        print("\nTop 10 Championship Contenders:")
        for i, (team, prob) in enumerate(list(results['champion_probs'].items())[:10], 1):
            # Get team seed
            team_data = self.teams_df[self.teams_df['TeamName'] == team].iloc[0]
            seed = int(team_data[' "seed"'])
            print(f"{i}. {team} (Seed {seed}): {prob:.1f}% championship probability")
        
        # Final Four probabilities
        print("\nTop 16 Final Four Contenders:")
        for i, (team, prob) in enumerate(list(results['final_four_probs'].items())[:16], 1):
            # Get team seed
            team_data = self.teams_df[self.teams_df['TeamName'] == team].iloc[0]
            seed = int(team_data[' "seed"'])
            print(f"{i}. {team} (Seed {seed}): {prob:.1f}% Final Four probability")
        
        # Compare with deep learning predictions
        print("\nComparison with Deep Learning Predictions:")
        print("Team                  | Seed | DL Predicted Exit | Simulation Avg Exit")
        print("-" * 70)
        
        # Get top 10 teams by championship probability
        top_teams = list(results['champion_probs'].keys())[:10]
        for team in top_teams:
            team_data = self.teams_df[self.teams_df['TeamName'] == team].iloc[0]
            seed = int(team_data[' "seed"'])
            dl_exit = int(team_data['PredictedExitRound'])
            dl_exit_name = self.exit_round_mapping[dl_exit]
            sim_exit = results['avg_exit_rounds'][team]
            sim_exit_name = self.exit_round_mapping[int(round(sim_exit))]
            
            print(f"{team:22} | {seed:4} | {dl_exit_name:16} | {sim_exit_name} ({sim_exit:.2f})")
        
        print("\nSimulation results saved to:", self.output_dir)
        
    def run(self):
        """Run the complete simulation pipeline"""
        # Run simulations
        results = self.run_simulations()
        
        # Save results
        self.save_results(results)
        
        # Print summary
        self.print_summary(results)
        
        return results


if __name__ == "__main__":
    # Run tournament simulations
    simulator = MarchMadnessSimulator(
        predictions_file='deep_learning_model/deep_learning_predictions.csv',
        output_dir='tournament_simulations',
        num_simulations=1000
    )
    simulator.run() 