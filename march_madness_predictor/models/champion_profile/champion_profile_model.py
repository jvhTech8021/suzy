import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

class ChampionProfilePredictor:
    """
    Model to identify teams that most closely resemble the statistical profile of 
    historical NCAA champions.
    """
    
    def __init__(self, model_save_path='model'):
        self.model_save_path = model_save_path
        
        # Historical champion profile (2009-2024 average)
        self.champion_profile = {
            'AdjEM': 28.72,
            'RankAdjEM': 5.4,
            'AdjOE': 120.6,
            'RankAdjOE': 4.8,  # Average rank of champion offensive efficiency
            'AdjDE': 91.8,
            'RankAdjDE': 11.2  # Average rank of champion defensive efficiency
        }
        
        # Success probabilities based on similarity rank from historical analysis
        self.champion_probs = {
            1: 14.3, 2: 7.1, 3: 7.1, 4: 0.0, 5: 14.3,
            6: 0.0, 7: 21.4, 8: 0.0, 9: 21.4, 10: 0.0
        }
        
        # Final Four probabilities based on similarity rank
        self.final_four_probs = {
            1: 14.3, 2: 14.3, 3: 35.7, 4: 0.0, 5: 28.6,
            6: 7.1, 7: 28.6, 8: 28.6, 9: 42.9, 10: 14.3
        }
        
        # Create directories for model artifacts if they don't exist
        os.makedirs(self.model_save_path, exist_ok=True)
    
    def calculate_similarity(self, team_stats, level_profile):
        """
        Calculate similarity of a team to the champion profile with increased emphasis on rankings
        """
        # Calculate differences for absolute values
        em_diff = (team_stats['AdjEM'] - level_profile['AdjEM'])**2
        oe_diff = (team_stats['AdjOE'] - level_profile['AdjOE'])**2
        de_diff = (team_stats['AdjDE'] - level_profile['AdjDE'])**2
        
        # Calculate differences for rankings (with higher weight)
        em_rank_diff = (team_stats['RankAdjEM'] - level_profile['RankAdjEM'])**2
        
        # Check if we have the ranking for offensive and defensive efficiency
        oe_rank_diff = 0
        de_rank_diff = 0
        
        if 'RankAdjOE' in team_stats:
            oe_rank_diff = (team_stats['RankAdjOE'] - level_profile['RankAdjOE'])**2
        
        if 'RankAdjDE' in team_stats:
            de_rank_diff = (team_stats['RankAdjDE'] - level_profile['RankAdjDE'])**2
        
        # Weight the differences - increased weight on rankings
        # Values: em_diff/100 + oe_diff/100 + de_diff/100 = normalized value differences (lower weight)
        # Rankings: em_rank_diff*2 + oe_rank_diff*1.5 + de_rank_diff*1.5 = ranking differences (higher weight)
        weighted_diff = np.sqrt(
            # Value differences (35% weight instead of 20%)
            (em_diff/100 + oe_diff/100 + de_diff/100) * 0.35 +
            # Ranking differences (65% weight instead of 80%)
            (em_rank_diff*2 + oe_rank_diff*1.5 + de_rank_diff*1.5) * 0.65
        )
        
        # Calculate similarity score (0-100 scale)
        similarity = max(0, 100 - (weighted_diff * 8))
        
        # Calculate percentage match for each component
        # Value matches (absolute values)
        em_match = 100 - min(100, abs(team_stats['AdjEM'] - level_profile['AdjEM']) / level_profile['AdjEM'] * 100)
        oe_match = 100 - min(100, abs(team_stats['AdjOE'] - level_profile['AdjOE']) / level_profile['AdjOE'] * 100)
        de_match = 100 - min(100, abs(team_stats['AdjDE'] - level_profile['AdjDE']) / level_profile['AdjDE'] * 100)
        
        # Rank matches
        rank_match = 100 - min(100, abs(team_stats['RankAdjEM'] - level_profile['RankAdjEM']) / 20 * 100)
        
        # Add rank matches for OE and DE if available
        oe_rank_match = 0
        de_rank_match = 0
        
        if 'RankAdjOE' in team_stats:
            oe_rank_match = 100 - min(100, abs(team_stats['RankAdjOE'] - level_profile['RankAdjOE']) / 20 * 100)
        
        if 'RankAdjDE' in team_stats:
            de_rank_match = 100 - min(100, abs(team_stats['RankAdjDE'] - level_profile['RankAdjDE']) / 20 * 100)
        
        # Overall rank match is the average of the available rank matches
        available_rank_matches = [rank_match]
        if oe_rank_match > 0:
            available_rank_matches.append(oe_rank_match)
        if de_rank_match > 0:
            available_rank_matches.append(de_rank_match)
        
        overall_rank_match = sum(available_rank_matches) / len(available_rank_matches)
        
        return {
            'Similarity': similarity,
            'WeightedDiff': weighted_diff,
            'EM_Match': em_match,
            'Rank_Match': overall_rank_match,
            'OE_Match': oe_match,
            'DE_Match': de_match,
            'OE_Rank_Match': oe_rank_match,
            'DE_Rank_Match': de_rank_match
        }
    
    def assess_team(self, team_stats):
        """
        Provide a qualitative assessment of a team compared to champion profile
        """
        assessment = []
        
        # Assess overall efficiency (AdjEM)
        if abs(team_stats['AdjEM'] - self.champion_profile['AdjEM']) < 3:
            assessment.append("Nearly perfect efficiency margin")
        elif team_stats['AdjEM'] > self.champion_profile['AdjEM'] + 5:
            assessment.append("Significantly stronger overall efficiency than typical champions")
        elif team_stats['AdjEM'] > self.champion_profile['AdjEM']:
            assessment.append("Stronger overall efficiency than typical champions")
        elif team_stats['AdjEM'] < self.champion_profile['AdjEM'] - 5:
            assessment.append("Significantly weaker overall efficiency than typical champions")
        else:
            assessment.append("Slightly weaker overall efficiency than typical champions")
        
        # Assess national ranking
        if abs(team_stats['RankAdjEM'] - self.champion_profile['RankAdjEM']) <= 2:
            assessment.append("Ideal national ranking (top 5-7)")
        elif team_stats['RankAdjEM'] <= 10:
            assessment.append("Strong national ranking (top 10)")
        elif team_stats['RankAdjEM'] <= 15:
            assessment.append("Good national ranking (top 15)")
        else:
            assessment.append("Ranking too low for typical champion")
        
        # Assess offensive efficiency
        if abs(team_stats['AdjOE'] - self.champion_profile['AdjOE']) < 5:
            assessment.append("Typical champion-level offense")
        elif team_stats['AdjOE'] > self.champion_profile['AdjOE'] + 5:
            assessment.append("Elite offense (better than typical champions)")
        elif team_stats['AdjOE'] < self.champion_profile['AdjOE'] - 5:
            assessment.append("Offense below champion standards")
        
        # Assess defensive efficiency
        if abs(team_stats['AdjDE'] - self.champion_profile['AdjDE']) < 3:
            assessment.append("Typical champion-level defense")
        elif team_stats['AdjDE'] < self.champion_profile['AdjDE'] - 3:
            assessment.append("Elite defense (better than typical champions)")
        elif team_stats['AdjDE'] > self.champion_profile['AdjDE'] + 5:
            assessment.append("Defense significantly below champion standards")
        else:
            assessment.append("Defense slightly below champion standards")
        
        return "; ".join(assessment)
    
    def load_current_data(self, data_path):
        """
        Load and clean current season KenPom data
        """
        df = pd.read_csv(data_path)
        
        # Clean data
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'TeamName':
                df[col] = df[col].str.replace('"', '').astype(float)
            elif col == 'TeamName':
                df[col] = df[col].str.replace('"', '')
        
        return df
    
    def predict_champion_similarity(self, current_data):
        """
        Calculate champion profile similarity for all teams
        """
        all_teams = []
        
        for _, row in current_data.iterrows():
            team_stats = {
                'AdjEM': row['AdjEM'],
                'RankAdjEM': row['RankAdjEM'],
                'AdjOE': row['AdjOE'],
                'AdjDE': row['AdjDE']
            }
            
            # Add rankings for OE and DE if available
            if 'RankAdjOE' in row:
                team_stats['RankAdjOE'] = row['RankAdjOE']
            
            if 'RankAdjDE' in row:
                team_stats['RankAdjDE'] = row['RankAdjDE']
            
            similarity_metrics = self.calculate_similarity(team_stats, self.champion_profile)
            assessment = self.assess_team(team_stats)
            
            team_data = {
                'TeamName': row['TeamName'],
                'Similarity': similarity_metrics['Similarity'],
                'AdjEM': row['AdjEM'],
                'RankAdjEM': row['RankAdjEM'],
                'AdjOE': row['AdjOE'],
                'AdjDE': row['AdjDE'],
                'EM_Match': similarity_metrics['EM_Match'],
                'Rank_Match': similarity_metrics['Rank_Match'],
                'OE_Match': similarity_metrics['OE_Match'],
                'DE_Match': similarity_metrics['DE_Match'],
                'Assessment': assessment
            }
            
            # Add OE and DE rank matches if available
            if similarity_metrics['OE_Rank_Match'] > 0:
                team_data['OE_Rank_Match'] = similarity_metrics['OE_Rank_Match']
            
            if similarity_metrics['DE_Rank_Match'] > 0:
                team_data['DE_Rank_Match'] = similarity_metrics['DE_Rank_Match']
            
            all_teams.append(team_data)
        
        # Sort by similarity (higher is better)
        all_teams.sort(key=lambda x: x['Similarity'], reverse=True)
        
        # Create DataFrame
        teams_df = pd.DataFrame(all_teams)
        
        # Add rank
        teams_df['SimilarityRank'] = range(1, len(teams_df) + 1)
        
        # Add success probabilities
        teams_df['ChampionPct'] = teams_df['SimilarityRank'].apply(
            lambda x: self.champion_probs.get(x, 1.0) if x <= 30 else 0.5
        )
        
        teams_df['FinalFourPct'] = teams_df['SimilarityRank'].apply(
            lambda x: self.final_four_probs.get(x, 5.0) if x <= 30 else 2.0
        )
        
        # Add profile difference columns
        teams_df['EM_Diff'] = teams_df['AdjEM'] - self.champion_profile['AdjEM']
        teams_df['Rank_Diff'] = teams_df['RankAdjEM'] - self.champion_profile['RankAdjEM']
        teams_df['OE_Diff'] = teams_df['AdjOE'] - self.champion_profile['AdjOE']
        teams_df['DE_Diff'] = teams_df['AdjDE'] - self.champion_profile['AdjDE']
        
        # Calculate tournament potential score with increased weight on rankings
        teams_df['TournamentPotential'] = (
            teams_df['Similarity'] * 0.35 +            # Similarity to champion profile (35%)
            (100 - teams_df['RankAdjEM']) * 0.45 +     # National ranking importance further increased (45%)
            teams_df['AdjEM'] * 0.2                    # Adjusted efficiency margin (20%)
        ) / 50  # Scale to a more readable number
        
        # Format percentage columns
        teams_df['SimilarityPct'] = teams_df['Similarity']
        teams_df['EM_MatchPct'] = teams_df['EM_Match']
        teams_df['Rank_MatchPct'] = teams_df['Rank_Match']
        teams_df['OE_MatchPct'] = teams_df['OE_Match']
        teams_df['DE_MatchPct'] = teams_df['DE_Match']
        
        return teams_df
    
    def generate_visualizations(self, predictions):
        """
        Generate visualizations for the predictions
        """
        # Create a scatter plot comparing teams to champion profile
        plt.figure(figsize=(12, 8))
        
        # Plot all teams
        plt.scatter(predictions['AdjOE'], predictions['AdjDE'], 
                  s=predictions['AdjEM']*3, alpha=0.3, c='gray')
        
        # Highlight top 30 teams
        top30 = predictions.iloc[:30]
        plt.scatter(top30['AdjOE'], top30['AdjDE'], 
                  s=top30['AdjEM']*3, alpha=0.8, c='blue')
        
        # Highlight top 10 teams with labels
        for i, row in predictions.iloc[:10].iterrows():
            plt.annotate(row['TeamName'], 
                      (row['AdjOE'], row['AdjDE']),
                      xytext=(5, 5), textcoords='offset points')
        
        # Plot champion profile
        plt.scatter([self.champion_profile['AdjOE']], [self.champion_profile['AdjDE']], 
                  s=self.champion_profile['AdjEM']*3, c='gold', edgecolors='black', marker='*')
        plt.annotate('Champion Profile', 
                   (self.champion_profile['AdjOE'], self.champion_profile['AdjDE']),
                   xytext=(10, -10), textcoords='offset points',
                   bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
        
        # Configure the plot
        plt.axhline(y=self.champion_profile['AdjDE'], color='r', linestyle='--', alpha=0.3)
        plt.axvline(x=self.champion_profile['AdjOE'], color='r', linestyle='--', alpha=0.3)
        plt.title('Team Comparison to Champion Profile (2025)')
        plt.xlabel('Offensive Efficiency')
        plt.ylabel('Defensive Efficiency (Lower is Better)')
        plt.grid(True, alpha=0.3)
        
        # Reverse Y-axis since lower defensive numbers are better
        plt.gca().invert_yaxis()
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'champion_profile_comparison.png'))
        
        # Create a bar chart of similarity percentages for top 20 teams
        plt.figure(figsize=(14, 8))
        top20 = predictions.head(20).copy()
        plt.barh(top20['TeamName'][::-1], top20['SimilarityPct'][::-1], color='skyblue')
        plt.xlabel('Similarity to Champion Profile (%)')
        plt.title('Top 20 Teams by Similarity to Champion Profile')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'top20_similarity.png'))
        
        # Create visualizations for championship probabilities
        plt.figure(figsize=(14, 8))
        top15 = predictions.head(15).copy()
        plt.barh(top15['TeamName'][::-1], top15['ChampionPct'][::-1], color='gold')
        plt.xlabel('Championship Probability (%)')
        plt.title('Top 15 Teams by Championship Probability')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'championship_probabilities.png'))
    
    def save_predictions(self, predictions):
        """
        Save prediction results to CSV files
        """
        # Select columns for output
        output_columns = [
            'SimilarityRank', 'TeamName', 'SimilarityPct', 'ChampionPct', 'FinalFourPct', 'TournamentPotential',
            'AdjEM', 'EM_Diff', 'EM_MatchPct',
            'RankAdjEM', 'Rank_Diff', 'Rank_MatchPct',
            'AdjOE', 'OE_Diff', 'OE_MatchPct',
            'AdjDE', 'DE_Diff', 'DE_MatchPct',
            'Assessment'
        ]
        
        # Save all teams
        predictions[output_columns].to_csv(
            os.path.join(self.model_save_path, 'all_teams_champion_profile.csv'), 
            index=False
        )
        
        # Save top 30 teams
        predictions.head(30)[output_columns].to_csv(
            os.path.join(self.model_save_path, 'top30_champion_resemblers.csv'),
            index=False
        )
        
        # Save champion profile as JSON for easy access
        with open(os.path.join(self.model_save_path, 'champion_profile.json'), 'w') as f:
            json.dump(self.champion_profile, f, indent=4)
        
        # Save performance statistics
        with open(os.path.join(self.model_save_path, 'historical_performance.json'), 'w') as f:
            performance_data = {
                'champion_probs': self.champion_probs,
                'final_four_probs': self.final_four_probs
            }
            json.dump(performance_data, f, indent=4)
    
    def create_bracket_visualization(self, predictions):
        """
        Create a mock tournament bracket based on similarity rankings
        """
        top16 = predictions.head(16)
        
        # Create bracket structure
        bracket = {
            'EAST': {
                1: top16.iloc[0]['TeamName'],
                8: top16.iloc[7]['TeamName'],
                4: top16.iloc[3]['TeamName'],
                5: top16.iloc[4]['TeamName']
            },
            'WEST': {
                1: top16.iloc[1]['TeamName'],
                8: top16.iloc[6]['TeamName'],
                4: top16.iloc[2]['TeamName'],
                5: top16.iloc[5]['TeamName']
            },
            'SOUTH': {
                1: top16.iloc[8]['TeamName'],
                8: top16.iloc[15]['TeamName'],
                4: top16.iloc[11]['TeamName'],
                5: top16.iloc[12]['TeamName']
            },
            'MIDWEST': {
                1: top16.iloc[9]['TeamName'],
                8: top16.iloc[14]['TeamName'],
                4: top16.iloc[10]['TeamName'],
                5: top16.iloc[13]['TeamName']
            }
        }
        
        # Save bracket as JSON
        with open(os.path.join(self.model_save_path, 'similarity_bracket.json'), 'w') as f:
            json.dump(bracket, f, indent=4)
        
        # Generate a text visualization
        bracket_text = [
            "Mock Tournament Bracket Based on Champion Profile Similarity:",
            "=" * 80,
            "EAST                      WEST",
            f"1. {bracket['EAST'][1]:<20} 1. {bracket['WEST'][1]}",
            f"8. {bracket['EAST'][8]:<20} 8. {bracket['WEST'][8]}",
            f"4. {bracket['EAST'][4]:<20} 4. {bracket['WEST'][4]}",
            f"5. {bracket['EAST'][5]:<20} 5. {bracket['WEST'][5]}",
            "",
            "SOUTH                     MIDWEST",
            f"1. {bracket['SOUTH'][1]:<20} 1. {bracket['MIDWEST'][1]}",
            f"8. {bracket['SOUTH'][8]:<20} 8. {bracket['MIDWEST'][8]}",
            f"4. {bracket['SOUTH'][4]:<20} 4. {bracket['MIDWEST'][4]}",
            f"5. {bracket['SOUTH'][5]:<20} 5. {bracket['MIDWEST'][5]}"
        ]
        
        # Save bracket text visualization
        with open(os.path.join(self.model_save_path, 'similarity_bracket.txt'), 'w') as f:
            f.write('\n'.join(bracket_text))
        
        return bracket_text
    
    def analyze_tournament_success_levels(self, historical_data_dir, prediction_data):
        """
        Analyze historical teams by tournament success level and match current teams
        to historical teams at each level.
        
        Parameters:
        -----------
        historical_data_dir : str
            Directory containing historical KenPom data
        prediction_data : pd.DataFrame
            DataFrame containing current season predictions
            
        Returns:
        --------
        dict
            Dictionary containing results for each tournament round
        """
        print("\nAnalyzing teams by tournament round performance...")
        
        # Define tournament round levels
        round_levels = {
            7: "National Champions",
            6: "Championship Game",
            5: "Final Four",
            4: "Elite Eight",
            3: "Sweet Sixteen",
            2: "Round of 32",
            1: "Tournament Qualifiers"
        }
        
        # Store results for each round level
        results = {}
        
        # Process each round level
        for round_num, round_name in round_levels.items():
            print(f"Processing {round_name}...")
            
            # Create profiles for teams reaching each round
            teams_at_level = []
            team_count = 0
            
            # Collect team data from historical seasons
            for year in range(2009, 2025):
                # Skip 2020 (COVID year)
                if year == 2020:
                    continue
                    
                file_path = os.path.join(historical_data_dir, f'processed_{year}.csv')
                if not os.path.exists(file_path):
                    continue
                
                df = pd.read_csv(file_path)
                
                # Filter teams that reached this round or further
                if round_num == 1:
                    # For tournament qualifiers, include all teams that made the tournament
                    round_teams = df[df['TournamentExitRound'].notna() & (df['TournamentExitRound'] > 0)]
                else:
                    # For other rounds, include teams that reached this round or further
                    round_teams = df[df['TournamentExitRound'] >= round_num]
                
                for _, team in round_teams.iterrows():
                    team_data = {
                        'Season': year,
                        'TeamName': team['TeamName'],
                        'AdjEM': team['AdjEM'],
                        'RankAdjEM': team['RankAdjEM'],
                        'AdjOE': team['AdjOE'],
                        'AdjDE': team['AdjDE'],
                        'ExitRound': team['TournamentExitRound']
                    }
                    
                    # Add rankings for OE and DE if available
                    if 'RankAdjOE' in team:
                        team_data['RankAdjOE'] = team['RankAdjOE']
                    
                    if 'RankAdjDE' in team:
                        team_data['RankAdjDE'] = team['RankAdjDE']
                    
                    teams_at_level.append(team_data)
                    team_count += 1
            
            # Skip if no teams found
            if not teams_at_level:
                print(f"No historical teams found for {round_name}")
                continue
                
            # Create a profile for this round level
            level_profile = {
                'AdjEM': np.mean([t['AdjEM'] for t in teams_at_level]),
                'RankAdjEM': np.mean([t['RankAdjEM'] for t in teams_at_level]),
                'AdjOE': np.mean([t['AdjOE'] for t in teams_at_level]),
                'AdjDE': np.mean([t['AdjDE'] for t in teams_at_level])
            }
            
            # Add rankings if available
            oe_ranks = [t.get('RankAdjOE') for t in teams_at_level if 'RankAdjOE' in t]
            de_ranks = [t.get('RankAdjDE') for t in teams_at_level if 'RankAdjDE' in t]
            
            if oe_ranks:
                level_profile['RankAdjOE'] = np.mean(oe_ranks)
            
            if de_ranks:
                level_profile['RankAdjDE'] = np.mean(de_ranks)
            
            # Calculate similarity scores for current teams compared to this level
            current_team_matches = []
            
            for _, team in prediction_data.iterrows():
                team_stats = {
                    'AdjEM': team['AdjEM'],
                    'RankAdjEM': team['RankAdjEM'],
                    'AdjOE': team['AdjOE'],
                    'AdjDE': team['AdjDE']
                }
                
                # Add rankings if available
                if 'RankAdjOE' in team:
                    team_stats['RankAdjOE'] = team['RankAdjOE']
                
                if 'RankAdjDE' in team:
                    team_stats['RankAdjDE'] = team['RankAdjDE']
                
                # Calculate similarity to this level's profile
                similarity_metrics = self.calculate_similarity(team_stats, level_profile)
                
                # Find most similar historical teams at this level
                similar_historical_teams = []
                for hist_team in teams_at_level:
                    hist_team_stats = {
                        'AdjEM': hist_team['AdjEM'],
                        'RankAdjEM': hist_team['RankAdjEM'],
                        'AdjOE': hist_team['AdjOE'],
                        'AdjDE': hist_team['AdjDE']
                    }
                    
                    # Add rankings if available
                    if 'RankAdjOE' in hist_team:
                        hist_team_stats['RankAdjOE'] = hist_team['RankAdjOE']
                    
                    if 'RankAdjDE' in hist_team:
                        hist_team_stats['RankAdjDE'] = hist_team['RankAdjDE']
                    
                    # Calculate similarity between current team and historical team
                    hist_similarity = self.calculate_similarity(team_stats, hist_team_stats)
                    
                    similar_historical_teams.append({
                        'Season': hist_team['Season'],
                        'TeamName': hist_team['TeamName'],
                        'Similarity': hist_similarity['Similarity'],
                        'ExitRound': hist_team['ExitRound']
                    })
                
                # Sort by similarity and get top 5
                similar_historical_teams.sort(key=lambda x: x['Similarity'], reverse=True)
                top_similar_teams = similar_historical_teams[:5]
                
                current_team_matches.append({
                    'TeamName': team['TeamName'],
                    'Similarity': similarity_metrics['Similarity'],
                    'SimilarTeams': top_similar_teams
                })
            
            # Sort by similarity
            current_team_matches.sort(key=lambda x: x['Similarity'], reverse=True)
            
            # Store results
            results[round_num] = {
                'RoundName': round_name,
                'TeamCount': team_count,
                'Profile': level_profile,
                'CurrentTeams': current_team_matches
            }
            
            # Save the results
            round_filename = round_name.lower().replace(' ', '_')
            with open(os.path.join(self.model_save_path, f'{round_filename}_analysis.json'), 'w') as f:
                # Convert to serializable format
                serializable_results = {
                    'RoundName': results[round_num]['RoundName'],
                    'TeamCount': results[round_num]['TeamCount'],
                    'Profile': results[round_num]['Profile'],
                    'CurrentTeams': [{
                        'TeamName': t['TeamName'],
                        'Similarity': t['Similarity'],
                        'SimilarTeams': [{
                            'Season': st['Season'],
                            'TeamName': st['TeamName'],
                            'Similarity': st['Similarity'],
                            'ExitRound': int(st['ExitRound'])
                        } for st in t['SimilarTeams']]
                    } for t in results[round_num]['CurrentTeams'][:30]]  # Save top 30 teams
                }
                json.dump(serializable_results, f, indent=4)
        
        print(f"Tournament success level analysis completed for {len(results)} round levels")
        return results
    
    def run_full_pipeline(self, data_path):
        """
        Run the complete champion profile analysis pipeline
        """
        print("Loading current season data...")
        current_data = self.load_current_data(data_path)
        print(f"Loaded data for {len(current_data)} teams")
        
        print("\nCalculating champion profile similarity for all teams...")
        predictions = self.predict_champion_similarity(current_data)
        
        print("\nGenerating visualizations...")
        self.generate_visualizations(predictions)
        
        print("\nCreating mock tournament bracket...")
        bracket_text = self.create_bracket_visualization(predictions)
        print('\n'.join(bracket_text))
        
        print("\nSaving prediction results...")
        self.save_predictions(predictions)
        
        print("\nTop 10 teams by champion profile similarity:")
        top_10 = predictions.head(10)
        for i, (_, team) in enumerate(top_10.iterrows(), 1):
            print(f"{i}. {team['TeamName']}: {team['SimilarityPct']:.1f}% similarity, {team['ChampionPct']:.1f}% championship")
        
        print("\nChampion profile:", self.champion_profile)
        
        return predictions


if __name__ == "__main__":
    # Create model directory
    model_dir = "../../models/champion_profile/model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Run the pipeline
    predictor = ChampionProfilePredictor(model_save_path=model_dir)
    predictions = predictor.run_full_pipeline('../../../susan_kenpom/summary25.csv')
    
    print("\nPredictions saved to model directory")
    print("Pipeline completed successfully!") 