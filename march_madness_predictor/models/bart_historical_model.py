import os
import pandas as pd
import numpy as np
from collections import defaultdict
import json

class BartHistoricalModel:
    """
    A model that uses historical BART (Bart Torvik) data to improve game predictions
    by analyzing performance trends and historical matchups
    """
    
    def __init__(self, bart_dir=None):
        """
        Initialize the BART historical model
        
        Parameters:
        -----------
        bart_dir : str or None
            Directory containing BART data files. If None, tries to find the BART folder
        """
        self.bart_dir = bart_dir
        self.historical_data = {}  # Dictionary mapping years to DataFrames
        self.team_trends = {}      # Dictionary mapping teams to their historical trends
        self.matchup_history = {}  # Dictionary mapping team pairs to their historical matchups
        
        # Load the historical data
        self._load_historical_data()
        
        # Calculate team trends and matchup histories
        self._calculate_team_trends()
        
    def _load_historical_data(self):
        """Load historical BART data for all available years"""
        if self.bart_dir is None:
            # Try to find the BART folder
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.bart_dir = os.path.join(base_path, "BART")
        
        if not os.path.exists(self.bart_dir):
            print(f"Error: BART directory not found at {self.bart_dir}")
            return
        
        # Load data for all available years
        for year in range(2009, 2025):
            file_path = os.path.join(self.bart_dir, f"{year}_team_results.csv")
            if os.path.exists(file_path):
                try:
                    data = pd.read_csv(file_path)
                    
                    # Standardize team names
                    if 'team' in data.columns:
                        data['TeamName'] = data['team'].str.strip()
                    elif 'Team' in data.columns:
                        data['TeamName'] = data['Team'].str.strip()
                    
                    # Store the year in the DataFrame
                    data['Year'] = year
                    
                    self.historical_data[year] = data
                    print(f"Loaded BART data for {year} with {len(data)} teams")
                except Exception as e:
                    print(f"Error loading BART data for {year}: {str(e)}")
        
        print(f"Loaded BART data for {len(self.historical_data)} years")
            
    def _calculate_team_trends(self):
        """Calculate historical trends for all teams"""
        if not self.historical_data:
            print("No historical data available to calculate trends")
            return
        
        # Combine all years of data
        all_years_data = pd.concat(self.historical_data.values())
        
        # Group by team and calculate trends
        for team_name in all_years_data['TeamName'].unique():
            team_data = all_years_data[all_years_data['TeamName'] == team_name]
            
            # Only calculate trends if we have at least 3 years of data
            if len(team_data) >= 3:
                # Sort by year
                team_data = team_data.sort_values('Year')
                
                # Calculate trends for key metrics
                trends = {
                    'years_available': team_data['Year'].tolist(),
                    'barthag_trend': self._calculate_metric_trend(team_data, 'barthag'),
                    'adj_o_trend': self._calculate_metric_trend(team_data, 'adj_o'),
                    'adj_d_trend': self._calculate_metric_trend(team_data, 'adj_d'),
                    'adj_t_trend': self._calculate_metric_trend(team_data, 'adj_t'),
                    'wab_trend': self._calculate_metric_trend(team_data, 'WAB'),
                    'tournament_history': self._extract_tournament_history(team_data),
                    'avg_barthag': team_data['barthag'].mean() if 'barthag' in team_data.columns else None,
                    'avg_seed': team_data['seed'].mean() if 'seed' in team_data.columns else None,
                    'consistency': self._calculate_consistency(team_data)
                }
                
                self.team_trends[team_name] = trends
        
        print(f"Calculated trends for {len(self.team_trends)} teams")
        
    def _calculate_metric_trend(self, team_data, metric):
        """
        Calculate the trend for a specific metric over time
        
        Parameters:
        -----------
        team_data : pd.DataFrame
            DataFrame containing team data over multiple years
        metric : str
            Metric to calculate the trend for
            
        Returns:
        --------
        dict
            Dictionary with trend information
        """
        if metric not in team_data.columns:
            return None
        
        # Get values and years
        values = team_data[metric].values
        years = team_data['Year'].values
        
        # Calculate year-over-year changes
        changes = []
        for i in range(1, len(values)):
            changes.append(values[i] - values[i-1])
        
        # Calculate trend (linear regression)
        if len(years) >= 3:
            # Normalize years for better numerical stability
            normalized_years = years - years.min()
            
            # Use numpy's polyfit to fit a linear trend
            trend, _ = np.polyfit(normalized_years, values, 1)
        else:
            trend = None
        
        # Calculate recent average (last 3 years if available)
        recent_values = values[-3:] if len(values) >= 3 else values
        recent_avg = np.mean(recent_values)
        
        return {
            'values': values.tolist(),
            'years': years.tolist(),
            'changes': changes,
            'trend': trend,
            'recent_avg': recent_avg,
            'all_time_avg': np.mean(values)
        }
    
    def _extract_tournament_history(self, team_data):
        """
        Extract tournament history for a team
        
        Parameters:
        -----------
        team_data : pd.DataFrame
            DataFrame containing team data over multiple years
            
        Returns:
        --------
        dict
            Dictionary with tournament history
        """
        tournament_history = []
        
        # Check if we have seed and tournament data
        has_seed = 'seed' in team_data.columns
        
        if has_seed:
            for _, row in team_data.iterrows():
                year = row['Year']
                seed = row['seed'] if not pd.isna(row['seed']) else None
                
                # Determine if the team made the tournament
                made_tournament = seed is not None and seed <= 16
                
                if made_tournament:
                    tournament_history.append({
                        'year': year,
                        'seed': seed,
                        'barthag': row['barthag'] if 'barthag' in row else None,
                        'adj_o': row['adj_o'] if 'adj_o' in row else None,
                        'adj_d': row['adj_d'] if 'adj_d' in row else None,
                        'wab': row['WAB'] if 'WAB' in row else None
                    })
        
        return tournament_history
    
    def _calculate_consistency(self, team_data):
        """
        Calculate consistency metrics for a team
        
        Parameters:
        -----------
        team_data : pd.DataFrame
            DataFrame containing team data over multiple years
            
        Returns:
        --------
        dict
            Dictionary with consistency metrics
        """
        consistency = {}
        
        # Calculate variance in key metrics over time
        for metric in ['barthag', 'adj_o', 'adj_d', 'WAB']:
            if metric in team_data.columns:
                variance = team_data[metric].var()
                std_dev = team_data[metric].std()
                consistency[f'{metric}_variance'] = variance
                consistency[f'{metric}_std_dev'] = std_dev
        
        return consistency
    
    def get_team_trend(self, team_name):
        """
        Get historical trend data for a specific team
        
        Parameters:
        -----------
        team_name : str
            Name of the team
            
        Returns:
        --------
        dict
            Dictionary with team trend data or None if no data is available
        """
        return self.team_trends.get(team_name)
    
    def calculate_trend_adjustment(self, team1_name, team2_name):
        """
        Calculate a point adjustment based on historical trends
        
        Parameters:
        -----------
        team1_name : str
            Name of the first team
        team2_name : str
            Name of the second team
            
        Returns:
        --------
        float
            Point adjustment for team1 (can be positive or negative)
        """
        team1_trend = self.get_team_trend(team1_name)
        team2_trend = self.get_team_trend(team2_name)
        
        if team1_trend is None or team2_trend is None:
            return 0.0
        
        adjustment = 0.0
        
        # Get trend directions for key metrics
        if team1_trend.get('barthag_trend') and team2_trend.get('barthag_trend'):
            team1_barthag_trend = team1_trend['barthag_trend']['trend'] or 0
            team2_barthag_trend = team2_trend['barthag_trend']['trend'] or 0
            
            # Teams with positive trends get a boost
            barthag_trend_diff = team1_barthag_trend - team2_barthag_trend
            
            # Convert trend difference to points
            # A barthag trend of 0.05 per year is substantial, worth about 0.5 points
            adjustment += barthag_trend_diff * 10
        
        # Get consistency metrics - teams with higher consistency (lower variance) tend to be more reliable
        if 'consistency' in team1_trend and 'consistency' in team2_trend:
            team1_consistency = team1_trend['consistency'].get('barthag_std_dev', 0)
            team2_consistency = team2_trend['consistency'].get('barthag_std_dev', 0)
            
            if team1_consistency and team2_consistency:
                # Lower standard deviation means more consistent
                consistency_diff = team2_consistency - team1_consistency
                
                # More consistent teams get a small boost
                # A standard deviation difference of 0.1 is worth about 0.3 points
                adjustment += consistency_diff * 3
        
        # Cap the adjustment at +/- 2 points to avoid overvaluing historical trends
        return max(-2.0, min(2.0, adjustment))
    
    def calculate_tournament_experience_adjustment(self, team1_name, team2_name):
        """
        Calculate an adjustment based on tournament experience
        
        Parameters:
        -----------
        team1_name : str
            Name of the first team
        team2_name : str
            Name of the second team
            
        Returns:
        --------
        float
            Point adjustment for team1 (can be positive or negative)
        """
        team1_trend = self.get_team_trend(team1_name)
        team2_trend = self.get_team_trend(team2_name)
        
        if team1_trend is None or team2_trend is None:
            return 0.0
        
        team1_tournament_history = team1_trend.get('tournament_history', [])
        team2_tournament_history = team2_trend.get('tournament_history', [])
        
        # Calculate tournament appearances in last 5 years
        recent_years = set(range(2018, 2025))  # Last ~5 years
        team1_recent_appearances = len([th for th in team1_tournament_history if th['year'] in recent_years])
        team2_recent_appearances = len([th for th in team2_tournament_history if th['year'] in recent_years])
        
        # Calculate average seed in those appearances
        team1_recent_seeds = [th['seed'] for th in team1_tournament_history if th['year'] in recent_years]
        team2_recent_seeds = [th['seed'] for th in team2_tournament_history if th['year'] in recent_years]
        
        team1_avg_seed = np.mean(team1_recent_seeds) if team1_recent_seeds else 16
        team2_avg_seed = np.mean(team2_recent_seeds) if team2_recent_seeds else 16
        
        # Calculate adjustment based on appearances and seed quality
        # More appearances and better (lower) seeds are better
        
        # Appearances difference (each appearance worth 0.25 points)
        appearances_diff = team1_recent_appearances - team2_recent_appearances
        appearances_adjustment = appearances_diff * 0.25
        
        # Seed quality difference (lower is better, worth 0.15 points per seed)
        seed_diff = team2_avg_seed - team1_avg_seed
        seed_adjustment = seed_diff * 0.15
        
        # Combine adjustments
        combined_adjustment = appearances_adjustment + seed_adjustment
        
        # Cap the adjustment
        return max(-1.5, min(1.5, combined_adjustment))
    
    def enhance_game_prediction(self, prediction):
        """
        Enhance a game prediction with historical trend data
        
        Parameters:
        -----------
        prediction : dict
            Game prediction from GamePredictor
            
        Returns:
        --------
        dict
            Enhanced prediction with additional adjustments
        """
        if not isinstance(prediction, dict):
            return prediction
        
        # Extract team names
        team1_name = prediction.get('team1', {}).get('name')
        team2_name = prediction.get('team2', {}).get('name')
        
        if not team1_name or not team2_name:
            return prediction
        
        # Calculate trend adjustment
        trend_adjustment = self.calculate_trend_adjustment(team1_name, team2_name)
        
        # Calculate tournament experience adjustment
        tournament_adjustment = self.calculate_tournament_experience_adjustment(team1_name, team2_name)
        
        # Get team trends
        team1_trend = self.get_team_trend(team1_name)
        team2_trend = self.get_team_trend(team2_name)
        
        # Apply adjustments to the prediction
        team1_score = prediction['team1']['predicted_score']
        team2_score = prediction['team2']['predicted_score']
        
        # Apply our new adjustments
        team1_score += (trend_adjustment + tournament_adjustment)
        
        # Recalculate spread and win probability
        new_spread = team1_score - team2_score
        
        # Update the prediction
        enhanced_prediction = prediction.copy()
        enhanced_prediction['team1']['predicted_score'] = team1_score
        enhanced_prediction['spread'] = new_spread
        enhanced_prediction['bart_historical'] = {
            'trend_adjustment': trend_adjustment,
            'tournament_adjustment': tournament_adjustment,
            'team1_trend': team1_trend,
            'team2_trend': team2_trend
        }
        
        # Recalculate win probability (approximation)
        old_wp = prediction['team1']['win_probability']
        adjustment = trend_adjustment + tournament_adjustment
        
        # Approximately adjust win probability (each point is worth ~4% win probability)
        wp_adjustment = adjustment * 0.04
        new_wp = max(0.01, min(0.99, old_wp + wp_adjustment))
        
        enhanced_prediction['team1']['win_probability'] = new_wp
        enhanced_prediction['team2']['win_probability'] = 1 - new_wp
        
        return enhanced_prediction
    
    def save_model(self, file_path='bart_historical_model.json'):
        """
        Save the model data to a JSON file
        
        Parameters:
        -----------
        file_path : str
            Path to save the model data
        """
        # Create a simplified version of the model for saving
        model_data = {
            'team_trends': self.team_trends
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(model_data, f, indent=4)
            print(f"Model saved to {file_path}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    def load_model(self, file_path='bart_historical_model.json'):
        """
        Load the model data from a JSON file
        
        Parameters:
        -----------
        file_path : str
            Path to load the model data from
            
        Returns:
        --------
        bool
            True if the model was loaded successfully, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                model_data = json.load(f)
            
            self.team_trends = model_data.get('team_trends', {})
            print(f"Model loaded from {file_path} with {len(self.team_trends)} team trends")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    # Create and train the model
    model = BartHistoricalModel()
    
    # Get trend data for a team
    trend = model.get_team_trend("Gonzaga")
    if trend:
        print(f"Gonzaga's BART trend information:")
        print(f"  Years available: {trend['years_available']}")
        print(f"  Barthag trend: {trend['barthag_trend']['trend'] if trend['barthag_trend'] else 'N/A'}")
        print(f"  Tournament appearances: {len(trend['tournament_history'])}")
    
    # Save the model
    model.save_model()
    
    # Example of enhancing a prediction
    sample_prediction = {
        'team1': {'name': 'Gonzaga', 'predicted_score': 75.5, 'win_probability': 0.65},
        'team2': {'name': 'Duke', 'predicted_score': 70.2, 'win_probability': 0.35},
        'spread': 5.3
    }
    
    enhanced = model.enhance_game_prediction(sample_prediction)
    print(f"Original spread: {sample_prediction['spread']}")
    print(f"Enhanced spread: {enhanced['spread']}")
    print(f"Adjustments: Trend = {enhanced['bart_historical']['trend_adjustment']}, Tournament = {enhanced['bart_historical']['tournament_adjustment']}") 