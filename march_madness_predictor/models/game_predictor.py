import numpy as np
import pandas as pd
import os

class GamePredictor:
    """
    Predicts the outcome of a game between two teams using KenPom metrics.
    Focuses on statistical analysis rather than historical tournament round data.
    """
    
    def __init__(self, data_loader=None):
        """
        Initialize the game predictor with data from the data loader
        
        Parameters:
        -----------
        data_loader : DataLoader
            Instance of the DataLoader class to load KenPom data
        """
        self.data_loader = data_loader
        self.current_data = None
        
        # Define core metrics that are most likely available in KenPom data
        self.core_metrics = [
            'AdjEM', 'AdjOE', 'AdjDE', 'AdjTempo'
        ]
        
        # Define supplementary metrics that may or may not be available
        self.supplementary_metrics = [
            'EFG_Pct', 'EFGD_Pct',  # May be 'eFG%' and 'eFG%D' or other variations
            'TOR', 'TORD',          # May be 'TO%' and 'TO%D'
            'ORB', 'DRB',           # May be 'OR%' and 'DR%'
            'FTR', 'FTRD',          # May be 'FTR' and 'FTRD'
            'Two_Pct', 'TwoD_Pct',  # May be '2P%' and '2P%D'
            'Three_Pct', 'ThreeD_Pct'  # May be '3P%' and '3P%D'
        ]
        
        # Define alternate column names that might be in the data
        self.column_mapping = {
            'EFG_Pct': ['EFG_Pct', 'eFG%', 'EFG%', 'Effective FG%', 'eFGPct'],
            'EFGD_Pct': ['EFGD_Pct', 'eFG%D', 'EFG%D', 'Defensive eFG%', 'eFGDPct'],
            'TOR': ['TOR', 'TO%', 'TO Pct', 'TOPct', 'Turnover Pct'],
            'TORD': ['TORD', 'TO%D', 'TO PctD', 'TODPct', 'Defensive TO Pct'],
            'ORB': ['ORB', 'OR%', 'OR Pct', 'ORPct', 'Offensive Reb Pct'],
            'DRB': ['DRB', 'DR%', 'DR Pct', 'DRPct', 'Defensive Reb Pct'],
            'FTR': ['FTR', 'FT Rate', 'FTRate', 'Free Throw Rate'],
            'FTRD': ['FTRD', 'FT RateD', 'FTRateD', 'Defensive FT Rate'],
            'Two_Pct': ['Two_Pct', '2P%', '2PPct', '2P Pct', 'Two Point Pct'],
            'TwoD_Pct': ['TwoD_Pct', '2P%D', '2PDPct', '2P PctD', 'Defensive Two Point Pct'],
            'Three_Pct': ['Three_Pct', '3P%', '3PPct', '3P Pct', 'Three Point Pct'],
            'ThreeD_Pct': ['ThreeD_Pct', '3P%D', '3PDPct', '3P PctD', 'Defensive Three Point Pct'],
            'AdjEM': ['AdjEM', 'AdjEff', 'Adjusted Efficiency Margin', 'Adj Efficiency Margin', 'AdjustEM'],
            'AdjOE': ['AdjOE', 'AdjO', 'Adjusted Offensive Efficiency', 'Adj Offensive Efficiency', 'AdjOffEff', 'AdjOff'],
            'AdjDE': ['AdjDE', 'AdjD', 'Adjusted Defensive Efficiency', 'Adj Defensive Efficiency', 'AdjDefEff', 'AdjDef'],
            'AdjTempo': ['AdjTempo', 'Adj Tempo', 'Adjusted Tempo', 'Tempo']
        }
        
        # Home court advantage in points
        self.home_court_advantage = 3.5
        
        self._load_data()
        self._detect_available_columns()
        self.active_metrics = self._get_active_metrics()
    
    def _load_data(self):
        """Load current season KenPom data"""
        if self.data_loader is None:
            # If no data loader provided, try to load data directly
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            # Try from susan_kenpom folder first
            data_file = os.path.join(base_path, "susan_kenpom/summary25.csv")
            if not os.path.exists(data_file):
                # Fallback to data folder
                data_file = os.path.join(base_path, "data/summary25.csv")
            
            try:
                self.current_data = pd.read_csv(data_file)
                # Clean data
                for col in self.current_data.columns:
                    if self.current_data[col].dtype == 'object' and col != 'TeamName':
                        self.current_data[col] = self.current_data[col].str.replace('"', '').astype(float)
                    elif col == 'TeamName':
                        self.current_data[col] = self.current_data[col].str.replace('"', '')
                
                # Print the actual columns in the data file
                print(f"Columns available in KenPom data: {', '.join(self.current_data.columns.tolist())}")
            except FileNotFoundError:
                print(f"Error: Required data files not found at {data_file}")
                return
        else:
            # Use data loader to get data
            self.current_data = self.data_loader.get_current_season_data()
    
    def _detect_available_columns(self):
        """
        Detect which columns are available in the dataset and create mappings
        to our standardized names
        """
        if self.current_data is None:
            return
        
        self.available_columns = {}
        
        # Check which columns are available in the data
        data_columns = self.current_data.columns.tolist()
        
        # Create a mapping from our standard names to actual column names
        for standard_name, possible_names in self.column_mapping.items():
            for name in possible_names:
                if name in data_columns:
                    self.available_columns[standard_name] = name
                    break
        
        # Print which metrics were found
        found_metrics = list(self.available_columns.keys())
        print(f"Found {len(found_metrics)} metrics in the data: {', '.join(found_metrics)}")
    
    def _get_active_metrics(self):
        """
        Get a list of metrics that are actually available in the data
        """
        active_metrics = []
        
        # Add core metrics that are available
        for metric in self.core_metrics:
            if metric in self.available_columns or metric in self.current_data.columns:
                active_metrics.append(metric)
        
        # Add supplementary metrics that are available
        for metric in self.supplementary_metrics:
            if metric in self.available_columns or metric in self.current_data.columns:
                active_metrics.append(metric)
        
        print(f"Using {len(active_metrics)} active metrics for predictions: {', '.join(active_metrics)}")
        return active_metrics
    
    def _safe_get(self, team, column):
        """
        Safely get a value from the team data, handling missing columns
        
        Parameters:
        -----------
        team : pd.Series
            Team data
        column : str
            Column name to retrieve
            
        Returns:
        --------
        float
            Column value or a reasonable default
        """
        # If the exact column exists, return it
        if column in team.index:
            return team[column]
        
        # If we have a mapping for this column, use it
        if column in self.available_columns:
            mapped_column = self.available_columns[column]
            if mapped_column in team.index:
                return team[mapped_column]
        
        # If we don't have the column, use defaults based on NCAA averages
        defaults = {
            'EFG_Pct': 50.0,     # Average effective FG%
            'EFGD_Pct': 50.0,    # Average defensive effective FG%
            'TOR': 18.0,         # Average turnover rate
            'TORD': 18.0,        # Average defensive turnover rate
            'ORB': 30.0,         # Average offensive rebound rate
            'DRB': 70.0,         # Average defensive rebound rate
            'FTR': 32.0,         # Average FT rate
            'FTRD': 32.0,        # Average defensive FT rate
            'Two_Pct': 50.0,     # Average 2-point %
            'TwoD_Pct': 50.0,    # Average defensive 2-point %
            'Three_Pct': 35.0,   # Average 3-point %
            'ThreeD_Pct': 35.0   # Average defensive 3-point %
        }
        
        # For core metrics, we need them, so compute them if not available
        if column == 'AdjEM' and 'AdjOE' in team.index and 'AdjDE' in team.index:
            return team['AdjOE'] - team['AdjDE']
        elif column == 'AdjOE' and 'AdjEM' in team.index and 'AdjDE' in team.index:
            return team['AdjEM'] + team['AdjDE']
        elif column == 'AdjDE' and 'AdjOE' in team.index and 'AdjEM' in team.index:
            return team['AdjOE'] - team['AdjEM']
        elif column == 'AdjTempo':
            return 68.0  # Average NCAA tempo
        
        # Use default if available
        if column in defaults:
            return defaults[column]
        
        # Fallback to 0 if nothing else works
        return 0.0
    
    def get_available_teams(self):
        """
        Get a list of all available teams
        
        Returns:
        --------
        list
            List of team names
        """
        if self.current_data is None:
            return []
        
        return sorted(self.current_data['TeamName'].unique())
    
    def predict_game(self, team1_name, team2_name, location='neutral'):
        """
        Predict the outcome of a game between two teams using KenPom metrics
        
        Parameters:
        -----------
        team1_name : str
            Name of the first team
        team2_name : str
            Name of the second team
        location : str
            Game location: 'home_1' (team1 at home), 'home_2' (team2 at home), or 'neutral'
            
        Returns:
        --------
        dict
            Dictionary with prediction results
        """
        if self.current_data is None:
            return {"error": "No data available for prediction"}
        
        # Find teams in data
        team1 = self.current_data[self.current_data['TeamName'] == team1_name]
        team2 = self.current_data[self.current_data['TeamName'] == team2_name]
        
        if len(team1) == 0:
            return {"error": f"Team not found: {team1_name}"}
        if len(team2) == 0:
            return {"error": f"Team not found: {team2_name}"}
            
        team1 = team1.iloc[0]
        team2 = team2.iloc[0]
        
        # Get historical matchup data
        historical_data = self._analyze_historical_matchups(team1_name, team2_name)
        
        # Calculate statistical advantages
        advantages = self._calculate_advantages(team1, team2)
        
        # Get key metrics safely
        team1_adjoe = self._safe_get(team1, 'AdjOE')
        team1_adjde = self._safe_get(team1, 'AdjDE')
        team1_adjtempo = self._safe_get(team1, 'AdjTempo')
        team2_adjoe = self._safe_get(team2, 'AdjOE')
        team2_adjde = self._safe_get(team2, 'AdjDE')
        team2_adjtempo = self._safe_get(team2, 'AdjTempo')
        
        # Determine predicted score using KenPom ratings
        avg_tempo = (team1_adjtempo + team2_adjtempo) / 2
        team1_predicted_score = (team1_adjoe * avg_tempo / 100) * (team2_adjde / 100)
        team2_predicted_score = (team2_adjoe * avg_tempo / 100) * (team1_adjde / 100)
        
        # Calculate spread and total
        spread = team1_predicted_score - team2_predicted_score
        total = team1_predicted_score + team2_predicted_score
        
        # Calculate win probability using logistic function based on AdjEM
        team1_adjem = self._safe_get(team1, 'AdjEM')
        team2_adjem = self._safe_get(team2, 'AdjEM')
        rating_diff = team1_adjem - team2_adjem
        team1_win_prob = 1 / (1 + np.exp(-rating_diff * 0.1))
        team2_win_prob = 1 - team1_win_prob
        
        # Apply historical matchup adjustment if available
        if historical_data['total_matchups'] > 0:
            # Calculate historical win rate for team1
            historical_win_rate = historical_data['team1_wins'] / historical_data['total_matchups']
            
            # Factor in historical average margin
            margin_adjustment = historical_data['avg_margin'] * 0.1  # Scale it down
            
            # Adjust the spread
            spread += margin_adjustment
            
            # Adjust win probability using historical data (10% weight)
            historical_weight = 0.1
            adjusted_team1_win_prob = (team1_win_prob * (1 - historical_weight)) + (historical_win_rate * historical_weight)
            
            # Ensure probability stays within reasonable bounds
            team1_win_prob = min(max(adjusted_team1_win_prob, 0.05), 0.95)
            team2_win_prob = 1 - team1_win_prob
        
        # Apply adjustments based on key statistics that are available
        # Start with a base adjustment
        spread_adjustment = 0
        total_adjustment = 0
        
        # Offensive efficiency has major impact
        if 'AdjOE' in self.active_metrics:
            oe_diff = team1_adjoe - team2_adjoe
            spread_adjustment += oe_diff * 0.05
        
        # Defensive efficiency has major impact (negative because lower is better)
        if 'AdjDE' in self.active_metrics:
            de_diff = team2_adjde - team1_adjde
            spread_adjustment += de_diff * 0.05
        
        # Tempo can affect total but not necessarily spread
        if 'AdjTempo' in self.active_metrics:
            tempo_diff = team1_adjtempo - team2_adjtempo
            total_adjustment += abs(tempo_diff) * 0.1
        
        # Add supplementary metrics adjustments if available
        if 'TOR' in self.active_metrics:
            # Turnover rate affects possession quality
            to_diff = self._safe_get(team2, 'TOR') - self._safe_get(team1, 'TOR')  # Lower is better
            spread_adjustment += to_diff * 0.2
        
        if 'Three_Pct' in self.active_metrics:
            # Three-point shooting can create variance
            three_diff = self._safe_get(team1, 'Three_Pct') - self._safe_get(team2, 'Three_Pct')
            spread_adjustment += three_diff * 0.3
        
        # Apply location-based adjustment
        location_adjustment = 0
        if location == 'home_1':
            location_adjustment = self.home_court_advantage
            spread += location_adjustment
            team1_win_prob = self._adjust_win_probability(team1_win_prob, location_adjustment)
        elif location == 'home_2':
            location_adjustment = -self.home_court_advantage
            spread += location_adjustment
            team1_win_prob = self._adjust_win_probability(team1_win_prob, location_adjustment)
        
        # Recalculate team2_win_prob based on adjusted team1_win_prob
        team2_win_prob = 1 - team1_win_prob
        
        # Apply other adjustments
        spread += spread_adjustment
        total += total_adjustment
        
        # Create team stats comparison
        team_stats = []
        for col in self.active_metrics:
            stat_name = col.replace('_', ' ')
            team1_value = self._safe_get(team1, col)
            team2_value = self._safe_get(team2, col)
            
            team_stats.append({
                'stat': stat_name,
                'team1_value': team1_value,
                'team2_value': team2_value,
                'difference': team1_value - team2_value
            })
        
        # Return prediction results with historical matchup data
        result = {
            "team1": {
                "name": team1_name,
                "win_probability": team1_win_prob,
                "predicted_score": team1_predicted_score
            },
            "team2": {
                "name": team2_name,
                "win_probability": team2_win_prob,
                "predicted_score": team2_predicted_score
            },
            "spread": spread,
            "total": total,
            "location": location,
            "location_adjustment": location_adjustment,
            "team_stats": team_stats,
            "predicted_winner": team1_name if team1_win_prob > team2_win_prob else team2_name,
            "key_factors": self._calculate_key_factors(team1, team2),
            "historical_matchups": {
                "total_matchups": historical_data['total_matchups'],
                "team1_wins": historical_data['team1_wins'],
                "team2_wins": historical_data['team2_wins'],
                "avg_margin": historical_data['avg_margin']
            }
        }
        
        return result
    
    def _adjust_win_probability(self, base_probability, point_adjustment):
        """
        Adjust win probability based on home court advantage
        3.5 points is roughly equivalent to about 15% win probability shift for even matchups
        
        Parameters:
        -----------
        base_probability : float
            Base win probability before adjustment
        point_adjustment : float
            Adjustment in points (positive for team1 advantage, negative for team2)
            
        Returns:
        --------
        float
            Adjusted win probability
        """
        if point_adjustment == 0:
            return base_probability
        
        # Scale factor controls how much impact points have on probability
        scale_factor = 0.04  # Approximately 4% probability shift per point
        
        # Apply logistic adjustment centered around 0.5 probability
        odds = base_probability / (1 - base_probability)
        adjusted_odds = odds * np.exp(scale_factor * point_adjustment)
        adjusted_probability = adjusted_odds / (1 + adjusted_odds)
        
        return adjusted_probability
    
    def _calculate_advantages(self, team1, team2):
        """
        Calculate statistical advantages between teams
        
        Parameters:
        -----------
        team1 : pd.Series
            Statistics for team 1
        team2 : pd.Series
            Statistics for team 2
            
        Returns:
        --------
        dict
            Dictionary with advantage metrics
        """
        advantages = {}
        
        # Only calculate advantages for metrics that are available
        if 'AdjOE' in self.active_metrics:
            advantages['offense'] = self._safe_get(team1, 'AdjOE') - self._safe_get(team2, 'AdjOE')
        
        if 'AdjDE' in self.active_metrics:
            advantages['defense'] = self._safe_get(team2, 'AdjDE') - self._safe_get(team1, 'AdjDE')
        
        if 'AdjTempo' in self.active_metrics:
            advantages['tempo'] = self._safe_get(team1, 'AdjTempo') - self._safe_get(team2, 'AdjTempo')
        
        if 'EFG_Pct' in self.active_metrics:
            advantages['shooting'] = self._safe_get(team1, 'EFG_Pct') - self._safe_get(team2, 'EFG_Pct')
        
        if 'TOR' in self.active_metrics:
            advantages['turnovers'] = self._safe_get(team2, 'TOR') - self._safe_get(team1, 'TOR')
        
        if 'ORB' in self.active_metrics:
            advantages['offensive_rebounding'] = self._safe_get(team1, 'ORB') - self._safe_get(team2, 'ORB')
        
        if 'DRB' in self.active_metrics:
            advantages['defensive_rebounding'] = self._safe_get(team1, 'DRB') - self._safe_get(team2, 'DRB')
        
        if 'Three_Pct' in self.active_metrics:
            advantages['three_point'] = self._safe_get(team1, 'Three_Pct') - self._safe_get(team2, 'Three_Pct')
        
        return advantages
    
    def _calculate_key_factors(self, team1, team2):
        """
        Calculate key factors that influence the game outcome
        
        Parameters:
        -----------
        team1 : pd.Series
            Statistics for team 1
        team2 : pd.Series
            Statistics for team 2
            
        Returns:
        --------
        list
            List of key factors
        """
        factors = []
        
        # Only calculate key factors for metrics that are available
        if 'AdjOE' in self.active_metrics:
            # Offensive efficiency difference
            oe_diff = self._safe_get(team1, 'AdjOE') - self._safe_get(team2, 'AdjOE')
            if abs(oe_diff) > 2:
                factors.append({
                    'factor': 'Offensive Efficiency',
                    'advantage': team1['TeamName'] if oe_diff > 0 else team2['TeamName'],
                    'magnitude': abs(oe_diff),
                    'description': f"{'Higher' if oe_diff > 0 else 'Lower'} offensive efficiency"
                })
        
        if 'AdjDE' in self.active_metrics:
            # Defensive efficiency difference
            de_diff = self._safe_get(team2, 'AdjDE') - self._safe_get(team1, 'AdjDE')  # Lower is better for defense
            if abs(de_diff) > 2:
                factors.append({
                    'factor': 'Defensive Efficiency',
                    'advantage': team1['TeamName'] if de_diff > 0 else team2['TeamName'],
                    'magnitude': abs(de_diff),
                    'description': f"{'Stronger' if de_diff > 0 else 'Weaker'} defensive efficiency"
                })
        
        if 'AdjTempo' in self.active_metrics:
            # Tempo advantage
            tempo_diff = self._safe_get(team1, 'AdjTempo') - self._safe_get(team2, 'AdjTempo')
            if abs(tempo_diff) > 3:
                faster_team = team1['TeamName'] if tempo_diff > 0 else team2['TeamName']
                slower_team = team2['TeamName'] if tempo_diff > 0 else team1['TeamName']
                factors.append({
                    'factor': 'Tempo',
                    'advantage': 'Neutral',
                    'magnitude': abs(tempo_diff),
                    'description': f"{faster_team} prefers a faster pace, {slower_team} plays slower"
                })
        
        if 'EFG_Pct' in self.active_metrics:
            # Shooting advantage
            efg_diff = self._safe_get(team1, 'EFG_Pct') - self._safe_get(team2, 'EFG_Pct')
            if abs(efg_diff) > 2:
                factors.append({
                    'factor': 'Shooting Efficiency',
                    'advantage': team1['TeamName'] if efg_diff > 0 else team2['TeamName'],
                    'magnitude': abs(efg_diff),
                    'description': f"{'Better' if efg_diff > 0 else 'Worse'} overall shooting efficiency"
                })
        
        if 'ORB' in self.active_metrics:
            # Rebounding advantage
            orb_diff = self._safe_get(team1, 'ORB') - self._safe_get(team2, 'ORB')
            if abs(orb_diff) > 3:
                factors.append({
                    'factor': 'Offensive Rebounding',
                    'advantage': team1['TeamName'] if orb_diff > 0 else team2['TeamName'],
                    'magnitude': abs(orb_diff),
                    'description': f"{'Stronger' if orb_diff > 0 else 'Weaker'} on the offensive glass"
                })
        
        if 'TOR' in self.active_metrics:
            # Turnover advantage
            to_diff = self._safe_get(team2, 'TOR') - self._safe_get(team1, 'TOR')  # Lower is better for turnovers
            if abs(to_diff) > 2:
                factors.append({
                    'factor': 'Ball Control',
                    'advantage': team1['TeamName'] if to_diff > 0 else team2['TeamName'],
                    'magnitude': abs(to_diff),
                    'description': f"{'Better' if to_diff > 0 else 'Worse'} at protecting the ball"
                })
        
        if 'Three_Pct' in self.active_metrics:
            # Three-point shooting
            three_diff = self._safe_get(team1, 'Three_Pct') - self._safe_get(team2, 'Three_Pct')
            if abs(three_diff) > 2:
                factors.append({
                    'factor': 'Three-Point Shooting',
                    'advantage': team1['TeamName'] if three_diff > 0 else team2['TeamName'],
                    'magnitude': abs(three_diff),
                    'description': f"{'Better' if three_diff > 0 else 'Worse'} three-point shooting"
                })
        
        # Sort factors by magnitude
        factors.sort(key=lambda x: x['magnitude'], reverse=True)
        
        return factors[:5]  # Return top 5 factors
    
    def _analyze_historical_matchups(self, team1_name, team2_name):
        """
        Analyze historical matchups between two teams
        
        Parameters:
        -----------
        team1_name : str
            Name of the first team
        team2_name : str
            Name of the second team
            
        Returns:
        --------
        dict
            Dictionary with historical matchup data
        """
        historical_matchups = []
        team1_wins = 0
        team2_wins = 0
        
        # Check historical data for each available year
        for year in range(2009, 2025):
            try:
                # Skip 2020 (COVID year)
                if year == 2020:
                    continue
                
                # Try to load from data_loader first if available
                historical_df = None
                if self.data_loader is not None:
                    try:
                        historical_df = self.data_loader.get_historical_data(year)
                    except:
                        pass
                
                # Fall back to direct file loading if data_loader didn't work
                if historical_df is None:
                    # Determine the base path
                    if self.data_loader is not None and hasattr(self.data_loader, 'kenpom_data_dir'):
                        base_dir = self.data_loader.kenpom_data_dir
                    else:
                        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        base_dir = os.path.join(base_path, "susan_kenpom")
                    
                    # Load from file
                    file_path = os.path.join(base_dir, f"processed_{year}.csv")
                    if not os.path.exists(file_path):
                        continue
                        
                    historical_df = pd.read_csv(file_path)
                    
                    # Clean data
                    for col in historical_df.columns:
                        if historical_df[col].dtype == 'object' and col != 'TeamName':
                            historical_df[col] = historical_df[col].str.replace('"', '').astype(float)
                        elif col == 'TeamName':
                            historical_df[col] = historical_df[col].str.replace('"', '')
                
                # Look for both teams in this year's data
                team1_data = historical_df[historical_df['TeamName'] == team1_name]
                team2_data = historical_df[historical_df['TeamName'] == team2_name]
                
                if len(team1_data) > 0 and len(team2_data) > 0:
                    # Both teams existed this year
                    team1_metrics = team1_data.iloc[0]
                    team2_metrics = team2_data.iloc[0]
                    
                    # Use adjusted efficiency margin to determine hypothetical winner
                    # Fall back to ranks if values aren't available
                    if 'AdjEM' in team1_metrics and 'AdjEM' in team2_metrics:
                        team1_em = team1_metrics['AdjEM']
                        team2_em = team2_metrics['AdjEM']
                        em_diff = team1_em - team2_em
                        
                        if em_diff > 0:
                            team1_wins += 1
                            winner = team1_name
                        else:
                            team2_wins += 1
                            winner = team2_name
                        
                        historical_matchups.append({
                            'year': year,
                            'team1_em': team1_em,
                            'team2_em': team2_em,
                            'diff': em_diff,
                            'winner': winner
                        })
            except Exception as e:
                # Skip any years with errors
                continue
                
        # Calculate average margin between the teams historically
        avg_margin = 0
        if historical_matchups:
            margins = [matchup['diff'] for matchup in historical_matchups]
            avg_margin = sum(margins) / len(margins) if margins else 0
            
        return {
            'team1_wins': team1_wins,
            'team2_wins': team2_wins,
            'total_matchups': len(historical_matchups),
            'matchups': historical_matchups,
            'avg_margin': avg_margin
        }

def predict_matchup(team1, team2, data_loader=None, location='neutral'):
    """
    Convenience function to predict a matchup without creating a GamePredictor instance
    
    Parameters:
    -----------
    team1 : str
        Name of the first team
    team2 : str
        Name of the second team
    data_loader : DataLoader
        Instance of the DataLoader class to load KenPom data
    location : str
        Game location: 'home_1' (team1 at home), 'home_2' (team2 at home), or 'neutral'
        
    Returns:
    --------
    dict
        Dictionary with prediction results
    """
    predictor = GamePredictor(data_loader)
    return predictor.predict_game(team1, team2, location) 