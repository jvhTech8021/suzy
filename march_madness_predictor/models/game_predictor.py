import numpy as np
import pandas as pd
import os
import json

class GamePredictor:
    """
    Predicts the outcome of a game between two teams using KenPom metrics.
    Focuses on statistical analysis rather than historical tournament round data.
    """
    
    # Tournament round scaling factors - moved from local variable to class property
    # These can be modified directly to adjust tournament predictions without code changes
    # Example: predictor.TOURNAMENT_SCALING_FACTORS["championship_pct"] = 0.15
    TOURNAMENT_SCALING_FACTORS = {
        "championship_pct": 0.15,  # Championship: 0.075 points per percentage point
        "final_four_pct": 0.10,     # Final Four: 0.05 points per percentage point
        "elite_eight_pct": 0.07,    # Elite Eight: 0.04 points per percentage point
        "sweet_sixteen_pct": 0.030, # Sweet Sixteen: 0.025 points per percentage point
        "round_32_pct": 0.015,     # Round of 32: 0.0125 points per percentage point
    }
    
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
        self.exit_round_data = None
        self.champion_profile_data = None
        self.height_data = None  # New field for height, experience, and bench data
        
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
        
        # Define height and experience metrics
        self.height_metrics = [
            'Size', 'Hgt5', 'HgtEff', 'Exp', 'Bench', 'Continuity'
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
        self._load_tournament_prediction_data()
    
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
            
            # Load height and experience data if available
            try:
                self.height_data = self.data_loader.get_height_data()
                print(f"Loaded height and experience data for {len(self.height_data)} teams")
            except Exception as e:
                print(f"Error loading height and experience data: {str(e)}")
                self.height_data = None
    
    def _load_tournament_prediction_data(self):
        """Load tournament prediction data from other models"""
        if self.data_loader is None:
            return
        
        # Try to load exit round predictions
        try:
            self.exit_round_data = self.data_loader.get_exit_round_predictions()
            print(f"Loaded exit round predictions for {len(self.exit_round_data)} teams")
        except Exception as e:
            print(f"Error loading exit round predictions: {str(e)}")
            self.exit_round_data = None
        
        # Try to load champion profile predictions
        try:
            self.champion_profile_data = self.data_loader.get_champion_profile_predictions()
            print(f"Loaded champion profile predictions for {len(self.champion_profile_data)} teams")
        except Exception as e:
            print(f"Error loading champion profile predictions: {str(e)}")
            self.champion_profile_data = None
    
    def _get_tournament_prediction_data(self, team_name):
        """
        Get tournament prediction data for a team
        
        Parameters:
        -----------
        team_name : str
            Name of the team
            
        Returns:
        --------
        dict
            Dictionary with tournament prediction data
        """
        result = {
            "has_exit_round_data": False,
            "has_champion_profile_data": False,
            "championship_pct": None,
            "final_four_pct": None,
            "elite_eight_pct": None,    # Added Elite Eight
            "sweet_sixteen_pct": None,  # Added Sweet Sixteen
            "round_32_pct": None,       # Added Round of 32
            "predicted_exit": None,
            "similarity_pct": None,
            "seed": None
        }
        
        # Get exit round data if available
        if self.exit_round_data is not None:
            team_data = self.exit_round_data[self.exit_round_data['TeamName'] == team_name]
            if len(team_data) > 0:
                result["has_exit_round_data"] = True
                result["championship_pct"] = team_data.iloc[0].get('ChampionshipPct', None)
                result["final_four_pct"] = team_data.iloc[0].get('FinalFourPct', None)
                result["predicted_exit"] = team_data.iloc[0].get('PredictedExit', None)
                result["seed"] = team_data.iloc[0].get('Seed', None)
                
                # Estimate probabilities for other rounds based on the available data
                # Only if they're not already provided in the dataset
                if result["championship_pct"] is not None and result["final_four_pct"] is not None:
                    # Estimate Elite Eight (only if not directly in the data)
                    if result["elite_eight_pct"] is None and result["final_four_pct"] > 0:
                        # Elite Eight is typically higher than Final Four
                        result["elite_eight_pct"] = min(100, result["final_four_pct"] * 1.8)  # ~1.8x Final Four chance
                    
                    # Estimate Sweet Sixteen (only if not directly in the data)
                    if result["sweet_sixteen_pct"] is None and (result["elite_eight_pct"] or 0) > 0:
                        # Sweet Sixteen is typically higher than Elite Eight
                        result["sweet_sixteen_pct"] = min(100, result["elite_eight_pct"] * 1.6)  # ~1.6x Elite Eight chance
                    
                    # Estimate Round of 32 (only if not directly in the data)
                    if result["round_32_pct"] is None and (result["sweet_sixteen_pct"] or 0) > 0:
                        # Round of 32 is typically higher than Sweet Sixteen
                        result["round_32_pct"] = min(100, result["sweet_sixteen_pct"] * 1.5)  # ~1.5x Sweet Sixteen chance
                
                # Apply penalty for rounds beyond predicted exit round (instead of setting to 0%)
                # Reduce probability by 5% for each round beyond predicted exit
                if result["predicted_exit"] is not None:
                    exit_round_map = {
                        "Did Not Make Tournament": 0,
                        "Round of 64": 1,
                        "Round of 32": 2,
                        "Sweet 16": 3,
                        "Elite 8": 4,
                        "Final Four": 5,
                        "Championship": 6
                    }
                    
                    # Only apply if exit round is in our map
                    if result["predicted_exit"] in exit_round_map:
                        exit_round = exit_round_map[result["predicted_exit"]]
                        
                        # Apply to Round of 32 (if needed)
                        if exit_round < 2 and result["round_32_pct"] is not None:
                            result["round_32_pct"] = max(0, result["round_32_pct"] - 5)
                            
                        # Apply to Sweet 16
                        if exit_round < 3 and result["sweet_sixteen_pct"] is not None:
                            result["sweet_sixteen_pct"] = max(0, result["sweet_sixteen_pct"] - 5)
                            
                        # Apply to Elite 8
                        if exit_round < 4 and result["elite_eight_pct"] is not None:
                            result["elite_eight_pct"] = max(0, result["elite_eight_pct"] - 5)
                            
                        # Apply to Final Four
                        if exit_round < 5 and result["final_four_pct"] is not None:
                            result["final_four_pct"] = max(0, result["final_four_pct"] - 5)
                            
                        # Apply to Championship
                        if exit_round < 6 and result["championship_pct"] is not None:
                            result["championship_pct"] = max(0, result["championship_pct"] - 5)
        
        # Get champion profile data if available
        if self.champion_profile_data is not None:
            team_data = self.champion_profile_data[self.champion_profile_data['TeamName'] == team_name]
            if len(team_data) > 0:
                result["has_champion_profile_data"] = True
                # Only override championship and final four percentages if they're not already set
                if result["championship_pct"] is None:
                    result["championship_pct"] = team_data.iloc[0].get('ChampionPct', None)
                if result["final_four_pct"] is None:
                    result["final_four_pct"] = team_data.iloc[0].get('FinalFourPct', None)
                result["similarity_pct"] = team_data.iloc[0].get('SimilarityPct', None)
                
                # If we have championship and final four percentages from champion profile
                # but no other round percentages, estimate them
                if result["championship_pct"] is not None and result["final_four_pct"] is not None:
                    # Estimate Elite Eight (only if not already set)
                    if result["elite_eight_pct"] is None and result["final_four_pct"] > 0:
                        result["elite_eight_pct"] = min(100, result["final_four_pct"] * 1.8)
                    
                    # Estimate Sweet Sixteen (only if not already set)
                    if result["sweet_sixteen_pct"] is None and (result["elite_eight_pct"] or 0) > 0:
                        result["sweet_sixteen_pct"] = min(100, result["elite_eight_pct"] * 1.6)
                    
                    # Estimate Round of 32 (only if not already set)
                    if result["round_32_pct"] is None and (result["sweet_sixteen_pct"] or 0) > 0:
                        result["round_32_pct"] = min(100, result["sweet_sixteen_pct"] * 1.5)
        
        return result
    
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
    
    def _get_height_data(self, team_name):
        """
        Get height and experience data for a team
        
        Parameters:
        -----------
        team_name : str
            Name of the team
            
        Returns:
        --------
        dict
            Dictionary with height and experience data
        """
        result = {
            "has_height_data": False,
            "size": None,
            "hgt5": None,
            "effhgt": None,
            "experience": None,
            "bench": None,
            "gt10": None
        }
        
        # Check if height data is available
        if self.height_data is None:
            return result
        
        # Look for the team in the height data
        team_data = self.height_data[self.height_data['TeamName'] == team_name]
        if len(team_data) == 0:
            return result
        
        # Extract data
        team_row = team_data.iloc[0]
        result["has_height_data"] = True
        result["size"] = team_row.get('Size', None)
        result["hgt5"] = team_row.get('Hgt5', None)
        result["effhgt"] = team_row.get('HgtEff', None)
        result["experience"] = team_row.get('Exp', None)
        result["bench"] = team_row.get('Bench', None)
        result["gt10"] = None
        
        return result
    
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
        
        # Get tournament prediction data for both teams
        team1_tournament_data = self._get_tournament_prediction_data(team1_name)
        team2_tournament_data = self._get_tournament_prediction_data(team2_name)
        
        # Get height and experience data for both teams
        team1_height_data = self._get_height_data(team1_name)
        team2_height_data = self._get_height_data(team2_name)
        
        # Get historical matchup data
        historical_data = self._analyze_historical_matchups(team1_name, team2_name)
        
        # Calculate statistical advantages
        advantages = self._calculate_advantages(team1, team2)
        
        # Get key metrics safely
        team1_adjoe = self._safe_get(team1, 'AdjOE')
        team1_adjde = self._safe_get(team1, 'AdjDE')
        team2_adjoe = self._safe_get(team2, 'AdjOE')
        team2_adjde = self._safe_get(team2, 'AdjDE')
        
        # Get tempo
        team1_tempo = self._safe_get(team1, 'AdjTempo')
        team2_tempo = self._safe_get(team2, 'AdjTempo')
        
        # Average of the two tempos (teams play at a pace somewhere in the middle)
        avg_tempo = (team1_tempo + team2_tempo) / 2
        
        # Calculate expected points per possession
        team1_off_ppp = team1_adjoe / 100
        team2_off_ppp = team2_adjoe / 100
        
        # Adjust for opponent defense
        team1_expected_ppp = ((team1_off_ppp * 2) + (team2_adjde / 100)) / 3
        team2_expected_ppp = ((team2_off_ppp * 2) + (team1_adjde / 100)) / 3
        
        # Calculate expected score
        team1_expected_score = team1_expected_ppp * avg_tempo
        team2_expected_score = team2_expected_ppp * avg_tempo
        
        # Apply home court advantage if applicable
        if location == 'home_1':
            team1_expected_score += self.home_court_advantage
        elif location == 'home_2':
            team2_expected_score += self.home_court_advantage
        
        # Apply tournament prediction adjustment if available
        tournament_adjustment_team1 = 0
        tournament_adjustment_team2 = 0
        # Track detailed adjustment breakdowns
        tournament_adjustment_detail_team1 = {}
        tournament_adjustment_detail_team2 = {}
        
        if team1_tournament_data["championship_pct"] is not None and team2_tournament_data["championship_pct"] is not None:
            # Use the class property for scaling factors
            scaling_factors = self.TOURNAMENT_SCALING_FACTORS
            
            # Initialize adjustments
            team1_total_adjustment = 0
            team2_total_adjustment = 0
            
            # Apply adjustments for each tournament round if data is available
            for round_key, scaling_factor in scaling_factors.items():
                # Team 1 adjustment for this round
                if round_key in team1_tournament_data and team1_tournament_data[round_key] is not None:
                    round_adjustment = team1_tournament_data[round_key] * scaling_factor
                    team1_total_adjustment += round_adjustment
                    # Store the detailed breakdown
                    tournament_adjustment_detail_team1[round_key] = {
                        "percentage": team1_tournament_data[round_key],
                        "factor": scaling_factor,
                        "points": round_adjustment
                    }
                
                # Team 2 adjustment for this round
                if round_key in team2_tournament_data and team2_tournament_data[round_key] is not None:
                    round_adjustment = team2_tournament_data[round_key] * scaling_factor
                    team2_total_adjustment += round_adjustment
                    # Store the detailed breakdown
                    tournament_adjustment_detail_team2[round_key] = {
                        "percentage": team2_tournament_data[round_key],
                        "factor": scaling_factor,
                        "points": round_adjustment
                    }
            
            # Store the final calculated adjustments
            tournament_adjustment_team1 = team1_total_adjustment
            tournament_adjustment_team2 = team2_total_adjustment
            
            # Apply the adjustments to the expected scores
            team1_expected_score += tournament_adjustment_team1
            team2_expected_score += tournament_adjustment_team2
        
        # Apply seed adjustment if available
        seed_adjustment = 0
        if team1_tournament_data["seed"] is not None and team2_tournament_data["seed"] is not None:
            # Lower seeds (better teams) get a boost
            seed_diff = team2_tournament_data["seed"] - team1_tournament_data["seed"]
            seed_adjustment = seed_diff / 7  # Scale appropriately
            team1_expected_score += seed_adjustment
        
        # Apply height and experience adjustments if available
        height_adjustment = 0
        experience_adjustment = 0
        bench_adjustment = 0  # Define this variable to avoid undefined errors
        
        # Only apply height and experience adjustments if valid data is available for both teams
        # But ensure this doesn't affect the base prediction
        if team1_height_data["has_height_data"] and team2_height_data["has_height_data"]:
            # Height advantage
            if team1_height_data["effhgt"] is not None and team2_height_data["effhgt"] is not None:
                # Make sure both values are not NaN before calculation
                if not pd.isna(team1_height_data["effhgt"]) and not pd.isna(team2_height_data["effhgt"]):
                    effhgt_diff = team1_height_data["effhgt"] - team2_height_data["effhgt"]
                    if abs(effhgt_diff) > 0.7:  # Lowered threshold from 1.0 to 0.7 inch
                        height_adjustment = effhgt_diff * 0.8  # Increased from 0.5 to 0.8 points per inch
                        team1_expected_score += height_adjustment
            
            # Experience advantage
            if team1_height_data["experience"] is not None and team2_height_data["experience"] is not None:
                # Make sure both values are not NaN before calculation
                if not pd.isna(team1_height_data["experience"]) and not pd.isna(team2_height_data["experience"]):
                    exp_diff = team1_height_data["experience"] - team2_height_data["experience"]
                    if abs(exp_diff) > 0.4:  # Lowered threshold from 0.5 to 0.4 years
                        experience_adjustment = exp_diff * 0.7  # Increased from 0.5 to 0.7 points per year
                        team1_expected_score += experience_adjustment
            
            # Bench depth
            if team1_height_data["bench"] is not None and team2_height_data["bench"] is not None:
                # Make sure both values are not NaN before calculation
                if not pd.isna(team1_height_data["bench"]) and not pd.isna(team2_height_data["bench"]):
                    bench_diff = team1_height_data["bench"] - team2_height_data["bench"]
                    if abs(bench_diff) > 6:  # Increased threshold from 4 to 6 percent
                        bench_adjustment = abs(bench_diff) / 5.0  # Decreased importance (increased divisor from 3.0 to 5.0)
                        team1_expected_score += bench_adjustment
        
        # Add a final check to make sure expected scores are not NaN
        if pd.isna(team1_expected_score) or pd.isna(team2_expected_score):
            # If we somehow got NaN values, recalculate without the adjustments
            # This is a fallback to ensure we always have a prediction
            team1_expected_score = team1_expected_ppp * avg_tempo
            team2_expected_score = team2_expected_ppp * avg_tempo
            
            # Re-apply only the home court advantage
            if location == 'home_1':
                team1_expected_score += self.home_court_advantage
            elif location == 'home_2':
                team2_expected_score += self.home_court_advantage
        
        # Calculate spread (always Team1 - Team2)
        spread = team1_expected_score - team2_expected_score
        
        # Calculate total
        total = team1_expected_score + team2_expected_score
        
        # Calculate win probability using log5 formula
        team1_raw_wp = 1 / (1 + 10 ** (-(team1_adjoe - team1_adjde - (team2_adjoe - team2_adjde)) / 100))
        
        # Adjust win probability for home court if applicable
        if location == 'home_1':
            team1_wp = self._adjust_win_probability(team1_raw_wp, self.home_court_advantage)
        elif location == 'home_2':
            team1_wp = self._adjust_win_probability(team1_raw_wp, -self.home_court_advantage)
        else:
            team1_wp = team1_raw_wp
            
        # Apply any tournament model adjustments
        if tournament_adjustment_team1 != 0 or tournament_adjustment_team2 != 0:
            total_point_adjustment = tournament_adjustment_team1 - tournament_adjustment_team2
            team1_wp = self._adjust_win_probability(team1_wp, total_point_adjustment)
        
        team2_wp = 1 - team1_wp
        
        # Calculate key factors for the matchup
        key_factors = self._calculate_key_factors(team1, team2)
        
        # Calculate stat comparisons for display
        stat_comparisons = []
        for stat in self.active_metrics:
            team1_value = self._safe_get(team1, stat)
            team2_value = self._safe_get(team2, stat)
            difference = team1_value - team2_value
            
            # Determine if higher or lower is better for this stat
            higher_is_better = True
            if stat in ['AdjDE', 'EFGD_Pct', 'TOR', 'TORD', 'TwoD_Pct', 'ThreeD_Pct']:
                higher_is_better = False
                
            # Format the display name
            display_name = stat.replace('_', ' ')
            
            stat_comparisons.append({
                "stat": display_name,
                "team1_value": team1_value,
                "team2_value": team2_value,
                "difference": difference,
                "advantage": team1_name if (higher_is_better and difference > 0) or (not higher_is_better and difference < 0) else team2_name
            })
            
        # Return the prediction data
        result = {
            "team1": {
                "name": team1_name,
                "predicted_score": team1_expected_score,
                "win_probability": team1_wp,
                "tournament_data": team1_tournament_data,
                "height_data": team1_height_data,
                "tournament_adjustment_detail": tournament_adjustment_detail_team1
            },
            "team2": {
                "name": team2_name,
                "predicted_score": team2_expected_score,
                "win_probability": team2_wp,
                "tournament_data": team2_tournament_data,
                "height_data": team2_height_data,
                "tournament_adjustment_detail": tournament_adjustment_detail_team2
            },
            "spread": spread,
            "total": total,
            "location": location,
            "key_factors": key_factors,
            "team_stats": stat_comparisons,
            "historical_matchups": historical_data,
            "tournament_adjustment": tournament_adjustment_team1 - tournament_adjustment_team2,  # Net adjustment for compatibility
            "tournament_adjustment_team1": tournament_adjustment_team1,
            "tournament_adjustment_team2": tournament_adjustment_team2,
            "seed_adjustment": seed_adjustment,
            "height_adjustment": height_adjustment,
            "experience_adjustment": experience_adjustment,
            "bench_adjustment": bench_adjustment,
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
        
        # Add height and experience factors if available
        team1_name = team1['TeamName']
        team2_name = team2['TeamName']
        
        # Get height data for both teams
        team1_height_data = self._get_height_data(team1_name)
        team2_height_data = self._get_height_data(team2_name)
        
        if team1_height_data["has_height_data"] and team2_height_data["has_height_data"]:
            # Height advantage
            if team1_height_data["effhgt"] is not None and team2_height_data["effhgt"] is not None:
                # Make sure both values are not NaN before calculation
                if not pd.isna(team1_height_data["effhgt"]) and not pd.isna(team2_height_data["effhgt"]):
                    effhgt_diff = team1_height_data["effhgt"] - team2_height_data["effhgt"]
                    if abs(effhgt_diff) > 0.7:  # Lowered threshold from 1.0 to 0.7 inch
                        factors.append({
                            'factor': 'Height',
                            'advantage': team1_name if effhgt_diff > 0 else team2_name,
                            'magnitude': abs(effhgt_diff) * 2.5,  # Increased from 2 to 2.5
                            'description': f"{'Taller' if effhgt_diff > 0 else 'Shorter'} team (by {abs(effhgt_diff):.1f}\")"
                        })
            
            # Experience advantage
            if team1_height_data["experience"] is not None and team2_height_data["experience"] is not None:
                # Make sure both values are not NaN before calculation
                if not pd.isna(team1_height_data["experience"]) and not pd.isna(team2_height_data["experience"]):
                    exp_diff = team1_height_data["experience"] - team2_height_data["experience"]
                    if abs(exp_diff) > 0.4:  # Lowered threshold from 0.5 to 0.4 years
                        factors.append({
                            'factor': 'Experience',
                            'advantage': team1_name if exp_diff > 0 else team2_name,
                            'magnitude': abs(exp_diff) * 3.5,  # Increased from 3 to 3.5
                            'description': f"{'More' if exp_diff > 0 else 'Less'} experienced (by {abs(exp_diff):.1f} years)"
                        })
            
            # Bench depth
            if team1_height_data["bench"] is not None and team2_height_data["bench"] is not None:
                # Make sure both values are not NaN before calculation
                if not pd.isna(team1_height_data["bench"]) and not pd.isna(team2_height_data["bench"]):
                    bench_diff = team1_height_data["bench"] - team2_height_data["bench"]
                    if abs(bench_diff) > 6:  # Increased threshold from 4 to 6 percent
                        factors.append({
                            'factor': 'Bench Depth',
                            'advantage': team1_name if bench_diff > 0 else team2_name,
                            'magnitude': abs(bench_diff) / 5.0,  # Decreased importance (increased divisor from 3.0 to 5.0)
                            'description': f"{'Deeper' if bench_diff > 0 else 'Thinner'} bench (by {abs(bench_diff):.1f}% minutes)"
                        })
        
        # Sort factors by magnitude
        factors.sort(key=lambda x: x['magnitude'], reverse=True)
        
        return factors[:6]  # Return top 6 factors instead of 5 to include height/experience
    
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

    def save_prediction(self, prediction, file_path='historical_picks.json'):
        """
        Save the prediction result to a JSON file.

        Parameters:
        -----------
        prediction : dict
            The prediction result to save.
        file_path : str
            The path to the JSON file where the prediction will be saved.
        """
        try:
            # Load existing data
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
            except FileNotFoundError:
                data = []

            # Ensure data is a list
            if not isinstance(data, list):
                print("Warning: Data is not a list. Resetting to an empty list.")
                data = []

            # Append new prediction
            data.append(prediction)

            # Save back to file
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            print(f"Error saving prediction: {e}")

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