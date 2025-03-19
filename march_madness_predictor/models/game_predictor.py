import numpy as np
import pandas as pd
import os
import json

# Import our new BartHistoricalModel
try:
    from march_madness_predictor.models.bart_historical_model import BartHistoricalModel
except ImportError:
    try:
        from models.bart_historical_model import BartHistoricalModel
    except ImportError:
        print("Warning: Could not import BartHistoricalModel, historical enhancements will be disabled")
        BartHistoricalModel = None

class GamePredictor:
    """
    Predicts the outcome of a game between two teams using KenPom metrics.
    Focuses on statistical analysis rather than historical tournament round data.
    """
    
    # Tournament round scaling factors - moved from local variable to class property
    # These can be modified directly to adjust tournament predictions without code changes
    # Example: predictor.TOURNAMENT_SCALING_FACTORS["championship_pct"] = 0.15
    TOURNAMENT_SCALING_FACTORS = {
        "championship_pct": 0.15,  # Championship: 0.15 points per percentage point
        "final_four_pct": 0.10,     # Final Four: 0.10 points per percentage point
        "elite_eight_pct": 0.07,    # Elite Eight: 0.07 points per percentage point
        "sweet_sixteen_pct": 0.030, # Sweet Sixteen: 0.030 points per percentage point
        "round_32_pct": 0.015,     # Round of 32: 0.015 points per percentage point
    }
    
    # Baseline expectations for tournament appearance
    # Teams below these baselines will be penalized
    TOURNAMENT_BASELINE_EXPECTATIONS = {
        "championship_pct": 0.5,    # Expected to have at least 0.5% championship chance
        "final_four_pct": 2.0,      # Expected to have at least 2% Final Four chance
        "elite_eight_pct": 5.0,     # Expected to have at least 5% Elite Eight chance
        "sweet_sixteen_pct": 10.0,  # Expected to have at least 10% Sweet Sixteen chance
        "round_32_pct": 20.0,       # Expected to have at least 20% Round of 32 chance
    }
    
    # Penalty factor for teams below baseline (as a proportion of the scaling factor)
    # A value of 1.0 means penalties are equally as strong as bonuses
    TOURNAMENT_PENALTY_FACTOR = 1.5
    
    # Thresholds for identifying potential "under" games
    UNDER_STRONG_DEFENSE_PERCENTILE = 75  # Teams above the 75th percentile in defense are "strong defensive teams"
    UNDER_SLOW_TEMPO_PERCENTILE = 25      # Teams below the 25th percentile in tempo are "slow-paced teams"
    UNDER_PROBABILITY_THRESHOLD = 0.65    # 65% confidence to recommend an under
    
    # Offensive vs defensive weighting ratio - now balanced 1:1 (was 2:1)
    OFF_DEF_WEIGHT_RATIO = 1.0
    
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
        self.bart_data = None    # New field for BART (Bart Torvik) data
        self.bart_historical_model = None  # Will hold our historical model instance
        
        # Define core metrics that are most likely available in KenPom data
        self.core_metrics = [
            'AdjEM', 'AdjOE', 'AdjDE', 'AdjTempo'
        ]
        
        # Define BART metrics that we'll try to use if available
        self.bart_metrics = [
            'barthag', 'WAB', 'adj_o', 'adj_d', 'adj_t', 
            'ov_rtg', 'wins', 'losses', 'conf_wins', 'conf_losses'
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
        
        # Cache for percentile calculations
        self.defensive_percentiles = None
        self.tempo_percentiles = None
        
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
            'AdjTempo': ['AdjTempo', 'Adj Tempo', 'Adjusted Tempo', 'Tempo'],
            # Add BART metrics mapping
            'barthag': ['barthag', 'barthag_rating', 'bart_power'],
            'WAB': ['WAB', 'wins_above_bubble', 'wins_vs_bubble'],
            'adj_o': ['adj_o', 'adj_offense', 'bart_offense'],
            'adj_d': ['adj_d', 'adj_defense', 'bart_defense'],
            'adj_t': ['adj_t', 'adj_tempo', 'bart_tempo']
        }
        
        # Home court advantage in points
        self.home_court_advantage = 3.5
        
        self._load_data()
        self._load_bart_data()  # Load BART data
        self._detect_available_columns()
        self.active_metrics = self._get_active_metrics()
        self._load_tournament_prediction_data()
        self._initialize_bart_historical_model()  # Initialize historical model
        self._calculate_percentiles()  # Calculate percentiles for defense and tempo
    
    def _initialize_bart_historical_model(self):
        """Initialize the BART historical model for enhanced predictions"""
        if BartHistoricalModel is not None:
            try:
                # First, try to load a pre-trained model if it exists
                model = BartHistoricalModel()
                
                # Check if a pre-saved model exists
                model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bart_historical_model.json')
                if os.path.exists(model_file):
                    if model.load_model(model_file):
                        self.bart_historical_model = model
                        print(f"Loaded pre-trained BART historical model")
                        return
                
                # If no pre-saved model, create a new one
                # This may take a while as it processes all historical data
                self.bart_historical_model = model
                print("Initialized BART historical model")
            except Exception as e:
                print(f"Error initializing BART historical model: {str(e)}")
                self.bart_historical_model = None
    
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
    
    def _load_bart_data(self):
        """
        Load BART (Bart Torvik) data for the current season
        This data provides additional insights beyond KenPom metrics
        """
        try:
            # Try to load from the BART folder
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            bart_file = os.path.join(base_path, "BART/2024_team_results.csv")
            
            if os.path.exists(bart_file):
                self.bart_data = pd.read_csv(bart_file)
                
                # Standardize team names for better matching
                if 'team' in self.bart_data.columns:
                    self.bart_data['TeamName'] = self.bart_data['team'].str.strip()
                elif 'Team' in self.bart_data.columns:
                    self.bart_data['TeamName'] = self.bart_data['Team'].str.strip()
                
                print(f"Loaded BART data for {len(self.bart_data)} teams")
                
                # Add mapping between KenPom team names and BART team names if needed
                self._create_team_name_mapping()
            else:
                print(f"BART data file not found at {bart_file}")
                self.bart_data = None
        except Exception as e:
            print(f"Error loading BART data: {str(e)}")
            self.bart_data = None
    
    def _create_team_name_mapping(self):
        """
        Create a mapping between KenPom team names and BART team names
        to handle differences in naming conventions
        """
        if self.bart_data is None or self.current_data is None:
            return
        
        # Common name variations to handle
        name_variations = {
            "NC State": ["North Carolina St.", "NC State", "N.C. State"],
            "UConn": ["Connecticut", "UConn", "Connecticut"],
            "USC": ["Southern California", "USC", "Southern Cal"],
            "SMU": ["Southern Methodist", "SMU"],
            "UCF": ["Central Florida", "UCF"],
            "UNC": ["North Carolina", "UNC"],
            "UNLV": ["Nevada Las Vegas", "UNLV"],
            "VCU": ["Virginia Commonwealth", "VCU"],
            "BYU": ["Brigham Young", "BYU"],
            "LSU": ["Louisiana State", "LSU"],
            "Ole Miss": ["Mississippi", "Ole Miss"],
            "Pitt": ["Pittsburgh", "Pitt"],
            "UMass": ["Massachusetts", "UMass"],
            "UTEP": ["Texas El Paso", "UTEP"],
            "UAB": ["Alabama Birmingham", "UAB"],
            "St. John's": ["St. John's (NY)", "St. John's", "Saint John's"],
            "UIC": ["Illinois Chicago", "UIC"],
            "ETSU": ["East Tennessee St.", "ETSU", "East Tennessee State"],
        }
        
        # Create a dictionary to map BART team names to KenPom team names
        self.team_name_mapping = {}
        
        # First, try exact matches
        kenpom_teams = set(self.current_data['TeamName'].str.strip())
        bart_teams = set(self.bart_data['TeamName'].str.strip())
        
        # For teams with exact matches, create direct mappings
        for team in bart_teams:
            if team in kenpom_teams:
                self.team_name_mapping[team] = team
        
        # For teams without exact matches, try variations
        for kenpom_name in kenpom_teams:
            if kenpom_name not in self.team_name_mapping.values():
                # Check if this team has known variations
                for common_name, variations in name_variations.items():
                    if kenpom_name in variations:
                        # Check if any variation exists in BART data
                        for var in variations:
                            if var in bart_teams:
                                self.team_name_mapping[var] = kenpom_name
                                break
        
        print(f"Created mapping for {len(self.team_name_mapping)} teams between BART and KenPom data")
    
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
    
    def _get_bart_data(self, team_name):
        """
        Get BART data for a specific team
        
        Parameters:
        -----------
        team_name : str
            Name of the team
            
        Returns:
        --------
        dict
            Dictionary with BART metrics or None if data not available
        """
        if self.bart_data is None:
            return None
        
        # Try to find the team in BART data
        team_data = self.bart_data[self.bart_data['TeamName'] == team_name]
        
        # If no direct match, try using the mapping
        if len(team_data) == 0 and hasattr(self, 'team_name_mapping'):
            for bart_name, kenpom_name in self.team_name_mapping.items():
                if kenpom_name == team_name:
                    team_data = self.bart_data[self.bart_data['TeamName'] == bart_name]
                    if len(team_data) > 0:
                        break
        
        if len(team_data) == 0:
            return None
        
        # Extract relevant metrics
        team_row = team_data.iloc[0]
        result = {}
        
        # Try to get each BART metric
        for metric in self.bart_metrics:
            if metric in team_row:
                result[metric] = team_row[metric]
            elif metric.lower() in team_row:
                result[metric] = team_row[metric.lower()]
        
        return result
    
    def _calculate_barthag_win_probability(self, team1_barthag, team2_barthag, location='neutral'):
        """
        Calculate win probability based on Bart Torvik's power ratings (barthag)
        
        Parameters:
        -----------
        team1_barthag : float
            barthag rating for team 1
        team2_barthag : float
            barthag rating for team 2
        location : str
            Game location: 'home_1' (team1 at home), 'home_2' (team2 at home), or 'neutral'
            
        Returns:
        --------
        float
            Win probability for team 1
        """
        if team1_barthag is None or team2_barthag is None:
            return None
        
        # Basic log5 formula with barthag ratings
        team1_wp = team1_barthag / (team1_barthag + (1 - team2_barthag))
        
        # Adjust for home court advantage
        if location == 'home_1':
            team1_wp = min(0.99, team1_wp + 0.04)  # ~4% boost for home team
        elif location == 'home_2':
            team1_wp = max(0.01, team1_wp - 0.04)  # ~4% reduction for away team
        
        return team1_wp
    
    def _calculate_wab_adjustment(self, team1_wab, team2_wab):
        """
        Calculate a point adjustment based on Wins Above Bubble (WAB)
        Teams with higher WAB tend to perform better in high-pressure situations
        
        Parameters:
        -----------
        team1_wab : float
            Wins Above Bubble for team 1
        team2_wab : float
            Wins Above Bubble for team 2
            
        Returns:
        --------
        float
            Point adjustment for team 1 (can be positive or negative)
        """
        if team1_wab is None or team2_wab is None:
            return 0.0
        
        # Calculate the difference in WAB
        wab_diff = team1_wab - team2_wab
        
        # Convert WAB difference to point adjustment
        # Each WAB difference of 3.0 is worth ~1 point
        adjustment = wab_diff / 3.0
        
        # Cap the adjustment at +/- 3 points
        return max(-3.0, min(3.0, adjustment))
    
    def _calculate_percentiles(self):
        """
        Calculate percentiles for defensive efficiency and tempo
        """
        if self.current_data is None:
            return
        
        # Calculate defensive percentiles (lower AdjDE is better defense)
        # We invert the values so that higher percentile = better defense
        if 'AdjDE' in self.current_data.columns:
            self.defensive_percentiles = {}
            # Invert values so higher percentile = better defense
            inverted_defense = self.current_data['AdjDE'].max() - self.current_data['AdjDE']
            for i, row in self.current_data.iterrows():
                team_name = row['TeamName']
                inverted_value = inverted_defense.iloc[i]
                percentile = (inverted_value >= inverted_defense).mean() * 100
                self.defensive_percentiles[team_name] = percentile
        
        # Calculate tempo percentiles (lower = slower)
        if 'AdjTempo' in self.current_data.columns:
            self.tempo_percentiles = {}
            for i, row in self.current_data.iterrows():
                team_name = row['TeamName']
                tempo_value = row['AdjTempo']
                percentile = (tempo_value >= self.current_data['AdjTempo']).mean() * 100
                self.tempo_percentiles[team_name] = percentile
    
    def _calculate_tempo_control_factor(self, team1, team2):
        """
        Calculate which team is likely to control the tempo and how much
        
        Parameters:
        -----------
        team1 : pd.Series
            Statistics for team 1
        team2 : pd.Series
            Statistics for team 2
            
        Returns:
        --------
        tuple
            (team that controls tempo, control factor 0-1)
        """
        team1_tempo = self._safe_get(team1, 'AdjTempo')
        team2_tempo = self._safe_get(team2, 'AdjTempo')
        
        # Get the difference in tempo
        tempo_diff = abs(team1_tempo - team2_tempo)
        
        # Calculate control factor (how much one team controls tempo)
        # Max control is 0.8 (80% control for one team), min is 0.5 (50/50 control)
        control_factor = min(0.8, 0.5 + (tempo_diff / 20) * 0.3)
        
        # Determine which team will control tempo more
        # Teams with extreme tempos tend to control games more
        team1_name = team1['TeamName'] if 'TeamName' in team1.index else 'Team 1'
        team2_name = team2['TeamName'] if 'TeamName' in team2.index else 'Team 2'
        
        # Check if either team has an extreme tempo (very fast or very slow)
        team1_is_extreme = False
        team2_is_extreme = False
        
        if self.tempo_percentiles:
            team1_tempo_pct = self.tempo_percentiles.get(team1_name, 50)
            team2_tempo_pct = self.tempo_percentiles.get(team2_name, 50)
            
            # Teams below 20th percentile (very slow) or above 80th percentile (very fast)
            # tend to control tempo more
            team1_is_extreme = team1_tempo_pct <= 20 or team1_tempo_pct >= 80
            team2_is_extreme = team2_tempo_pct <= 20 or team2_tempo_pct >= 80
        
        # If one team has extreme tempo and the other doesn't, they control the tempo
        if team1_is_extreme and not team2_is_extreme:
            return (team1_name, control_factor)
        elif team2_is_extreme and not team1_is_extreme:
            return (team2_name, control_factor)
        
        # If both or neither have extreme tempos, the team with better tempo consistency controls more
        # For now, use a basic approach (could be enhanced with BART historical data)
        if team1_tempo > team2_tempo:
            return (team1_name, control_factor)
        else:
            return (team2_name, control_factor)
    
    def _calculate_defensive_matchup_strength(self, team1, team2):
        """
        Calculate the strength of the defensive matchup between teams
        
        Parameters:
        -----------
        team1 : pd.Series
            Statistics for team 1
        team2 : pd.Series
            Statistics for team 2
            
        Returns:
        --------
        float
            Defensive matchup score (higher means stronger defensive matchup)
        """
        team1_offense = self._safe_get(team1, 'AdjOE')
        team1_defense = self._safe_get(team1, 'AdjDE')
        team2_offense = self._safe_get(team2, 'AdjOE')
        team2_defense = self._safe_get(team2, 'AdjDE')
        
        # Get team names
        team1_name = team1['TeamName'] if 'TeamName' in team1.index else 'Team 1'
        team2_name = team2['TeamName'] if 'TeamName' in team2.index else 'Team 2'
        
        # Basic defensive strength (lower AdjDE is better defense)
        defensive_strength = (200 - team1_defense - team2_defense) / 2
        
        # Both teams having good defense increases the defensive matchup strength
        both_good_defense = 0
        if self.defensive_percentiles:
            team1_def_pct = self.defensive_percentiles.get(team1_name, 50)
            team2_def_pct = self.defensive_percentiles.get(team2_name, 50)
            
            # Bonus if both teams have strong defense (above 70th percentile)
            if team1_def_pct > 70 and team2_def_pct > 70:
                both_good_defense = (team1_def_pct + team2_def_pct) / 20
        
        # Offensive vs. defensive strength mismatch
        # If teams have much better defense than offense, it's a stronger defensive matchup
        offense_defense_mismatch = ((team1_defense / team1_offense) + (team2_defense / team2_offense)) / 2
        offense_defense_mismatch = max(0, 2 - offense_defense_mismatch) * 5
        
        # Combine factors with weights
        matchup_score = defensive_strength * 0.5 + both_good_defense * 0.3 + offense_defense_mismatch * 0.2
        
        return matchup_score
    
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
        
        # Get BART data for both teams
        team1_bart_data = self._get_bart_data(team1_name)
        team2_bart_data = self._get_bart_data(team2_name)
        
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
        
        # Calculate tempo control (which team controls pace and by how much)
        controlling_team, control_factor = self._calculate_tempo_control_factor(team1, team2)
        
        # Calculate the expected tempo based on control factor
        if controlling_team == team1_name:
            # Team 1 controls more
            avg_tempo = (team1_tempo * control_factor) + (team2_tempo * (1 - control_factor))
        else:
            # Team 2 controls more
            avg_tempo = (team1_tempo * (1 - control_factor)) + (team2_tempo * control_factor)
        
        # Calculate expected points per possession
        team1_off_ppp = team1_adjoe / 100
        team2_off_ppp = team2_adjoe / 100
        
        # Adjusted based on balanced offensive-defensive weighting (now 1:1 instead of 2:1)
        team1_expected_ppp = ((team1_off_ppp * self.OFF_DEF_WEIGHT_RATIO) + (team2_adjde / 100)) / (self.OFF_DEF_WEIGHT_RATIO + 1)
        team2_expected_ppp = ((team2_off_ppp * self.OFF_DEF_WEIGHT_RATIO) + (team1_adjde / 100)) / (self.OFF_DEF_WEIGHT_RATIO + 1)
        
        # Calculate expected score
        team1_expected_score = team1_expected_ppp * avg_tempo
        team2_expected_score = team2_expected_ppp * avg_tempo
        
        # Calculate defensive matchup strength
        defensive_matchup_score = self._calculate_defensive_matchup_strength(team1, team2)
        
        # Apply defensive matchup adjustment
        # Strong defensive matchups tend to reduce scoring
        # Scale: 0-100, with higher values meaning stronger defense
        if defensive_matchup_score > 50:
            defensive_adjustment = (defensive_matchup_score - 50) * 0.10
            team1_expected_score -= defensive_adjustment
            team2_expected_score -= defensive_adjustment
        
        # Apply home court advantage if applicable
        if location == 'home_1':
            team1_expected_score += self.home_court_advantage
        elif location == 'home_2':
            team2_expected_score += self.home_court_advantage
        
        # Apply BART adjustments if available
        bart_adjustment = 0
        barthag_win_prob = None
        wab_adjustment = 0
        
        if team1_bart_data is not None and team2_bart_data is not None:
            # Get barthag ratings
            team1_barthag = team1_bart_data.get('barthag')
            team2_barthag = team2_bart_data.get('barthag')
            
            # Calculate win probability using barthag
            if team1_barthag is not None and team2_barthag is not None:
                barthag_win_prob = self._calculate_barthag_win_probability(team1_barthag, team2_barthag, location)
                
                # Adjust score based on barthag difference
                barthag_diff = team1_barthag - team2_barthag
                # Each 0.1 difference in barthag is ~1.5 points
                bart_adjustment = barthag_diff * 15
                team1_expected_score += bart_adjustment
            
            # Apply WAB adjustment
            team1_wab = team1_bart_data.get('WAB')
            team2_wab = team2_bart_data.get('WAB')
            
            if team1_wab is not None and team2_wab is not None:
                wab_adjustment = self._calculate_wab_adjustment(team1_wab, team2_wab)
                team1_expected_score += wab_adjustment
        
        # Apply tournament prediction adjustment if available
        tournament_adjustment_team1 = 0
        tournament_adjustment_team2 = 0
        # Track detailed adjustment breakdowns
        tournament_adjustment_detail_team1 = {}
        tournament_adjustment_detail_team2 = {}
        
        if team1_tournament_data["championship_pct"] is not None and team2_tournament_data["championship_pct"] is not None:
            # Use the class property for scaling factors
            scaling_factors = self.TOURNAMENT_SCALING_FACTORS
            baseline_expectations = self.TOURNAMENT_BASELINE_EXPECTATIONS
            penalty_factor = self.TOURNAMENT_PENALTY_FACTOR
            
            # Initialize adjustments
            team1_total_adjustment = 0
            team2_total_adjustment = 0
            
            # Apply adjustments for each tournament round if data is available
            for round_key, scaling_factor in scaling_factors.items():
                baseline = baseline_expectations.get(round_key, 0)
                
                # Team 1 adjustment for this round
                if round_key in team1_tournament_data and team1_tournament_data[round_key] is not None:
                    team1_pct = team1_tournament_data[round_key]
                    
                    # If above baseline, apply bonus
                    if team1_pct >= baseline:
                        round_adjustment = team1_pct * scaling_factor
                        adjustment_type = "bonus"
                    # If below baseline, apply penalty
                    else:
                        # Calculate the shortfall from the baseline
                        shortfall = baseline - team1_pct
                        # Apply penalty (with potentially different scaling)
                        round_adjustment = -1 * shortfall * scaling_factor * penalty_factor
                        adjustment_type = "penalty"
                    
                    team1_total_adjustment += round_adjustment
                    # Store the detailed breakdown
                    tournament_adjustment_detail_team1[round_key] = {
                        "percentage": team1_pct,
                        "baseline": baseline,
                        "factor": scaling_factor,
                        "points": round_adjustment,
                        "type": adjustment_type
                    }
                
                # Team 2 adjustment for this round
                if round_key in team2_tournament_data and team2_tournament_data[round_key] is not None:
                    team2_pct = team2_tournament_data[round_key]
                    
                    # If above baseline, apply bonus
                    if team2_pct >= baseline:
                        round_adjustment = team2_pct * scaling_factor
                        adjustment_type = "bonus"
                    # If below baseline, apply penalty
                    else:
                        # Calculate the shortfall from the baseline
                        shortfall = baseline - team2_pct
                        # Apply penalty (with potentially different scaling)
                        round_adjustment = -1 * shortfall * scaling_factor * penalty_factor
                        adjustment_type = "penalty"
                    
                    team2_total_adjustment += round_adjustment
                    # Store the detailed breakdown
                    tournament_adjustment_detail_team2[round_key] = {
                        "percentage": team2_pct,
                        "baseline": baseline,
                        "factor": scaling_factor,
                        "points": round_adjustment,
                        "type": adjustment_type
                    }
            
            # Store the final calculated adjustments
            tournament_adjustment_team1 = team1_total_adjustment
            tournament_adjustment_team2 = team2_total_adjustment
            
            # Apply the adjustments to the expected scores
            team1_expected_score += tournament_adjustment_team1
            team2_expected_score += tournament_adjustment_team2
        
        # Calculate spread (always Team1 - Team2)
        spread = team1_expected_score - team2_expected_score
        
        # Calculate total
        total = team1_expected_score + team2_expected_score
        
        # Analyze if this is a potential under game
        under_analysis = self._analyze_potential_under(team1, team2, team1_name, team2_name, 
                                                      total, defensive_matchup_score, 
                                                      control_factor, controlling_team)
        
        # Calculate win probability using log5 formula
        team1_raw_wp = 1 / (1 + 10 ** (-(team1_adjoe - team1_adjde - (team2_adjoe - team2_adjde)) / 100))
        
        # Adjust win probability for home court if applicable
        if location == 'home_1':
            team1_wp = self._adjust_win_probability(team1_raw_wp, self.home_court_advantage)
        elif location == 'home_2':
            team1_wp = self._adjust_win_probability(team1_raw_wp, -self.home_court_advantage)
        else:
            team1_wp = team1_raw_wp
        
        # If we have barthag win probability, blend it with the KenPom-based probability
        if barthag_win_prob is not None:
            # Weight barthag win probability at 40% and KenPom at 60%
            team1_wp = (0.6 * team1_wp) + (0.4 * barthag_win_prob)
        
        # Apply any tournament model adjustments
        if tournament_adjustment_team1 != 0 or tournament_adjustment_team2 != 0:
            total_point_adjustment = tournament_adjustment_team1 - tournament_adjustment_team2
            team1_wp = self._adjust_win_probability(team1_wp, total_point_adjustment)
        
        team2_wp = 1 - team1_wp
        
        # Calculate key factors for the matchup
        key_factors = self._calculate_key_factors(team1, team2)
        
        # Calculate stat comparisons for display
        stat_comparisons = self._calculate_stat_comparisons(team1, team2)
        
        # Return the prediction data
        result = {
            "team1": {
                "name": team1_name,
                "predicted_score": team1_expected_score,
                "win_probability": team1_wp,
                "tournament_data": team1_tournament_data,
                "height_data": team1_height_data,
                "bart_data": team1_bart_data,  # Add BART data
                "tournament_adjustment_detail": tournament_adjustment_detail_team1
            },
            "team2": {
                "name": team2_name,
                "predicted_score": team2_expected_score,
                "win_probability": team2_wp,
                "tournament_data": team2_tournament_data,
                "height_data": team2_height_data,
                "bart_data": team2_bart_data,  # Add BART data
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
            "bart_adjustment": bart_adjustment,
            "wab_adjustment": wab_adjustment,
            "barthag_win_prob": barthag_win_prob,
            "tempo_control": {
                "controlling_team": controlling_team,
                "control_factor": control_factor
            },
            "defensive_matchup": {
                "score": defensive_matchup_score
            },
            "total_analysis": under_analysis
        }
        
        return result
    
    def _analyze_potential_under(self, team1, team2, team1_name, team2_name, 
                                predicted_total, defensive_matchup_score, 
                                control_factor, controlling_team):
        """
        Analyze if this is likely to be an "under" game
        
        Parameters:
        -----------
        team1, team2 : pd.Series
            Statistics for both teams
        team1_name, team2_name : str
            Team names
        predicted_total : float
            Predicted total score
        defensive_matchup_score : float
            Score for the defensive matchup strength
        control_factor : float
            How much the controlling team controls the pace (0.5-0.8)
        controlling_team : str
            Name of the team controlling the pace
            
        Returns:
        --------
        dict
            Dictionary with under analysis results
        """
        # Initialize factors that contribute to under potential
        factors = []
        factor_weights = []
        
        # Check if either team has strong defense
        team1_defense_strong = False
        team2_defense_strong = False
        
        if self.defensive_percentiles:
            team1_def_pct = self.defensive_percentiles.get(team1_name, 50)
            team2_def_pct = self.defensive_percentiles.get(team2_name, 50)
            
            team1_defense_strong = team1_def_pct >= self.UNDER_STRONG_DEFENSE_PERCENTILE
            team2_defense_strong = team2_def_pct >= self.UNDER_STRONG_DEFENSE_PERCENTILE
            
            # Factor 1: Both teams have strong defense
            if team1_defense_strong and team2_defense_strong:
                factors.append("Both teams have strong defense")
                factor_weights.append(0.25)
            # Factor 2: At least one team has elite defense (top 10%)
            elif team1_def_pct >= 90 or team2_def_pct >= 90:
                factors.append("One team has elite defense")
                factor_weights.append(0.20)
            # Factor 3: One team has strong defense
            elif team1_defense_strong or team2_defense_strong:
                factors.append("One team has strong defense")
                factor_weights.append(0.10)
        
        # Check if the controlling team is slow-paced
        if self.tempo_percentiles:
            controlling_team_tempo_pct = self.tempo_percentiles.get(controlling_team, 50)
            strong_control = control_factor >= 0.65
            
            # Factor 4: Controlling team is slow-paced and has strong control
            if controlling_team_tempo_pct <= self.UNDER_SLOW_TEMPO_PERCENTILE and strong_control:
                factors.append("Slow-paced team strongly controls tempo")
                factor_weights.append(0.25)
            # Factor 5: Controlling team is slow-paced
            elif controlling_team_tempo_pct <= self.UNDER_SLOW_TEMPO_PERCENTILE:
                factors.append("Slow-paced team controls tempo")
                factor_weights.append(0.15)
            # Factor 6: Controlling team has strong control
            elif strong_control:
                factors.append("Strong tempo control")
                factor_weights.append(0.05)
        
        # Factor 7: Strong defensive matchup score
        if defensive_matchup_score >= 70:
            factors.append("Exceptionally strong defensive matchup")
            factor_weights.append(0.25)
        elif defensive_matchup_score >= 60:
            factors.append("Strong defensive matchup")
            factor_weights.append(0.15)
        
        # Factor 8: Historical matchup data suggests low scoring
        if 'historical_data' in locals() and isinstance(historical_data, dict) and historical_data.get('total_matchups', 0) >= 3:
            avg_total = historical_data.get('avg_total', None)
            if avg_total and avg_total < predicted_total * 0.9:  # Historical average at least 10% below prediction
                factors.append("Historical matchups have been low-scoring")
                factor_weights.append(0.15)
        
        # Calculate overall under probability based on weights
        under_probability = 0
        if factors:
            under_probability = sum(factor_weights)
        
        # Determine recommendation
        recommendation = "No clear indication"
        if under_probability >= self.UNDER_PROBABILITY_THRESHOLD:
            recommendation = "Strong under opportunity"
        elif under_probability >= 0.4:
            recommendation = "Possible under opportunity"
        
        # Return analysis
        return {
            "predicted_total": predicted_total,
            "under_probability": under_probability,
            "recommendation": recommendation,
            "factors": factors,
            "factor_weights": factor_weights
        }
    
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
        total_points = []
        
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
                        
                        # Estimate score
                        if 'AdjOE' in team1_metrics and 'AdjOE' in team2_metrics and 'AdjDE' in team1_metrics and 'AdjDE' in team2_metrics and 'AdjTempo' in team1_metrics and 'AdjTempo' in team2_metrics:
                            team1_o = team1_metrics['AdjOE']
                            team1_d = team1_metrics['AdjDE']
                            team2_o = team2_metrics['AdjOE']
                            team2_d = team2_metrics['AdjDE']
                            avg_tempo = (team1_metrics['AdjTempo'] + team2_metrics['AdjTempo']) / 2
                            
                            # Estimate points
                            team1_pts = (team1_o / 100) * avg_tempo
                            team2_pts = (team2_o / 100) * avg_tempo
                            game_total = team1_pts + team2_pts
                            total_points.append(game_total)
                        
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
        avg_total = 0
        if historical_matchups:
            margins = [matchup['diff'] for matchup in historical_matchups]
            avg_margin = sum(margins) / len(margins) if margins else 0
            
            if total_points:
                avg_total = sum(total_points) / len(total_points)
            
        return {
            'team1_wins': team1_wins,
            'team2_wins': team2_wins,
            'total_matchups': len(historical_matchups),
            'matchups': historical_matchups,
            'avg_margin': avg_margin,
            'avg_total': avg_total
        }
    
    def _calculate_stat_comparisons(self, team1, team2):
        """Calculate statistical comparisons between teams for display"""
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
                "advantage": team1['TeamName'] if (higher_is_better and difference > 0) or (not higher_is_better and difference < 0) else team2['TeamName']
            })
        
        return stat_comparisons
    
    def predict_game_with_history(self, team1_name, team2_name, location='neutral'):
        """
        Predict a game with additional historical trend analysis
        
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
            Enhanced prediction with historical adjustments
        """
        # First, get the base prediction
        base_prediction = self.predict_game(team1_name, team2_name, location)
        
        # If no historical model is available, return the base prediction
        if self.bart_historical_model is None:
            base_prediction['historical_model_available'] = False
            return base_prediction
        
        # Enhance the prediction with historical data
        enhanced_prediction = self.bart_historical_model.enhance_game_prediction(base_prediction)
        enhanced_prediction['historical_model_available'] = True
        
        return enhanced_prediction

def load_bart_historical_data(year=None, base_dir=None):
    """
    Utility function to load historical BART data for analysis
    
    Parameters:
    -----------
    year : int or None
        Year to load (e.g., 2019, 2020). If None, loads data for all available years
    base_dir : str or None
        Base directory for BART data. If None, tries to find the BART folder
        
    Returns:
    --------
    dict or pd.DataFrame
        Dictionary mapping years to DataFrames or a single DataFrame if year specified
    """
    if base_dir is None:
        # Try to find the BART folder
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "BART")
    
    if year is not None:
        # Load data for specific year
        file_path = os.path.join(base_dir, f"{year}_team_results.csv")
        if os.path.exists(file_path):
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                print(f"Error loading BART data for {year}: {str(e)}")
                return None
        else:
            print(f"No BART data available for {year}")
            return None
    else:
        # Load data for all available years
        result = {}
        for potential_year in range(2009, 2025):
            file_path = os.path.join(base_dir, f"{potential_year}_team_results.csv")
            if os.path.exists(file_path):
                try:
                    result[potential_year] = pd.read_csv(file_path)
                except Exception as e:
                    print(f"Error loading BART data for {potential_year}: {str(e)}")
        
        if result:
            print(f"Loaded BART data for {len(result)} years")
        else:
            print("No BART data found")
        
        return result

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