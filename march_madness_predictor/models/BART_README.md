# BART-Enhanced NCAA Tournament Prediction System

This enhancement to the March Madness Predictor incorporates data from Bart Torvik's analysis (BART data) to significantly improve prediction accuracy. By leveraging both the current season BART metrics and historical trends, we can make more nuanced predictions that account for a team's trajectory and tournament experience.

## Features

- **BART Metrics Integration**: Uses key metrics like barthag (Bart Torvik's power rating), WAB (Wins Above Bubble), adjusted offense/defense, and more
- **Historical Trend Analysis**: Analyzes multi-year trends to identify teams that are improving or declining
- **Tournament Experience Modeling**: Evaluates teams' historical tournament performance to better predict high-pressure games
- **Team Consistency Metrics**: Quantifies how consistent a team has been over time to assess reliability
- **Enhanced Win Probability**: Incorporates BART data into win probability calculations for more accurate predictions

## How It Works

The system builds on the existing GamePredictor by adding:

1. **BartHistoricalModel**: Loads and processes multi-year BART data to identify trends
2. **Improved GamePredictor**: Enhanced with BART metrics for current season analysis
3. **Integration Layer**: Combines basic KenPom predictions with BART adjustments

## Key Files

- `game_predictor.py`: Enhanced predictor that incorporates BART data
- `bart_historical_model.py`: Model for analyzing historical BART data and team trends
- `examples/bart_prediction_example.py`: Example script demonstrating the enhanced predictions

## Usage

### Basic Game Prediction

```python
from march_madness_predictor.models.game_predictor import GamePredictor

# Create predictor
predictor = GamePredictor()

# Basic prediction (uses BART data if available)
basic_prediction = predictor.predict_game("Gonzaga", "Duke", "neutral")

# Enhanced prediction with historical analysis
enhanced_prediction = predictor.predict_game_with_history("Gonzaga", "Duke", "neutral")

# Analyze the results
print(f"Basic prediction: {basic_prediction['team1']['name']} {basic_prediction['team1']['predicted_score']:.1f}, "
      f"{basic_prediction['team2']['name']} {basic_prediction['team2']['predicted_score']:.1f}")
print(f"Enhanced prediction: {enhanced_prediction['team1']['name']} {enhanced_prediction['team1']['predicted_score']:.1f}, "
      f"{enhanced_prediction['team2']['name']} {enhanced_prediction['team2']['predicted_score']:.1f}")
```

### Historical Team Analysis

```python
from march_madness_predictor.models.bart_historical_model import BartHistoricalModel

# Create the historical model
model = BartHistoricalModel()

# Get trend data for a specific team
gonzaga_trend = model.get_team_trend("Gonzaga")

# Print barthag trend
print(f"Gonzaga barthag trend direction: {gonzaga_trend['barthag_trend']['trend']:.4f} per year")
print(f"Recent tournament appearances: {[th['year'] for th in gonzaga_trend['tournament_history'][-5:]]}")
```

### Tournament Simulation

The example script `examples/bart_prediction_example.py` shows how to simulate an entire tournament region using the enhanced predictions:

```python
# Simulate a tournament game
winner = simulate_tournament_game("Duke", "UCLA", 1, 2)

# Simulate an entire region
east_region = [
    ("Duke", 1), ("Norfolk State", 16),
    ("Tennessee", 8), ("Saint Joseph's", 9),
    # ... other teams in bracket order ...
]
east_winner, east_seed = simulate_tournament_region("East", east_region)
```

## Improvements Over Basic KenPom Predictions

1. **Better Handles Trending Teams**: Teams that are significantly improving or declining throughout the season are more accurately predicted
2. **Accounts for Tournament Experience**: Teams with strong tournament history get appropriate adjustments
3. **Consistency Factor**: Teams with high performance consistency get a slight boost in high-pressure games
4. **WAB Integration**: Integrates Wins Above Bubble data, a strong indicator of team quality
5. **More Accurate Upsets**: Better identifies conditions where lower-seeded teams are likely to pull off upsets

## Data Requirements

The system looks for BART data in the following directory structure:

```
/BART/
  2009_team_results.csv
  2010_team_results.csv
  ...
  2024_team_results.csv
```

Each CSV should contain team statistics from Bart Torvik's analysis for that season, including:
- barthag (team power rating)
- WAB (Wins Above Bubble)
- adj_o (adjusted offense)
- adj_d (adjusted defense)
- Tournament seed (if applicable)

## Running the Example

To see the system in action:

```bash
cd march_madness_predictor
python examples/bart_prediction_example.py
```

This will demonstrate:
1. Prediction comparisons with and without BART data
2. Team historical analysis
3. Tournament region simulation
4. Final Four and Championship predictions

## Technical Details

- The system caches the historical model to avoid reprocessing data on each prediction
- Team name matching handles different naming conventions between datasets
- All adjustments are applied in points first, then converted to win probability adjustments
- Historical data requires at least 3 years of data for a team to calculate reliable trends

## Future Enhancements

Possible future improvements include:
- Incorporating coach-specific historical performance data
- Evaluating team strength in specific tournament contexts (fast-paced games, defensive battles, etc.)
- Machine learning enhancements that automatically weight the various factors
- Integration with network-based modeling of team relationships and transitive results 