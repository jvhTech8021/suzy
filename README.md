# March Madness Predictor Dashboard

A comprehensive dashboard for predicting NCAA March Madness tournament outcomes using KenPom statistics.

## Features

- **Champion Profile Analysis**: Compare teams to historical champions to determine how closely they match the profile of a typical champion.
- **Exit Round Predictions**: Predict how far teams will advance in the tournament.
- **Combined Model**: View combined predictions from both the Champion Profile and Exit Round models.
- **Team Explorer**: Explore detailed statistics for all teams.
- **Full Bracket Generator**: Generate a complete tournament bracket based on model predictions.
- **Game Predictor**: Analyze individual matchups between any two teams.

## Getting Started

### Prerequisites

- Python 3.7+
- Required Python packages (install with `pip install -r requirements.txt`)

### Installation

1. Clone this repository or download the source code
2. Navigate to the project directory
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Running the Dashboard

1. Make sure you have the required data in the `susan_kenpom` folder
2. Run the models to generate predictions:
   ```
   python march_madness_predictor/models/champion_profile/run_champion_profile_model.py
   python march_madness_predictor/models/exit_round/run_exit_round_model.py
   python march_madness_predictor/models/full_bracket/generate_bracket.py
   ```
3. Start the dashboard:
   ```
   python march_madness_predictor/app.py
   ```
4. Open your browser and navigate to `http://127.0.0.1:8050/`

## Troubleshooting

### "Data Not Available" Messages

If you see "Data Not Available" messages on any of the dashboard pages:

1. Make sure you've run the corresponding model script (see above)
2. Check that the model output files exist in the appropriate directories
3. Run the reset_cache.py script to clear any cached data and restart the dashboard:
   ```
   python reset_cache.py
   ```

### Game Predictor Issues

If the Game Predictor shows "No data available" for certain teams:

1. Make sure the KenPom data is correctly formatted in the `susan_kenpom` folder
2. Check that the `summary25.csv` file contains data for all teams
3. Run the reset_cache.py script to restart the dashboard:
   ```
   python reset_cache.py
   ```

### Dashboard Won't Start

If the dashboard fails to start:

1. Check that all required Python packages are installed
2. Verify that the data files exist in the correct locations
3. Look for error messages in the console output
4. Try running with debug mode:
   ```
   python march_madness_predictor/app.py --debug
   ```

## Data Structure

The dashboard expects data in the following structure:

- `susan_kenpom/summary25.csv`: Current season KenPom data
- `susan_kenpom/processed_YYYY.csv`: Historical KenPom data for previous years
- `march_madness_predictor/models/champion_profile/model/`: Output files from the Champion Profile model
- `march_madness_predictor/models/exit_round/model/`: Output files from the Exit Round model
- `march_madness_predictor/models/full_bracket/model/`: Output files from the Full Bracket generator

## Models

### Champion Profile Model

Compares current teams to historical champions based on key metrics:
- Adjusted Efficiency Margin (AdjEM)
- National Ranking
- Offensive Efficiency (AdjOE)
- Defensive Efficiency (AdjDE)

### Exit Round Model

Predicts how far teams will advance in the tournament based on:
- Seed
- KenPom metrics
- Historical performance

### Game Predictor

Predicts the outcome of individual matchups using:
- Team efficiency metrics
- Tempo adjustments
- Round-specific adjustments

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- KenPom.com for the statistical data
- All contributors to the project 