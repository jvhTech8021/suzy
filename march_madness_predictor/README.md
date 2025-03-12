# March Madness Predictor 2025

A comprehensive tool for analyzing and predicting NCAA tournament outcomes for the 2025 March Madness tournament.

## Overview

The March Madness Predictor combines multiple prediction models to provide insights into which teams are most likely to succeed in the tournament. It uses KenPom efficiency metrics and historical tournament data to identify patterns that correlate with tournament success.

The project includes:

- **Champion Profile Model**: Identifies teams that most closely resemble the statistical profile of historical NCAA champions.
- **Exit Round Model**: Uses deep learning to predict how far teams will advance in the tournament.
- **Combined Model**: Integrates predictions from both models for a comprehensive view of tournament potential.
- **Interactive Dashboard**: Visualize and explore predictions and team statistics.

## Project Structure

```
march_madness_predictor/
├── app.py                          # Main dashboard application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/                           # Data directory (not included in repo)
├── models/                         # Model implementations
│   ├── champion_profile/           # Champion profile model
│   │   ├── champion_profile_model.py
│   │   └── run_champion_profile_model.py
│   └── exit_round/                 # Exit round prediction model
│       ├── exit_round_model.py
│       └── run_exit_round_model.py
├── dashboard/                      # Dashboard components
│   ├── components/                 # Reusable dashboard components
│   │   └── navbar.py
│   └── pages/                      # Dashboard pages
│       ├── home.py
│       ├── champion_profile.py
│       ├── exit_round.py
│       ├── combined_model.py
│       ├── team_explorer.py
│       └── about.py
└── utils/                          # Utility functions
    └── data_loader.py              # Data loading utilities
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd march_madness_predictor
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the KenPom data files in the `susan_kenpom` directory:
   - `summary25.csv`: Current season KenPom data
   - `processed_{year}.csv`: Historical KenPom data with tournament results (for years 2009-2024)

## Usage

### Running the Models

1. Run the Champion Profile model:
```bash
python models/champion_profile/run_champion_profile_model.py
```

2. Run the Exit Round model:
```bash
python models/exit_round/run_exit_round_model.py
```

### Running the Dashboard

Start the dashboard application:
```bash
python app.py
```

Then open your browser and navigate to `http://127.0.0.1:8050/` to view the dashboard.

## Dashboard Features

- **Home**: Overview of top championship and Final Four contenders.
- **Champion Profile**: Teams that most closely resemble historical champions.
- **Exit Round**: Predictions for how far teams will advance in the tournament.
- **Combined Model**: Integrated predictions from both models.
- **Team Explorer**: Detailed analysis of individual teams.
- **About**: Information about the project and methodology.

## Data Sources

The predictions are based on KenPom efficiency metrics, which adjust for pace of play and strength of schedule to provide an accurate assessment of team strength. The historical data used for training includes tournament results from 2009-2024 (excluding 2020 due to COVID-19 cancellation).

## Limitations

While the models provide valuable insights, it's important to recognize their limitations:

- The NCAA tournament is inherently unpredictable, with upsets being a defining characteristic.
- The models are based on historical data and may not account for unique circumstances.
- Team dynamics, injuries, and other intangible factors can significantly impact tournament performance.
- Tournament seeds for 2025 are estimated based on team rankings, as the actual seeding has not yet been determined.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 