# NCAA Basketball Under Prediction System

This enhancement to the March Madness Predictor adds powerful capabilities for identifying potential "under" betting opportunities in NCAA basketball games. By analyzing defensive strength, tempo control, and historical matchup data, the system can identify games with a high probability of going under the total.

## Features

- **Balanced Offense/Defense Weighting**: The prediction model now gives equal weight to offensive and defensive metrics (1:1 ratio instead of 2:1), leading to more accurate total predictions.
- **Defensive Matchup Scoring**: A sophisticated scoring system evaluates the defensive strength of a matchup using multiple factors.
- **Tempo Control Analysis**: Evaluates which team is likely to control the pace and by how much, rather than using a simple average.
- **Historical Under Performance**: Analyzes historical matchups to identify patterns in scoring.
- **Under Probability Assessment**: Calculates the probability of a game going under the predicted total based on multiple factors.
- **Defense and Tempo Percentiles**: Ranks all teams by defensive efficiency and tempo to identify exceptional teams.

## Key Concepts

### 1. Defensive Matchup Score

The system calculates a defensive matchup score that considers:
- The defensive efficiency of both teams
- How many standard deviations above average each team's defense is
- The bonus when both teams have strong defense
- The ratio of defensive strength to offensive strength for both teams

### 2. Tempo Control

Instead of simply averaging both teams' preferred tempos, the system determines:
- Which team is likely to control the pace based on their tempo extremity
- How strongly they will control it (control factor from 0.5 to 0.8)
- How this control impacts the expected total possessions in the game

### 3. Under Probability Factors

The system considers 8 key factors to determine the probability of an under:

1. **Both teams have strong defense** (high percentile ranking)
2. **One team has elite defense** (top 10%)
3. **One team has strong defense** (above 75th percentile)
4. **Slow-paced team strongly controls tempo**
5. **Slow-paced team controls tempo**
6. **Strong tempo control** by either team
7. **Exceptionally strong defensive matchup score**
8. **Historical matchups have been low-scoring**

## Using the Under Prediction System

### Basic Usage

```python
from march_madness_predictor.models.game_predictor import GamePredictor

# Create predictor
predictor = GamePredictor()

# Predict a game with under analysis
prediction = predictor.predict_game("Virginia", "Wisconsin", "neutral")

# Access the under analysis
under_analysis = prediction['total_analysis']
print(f"Under probability: {under_analysis['under_probability']*100:.1f}%")
print(f"Recommendation: {under_analysis['recommendation']}")
```

### Example Output

```
Predicted Total: 126.8
Under Probability: 72.5%
Recommendation: Strong under opportunity

Factors contributing to under potential:
  1. Both teams have strong defense (25.0%)
  2. Slow-paced team strongly controls tempo (25.0%)
  3. Exceptionally strong defensive matchup (22.5%)
```

### Running the Example Script

For a comprehensive demonstration, run the example script:

```bash
cd march_madness_predictor
python examples/under_prediction_example.py
```

This will:
1. Analyze individual matchups for under potential
2. Perform batch analysis on multiple games
3. Identify the best under opportunities from top defensive teams

## Technical Details

### Defensive Percentiles

Teams are ranked into percentiles based on adjusted defensive efficiency. Higher percentiles indicate stronger defensive teams. The system considers teams above the 75th percentile as "strong defensive teams" and those above the 90th percentile as "elite defensive teams."

### Tempo Percentiles

Teams are ranked into percentiles based on adjusted tempo. Higher percentiles indicate faster-paced teams. The system considers teams below the 25th percentile as "slow-paced teams."

### Under Probability Threshold

The system uses a 65% confidence threshold to recommend a "Strong under opportunity." Games with a 40-65% probability are labeled as "Possible under opportunity."

### Historical Matchup Analysis

The system now tracks scoring in historical matchups between teams to identify patterns. If historical matchups have averaged at least 10% below the current predicted total, this is considered a strong indicator for an under.

## Tips for Best Results

1. **Focus on games between strong defensive teams**: Matchups where both teams rank above the 75th percentile in defense are especially promising.

2. **Look for slow-paced teams with control**: When a slow-paced team (below 25th percentile in tempo) has a high control factor (above 0.65), they're likely to drag the pace down.

3. **Consider defensive consistency**: Teams with consistent defensive performance (low standard deviation) are more reliable for under predictions.

4. **Pay attention to tournament settings**: Conference tournaments and NCAA tournament games often feature stronger defensive focus and lower scoring.

5. **Watch for rematches**: When teams have played before and scored well below the current predicted total, there's a strong likelihood of another under.

## Future Enhancements

Possible future improvements include:
- Incorporating defensive consistency across seasons
- Analyzing coaching styles in defensive matchups
- Tracking referee tendencies and their impact on scoring
- More sophisticated historical pattern matching 