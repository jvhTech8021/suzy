import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from models.game_predictor import GamePredictor
from app_components.tournament_breakdown import tournament_adjustment_breakdown

def layout(data_loader=None):
    """
    Create the layout for the game predictor page
    
    Parameters:
    -----------
    data_loader : DataLoader
        Instance of the DataLoader class
        
    Returns:
    --------
    html.Div
        Layout for the game predictor page
    """
    # Get the data loader
    if data_loader is None:
        from march_madness_predictor.utils.data_loader import DataLoader
        data_loader = DataLoader()
    
    # Create the game predictor
    predictor = GamePredictor(data_loader)
    
    # Get the list of teams
    teams = predictor.get_available_teams()
    
    return dbc.Container([
        html.H1("Game Predictor", className="mt-4 mb-4"),
        html.P("Select two teams and predict the outcome of a game between them."),
        
        # Team selection section
        dbc.Row([
            # Favorite Team selection
            dbc.Col([
                html.H4("Favorite Team", className="text-success"),
                dcc.Dropdown(
                    id="favorite-team-dropdown",
                    options=[{"label": team, "value": team} for team in teams],
                    placeholder="Select Favorite Team",
                    className="mb-3"
                )
            ], md=4),
            
            # Underdog Team selection
            dbc.Col([
                html.H4("Underdog Team", className="text-danger"),
                dcc.Dropdown(
                    id="underdog-team-dropdown",
                    options=[{"label": team, "value": team} for team in teams],
                    placeholder="Select Underdog Team",
                    className="mb-3"
                )
            ], md=4),
            
            # Location selection
            dbc.Col([
                html.H4("Location"),
                dcc.RadioItems(
                    id="location-radio",
                    options=[
                        {"label": "Neutral", "value": "neutral"},
                        {"label": "Favorite Home", "value": "home_favorite"},
                        {"label": "Underdog Home", "value": "home_underdog"}
                    ],
                    value="neutral",
                    labelStyle={"marginRight": "15px"},
                    className="mb-3"
                )
            ], md=4)
        ]),
        
        # Vegas odds input section
        dbc.Row([
            dbc.Col([
                html.H4("Vegas Odds (Optional)", className="mt-3 mb-2"),
                html.P("Enter the current spread and total to compare with model predictions.", className="mb-2")
            ], width=12)
        ]),
        
        dbc.Row([
            # Spread input - simplified to always be positive
            dbc.Col([
                html.Label([
                    "Vegas Spread ",
                    html.Span("(Points favorite team is favored by)", className="text-muted"),
                    html.Br(),
                    html.Small("Example: Enter 4 if favorite is favored by 4 points", className="text-muted")
                ]),
                dbc.Input(
                    id="vegas-spread-input",
                    type="number",
                    placeholder="e.g. 3.5",
                    step=0.5,
                    min=0,  # Only allow positive values
                    className="mb-3"
                )
            ], md=4),
            
            # Total input
            dbc.Col([
                html.Label("Vegas Total"),
                dbc.Input(
                    id="vegas-total-input",
                    type="number",
                    placeholder="e.g. 140.5",
                    step=0.5,
                    className="mb-3"
                )
            ], md=4),
            
            # Predict button
            dbc.Col([
                html.Div([
                    dbc.Button(
                        "Predict Game",
                        id="predict-button",
                        color="primary",
                        size="lg",
                        className="mt-4 w-100"
                    )
                ], className="d-flex align-items-end h-100")
            ], md=4),
            
            # Save prediction button
            dbc.Col([
                html.Div([
                    dbc.Button(
                        "Save Prediction",
                        id="save-prediction-button",
                        color="success",
                        size="lg",
                        className="mt-4 w-100"
                    )
                ], className="d-flex align-items-end h-100")
            ], md=4)
        ]),
        
        # Prediction results
        html.Div(id="prediction-results", className="mt-4")
    ])

# Callback to predict the game and display results
@callback(
    Output("prediction-results", "children"),
    [Input("predict-button", "n_clicks")],
    [State("favorite-team-dropdown", "value"),
     State("underdog-team-dropdown", "value"),
     State("location-radio", "value"),
     State("vegas-spread-input", "value"),
     State("vegas-total-input", "value")],
    prevent_initial_call=True
)
def predict_game(n_clicks, favorite_team, underdog_team, location, vegas_spread, vegas_total):
    """
    Predict the game and display results
    
    Parameters:
    -----------
    n_clicks : int
        Number of times the button has been clicked
    favorite_team : str
        Name of the favorite team
    underdog_team : str
        Name of the underdog team
    location : str
        Game location (neutral, home_favorite, or home_underdog)
    vegas_spread : float
        Vegas spread for the game (positive number favoring the favorite team)
    vegas_total : float
        Vegas total for the game
        
    Returns:
    --------
    html.Div
        Prediction results
    """
    # Access the data_loader from the app context
    from dash import callback_context
    from dash.exceptions import PreventUpdate
    import dash
    
    # Get the current app
    app = dash.get_app()
    
    # Get data_loader from app's server object
    data_loader = app.server.data_loader if hasattr(app.server, 'data_loader') else None
    
    if not favorite_team or not underdog_team:
        return html.Div([
            html.P("Please select both teams to predict a game.", className="text-danger")
        ])
    
    if favorite_team == underdog_team:
        return html.Div([
            html.P("Please select different teams for the prediction.", className="text-danger")
        ])
    
    # Map the location value to the legacy format expected by the model
    if location == "home_favorite":
        model_location = "home_1"
    elif location == "home_underdog":
        model_location = "home_2"
    else:
        model_location = "neutral"
    
    # Convert favorite/underdog to team1/team2 format for the prediction model
    team1 = favorite_team
    team2 = underdog_team
    
    # Create predictor with data_loader
    predictor = GamePredictor(data_loader)
    prediction = predictor.predict_game(team1, team2, model_location)
    
    # Check for errors
    if "error" in prediction:
        return html.Div([
            html.P(f"Error: {prediction['error']}", className="text-danger")
        ])
    
    # Convert the Vegas spread to the format expected by the model
    # In the model, +spread means team1 is favored. Since team1 is now always the favorite,
    # we just need to ensure the spread is positive
    model_vegas_spread = abs(float(vegas_spread)) if vegas_spread is not None else None
    
    # Determine winner and loser for display
    if prediction["team1"]["win_probability"] > prediction["team2"]["win_probability"]:
        winner = prediction["team1"]
        loser = prediction["team2"]
        winner_name = team1
        loser_name = team2
    else:
        winner = prediction["team2"]
        loser = prediction["team1"]
        winner_name = team2
        loser_name = team1
    
    # Check if tournament data is available
    team1_tournament_data = prediction["team1"]["tournament_data"] if "tournament_data" in prediction["team1"] else {}
    team2_tournament_data = prediction["team2"]["tournament_data"] if "tournament_data" in prediction["team2"] else {}
    
    has_tournament_data = False
    if "tournament_data" in prediction["team1"] and "tournament_data" in prediction["team2"]:
        has_tournament_data = (
            team1_tournament_data.get("has_exit_round_data", False) or 
            team1_tournament_data.get("has_champion_profile_data", False) or
            team2_tournament_data.get("has_exit_round_data", False) or 
            team2_tournament_data.get("has_champion_profile_data", False)
        )
    
    # Check if height data is available
    team1_height_data = prediction["team1"].get("height_data", {})
    team2_height_data = prediction["team2"].get("height_data", {})
    has_height_data = (
        "height_data" in prediction["team1"] and 
        "height_data" in prediction["team2"] and
        team1_height_data.get("has_height_data", False) and 
        team2_height_data.get("has_height_data", False)
    )
    
    # Format scores and spread
    # Add checks for NaN values before rounding
    # Use the full scores (WITH tournament adjustments) for display to match the spread calculation
    team1_score = round(prediction["team1"]["predicted_score"]) if not pd.isna(prediction["team1"]["predicted_score"]) else "N/A"
    team2_score = round(prediction["team2"]["predicted_score"]) if not pd.isna(prediction["team2"]["predicted_score"]) else "N/A"
    spread = abs(round(prediction["spread"], 1)) if not pd.isna(prediction["spread"]) else "N/A"
    total = round(prediction["total"]) if not pd.isna(prediction["total"]) else "N/A"
    
    # Determine the spread text
    if isinstance(team1_score, int) and isinstance(team2_score, int):
        if team1_score > team2_score:
            spread_text = f"{team1} by {spread}"
        else:
            spread_text = f"{team2} by {spread}"
    else:
        spread_text = "Unable to calculate"
    
    # Create spread gauge
    if not isinstance(spread, str):  # Only create gauge if spread is numeric
        # Get the model's predicted favorite
        model_favored_team = team1 if prediction["team1"]["predicted_score"] > prediction["team2"]["predicted_score"] else team2
        
        # Since we now always enter spreads with favorite/underdog designation:
        # Vegas favored team is always the favorite_team (team1)
        vegas_favored_team = favorite_team
        
        # Calculate the difference between model and Vegas spreads
        model_spread = abs(spread)
        
        # Print detailed debug info about the spreads
        print(f"\nDEBUG INFO:")
        print(f"Model spread (variable): {model_spread} ({type(model_spread)})")
        print(f"Vegas spread (variable): {model_vegas_spread} ({type(model_vegas_spread)})")
        print(f"Team1 score: {team1_score}, Team2 score: {team2_score}")
        print(f"Raw prediction spread: {prediction['spread']}")
        
        # Calculate the spread difference (positive means model predicts favorite to cover)
        # If model agrees with Vegas on favorite team
        if model_vegas_spread is not None and not pd.isna(model_vegas_spread):
            if model_favored_team == vegas_favored_team:
                # Positive spread_diff means model thinks favorite will win by more than Vegas spread
                # Negative spread_diff means model thinks favorite will win by less than Vegas spread
                spread_diff = model_spread - model_vegas_spread
                
                if spread_diff > 0:
                    spread_diff_text = f"{abs(round(spread_diff, 1))} points more than Vegas (model favors {model_favored_team} by more)"
                else:
                    spread_diff_text = f"{abs(round(spread_diff, 1))} points less than Vegas (model favors {model_favored_team} by less)"
            else:
                # Model and Vegas disagree on who's favored
                spread_diff = model_spread + model_vegas_spread
                spread_diff_text = f"{abs(round(spread_diff, 1))} points different (model favors {model_favored_team} instead)"
            
            # Log values for debugging
            print(f"CALCULATION: Model spread: {model_spread}, Vegas spread: {model_vegas_spread}")
            print(f"CALCULATION: Calculated spread_diff = {spread_diff}, abs(spread_diff) = {abs(spread_diff)}")
            print(f"CALCULATION: Model favors: {model_favored_team}, Vegas favors: {vegas_favored_team}")
        else:
            # Handle case when Vegas spread is not provided
            spread_diff = None
            spread_diff_text = None
            print("CALCULATION: Vegas spread not provided, skipping spread difference calculation")
        
        # Determine confidence level based on the absolute difference
        if spread_diff is None:
            spread_confidence = None
            spread_confidence_color = "secondary"
        elif abs(spread_diff) < 0.5:
            spread_confidence = "Very Low"
            spread_confidence_color = "secondary"
        elif abs(spread_diff) < 1.0:
            spread_confidence = "Low"
            spread_confidence_color = "warning"
        elif abs(spread_diff) < 1.5:
            spread_confidence = "Medium"
            spread_confidence_color = "primary"
        else:
            # Anything above 1.5 points difference is high confidence
            spread_confidence = "High"
            spread_confidence_color = "success"
    
    total_diff = None
    total_diff_text = None
    total_confidence = None
    if vegas_total is not None and not pd.isna(total):
        total_diff = total - vegas_total
        total_diff_text = f"{abs(round(total_diff, 1))} points {'higher' if total_diff > 0 else 'lower'}"
        
        # Determine confidence level for total
        if abs(total_diff) < 2:
            total_confidence = "Very Low"
            total_confidence_color = "secondary"
        elif abs(total_diff) < 4:
            total_confidence = "Low"
            total_confidence_color = "warning"
        elif abs(total_diff) < 8:
            total_confidence = "Medium"
            total_confidence_color = "primary"
        else:
            # Anything above 8 points difference is high confidence
            total_confidence = "High"
            total_confidence_color = "success"
    
    # Tournament data section (if available)
    tournament_data_section = []
    if has_tournament_data:
        # Create table rows for tournament data
        tournament_rows = []
        
        # Championship probability row
        if team1_tournament_data.get("championship_pct") is not None or team2_tournament_data.get("championship_pct") is not None:
            team1_champ = team1_tournament_data.get("championship_pct") if team1_tournament_data.get("championship_pct") is not None else "N/A"
            team2_champ = team2_tournament_data.get("championship_pct") if team2_tournament_data.get("championship_pct") is not None else "N/A"
            
            if isinstance(team1_champ, (int, float)):
                team1_champ = f"{team1_champ:.1f}%"
            if isinstance(team2_champ, (int, float)):
                team2_champ = f"{team2_champ:.1f}%"
                
            tournament_rows.append(
                html.Tr([
                    html.Td("Championship Probability"),
                    html.Td(team1_champ),
                    html.Td(team2_champ)
                ])
            )
        
        # Final Four probability row
        if team1_tournament_data.get("final_four_pct") is not None or team2_tournament_data.get("final_four_pct") is not None:
            team1_ff = team1_tournament_data.get("final_four_pct") if team1_tournament_data.get("final_four_pct") is not None else "N/A"
            team2_ff = team2_tournament_data.get("final_four_pct") if team2_tournament_data.get("final_four_pct") is not None else "N/A"
            
            if isinstance(team1_ff, (int, float)):
                team1_ff = f"{team1_ff:.1f}%"
            if isinstance(team2_ff, (int, float)):
                team2_ff = f"{team2_ff:.1f}%"
                
            tournament_rows.append(
                html.Tr([
                    html.Td("Final Four Probability"),
                    html.Td(team1_ff),
                    html.Td(team2_ff)
                ])
            )
        
        # Create the tournament data section if we have rows
        if tournament_rows:
            tournament_data_section = [
                dbc.Row([
                    dbc.Col([
                        html.H4("Tournament Predictions", className="mb-3"),
                        dbc.Table([
                            html.Thead(
                                html.Tr([
                                    html.Th("Metric"),
                                    html.Th(team1),
                                    html.Th(team2)
                                ])
                            ),
                            html.Tbody(tournament_rows)
                        ], bordered=True, hover=True, striped=True),
                        
                        # Add info about how these factors affected the prediction
                        html.Div([
                            html.P([
                                "Tournament prediction data impact on this matchup:"
                            ], className="mt-3 mb-2"),
                            html.Ul([
                                html.Li([
                                    "Net tournament adjustment: ",
                                    html.Strong(f"{prediction.get('tournament_adjustment', 0):.2f} points") 
                                    if prediction.get('tournament_adjustment', 0) != 0 else "None"
                                ]),
                                html.Li([
                                    "Seed-based adjustment: ",
                                    html.Strong(f"{prediction.get('seed_adjustment', 0):.2f} points") 
                                    if prediction.get('seed_adjustment', 0) != 0 else "None"
                                ])
                            ])
                        ]) if prediction.get('tournament_adjustment', 0) != 0 or prediction.get('seed_adjustment', 0) != 0 else html.Div()
                    ])
                ], className="mb-4"),
                
                # Add the detailed tournament adjustment breakdown if available
                tournament_adjustment_breakdown(prediction) if has_tournament_data else html.Div()
            ]
    
    # Height and experience data section (if available)
    height_data_section = []
    if has_height_data:
        # Create table rows for height data
        height_rows = []
        
        # Overall Size
        if team1_height_data.get('size') is not None and team2_height_data.get('size') is not None:
            height_rows.append(
                html.Tr([
                    html.Td("Overall Size"),
                    html.Td(f"{team1_height_data.get('size', 0):.1f}"),
                    html.Td(f"{team2_height_data.get('size', 0):.1f}")
                ])
            )
        
        # Average Height
        if team1_height_data.get('hgt5') is not None and team2_height_data.get('hgt5') is not None:
            height_rows.append(
                html.Tr([
                    html.Td("Starting 5 Height"),
                    html.Td(f"{team1_height_data.get('hgt5', 0):.1f}\""),
                    html.Td(f"{team2_height_data.get('hgt5', 0):.1f}\"")
                ])
            )
        
        # Effective Height
        if team1_height_data.get('effhgt') is not None and team2_height_data.get('effhgt') is not None:
            height_rows.append(
                html.Tr([
                    html.Td("Effective Height"),
                    html.Td(f"{team1_height_data.get('effhgt', 0):.1f}\""),
                    html.Td(f"{team2_height_data.get('effhgt', 0):.1f}\"")
                ])
            )
        
        # Experience
        if team1_height_data.get('experience') is not None and team2_height_data.get('experience') is not None:
            height_rows.append(
                html.Tr([
                    html.Td("Experience"),
                    html.Td(f"{team1_height_data.get('experience', 0):.2f} years"),
                    html.Td(f"{team2_height_data.get('experience', 0):.2f} years")
                ])
            )
        
        # Create the height data section if we have rows
        if height_rows:
            height_data_section = [
                dbc.Row([
                    dbc.Col([
                        html.H4("Height and Experience Data", className="mb-3"),
                        dbc.Table([
                            html.Thead(
                                html.Tr([
                                    html.Th("Metric"),
                                    html.Th(team1),
                                    html.Th(team2)
                                ])
                            ),
                            html.Tbody(height_rows)
                        ], bordered=True, hover=True, striped=True),
                        
                        # Add info about how these factors affected the prediction
                        html.Div([
                            html.P([
                                "Height and experience impact on this matchup:"
                            ], className="mt-3 mb-2"),
                            html.Ul([
                                html.Li([
                                    "Height adjustment: ",
                                    html.Strong(f"{prediction.get('height_adjustment', 0):.2f} points") 
                                    if prediction.get('height_adjustment', 0) != 0 else "None"
                                ]),
                                html.Li([
                                    "Experience adjustment: ",
                                    html.Strong(f"{prediction.get('experience_adjustment', 0):.2f} points")
                                    if prediction.get('experience_adjustment', 0) != 0 else "None"
                                ])
                            ])
                        ])
                    ])
                ], className="mb-4")
            ]
    
    # Location description
    location_description = ""
    if location == "neutral":
        location_description = "This prediction is for a game at a neutral site."
    elif location == "home_favorite":
        location_description = f"This prediction includes a home court advantage for {team1}."
    elif location == "home_underdog":
        location_description = f"This prediction includes a home court advantage for {team2}."
    
    # Create spread gauge (default to empty figure)
    spread_fig = go.Figure()
    
    if not isinstance(spread, str):  # Only create gauge if spread is numeric
        # Determine which team is favored for display
        favored_team = team1 if team1_score > team2_score else team2
        underdog_team = team2 if team1_score > team2_score else team1
        
        # Create a more descriptive title with team names
        gauge_title = f"{favored_team} favored by {abs(spread):.1f} points"
        
        # Add Vegas comparison if available
        if model_vegas_spread is not None and not pd.isna(model_vegas_spread):
            agreement = (model_favored_team == vegas_favored_team)
            
            # Visual marker for agreement/disagreement
            if agreement:
                if spread_diff is not None and spread_diff > 2:
                    agreement_text = f"✓ Model predicts {favored_team} to cover by {spread_diff:.1f} points"
                    agreement_color = "green"
                elif spread_diff is not None and spread_diff < -2:
                    agreement_text = f"✓ Model predicts {underdog_team} to beat the spread by {abs(spread_diff):.1f} points"
                    agreement_color = "green"
                else:
                    agreement_text = f"✓ Model and Vegas are close ({abs(spread_diff):.1f} points difference)" if spread_diff is not None else "✓ Model and Vegas are close (0 points difference)"
                    agreement_color = "blue"
            else:
                agreement_text = f"✗ Model predicts {model_favored_team} to win (opposite of Vegas)"
                agreement_color = "red"
            
            gauge_title = f"{gauge_title}<br><span style='color:{agreement_color};'>{agreement_text}</span>"
        
        spread_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=abs(spread),
            title={"text": gauge_title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 30]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 5], "color": "lightgray"},
                    {"range": [5, 15], "color": "gray"},
                    {"range": [15, 30], "color": "lightblue"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 10
                }
            },
            number={"font": {"size": 26}, "suffix": " pts"}
        ))
        
        # Add confidence level annotation if Vegas spread provided
        if model_vegas_spread is not None and not pd.isna(model_vegas_spread):
            # Determine betting recommendation text
            if spread_diff is None or abs(spread_diff) < 1.5:
                confidence_text = "No Strong Edge"
                confidence_color = "#6c757d"  # secondary
            else:
                # Format the recommendation text
                if agreement:
                    if spread_diff > 1.5:
                        confidence_text = f"Take {favorite_team} -{model_vegas_spread}"
                    else:  # spread_diff < -1.5
                        confidence_text = f"Take {underdog_team} +{model_vegas_spread}"
                else:
                    confidence_text = f"Take {model_favored_team}"
                
                # Set color based on confidence level
                if abs(spread_diff) < 4:
                    confidence_color = "#007bff"  # primary (medium)
                else:
                    confidence_color = "#28a745"  # success (high)
                
            # Add the annotation with a more informative betting recommendation
            spread_fig.add_annotation(
                x=0.5,
                y=0.05,  # Lowered position to avoid overlap with the pts display
                text=confidence_text,
                showarrow=False,
                font=dict(color=confidence_color, size=16)
            )
        
        spread_fig.update_layout(
            height=300,
            margin=dict(l=50, r=50, t=80, b=30)
        )
    else:
        # Create an empty figure with a text message
        spread_fig.add_annotation(
            text="Spread calculation not available",
            showarrow=False,
            font=dict(size=16)
        )
        spread_fig.update_layout(
            height=300,
            margin=dict(l=50, r=50, t=80, b=30)
        )
    
    # Create win probability gauge
    win_prob_fig = go.Figure()
    
    if not pd.isna(winner["win_probability"]):  # Only create gauge if win probability is not NaN
        win_prob_value = winner["win_probability"] * 100
        win_prob_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=win_prob_value,
            title={"text": f"{winner_name} Win Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 60], "color": "lightgray"},
                    {"range": [60, 80], "color": "gray"},
                    {"range": [80, 100], "color": "lightblue"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 70
                }
            },
            number={"suffix": "%", "font": {"size": 26}}
        ))
        
        win_prob_fig.update_layout(
            height=300,
            margin=dict(l=50, r=50, t=80, b=30)
        )
    else:
        # Create an empty figure with a text message
        win_prob_fig.add_annotation(
            text="Win probability calculation not available",
            showarrow=False,
            font=dict(size=16)
        )
        win_prob_fig.update_layout(
            height=300,
            margin=dict(l=50, r=50, t=80, b=30)
        )
    
    # Create key stats comparison
    stats_comparison = []
    if "team_stats" in prediction:
        for stat in prediction["team_stats"]:
            stats_comparison.append({
                "Statistic": stat["stat"],
                team1: stat["team1_value"],
                team2: stat["team2_value"],
                "Difference": stat["difference"]
            })
        
        stats_df = pd.DataFrame(stats_comparison)
        
        # Create stats comparison figure
        fig_stats = go.Figure()
        
        # Add bars for team 1
        fig_stats.add_trace(go.Bar(
            y=stats_df["Statistic"],
            x=stats_df[team1],
            name=team1,
            orientation='h',
            marker=dict(color='rgba(58, 71, 180, 0.6)')
        ))
        
        # Add bars for team 2
        fig_stats.add_trace(go.Bar(
            y=stats_df["Statistic"],
            x=stats_df[team2],
            name=team2,
            orientation='h',
            marker=dict(color='rgba(246, 78, 139, 0.6)')
        ))
        
        fig_stats.update_layout(
            title="Key Statistics Comparison",
            height=500,
            margin=dict(l=0, r=0, t=40, b=0),
            barmode='group',
            xaxis_title="Value",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    else:
        fig_stats = go.Figure()
        fig_stats.add_annotation(
            text="Statistics comparison not available",
            showarrow=False,
            font=dict(size=16)
        )
        fig_stats.update_layout(
            height=300,
            margin=dict(l=50, r=50, t=80, b=30)
        )
    
    # Create key factors table
    key_factors_rows = []
    if "key_factors" in prediction:
        for factor in prediction["key_factors"]:
            key_factors_rows.append(
                html.Tr([
                    html.Td(factor["factor"]),
                    html.Td(factor["advantage"]),
                    html.Td(factor["description"])
                ])
            )
    
    # Set up Vegas comparison section if provided
    vegas_comparison = None
    if vegas_spread is not None or vegas_total is not None:
        vegas_comparison = dbc.Card([
            dbc.CardHeader(html.H4("Vegas Odds Comparison", className="mb-0")),
            dbc.CardBody([
                html.Div([
                    html.H5("Spread Comparison"),
                    html.P([
                        f"Model Spread: {model_favored_team} by {abs(round(spread, 1))} points",
                        html.Br(),
                        f"Vegas Spread: {vegas_favored_team} by {abs(round(vegas_spread, 1))} points" if vegas_spread is not None else "Vegas Spread: Not provided",
                        html.Br(),
                        html.Strong(f"Difference: Model is {spread_diff_text}") if spread_diff is not None else None,
                        html.Br() if spread_diff is not None else None,
                        dbc.Badge(spread_confidence, color=spread_confidence_color, className="mr-1") if spread_confidence is not None else None,
                        html.Span(f" confidence - based on {abs(spread_diff):.1f} point difference", className="text-muted") if spread_diff is not None else None
                    ])
                ]) if vegas_spread is not None else None,
                
                html.Div([
                    html.H5("Total Comparison", className="mt-3"),
                    html.P([
                        f"Model Total: {total:.1f} points",
                        html.Br(),
                        f"Vegas Total: {vegas_total:.1f} points" if vegas_total is not None else "Vegas Total: Not provided",
                        html.Br(),
                        html.Strong(f"Difference: Model is {total_diff_text}") if total_diff is not None else None,
                        html.Br() if total_diff is not None else None,
                        dbc.Badge(total_confidence, color=total_confidence_color, className="mr-1") if total_confidence is not None else None,
                        html.Span(f" confidence - based on {abs(total_diff):.1f} point difference", className="text-muted") if total_diff is not None else None
                    ])
                ]) if vegas_total is not None else None,
                
                html.Div([
                    html.H5("Betting Recommendations", className="mt-3"),
                    html.Div([
                        html.Div([
                            html.Strong("Spread: "), 
                            html.Span([
                                # If model favors same team as Vegas but by MORE points
                                f"Take {favorite_team} -{model_vegas_spread}" if model_favored_team == vegas_favored_team and spread_diff is not None and spread_diff > 1.5 else
                                # If model favors same team as Vegas but by FEWER points
                                f"Take {underdog_team} +{model_vegas_spread}" if model_favored_team == vegas_favored_team and spread_diff is not None and spread_diff < -1.5 else
                                # If model disagrees on who will win
                                f"Take {model_favored_team} {'+' + str(model_vegas_spread) if model_favored_team == underdog_team else '-' + str(model_vegas_spread) if model_favored_team == favorite_team else ''}",
                                dbc.Badge(spread_confidence, color=spread_confidence_color, className="ml-1")
                            # Always show a recommendation if the difference is significant (> 1.5 points)
                            ]) if spread_diff is not None and abs(spread_diff) > 1.5 else 
                              "No strong edge on spread (model and Vegas are within 1.5 points)",
                        ]),
                        html.Div([
                            html.Strong("Total: "), 
                            html.Span([
                                f"Take {'Over' if total_diff > 0 else 'Under'} {vegas_total}",
                                dbc.Badge(total_confidence, color=total_confidence_color, className="ml-1")
                            ]) if total_diff is not None and abs(total_diff) > 3 else "No strong edge on total",
                        ], className="mt-2"),
                        html.Div([
                            html.Span("Note: ", className="font-weight-bold"),
                            "Strong edge exists when model differs from Vegas by more than 1.5 points (spread) or 3 points (total)."
                        ], className="mt-3 text-muted small")
                    ])
                ]) if (vegas_spread is not None or vegas_total is not None) else None
            ])
        ], className="mb-4")
    
    # Explanation and confidence section
    explanation_card = dbc.Card([
        dbc.CardHeader(html.H4("Explanation & Confidence", className="mb-0")),
        dbc.CardBody([
            # ... existing explanation code ...
            
            # Add insights about Vegas line comparison if provided
            html.Div([
                html.H5("Vegas Comparison Insight", className="mt-3"),
                html.Div([
                    html.P([
                        html.Strong("Spread: "),
                        f"The model predicts {model_favored_team} by {abs(round(spread, 1))} ",
                        html.Strong("vs"), 
                        f" Vegas has {vegas_favored_team} by {model_vegas_spread}"
                    ]) if model_vegas_spread is not None else None,
                    html.P([
                        html.Strong("Recommended Action: "),
                        html.Span([
                            # If model favors same team as Vegas but by MORE points
                            f"Take {favorite_team} -{model_vegas_spread}" if model_favored_team == vegas_favored_team and spread_diff is not None and spread_diff > 1.5 else
                            # If model favors same team as Vegas but by FEWER points
                            f"Take {underdog_team} +{model_vegas_spread}" if model_favored_team == vegas_favored_team and spread_diff is not None and spread_diff < -1.5 else
                            # If model disagrees with Vegas on who will win
                            f"Take {model_favored_team} {'+' + str(model_vegas_spread) if model_favored_team == underdog_team else '-' + str(model_vegas_spread) if model_favored_team == favorite_team else ''}",
                            f" ({spread_confidence} confidence)" if spread_confidence is not None else ""
                        # Always show a recommendation if the difference is significant
                        ]) if spread_diff is not None and abs(spread_diff) > 1.5 else 
                           "No action recommended on spread (model and Vegas are within 1.5 points)",
                    ]) if model_vegas_spread is not None else None,
                    html.P([
                        html.Strong("Total: "),
                        f"The model predicts {total:.1f} points ",
                        html.Strong("vs"),
                        f" Vegas total of {vegas_total:.1f}"
                    ]) if vegas_total is not None else None,
                    html.P([
                        html.Strong("Recommended Action: "),
                        html.Span([
                            f"Take {'Over' if total_diff > 0 else 'Under'} {vegas_total}",
                            f" ({total_confidence} confidence)" if total_confidence is not None else ""
                        ]) if total_diff is not None and abs(total_diff) > 3 else "No action recommended on total",
                    ]) if vegas_total is not None else None
                ])
            ]) if vegas_spread is not None or vegas_total is not None else None
        ])
    ], className="mb-4")
    
    # Return the prediction results
    return html.Div([
        html.H3("Game Prediction", className="mb-4"),
        
        # Game context
        html.P(location_description, className="lead"),
        
        # Predicted score, spread and over/under
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Predicted Score"),
                    dbc.CardBody([
                        html.H2([
                            html.Span(team1, className="text-primary"),
                            f" {team1_score} - {team2_score} " if isinstance(team1_score, (int, float)) and isinstance(team2_score, (int, float)) else " Score unavailable ",
                            html.Span(team2, className="text-danger")
                        ], className="text-center"),
                        html.H4([
                            f"Predicted Spread: ",
                            html.Strong(spread_text)
                        ], className="text-center mt-3") if spread_text != "Unable to calculate" else html.H4("Spread unavailable", className="text-center mt-3"),
                        html.P([
                            f"Over/Under: {total} points"
                        ], className="text-center"),
                        html.P([
                            "Note: Scores include tournament probability adjustments"
                        ], className="text-center text-muted small mt-2") if prediction['tournament_adjustment'] != 0 else html.Div()
                    ])
                ], className="mb-4")
            ], md=12, lg=4),
            
            dbc.Col([
                dcc.Graph(
                    figure=spread_fig,
                    config={"displayModeBar": False}
                )
            ], md=12, lg=4),
            
            dbc.Col([
                dcc.Graph(
                    figure=win_prob_fig,
                    config={"displayModeBar": False}
                )
            ], md=12, lg=4)
        ]),
        
        # Add betting recommendation highlight card if Vegas odds are provided
        html.Div([
            dbc.Card([
                dbc.CardHeader(html.H4("Betting Recommendation Summary", className="mb-0 text-center")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Spread Bet", className="text-center"),
                            html.Div([
                                html.H4([
                                    # Print debug info right before making this decision
                                    (print(f"DEBUG - Final check: spread_diff={spread_diff}, abs(spread_diff)={abs(spread_diff) if spread_diff is not None else None}") or True) and (
                                    # Only show "No Strong Edge" when the difference is small
                                    (spread_diff is None or abs(spread_diff) <= 1.5) 
                                    and html.Span("No Strong Edge", className="text-muted")
                                    # Otherwise show a recommendation
                                    or dbc.Badge(
                                        # If model favors same team as Vegas but by MORE points
                                        f"Take {favorite_team} -{model_vegas_spread}" if model_favored_team == vegas_favored_team and spread_diff > 1.5 else
                                        # If model favors same team as Vegas but by FEWER points
                                        f"Take {underdog_team} +{model_vegas_spread}" if model_favored_team == vegas_favored_team and spread_diff < -1.5 else
                                        # If model disagrees with Vegas on who will win
                                        f"Take {model_favored_team} {'+' + str(model_vegas_spread) if model_favored_team == underdog_team else '-' + str(model_vegas_spread) if model_favored_team == favorite_team else ''}",
                                        color=spread_confidence_color,
                                        className="p-2",
                                        style={"font-size": "1rem"}
                                    )
                                    )
                                ], className="text-center mt-3"),
                                html.Div([
                                    html.Strong("Confidence: "),
                                    html.Span(spread_confidence or "Very Low") 
                                ], className="text-center mt-2"),
                                html.Div([
                                    html.Strong("Difference: "),
                                    html.Span(f"{abs(spread_diff):.1f} points" if spread_diff is not None else "N/A")
                                ], className="text-center")
                            ])
                        ], width=6, className="border-right"),
                        
                        dbc.Col([
                            html.H5("Total Bet", className="text-center"),
                            html.Div([
                                html.H4([
                                    # Only show "No Strong Edge" when the difference is small
                                    (total_diff is None or abs(total_diff) <= 3)
                                    and html.Span("No Strong Edge", className="text-muted") 
                                    # Otherwise show a recommendation
                                    or dbc.Badge(
                                        f"Take {'Over' if total_diff > 0 else 'Under'} {vegas_total}",
                                        color=total_confidence_color,
                                        className="p-2",
                                        style={"font-size": "1rem"}
                                    )
                                ], className="text-center mt-3"),
                                html.Div([
                                    html.Strong("Confidence: "),
                                    html.Span(total_confidence or "Very Low")
                                ], className="text-center mt-2"),
                                html.Div([
                                    html.Strong("Difference: "),
                                    html.Span(f"{abs(total_diff):.1f} points" if total_diff is not None else "N/A")
                                ], className="text-center")
                            ])
                        ], width=6)
                    ]),
                    html.Div([
                        html.Hr(),
                        html.P([
                            html.I(className="fas fa-info-circle mr-2"),
                            "Recommendations are based on the difference between model predictions and Vegas lines. ",
                            "Strong edge exists when model differs from Vegas by more than 1.5 points (spread) or 3 points (total)."
                        ], className="text-muted small text-center mt-3 mb-0")
                    ])
                ])
            ], className="mb-4")
        ]) if vegas_spread is not None or vegas_total is not None else html.Div(),
        
        # Vegas comparison card (detailed analysis)
        html.Div([vegas_comparison]) if vegas_comparison is not None else html.Div(),
        
        # Tournament data section (if available)
        html.Div(tournament_data_section) if has_tournament_data else html.Div(),
        
        # Height and experience data section (if available)
        html.Div(height_data_section) if has_height_data else html.Div(),
        
        # Key statistics comparison
        dbc.Row([
            dbc.Col([
                html.H4("Team Statistics Comparison", className="mb-3"),
                dcc.Graph(
                    figure=fig_stats,
                    config={"displayModeBar": False}
                )
            ])
        ], className="mb-4"),
        
        # Key factors
        dbc.Row([
            dbc.Col([
                html.H4("Key Factors", className="mb-3"),
                dbc.Table([
                    html.Thead(
                        html.Tr([
                            html.Th("Factor"),
                            html.Th("Advantage"),
                            html.Th("Description")
                        ])
                    ),
                    html.Tbody(key_factors_rows)
                ], bordered=True, hover=True, striped=True)
            ])
        ]),
        
        # Matchup analysis
        dbc.Row([
            dbc.Col([
                html.H4("Matchup Analysis", className="mb-3"),
                html.P([
                    html.Strong(winner_name), 
                    f" is predicted to defeat ", 
                    html.Strong(loser_name), 
                    f" with a {round(winner['win_probability'] * 100)}% probability. " if not pd.isna(winner['win_probability']) else ". Win probability calculation unavailable. ",
                    f"The projected score is {team1_score}-{team2_score}" if isinstance(team1_score, (int, float)) and isinstance(team2_score, (int, float)) else "Score projection unavailable",
                    f", with a point spread of {spread} in favor of {winner_name}." if isinstance(spread, (int, float)) else "."
                ]),
                html.P([
                    "Key advantages for the predicted winner include: ",
                    ", ".join([f"{factor['factor']}" for factor in prediction["key_factors"] 
                            if factor['advantage'] == winner_name]) if any(factor['advantage'] == winner_name for factor in prediction["key_factors"]) else "None identified"
                ])
            ])
        ]),
        
        # Historical matchups section (if available)
        dbc.Row([
            dbc.Col([
                html.H4("Historical Matchup Data", className="mb-3"),
                html.Div([
                    # Display historical matchup data if available
                    html.P([
                        f"Based on historical data from 2009-2024, {team1} and {team2} have similar statistical profiles in ",
                        html.Strong(f"{prediction['historical_matchups']['total_matchups']}"), " seasons."
                    ]) if prediction['historical_matchups']['total_matchups'] > 0 else html.P("No historical matchup data available."),
                    
                    # Show the win rates if there are historical matchups
                    html.Div([
                        html.P([
                            f"In those seasons, based on efficiency metrics, ",
                            html.Strong(f"{team1}"), f" would have won ",
                            html.Strong(f"{prediction['historical_matchups']['team1_wins']}"),
                            f" times ({round(prediction['historical_matchups']['team1_wins'] / prediction['historical_matchups']['total_matchups'] * 100) if not pd.isna(prediction['historical_matchups']['team1_wins']) and prediction['historical_matchups']['total_matchups'] > 0 else 0}%), and ",
                            html.Strong(f"{team2}"), f" would have won ",
                            html.Strong(f"{prediction['historical_matchups']['team2_wins']}"),
                            f" times ({round(prediction['historical_matchups']['team2_wins'] / prediction['historical_matchups']['total_matchups'] * 100) if not pd.isna(prediction['historical_matchups']['team2_wins']) and prediction['historical_matchups']['total_matchups'] > 0 else 0}%)."
                        ]),
                        
                        # Show the average margin
                        html.P([
                            f"The average adjusted efficiency margin between the teams was ",
                            html.Strong(f"{abs(round(prediction['historical_matchups']['avg_margin'], 1)) if not pd.isna(prediction['historical_matchups']['avg_margin']) else 0}"),
                            f" points in favor of ",
                            html.Strong(f"{team1 if prediction['historical_matchups']['avg_margin'] > 0 else team2}" if not pd.isna(prediction['historical_matchups']['avg_margin']) else "neither team"), "."
                        ]),
                        
                        # Explain how this impacted the prediction
                        html.P([
                            "These historical matchups were factored into the final prediction with a 10% weight, ",
                            "strengthening" if not pd.isna(prediction['historical_matchups']['team1_wins']) and prediction['historical_matchups']['total_matchups'] > 0 and 
                                              not pd.isna(prediction['team1']['win_probability']) and 
                                              (prediction['historical_matchups']['team1_wins'] / prediction['historical_matchups']['total_matchups'] > 0.5) == (prediction['team1']['win_probability'] > 0.5) 
                                           else "tempering",
                            " the statistical model's conclusion."
                        ], className="text-muted")
                    ]) if prediction['historical_matchups']['total_matchups'] > 0 else html.Div()
                ])
            ])
        ]) if 'historical_matchups' in prediction else html.Div(),
        
        # Explanation and confidence section
        explanation_card
    ]) 

# Callback to save the prediction
@callback(
    Output("prediction-results", "children", allow_duplicate=True),
    [Input("save-prediction-button", "n_clicks")],
    [State("favorite-team-dropdown", "value"),
     State("underdog-team-dropdown", "value"),
     State("location-radio", "value"),
     State("vegas-spread-input", "value"),
     State("vegas-total-input", "value")],
    prevent_initial_call=True
)
def save_prediction(n_clicks, favorite_team, underdog_team, location, vegas_spread, vegas_total):
    if not favorite_team or not underdog_team:
        return html.Div([
            html.P("Please select both teams to save a prediction.", className="text-danger")
        ])

    if favorite_team == underdog_team:
        return html.Div([
            html.P("Please select different teams for the prediction.", className="text-danger")
        ])

    # Map the location value to the legacy format expected by the model
    if location == "home_favorite":
        model_location = "home_1"
    elif location == "home_underdog":
        model_location = "home_2"
    else:
        model_location = "neutral"
    
    # Convert favorite/underdog to team1/team2 format for the prediction model
    team1 = favorite_team
    team2 = underdog_team

    # Create predictor with data_loader
    predictor = GamePredictor()
    prediction = predictor.predict_game(team1, team2, model_location)
    
    # Convert the Vegas spread to the format expected by the model
    model_vegas_spread = abs(float(vegas_spread)) if vegas_spread is not None else None
    
    # Add Vegas odds information
    if model_vegas_spread is not None:
        prediction['vegas_spread'] = model_vegas_spread
    if vegas_total is not None:
        prediction['vegas_total'] = vegas_total
        
    predictor.save_prediction(prediction)

    return html.Div([
        html.P("Prediction saved successfully!", className="text-success")
    ])

def save_and_predict_game(team1_name, team2_name, location='neutral', data_loader=None):
    """
    Predict the outcome of a game and save the prediction.

    Parameters:
    -----------
    team1_name : str
        Name of the first team
    team2_name : str
        Name of the second team
    location : str
        Game location: 'home_1' (team1 at home), 'home_2' (team2 at home), or 'neutral'
    data_loader : DataLoader
        Instance of the DataLoader class to load KenPom data

    Returns:
    --------
    dict
        Dictionary with prediction results
    """
    predictor = GamePredictor(data_loader)
    prediction = predictor.predict_game(team1_name, team2_name, location)
    predictor.save_prediction(prediction)
    return prediction

# Example usage
# prediction = save_and_predict_game('Team A', 'Team B') 