import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from models.game_predictor import GamePredictor

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
        
        dbc.Row([
            # Team 1 selection
            dbc.Col([
                html.H4("Team 1"),
                dcc.Dropdown(
                    id="team1-dropdown",
                    options=[{"label": team, "value": team} for team in teams],
                    placeholder="Select Team 1",
                    className="mb-3"
                )
            ], md=4),
            
            # Team 2 selection
            dbc.Col([
                html.H4("Team 2"),
                dcc.Dropdown(
                    id="team2-dropdown",
                    options=[{"label": team, "value": team} for team in teams],
                    placeholder="Select Team 2",
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
                        {"label": "Team 1 Home", "value": "home_1"},
                        {"label": "Team 2 Home", "value": "home_2"}
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
            # Spread input
            dbc.Col([
                html.Label("Vegas Spread (positive if Team 1 favored, negative if Team 2 favored)"),
                dbc.Input(
                    id="vegas-spread-input",
                    type="number",
                    placeholder="e.g. -3.5",
                    step=0.5,
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
            ], md=4)
        ]),
        
        # Prediction results
        html.Div(id="prediction-results", className="mt-4")
    ])

# Callback to predict the game and display results
@callback(
    Output("prediction-results", "children"),
    [Input("predict-button", "n_clicks")],
    [State("team1-dropdown", "value"),
     State("team2-dropdown", "value"),
     State("location-radio", "value"),
     State("vegas-spread-input", "value"),
     State("vegas-total-input", "value")],
    prevent_initial_call=True
)
def predict_game(n_clicks, team1, team2, location, vegas_spread, vegas_total):
    """
    Predict the game and display results
    
    Parameters:
    -----------
    n_clicks : int
        Number of times the button has been clicked
    team1 : str
        Name of team 1
    team2 : str
        Name of team 2
    location : str
        Game location (neutral, home_1, or home_2)
    vegas_spread : float
        Vegas spread for the game
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
    
    if not team1 or not team2:
        return html.Div([
            html.P("Please select both teams to predict a game.", className="text-danger")
        ])
    
    if team1 == team2:
        return html.Div([
            html.P("Please select different teams for the prediction.", className="text-danger")
        ])
    
    # Create predictor with data_loader
    predictor = GamePredictor(data_loader)
    prediction = predictor.predict_game(team1, team2, location)
    
    # Check for errors
    if "error" in prediction:
        return html.Div([
            html.P(f"Error: {prediction['error']}", className="text-danger")
        ])
    
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
    
    # Format scores and spread
    # Add checks for NaN values before rounding
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
        spread_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=abs(spread),
            title={"text": f"Predicted Spread ({spread_text})"},
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
            number={"font": {"size": 26}}
        ))
        
        spread_fig.update_layout(
            height=300,
            margin=dict(l=50, r=50, t=80, b=30)
        )
    else:
        # Create an empty figure with a text message
        spread_fig = go.Figure()
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
        win_prob_fig = go.Figure()
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
    
    # Create key factors table
    key_factors_rows = []
    for factor in prediction["key_factors"]:
        key_factors_rows.append(
            html.Tr([
                html.Td(factor["factor"]),
                html.Td(factor["advantage"]),
                html.Td(factor["description"])
            ])
        )
    
    # Location description
    location_description = ""
    if location == "neutral":
        location_description = "This prediction is for a game at a neutral site."
    elif location == "home_1":
        location_description = f"This prediction includes a home court advantage for {team1}."
    elif location == "home_2":
        location_description = f"This prediction includes a home court advantage for {team2}."
    
    # Tournament prediction data
    team1_tournament_data = prediction["team1"]["tournament_data"]
    team2_tournament_data = prediction["team2"]["tournament_data"]
    has_tournament_data = (team1_tournament_data["has_exit_round_data"] or team1_tournament_data["has_champion_profile_data"] or
                           team2_tournament_data["has_exit_round_data"] or team2_tournament_data["has_champion_profile_data"])
    
    # Height and experience data
    team1_height_data = prediction["team1"]["height_data"]
    team2_height_data = prediction["team2"]["height_data"]
    has_height_data = team1_height_data["has_height_data"] and team2_height_data["has_height_data"]
    
    # Generate tournament data display
    tournament_data_section = []
    if has_tournament_data:
        # Create table rows for tournament data
        tournament_rows = []
        
        # Championship probability row
        if team1_tournament_data["championship_pct"] is not None or team2_tournament_data["championship_pct"] is not None:
            team1_champ = team1_tournament_data["championship_pct"] if team1_tournament_data["championship_pct"] is not None else "N/A"
            team2_champ = team2_tournament_data["championship_pct"] if team2_tournament_data["championship_pct"] is not None else "N/A"
            
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
        if team1_tournament_data["final_four_pct"] is not None or team2_tournament_data["final_four_pct"] is not None:
            team1_ff = team1_tournament_data["final_four_pct"] if team1_tournament_data["final_four_pct"] is not None else "N/A"
            team2_ff = team2_tournament_data["final_four_pct"] if team2_tournament_data["final_four_pct"] is not None else "N/A"
            
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
        
        # Predicted Exit round
        if team1_tournament_data["predicted_exit"] is not None or team2_tournament_data["predicted_exit"] is not None:
            team1_exit = team1_tournament_data["predicted_exit"] if team1_tournament_data["predicted_exit"] is not None else "N/A"
            team2_exit = team2_tournament_data["predicted_exit"] if team2_tournament_data["predicted_exit"] is not None else "N/A"
                
            tournament_rows.append(
                html.Tr([
                    html.Td("Predicted Tournament Exit"),
                    html.Td(team1_exit),
                    html.Td(team2_exit)
                ])
            )
        
        # Seed row
        if team1_tournament_data["seed"] is not None or team2_tournament_data["seed"] is not None:
            team1_seed = team1_tournament_data["seed"] if team1_tournament_data["seed"] is not None else "N/A"
            team2_seed = team2_tournament_data["seed"] if team2_tournament_data["seed"] is not None else "N/A"
            
            if isinstance(team1_seed, (int, float)) and not pd.isna(team1_seed):
                team1_seed = f"{int(team1_seed)}"
            if isinstance(team2_seed, (int, float)) and not pd.isna(team2_seed):
                team2_seed = f"{int(team2_seed)}"
                
            tournament_rows.append(
                html.Tr([
                    html.Td("Tournament Seed"),
                    html.Td(team1_seed),
                    html.Td(team2_seed)
                ])
            )
        
        # Create the tournament data section
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
                                "Championship probability adjustment: ",
                                html.Strong(f"{prediction['tournament_adjustment']:.2f} points") if prediction['tournament_adjustment'] != 0 else "None"
                            ]),
                            html.Li([
                                "Seed-based adjustment: ",
                                html.Strong(f"{prediction['seed_adjustment']:.2f} points") if prediction['seed_adjustment'] != 0 else "None"
                            ])
                        ])
                    ]) if prediction['tournament_adjustment'] != 0 or prediction['seed_adjustment'] != 0 else html.Div()
                ])
            ], className="mb-4")
        ]
    
    # Generate height and experience data display
    height_data_section = []
    if has_height_data:
        # Create table rows for height data
        height_rows = []
        
        # Overall Size
        if team1_height_data['size'] is not None and team2_height_data['size'] is not None:
            height_rows.append(
                html.Tr([
                    html.Td("Overall Size"),
                    html.Td(f"{team1_height_data['size']:.1f}"),
                    html.Td(f"{team2_height_data['size']:.1f}")
                ])
            )
        
        # Average Height
        if team1_height_data['hgt5'] is not None and team2_height_data['hgt5'] is not None:
            height_rows.append(
                html.Tr([
                    html.Td("Starting 5 Height"),
                    html.Td(f"{team1_height_data['hgt5']:.1f}\""),
                    html.Td(f"{team2_height_data['hgt5']:.1f}\"")
                ])
            )
        
        # Effective Height
        if team1_height_data['effhgt'] is not None and team2_height_data['effhgt'] is not None:
            height_rows.append(
                html.Tr([
                    html.Td("Effective Height"),
                    html.Td(f"{team1_height_data['effhgt']:.1f}\""),
                    html.Td(f"{team2_height_data['effhgt']:.1f}\"")
                ])
            )
        
        # Experience
        if team1_height_data['experience'] is not None and team2_height_data['experience'] is not None:
            height_rows.append(
                html.Tr([
                    html.Td("Experience"),
                    html.Td(f"{team1_height_data['experience']:.2f} years"),
                    html.Td(f"{team2_height_data['experience']:.2f} years")
                ])
            )
        
        # Bench Minutes
        if team1_height_data['bench'] is not None and team2_height_data['bench'] is not None:
            height_rows.append(
                html.Tr([
                    html.Td("Bench Minutes"),
                    html.Td(f"{team1_height_data['bench']:.1f}%"),
                    html.Td(f"{team2_height_data['bench']:.1f}%")
                ])
            )
        
        # Create the height data section
        if height_rows:  # Only show the section if we have at least one row
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
    
    # Set up Vegas comparison section if provided
    vegas_comparison = None
    if vegas_spread is not None or vegas_total is not None:
        spread_diff = None
        spread_diff_text = None
        if vegas_spread is not None and not pd.isna(spread):
            # Note: Vegas spread is positive when team1 is favored, our spread is team1 - team2
            spread_diff = spread - vegas_spread
            spread_diff_text = f"{abs(round(spread_diff, 1))} points {'higher' if spread_diff > 0 else 'lower'}"
        
        total_diff = None
        total_diff_text = None
        if vegas_total is not None and not pd.isna(total):
            total_diff = total - vegas_total
            total_diff_text = f"{abs(round(total_diff, 1))} points {'higher' if total_diff > 0 else 'lower'}"
        
        vegas_comparison = dbc.Card([
            dbc.CardHeader(html.H4("Vegas Odds Comparison", className="mb-0")),
            dbc.CardBody([
                html.Div([
                    html.H5("Spread Comparison"),
                    html.P([
                        f"Model Spread: {team1} {spread:+.1f} points",
                        html.Br(),
                        f"Vegas Spread: {team1} {vegas_spread:+.1f} points" if vegas_spread is not None else "Vegas Spread: Not provided",
                        html.Br(),
                        html.Strong(f"Difference: Model is {spread_diff_text} than Vegas") if spread_diff is not None else None
                    ])
                ]) if vegas_spread is not None else None,
                
                html.Div([
                    html.H5("Total Comparison", className="mt-3"),
                    html.P([
                        f"Model Total: {total:.1f} points",
                        html.Br(),
                        f"Vegas Total: {vegas_total:.1f} points" if vegas_total is not None else "Vegas Total: Not provided",
                        html.Br(),
                        html.Strong(f"Difference: Model is {total_diff_text} than Vegas") if total_diff is not None else None
                    ])
                ]) if vegas_total is not None else None,
                
                html.Div([
                    html.H5("Betting Analysis", className="mt-3"),
                    html.P([
                        html.Strong("Spread: "), 
                        f"Consider {team1 if spread_diff > 0 else team2}" if spread_diff is not None and abs(spread_diff) > 2 else "No strong edge",
                        html.Br(),
                        html.Strong("Total: "), 
                        f"Consider {'Over' if total_diff > 0 else 'Under'}" if total_diff is not None and abs(total_diff) > 3 else "No strong edge"
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
                html.P([
                    f"The model {'agrees' if spread_diff is not None and abs(spread_diff) < 2 else 'disagrees'} with the Vegas spread. ",
                    f"The model {'agrees' if total_diff is not None and abs(total_diff) < 3 else 'disagrees'} with the Vegas total."
                ]) if vegas_spread is not None or vegas_total is not None else None
            ])
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
                        ], className="text-center") if isinstance(total, (int, float)) else html.P("Total unavailable", className="text-center")
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
        
        # Add Vegas comparison section to the results if provided
        vegas_comparison if vegas_comparison is not None else None,
        
        # Explanation and confidence section
        explanation_card
    ]) 