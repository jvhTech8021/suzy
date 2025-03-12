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
    # Initialize the game predictor
    predictor = GamePredictor(data_loader)
    
    # Get available teams for dropdowns
    teams = predictor.get_available_teams()
    
    # Define the layout
    layout = html.Div([
        html.H1("NCAA Game Predictor", className="mb-4"),
        
        html.P([
            "This page allows you to predict the outcome of a game between any two teams using KenPom metrics. ",
            "The prediction model takes into account offensive and defensive efficiencies, tempo, shooting percentages, ",
            "rebounding, turnovers, and other key statistics to generate spread and win probability predictions."
        ], className="lead"),
        
        html.Hr(),
        
        # Team selection
        dbc.Row([
            dbc.Col([
                html.H5("Select Teams for Prediction", className="mb-3"),
                
                # Team 1 selection
                html.Label("Team 1:"),
                dcc.Dropdown(
                    id="team1-dropdown",
                    options=[{"label": team, "value": team} for team in teams],
                    value=teams[0] if len(teams) > 0 else None,
                    clearable=False,
                    className="mb-3"
                ),
                
                # Team 2 selection
                html.Label("Team 2:"),
                dcc.Dropdown(
                    id="team2-dropdown",
                    options=[{"label": team, "value": team} for team in teams],
                    value=teams[1] if len(teams) > 1 else None,
                    clearable=False,
                    className="mb-3"
                ),
                
                # Location selection
                html.Label("Game Location:"),
                dbc.RadioItems(
                    id="location-radio",
                    options=[
                        {"label": "Neutral Site", "value": "neutral"},
                        {"label": "Team 1 Home", "value": "home_1"},
                        {"label": "Team 2 Home", "value": "home_2"}
                    ],
                    value="neutral",
                    inline=True,
                    className="mb-3"
                ),
                
                # Predict button
                dbc.Button(
                    "Predict Game",
                    id="predict-button",
                    color="primary",
                    className="mt-3"
                )
            ], md=6, lg=5),
            
            dbc.Col([
                # Information about the predictor
                dbc.Card([
                    dbc.CardHeader("About the Game Predictor"),
                    dbc.CardBody([
                        html.P([
                            "The NCAA Game Predictor uses a mathematical model based on KenPom metrics ",
                            "to predict the outcome of a game between any two teams."
                        ]),
                        html.P([
                            "The model considers various factors, including:"
                        ]),
                        html.Ul([
                            html.Li("Team efficiency ratings (offense and defense)"),
                            html.Li("Pace of play (tempo)"),
                            html.Li("Shooting percentages (overall and three-point)"),
                            html.Li("Rebounding rates"),
                            html.Li("Turnover rates"),
                            html.Li("Home court advantage (+3.5 points for home team)")
                        ]),
                        html.P([
                            "The spread prediction takes into account statistical advantages of each team ",
                            "and adjusts for matchup-specific factors that could influence the outcome."
                        ])
                    ])
                ])
            ], md=6, lg=7)
        ]),
        
        html.Hr(),
        
        # Prediction results section
        html.Div(id="prediction-results")
    ])
    
    return layout

# Callback to predict the game and display results
@callback(
    Output("prediction-results", "children"),
    [Input("predict-button", "n_clicks")],
    [State("team1-dropdown", "value"),
     State("team2-dropdown", "value"),
     State("location-radio", "value")],
    prevent_initial_call=True
)
def predict_game(n_clicks, team1, team2, location):
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
    team1_score = round(prediction["team1"]["predicted_score"])
    team2_score = round(prediction["team2"]["predicted_score"])
    spread = abs(round(prediction["spread"], 1))
    total = round(prediction["total"])
    
    # Determine the spread text
    if team1_score > team2_score:
        spread_text = f"{team1} by {spread}"
    else:
        spread_text = f"{team2} by {spread}"
    
    # Create spread gauge
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
    
    # Create win probability gauge
    win_prob_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=winner["win_probability"] * 100,
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
                            f" {team1_score} - {team2_score} ",
                            html.Span(team2, className="text-danger")
                        ], className="text-center"),
                        html.H4([
                            f"Predicted Spread: ",
                            html.Strong(spread_text)
                        ], className="text-center mt-3"),
                        html.P([
                            f"Over/Under: {total} points"
                        ], className="text-center")
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
                    f" with a {round(winner['win_probability'] * 100)}% probability. ",
                    f"The projected score is {team1_score}-{team2_score}, with a point spread of {spread} ",
                    f"in favor of {winner_name}."
                ]),
                html.P([
                    "Key advantages for the predicted winner include: ",
                    ", ".join([f"{factor['factor']}" for factor in prediction["key_factors"] 
                            if factor['advantage'] == winner_name])
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
                            f" times ({round(prediction['historical_matchups']['team1_wins'] / prediction['historical_matchups']['total_matchups'] * 100)}%), and ",
                            html.Strong(f"{team2}"), f" would have won ",
                            html.Strong(f"{prediction['historical_matchups']['team2_wins']}"),
                            f" times ({round(prediction['historical_matchups']['team2_wins'] / prediction['historical_matchups']['total_matchups'] * 100)}%)."
                        ]),
                        
                        # Show the average margin
                        html.P([
                            f"The average adjusted efficiency margin between the teams was ",
                            html.Strong(f"{abs(round(prediction['historical_matchups']['avg_margin'], 1))}"),
                            f" points in favor of ",
                            html.Strong(f"{team1 if prediction['historical_matchups']['avg_margin'] > 0 else team2}"), "."
                        ]),
                        
                        # Explain how this impacted the prediction
                        html.P([
                            "These historical matchups were factored into the final prediction with a 10% weight, ",
                            f"{'strengthening' if (prediction['historical_matchups']['team1_wins'] / prediction['historical_matchups']['total_matchups'] > 0.5) == (prediction['team1']['win_probability'] > 0.5) else 'tempering'} ",
                            "the statistical model's conclusion."
                        ], className="text-muted")
                    ]) if prediction['historical_matchups']['total_matchups'] > 0 else html.Div()
                ])
            ])
        ]) if 'historical_matchups' in prediction else html.Div()
    ]) 