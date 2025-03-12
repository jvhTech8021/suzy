import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table, callback, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def layout(data_loader):
    """
    Create the layout for the team explorer page
    
    Parameters:
    -----------
    data_loader : DataLoader
        Instance of the DataLoader class
        
    Returns:
    --------
    html.Div
        Layout for the team explorer page
    """
    try:
        # Load the data
        current_data = data_loader.get_current_season_data()
        
        # Try to load the combined predictions
        try:
            combined_predictions = data_loader.get_combined_predictions()
            has_predictions = True
        except:
            has_predictions = False
        
        # Create a dropdown of all teams
        team_options = [{'label': team, 'value': team} for team in sorted(current_data['TeamName'])]
        
        # Create the layout
        content = [
            html.H1("Team Explorer", className="text-center mb-4"),
            html.P(
                "Select a team to view detailed statistics and tournament predictions.",
                className="lead text-center mb-4"
            ),
            
            dbc.Card([
                dbc.CardHeader("Team Selection"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select a Team:"),
                            dcc.Dropdown(
                                id='team-dropdown',
                                options=team_options,
                                value=team_options[0]['value'] if team_options else None,
                                clearable=False,
                                className="mb-3"
                            ),
                        ], md=6),
                        
                        dbc.Col([
                            html.Label("Or Filter by Conference:"),
                            dcc.Dropdown(
                                id='conference-dropdown',
                                options=[{'label': conf, 'value': conf} for conf in sorted(current_data['Conference'].unique())],
                                value=None,
                                clearable=True,
                                placeholder="Select a conference...",
                                className="mb-3"
                            ),
                        ], md=6),
                    ]),
                    
                    dbc.Button("View Team Analysis", id="view-team-button", color="primary", className="mt-2"),
                ])
            ], className="mb-4"),
            
            html.Div(id="team-analysis-content")
        ]
    
    except Exception as e:
        # If data isn't available, show a message
        content = [
            html.H1("Team Explorer", className="text-center mb-4"),
            
            dbc.Alert([
                html.H4("Data Not Available", className="alert-heading"),
                html.P(
                    "The team data is not yet available. Please make sure the KenPom data is available in the susan_kenpom directory."
                ),
                html.Hr(),
                html.P(
                    "Once the data is available, refresh this page to explore teams.",
                    className="mb-0"
                ),
            ], color="warning", className="mb-4"),
        ]
    
    return html.Div(content)

# Callback to update the team dropdown based on conference selection
@callback(
    Output('team-dropdown', 'options'),
    Output('team-dropdown', 'value'),
    Input('conference-dropdown', 'value'),
    State('team-dropdown', 'value'),
    prevent_initial_call=True
)
def update_team_dropdown(selected_conference, current_team):
    """
    Update the team dropdown based on the selected conference
    """
    # Load the data
    try:
        data_loader = DataLoader()
        current_data = data_loader.get_current_season_data()
        
        if selected_conference:
            # Filter teams by conference
            filtered_teams = current_data[current_data['Conference'] == selected_conference]['TeamName']
            team_options = [{'label': team, 'value': team} for team in sorted(filtered_teams)]
            
            # If the current team is not in the filtered list, select the first team
            if current_team not in filtered_teams.values:
                new_value = team_options[0]['value'] if team_options else None
            else:
                new_value = current_team
        else:
            # Show all teams
            team_options = [{'label': team, 'value': team} for team in sorted(current_data['TeamName'])]
            new_value = current_team
        
        return team_options, new_value
    
    except Exception as e:
        # If there's an error, return empty options
        return [], None

# Callback to display team analysis
@callback(
    Output('team-analysis-content', 'children'),
    Input('view-team-button', 'n_clicks'),
    State('team-dropdown', 'value'),
    prevent_initial_call=True
)
def display_team_analysis(n_clicks, selected_team):
    """
    Display detailed analysis for the selected team
    """
    if not n_clicks or not selected_team:
        return html.Div()
    
    try:
        # Load the data
        data_loader = DataLoader()
        current_data = data_loader.get_current_season_data()
        
        # Get the team data
        team_data = current_data[current_data['TeamName'] == selected_team].iloc[0]
        
        # Try to load the combined predictions
        try:
            combined_predictions = data_loader.get_combined_predictions()
            team_predictions = combined_predictions[combined_predictions['TeamName'] == selected_team].iloc[0]
            has_predictions = True
        except:
            has_predictions = False
        
        # Create a table of team statistics
        stats_table = dash_table.DataTable(
            id='team-stats-table',
            columns=[
                {'name': 'Metric', 'id': 'Metric'},
                {'name': 'Value', 'id': 'Value'},
                {'name': 'National Rank', 'id': 'Rank'},
            ],
            data=[
                {'Metric': 'Adjusted Efficiency Margin (AdjEM)', 'Value': f"{team_data['AdjEM']:.2f}", 'Rank': f"{team_data['RankAdjEM']:.0f}"},
                {'Metric': 'Offensive Efficiency (AdjOE)', 'Value': f"{team_data['AdjOE']:.1f}", 'Rank': f"{team_data['RankAdjOE']:.0f}"},
                {'Metric': 'Defensive Efficiency (AdjDE)', 'Value': f"{team_data['AdjDE']:.1f}", 'Rank': f"{team_data['RankAdjDE']:.0f}"},
                {'Metric': 'Tempo (AdjTempo)', 'Value': f"{team_data['AdjTempo']:.1f}", 'Rank': f"{team_data['RankAdjTempo']:.0f}"},
                {'Metric': 'Luck', 'Value': f"{team_data['Luck']:.3f}", 'Rank': f"{team_data['RankLuck']:.0f}"},
                {'Metric': 'Strength of Schedule', 'Value': f"{team_data['SOS_AdjEM']:.2f}", 'Rank': f"{team_data['RankSOS_AdjEM']:.0f}"},
                {'Metric': 'Non-Conference SOS', 'Value': f"{team_data['NCSOS_AdjEM']:.2f}", 'Rank': f"{team_data['RankNCSOSAdjEM']:.0f}"},
            ],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_cell={
                'textAlign': 'left',
                'padding': '10px'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                    'if': {'filter_query': '{Rank} <= 10', 'column_id': 'Rank'},
                    'backgroundColor': '#CCFFCC'
                },
                {
                    'if': {'filter_query': '{Rank} > 300', 'column_id': 'Rank'},
                    'backgroundColor': '#FFCCCC'
                }
            ],
        )
        
        # Create a radar chart of team strengths
        categories = ['Offense', 'Defense', 'Tempo', 'Experience', 'Bench', 'Height', 'Luck']
        
        # Normalize values to 0-100 scale (higher is better)
        # For defense, lower is better, so we invert it
        values = [
            100 - min(100, team_data['RankAdjOE'] / 3.64),  # Offense (364 teams)
            100 - min(100, team_data['RankAdjDE'] / 3.64),  # Defense
            100 - min(100, team_data['RankAdjTempo'] / 3.64),  # Tempo
            100 - min(100, team_data['RankExp'] / 3.64),  # Experience
            100 - min(100, team_data['RankBench'] / 3.64),  # Bench
            100 - min(100, team_data['RankHeight'] / 3.64),  # Height
            100 - min(100, team_data['RankLuck'] / 3.64),  # Luck
        ]
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=selected_team
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title=f"{selected_team} Team Profile",
            showlegend=False
        )
        
        # Create tournament prediction content if available
        if has_predictions:
            # Get the champion profile
            champion_profile = data_loader.get_champion_profile()
            
            # Create a comparison to champion profile
            comparison_table = dash_table.DataTable(
                id='champion-comparison-table',
                columns=[
                    {'name': 'Metric', 'id': 'Metric'},
                    {'name': selected_team, 'id': 'Team'},
                    {'name': 'Champion Profile', 'id': 'Champion'},
                    {'name': 'Difference', 'id': 'Difference'},
                ],
                data=[
                    {
                        'Metric': 'Adjusted Efficiency Margin (AdjEM)',
                        'Team': f"{team_data['AdjEM']:.2f}",
                        'Champion': f"{champion_profile['AdjEM']:.2f}",
                        'Difference': f"{team_data['AdjEM'] - champion_profile['AdjEM']:.2f}"
                    },
                    {
                        'Metric': 'National Ranking',
                        'Team': f"{team_data['RankAdjEM']:.0f}",
                        'Champion': f"{champion_profile['RankAdjEM']:.1f}",
                        'Difference': f"{team_data['RankAdjEM'] - champion_profile['RankAdjEM']:.1f}"
                    },
                    {
                        'Metric': 'Offensive Efficiency (AdjOE)',
                        'Team': f"{team_data['AdjOE']:.1f}",
                        'Champion': f"{champion_profile['AdjOE']:.1f}",
                        'Difference': f"{team_data['AdjOE'] - champion_profile['AdjOE']:.1f}"
                    },
                    {
                        'Metric': 'Defensive Efficiency (AdjDE)',
                        'Team': f"{team_data['AdjDE']:.1f}",
                        'Champion': f"{champion_profile['AdjDE']:.1f}",
                        'Difference': f"{team_data['AdjDE'] - champion_profile['AdjDE']:.1f}"
                    },
                ],
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_cell={
                    'textAlign': 'center',
                    'padding': '10px'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    },
                    {
                        'if': {'column_id': 'Metric'},
                        'textAlign': 'left'
                    },
                    {
                        'if': {'filter_query': '{Difference} > 0', 'column_id': 'Difference'},
                        'color': 'green'
                    },
                    {
                        'if': {'filter_query': '{Difference} < 0', 'column_id': 'Difference'},
                        'color': 'red'
                    }
                ],
            )
            
            # Create a tournament predictions card
            tournament_predictions = dbc.Card([
                dbc.CardHeader("Tournament Predictions"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Champion Profile Model"),
                            html.P(f"Similarity to Champion Profile: {team_predictions['SimilarityPct']:.1f}%"),
                            html.P(f"Championship Probability: {team_predictions['ChampionshipPct_ChampProfile']:.1f}%"),
                            html.P(f"Final Four Probability: {team_predictions['FinalFourPct_ChampProfile']:.1f}%"),
                            html.P(f"Assessment: {team_predictions['Assessment']}"),
                        ], md=6),
                        
                        dbc.Col([
                            html.H5("Exit Round Model"),
                            html.P(f"Estimated Seed: {team_predictions['Seed']:.0f}" if not pd.isna(team_predictions['Seed']) else "Not projected to make tournament"),
                            html.P(f"Predicted Exit Round: {team_predictions['PredictedExit']}" if not pd.isna(team_predictions['PredictedExit']) else "N/A"),
                            html.P(f"Championship Probability: {team_predictions['ChampionshipPct_ExitRound']:.1f}%" if not pd.isna(team_predictions['ChampionshipPct_ExitRound']) else "N/A"),
                            html.P(f"Final Four Probability: {team_predictions['FinalFourPct_ExitRound']:.1f}%" if not pd.isna(team_predictions['FinalFourPct_ExitRound']) else "N/A"),
                        ], md=6),
                    ]),
                    
                    html.Hr(),
                    
                    html.H5("Combined Model Prediction"),
                    html.P(f"Championship Probability: {team_predictions['ChampionshipPct_Combined']:.1f}%"),
                    html.P(f"Final Four Probability: {team_predictions['FinalFourPct_Combined']:.1f}%"),
                    
                    # Add a gauge chart for championship probability
                    dcc.Graph(
                        figure=go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=team_predictions['ChampionshipPct_Combined'],
                            title={'text': "Championship Probability"},
                            gauge={
                                'axis': {'range': [None, 25]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 5], 'color': "lightgray"},
                                    {'range': [5, 10], 'color': "gray"},
                                    {'range': [10, 25], 'color': "gold"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 20
                                }
                            }
                        ))
                    )
                ])
            ], className="mb-4")
        
        # Create the team analysis content
        team_analysis = [
            dbc.Card([
                dbc.CardHeader(f"{selected_team} - {team_data['Conference']}"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Team Statistics"),
                            stats_table
                        ], md=8),
                        
                        dbc.Col([
                            dcc.Graph(figure=fig_radar)
                        ], md=4),
                    ]),
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Comparison to Champion Profile"),
                dbc.CardBody([
                    html.P(
                        "This table compares the team's key metrics to the average profile of NCAA champions from 2009-2024. "
                        "Positive differences indicate the team is better than the average champion in that metric."
                    ),
                    comparison_table if has_predictions else html.P("Champion profile data not available. Run the champion profile model first.")
                ])
            ], className="mb-4"),
            
            tournament_predictions if has_predictions else dbc.Card([
                dbc.CardHeader("Tournament Predictions"),
                dbc.CardBody([
                    html.P("Tournament prediction data not available. Run the prediction models first.")
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Team Strengths and Weaknesses"),
                dbc.CardBody([
                    html.H5("Strengths"),
                    html.Ul([
                        html.Li(f"Offensive Efficiency: {team_data['AdjOE']:.1f} (Rank: {team_data['RankAdjOE']:.0f})") 
                            if team_data['RankAdjOE'] <= 50 else None,
                        html.Li(f"Defensive Efficiency: {team_data['AdjDE']:.1f} (Rank: {team_data['RankAdjDE']:.0f})")
                            if team_data['RankAdjDE'] <= 50 else None,
                        html.Li(f"Overall Efficiency: {team_data['AdjEM']:.2f} (Rank: {team_data['RankAdjEM']:.0f})")
                            if team_data['RankAdjEM'] <= 50 else None,
                        html.Li(f"Experience: (Rank: {team_data['RankExp']:.0f})")
                            if team_data['RankExp'] <= 50 else None,
                        html.Li(f"Height: (Rank: {team_data['RankHeight']:.0f})")
                            if team_data['RankHeight'] <= 50 else None,
                        html.Li(f"Bench Minutes: (Rank: {team_data['RankBench']:.0f})")
                            if team_data['RankBench'] <= 50 else None,
                    ]),
                    
                    html.H5("Weaknesses"),
                    html.Ul([
                        html.Li(f"Offensive Efficiency: {team_data['AdjOE']:.1f} (Rank: {team_data['RankAdjOE']:.0f})")
                            if team_data['RankAdjOE'] > 200 else None,
                        html.Li(f"Defensive Efficiency: {team_data['AdjDE']:.1f} (Rank: {team_data['RankAdjDE']:.0f})")
                            if team_data['RankAdjDE'] > 200 else None,
                        html.Li(f"Overall Efficiency: {team_data['AdjEM']:.2f} (Rank: {team_data['RankAdjEM']:.0f})")
                            if team_data['RankAdjEM'] > 200 else None,
                        html.Li(f"Experience: (Rank: {team_data['RankExp']:.0f})")
                            if team_data['RankExp'] > 200 else None,
                        html.Li(f"Height: (Rank: {team_data['RankHeight']:.0f})")
                            if team_data['RankHeight'] > 200 else None,
                        html.Li(f"Bench Minutes: (Rank: {team_data['RankBench']:.0f})")
                            if team_data['RankBench'] > 200 else None,
                    ]),
                ])
            ], className="mb-4"),
        ]
        
        return html.Div(team_analysis)
    
    except Exception as e:
        # If there's an error, return an error message
        return dbc.Alert(
            f"Error loading team data: {str(e)}",
            color="danger",
            className="mb-4"
        )

# Import the DataLoader class for the callbacks
from utils.data_loader import DataLoader 