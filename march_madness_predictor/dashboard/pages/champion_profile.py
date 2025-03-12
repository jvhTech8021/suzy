import dash
from dash import dcc, html, Output, Input, State, callback, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import os
import traceback
import time
from pathlib import Path

# Remove problematic relative import
# Constants can be defined directly if needed
CENTER_STYLE = {'textAlign': 'center'}
CENTER_ALIGN_STYLE = {'margin': '0 auto', 'textAlign': 'center'}

def layout(data_loader):
    """
    Create the layout for the champion profile page
    
    Parameters:
    -----------
    data_loader : DataLoader
        Instance of the DataLoader class
        
    Returns:
    --------
    html.Div
        Layout for the champion profile page
    """
    try:
        # Debug info
        print("\n=== CHAMPION PROFILE PAGE DEBUG ===")
        print(f"Champion profile dir: {data_loader.champion_profile_dir}")
        json_path = os.path.join(data_loader.champion_profile_dir, "champion_profile.json")
        csv_path = os.path.join(data_loader.champion_profile_dir, "all_teams_champion_profile.csv")
        print(f"Files exist? champion_profile.json: {os.path.exists(json_path)}, all_teams_champion_profile.csv: {os.path.exists(csv_path)}")
        
        # Load the champion profile data
        print("Attempting to load champion_profile...")
        champion_profile = data_loader.get_champion_profile()
        print(f"Loaded champion_profile: {champion_profile}")
        
        print("Attempting to load predictions...")
        predictions = data_loader.get_champion_profile_predictions()
        print(f"Loaded predictions with shape: {predictions.shape}")
        print("=== END DEBUG ===\n")
        
        # Create a table of the champion profile
        profile_table = dash_table.DataTable(
            id='champion-profile-table',
            columns=[
                {'name': 'KenPom Metric', 'id': 'Metric'},
                {'name': 'Value', 'id': 'Value'},
                {'name': 'KenPom Ranking', 'id': 'Ranking'}
            ],
            data=[
                {'Metric': 'Adjusted Efficiency Margin (AdjEM)', 'Value': f"{champion_profile['AdjEM']:.2f}", 'Ranking': f"{champion_profile['RankAdjEM']:.1f}"},
                {'Metric': 'Offensive Efficiency (AdjOE)', 'Value': f"{champion_profile['AdjOE']:.1f}", 'Ranking': f"{champion_profile['RankAdjOE']:.1f}"},
                {'Metric': 'Defensive Efficiency (AdjDE)', 'Value': f"{champion_profile['AdjDE']:.1f}", 'Ranking': f"{champion_profile['RankAdjDE']:.1f}"},
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
                }
            ],
        )
        
        # Create a bar chart of the top 20 teams by similarity
        top_20 = predictions.head(20)
        fig_similarity = px.bar(
            top_20,
            y='TeamName',
            x='SimilarityPct',
            orientation='h',
            title='Top 20 Teams by Similarity to Champion Profile',
            labels={'TeamName': 'Team', 'SimilarityPct': 'Similarity (%)'},
            color='SimilarityPct',
            color_continuous_scale='Viridis',
        )
        fig_similarity.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        # Create a scatter plot of offensive vs defensive efficiency
        # Fix the negative size issue by ensuring positive values
        df_plot = predictions.head(50).copy()
        df_plot['size_value'] = np.abs(df_plot['AdjEM']) + 5  # Ensure all values are positive by using absolute value and adding a base size
        
        fig_scatter = px.scatter(
            df_plot,
            x='AdjOE',
            y='AdjDE',
            title='Offensive vs Defensive Efficiency (Top 50 Teams)',
            labels={'AdjOE': 'Offensive Efficiency', 'AdjDE': 'Defensive Efficiency'},
            color='SimilarityPct',
            size='size_value',  # Use the positive size values
            hover_name='TeamName',
            hover_data=['SimilarityRank', 'ChampionPct', 'FinalFourPct'],
            color_continuous_scale='Viridis',
        )
        
        # Add champion profile point
        fig_scatter.add_trace(
            go.Scatter(
                x=[champion_profile['AdjOE']],
                y=[champion_profile['AdjDE']],
                mode='markers',
                marker=dict(
                    color='gold',
                    size=15,
                    symbol='star',
                    line=dict(color='black', width=2)
                ),
                name='Champion Profile',
                hoverinfo='name',
                hovertemplate='Champion Profile<br>AdjOE: %{x:.1f}<br>AdjDE: %{y:.1f}'
            )
        )
        
        # Reverse y-axis (lower defensive efficiency is better)
        fig_scatter.update_layout(yaxis_autorange="reversed")
        
        # Add reference lines for champion profile
        fig_scatter.add_shape(
            type="line",
            x0=champion_profile['AdjOE'],
            y0=predictions['AdjDE'].min(),
            x1=champion_profile['AdjOE'],
            y1=predictions['AdjDE'].max(),
            line=dict(color="gold", width=1, dash="dash"),
        )
        fig_scatter.add_shape(
            type="line",
            x0=predictions['AdjOE'].min(),
            y0=champion_profile['AdjDE'],
            x1=predictions['AdjOE'].max(),
            y1=champion_profile['AdjDE'],
            line=dict(color="gold", width=1, dash="dash"),
        )
        
        # Create a table of the top teams with actual similarity
        # Filter predictions to only include teams with similarity > 0
        similar_teams = predictions[predictions['SimilarityPct'] > 0].copy()
        
        # Get top 15 or all similar teams if less than 15
        top_teams_count = min(15, len(similar_teams))
        top_teams_df = similar_teams.head(top_teams_count)[['SimilarityRank', 'TeamName', 'SimilarityPct', 'ChampionPct', 'FinalFourPct', 
                                        'AdjEM', 'AdjOE', 'AdjDE', 'RankAdjEM']]
        
        # Format the columns
        top_teams_df['SimilarityPct'] = top_teams_df['SimilarityPct'].round(1)
        top_teams_df['ChampionPct'] = top_teams_df['ChampionPct'].round(1)
        top_teams_df['FinalFourPct'] = top_teams_df['FinalFourPct'].round(1)
        top_teams_df['AdjEM'] = top_teams_df['AdjEM'].round(1)
        top_teams_df['AdjOE'] = top_teams_df['AdjOE'].round(1)
        top_teams_df['AdjDE'] = top_teams_df['AdjDE'].round(1)
        
        # Rename columns for display
        top_teams_df.columns = ['Rank', 'Team', 'Similarity (%)', 'Champion (%)', 'Final Four (%)', 
                            'Adj EM', 'Off Eff', 'Def Eff', 'KenPom Rank']
        
        top_teams_table = dash_table.DataTable(
            id='top-teams-table',
            columns=[{"name": i, "id": i} for i in top_teams_df.columns],
            data=top_teams_df.to_dict('records'),
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
                    'if': {'column_id': 'Team'},
                    'textAlign': 'left'
                },
                {
                    'if': {'filter_query': '{Rank} <= 3'},
                    'backgroundColor': '#FFFFCC'
                }
            ],
            sort_action='native',
            filter_action='native',
            page_size=15,
        )
        
        # Add explanation of the ranking-based similarity model
        overview_card = dbc.Card([
            dbc.CardHeader("About the Champion Profile Model"),
            dbc.CardBody([
                html.P([
                    "The Champion Profile model identifies teams that most closely resemble historical NCAA champions based on both statistical values and rankings.",
                ]),
                html.P([
                    "Recent update: The model now places greater emphasis (65% weight) on a team's ranking position rather than just absolute values. ",
                    "This approach better reflects how a team's relative excellence within its season is more indicative of championship potential than raw metrics."
                ]),
                html.P([
                    "Key ranking metrics now considered:",
                ]),
                html.Ul([
                    html.Li(f"KenPom National Ranking: Champions typically ranked around #{champion_profile['RankAdjEM']:.1f} overall"),
                    html.Li(f"KenPom Offensive Efficiency Ranking: Champions typically ranked around #{champion_profile['RankAdjOE']:.1f}"),
                    html.Li(f"KenPom Defensive Efficiency Ranking: Champions typically ranked around #{champion_profile['RankAdjDE']:.1f}")
                ]),
                html.P([
                    "Teams with better rankings in these key metrics receive higher similarity scores even if their raw numbers differ from historical champions."
                ])
            ])
        ], className="mb-4")
        
        # Update the layout to include the overview card after the profile section
        content = [
            html.H1("Champion Profile Analysis", className="text-center mb-4"),
            html.P(
                "This page analyzes teams based on their similarity to the statistical profile of historical NCAA champions (2009-2024).",
                className="lead text-center mb-4"
            ),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Historical Champion Profile"),
                        dbc.CardBody([
                            html.P(
                                "The champion profile represents the average statistical profile of NCAA champions from 2009-2024. "
                                "Teams that closely match this profile have historically had greater success in the tournament."
                            ),
                            profile_table
                        ])
                    ], className="mb-4"),
                    
                ], md=6),
                
                dbc.Col([
                    overview_card
                ], md=6)
            ]),
            
            dbc.Card([
                dbc.CardHeader("Top Teams by Similarity"),
                dbc.CardBody([
                    dcc.Graph(figure=fig_similarity)
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Offensive vs Defensive Efficiency"),
                dbc.CardBody([
                    html.P(
                        "This chart shows how teams compare to the champion profile in terms of offensive and defensive efficiency. "
                        "The gold star represents the champion profile. Teams closer to this point have a more balanced profile."
                    ),
                    dcc.Graph(figure=fig_scatter)
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Top Teams by Champion Profile Similarity"),
                dbc.CardBody([
                    html.P(
                        f"This table shows the top {top_teams_count} teams that have measurable similarity to the statistical profile of historical NCAA champions. "
                        "Only teams with similarity scores greater than 0% are included. "
                        "The similarity percentage indicates how closely each team matches the champion profile."
                    ),
                    top_teams_table
                ])
            ], className="mb-4"),
            
            # New detailed analysis card
            dbc.Card([
                dbc.CardHeader("Detailed Analysis: How Champion Profile Similarity Works"),
                dbc.CardBody([
                    html.H5("How the Champion Profile is Compiled", className="mb-3"),
                    html.P(
                        "The champion profile represents the average statistical values of all NCAA champions from 2009-2024. "
                        "By analyzing these historical champions, we've identified the statistical fingerprint of a champion team:"
                    ),
                    html.Ul([
                        html.Li([
                            html.Strong("Balanced Excellence: "),
                            f"Champions typically have an Adjusted Efficiency Margin around {champion_profile['AdjEM']:.1f}, ",
                            f"ranking in the top {champion_profile['RankAdjEM']:.0f} nationally."
                        ]),
                        html.Li([
                            html.Strong("Elite Offense: "),
                            f"Champions average an Offensive Efficiency of {champion_profile['AdjOE']:.1f} points per 100 possessions, ",
                            f"typically ranking around #{champion_profile['RankAdjOE']:.0f} nationally."
                        ]),
                        html.Li([
                            html.Strong("Strong Defense: "),
                            f"Champions average a Defensive Efficiency of {champion_profile['AdjDE']:.1f} points allowed per 100 possessions, ",
                            f"typically ranking around #{champion_profile['RankAdjDE']:.0f} nationally."
                        ]),
                    ]),
                    
                    html.H5("How Similarity is Calculated", className="mt-4 mb-3"),
                    html.P(
                        "The similarity score uses a weighted formula that balances two key components:"
                    ),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H6("Rankings Component (65% Weight)", className="text-center"),
                                html.P(
                                    "This measures how closely a team's rankings match the typical champion profile. "
                                    "For example, if champions typically rank #5 nationally and a team is ranked #6, "
                                    "they receive a high similarity score for this component.",
                                    className="small"
                                ),
                                html.Ul([
                                    html.Li("National Ranking (Highest weight)", className="small"),
                                    html.Li("Offensive Efficiency Ranking", className="small"),
                                    html.Li("Defensive Efficiency Ranking", className="small"),
                                ])
                            ], className="p-3 border rounded")
                        ], md=6),
                        dbc.Col([
                            html.Div([
                                html.H6("Raw Values Component (35% Weight)", className="text-center"),
                                html.P(
                                    "This measures how closely a team's raw statistical values match the champion profile. "
                                    "For example, if champions average 120.6 points per 100 possessions and a team averages 119.8, "
                                    "they receive a high similarity score for this component.",
                                    className="small"
                                ),
                                html.Ul([
                                    html.Li("Adjusted Efficiency Margin", className="small"),
                                    html.Li("Offensive Efficiency Value", className="small"),
                                    html.Li("Defensive Efficiency Value", className="small"),
                                ])
                            ], className="p-3 border rounded")
                        ], md=6)
                    ]),
                    
                    html.P([
                        html.Strong("Technical detail: "),
                        "The similarity algorithm squares the differences between team values and champion profile values, "
                        "applies the appropriate weights (65% for rankings, 35% for raw values), and converts the result to "
                        "a 0-100 scale with higher numbers indicating greater similarity."
                    ], className="mt-3 small text-muted"),
                    
                    html.H5("Why Top Teams Are Similar", className="mt-4 mb-3"),
                    html.P(
                        f"Let's examine why the top teams receive high similarity scores:"
                    ),
                    
                    dbc.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Team"),
                                html.Th("Key Strengths"),
                                html.Th("Similarity"),
                                html.Th("Championship %")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td(similar_teams.iloc[0]['TeamName'] if len(similar_teams) > 0 else "No team"),
                                html.Td(similar_teams.iloc[0]['Assessment'] if len(similar_teams) > 0 else "-"),
                                html.Td(f"{similar_teams.iloc[0]['SimilarityPct']:.1f}%" if len(similar_teams) > 0 else "0%"),
                                html.Td(f"{similar_teams.iloc[0]['ChampionPct']:.1f}%" if len(similar_teams) > 0 else "0%")
                            ]),
                            html.Tr([
                                html.Td(similar_teams.iloc[1]['TeamName'] if len(similar_teams) > 1 else "No team"),
                                html.Td(similar_teams.iloc[1]['Assessment'] if len(similar_teams) > 1 else "-"),
                                html.Td(f"{similar_teams.iloc[1]['SimilarityPct']:.1f}%" if len(similar_teams) > 1 else "0%"),
                                html.Td(f"{similar_teams.iloc[1]['ChampionPct']:.1f}%" if len(similar_teams) > 1 else "0%")
                            ]),
                            html.Tr([
                                html.Td(similar_teams.iloc[2]['TeamName'] if len(similar_teams) > 2 else "No team"),
                                html.Td(similar_teams.iloc[2]['Assessment'] if len(similar_teams) > 2 else "-"),
                                html.Td(f"{similar_teams.iloc[2]['SimilarityPct']:.1f}%" if len(similar_teams) > 2 else "0%"),
                                html.Td(f"{similar_teams.iloc[2]['ChampionPct']:.1f}%" if len(similar_teams) > 2 else "0%")
                            ])
                        ])
                    ], bordered=True, hover=True),
                    
                    html.H5("What Similarity Means for Tournament Success", className="mt-4 mb-3"),
                    html.P(
                        "Historical analysis shows that similarity to the champion profile is strongly correlated with tournament success:"
                    ),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H6("Championship Probability", className="text-center"),
                                html.P([
                                    "Teams ranking in the top 10 by similarity have won ",
                                    html.Strong("93% of championships"),
                                    " since 2009. The #1 most similar team has a 14.3% chance of winning the title."
                                ], className="small")
                            ], className="p-3 border rounded")
                        ], md=6),
                        dbc.Col([
                            html.Div([
                                html.H6("Final Four Probability", className="text-center"),
                                html.P([
                                    "Teams in the top 10 by similarity have filled ",
                                    html.Strong("57% of Final Four spots"),
                                    " since 2009. Top 3 teams have a 21.4% chance or higher of reaching the Final Four."
                                ], className="small")
                            ], className="p-3 border rounded")
                        ], md=6)
                    ]),
                    
                    html.H5("The Tournament Potential Score", className="mt-4 mb-3"),
                    html.P([
                        "Beyond similarity alone, we calculate a Tournament Potential score that combines three factors:"
                    ]),
                    html.Ul([
                        html.Li(["Similarity to champion profile (", html.Strong("35%"), " weight)"]),
                        html.Li(["KenPom national ranking (", html.Strong("45%"), " weight)"]),
                        html.Li(["Raw efficiency margin (", html.Strong("20%"), " weight)"])
                    ]),
                    html.P([
                        "This score provides an overall assessment of championship potential that balances statistical similarity ",
                        "with the importance of being ranked highly in the current season."
                    ])
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Methodology"),
                dbc.CardBody([
                    html.P(
                        "The champion profile similarity model analyzes how closely each team's statistical profile matches "
                        "that of historical NCAA champions from 2009-2024. The model considers several key KenPom metrics:"
                    ),
                    html.Ul([
                        html.Li("KenPom Adjusted Efficiency Margin (AdjEM): Overall team strength"),
                        html.Li("KenPom National Ranking: Where the team ranks in KenPom ratings"),
                        html.Li("KenPom Offensive Efficiency (AdjOE): Points scored per 100 possessions, adjusted for opponent"),
                        html.Li("KenPom Defensive Efficiency (AdjDE): Points allowed per 100 possessions, adjusted for opponent"),
                    ]),
                    html.P(
                        "Teams are ranked by their similarity to the champion profile, with higher percentages indicating "
                        "a closer match. Historical analysis shows that teams ranking in the top 10 by similarity have a "
                        "significantly higher probability of winning the championship."
                    ),
                ])
            ], className="mb-4"),
        ]
    
    except Exception as e:
        # Debug info for the exception
        print("\n=== CHAMPION PROFILE PAGE ERROR ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        print("=== END ERROR ===\n")
        
        # If data isn't available, show a message to run the models first
        content = [
            html.H1("Champion Profile Analysis", className="text-center mb-4"),
            
            dbc.Alert([
                html.H4("Data Not Available", className="alert-heading"),
                html.P(
                    "The champion profile data is not yet available. Please run the champion profile model first:"
                ),
                html.Pre(
                    "python march_madness_predictor/models/champion_profile/run_champion_profile_model.py",
                    className="bg-light p-2 border"
                ),
                html.Hr(),
                html.P(
                    "Once the model has been run, refresh this page to see the analysis.",
                    className="mb-0"
                ),
            ], color="warning", className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("About the Champion Profile Model"),
                dbc.CardBody([
                    html.P(
                        "The champion profile model identifies teams that most closely resemble the statistical profile "
                        "of historical NCAA champions based on key KenPom metrics. This approach is based on the observation "
                        "that championship teams tend to have similar statistical profiles, particularly in terms of "
                        "adjusted efficiency margin, offensive efficiency, and defensive efficiency."
                    ),
                    html.P(
                        "By comparing current teams to this historical profile, we can identify which teams have the "
                        "statistical makeup of a potential champion."
                    ),
                ])
            ], className="mb-4")
        ]
    
    # Return the layout wrapped in a div
    return html.Div(content)

def old_layout(data_loader):
    """
    Create the layout for the champion profile page
    
    Parameters:
    -----------
    data_loader : DataLoader
        Instance of the DataLoader class
        
    Returns:
    --------
    html.Div
        Layout for the champion profile page
    """
    try:
        # Debug info
        print("\n=== CHAMPION PROFILE PAGE DEBUG ===")
        print(f"Champion profile dir: {data_loader.champion_profile_dir}")
        json_path = os.path.join(data_loader.champion_profile_dir, "champion_profile.json")
        csv_path = os.path.join(data_loader.champion_profile_dir, "all_teams_champion_profile.csv")
        print(f"Files exist? champion_profile.json: {os.path.exists(json_path)}, all_teams_champion_profile.csv: {os.path.exists(csv_path)}")
        
        # Load the champion profile data
        print("Attempting to load champion_profile...")
        champion_profile = data_loader.get_champion_profile()
        print(f"Loaded champion_profile: {champion_profile}")
        
        print("Attempting to load predictions...")
        predictions = data_loader.get_champion_profile_predictions()
        print(f"Loaded predictions with shape: {predictions.shape}")
        print("=== END DEBUG ===\n")
        
        # Create a table of the champion profile
        profile_table = dash_table.DataTable(
            id='champion-profile-table',
            columns=[
                {'name': 'Metric', 'id': 'Metric'},
                {'name': 'Value', 'id': 'Value'},
            ],
            data=[
                {'Metric': 'Adjusted Efficiency Margin (AdjEM)', 'Value': f"{champion_profile['AdjEM']:.2f}"},
                {'Metric': 'National Ranking', 'Value': f"{champion_profile['RankAdjEM']:.1f}"},
                {'Metric': 'Offensive Efficiency (AdjOE)', 'Value': f"{champion_profile['AdjOE']:.1f}"},
                {'Metric': 'Defensive Efficiency (AdjDE)', 'Value': f"{champion_profile['AdjDE']:.1f}"},
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
                }
            ],
        )
        
        # Create a bar chart of the top 20 teams by similarity
        top_20 = predictions.head(20)
        fig_similarity = px.bar(
            top_20,
            y='TeamName',
            x='SimilarityPct',
            orientation='h',
            title='Top 20 Teams by Similarity to Champion Profile',
            labels={'TeamName': 'Team', 'SimilarityPct': 'Similarity (%)'},
            color='SimilarityPct',
            color_continuous_scale='Viridis',
        )
        fig_similarity.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        # Create a scatter plot of offensive vs defensive efficiency
        # Ensure we have positive values for size by using absolute values
        df_plot = predictions.head(50).copy()
        df_plot['size_value'] = df_plot['AdjEM'].abs()  # Make sure size values are positive
        
        fig_scatter = px.scatter(
            df_plot,
            x='AdjOE',
            y='AdjDE',
            title='Offensive vs Defensive Efficiency (Top 50 Teams)',
            labels={'AdjOE': 'Offensive Efficiency', 'AdjDE': 'Defensive Efficiency'},
            color='SimilarityPct',
            size='size_value',  # Use the positive values for size
            hover_name='TeamName',
            hover_data=['SimilarityRank', 'ChampionPct', 'FinalFourPct'],
            color_continuous_scale='Viridis',
        )
        
        # Add champion profile point
        fig_scatter.add_trace(
            go.Scatter(
                x=[champion_profile['AdjOE']],
                y=[champion_profile['AdjDE']],
                mode='markers',
                marker=dict(
                    color='gold',
                    size=15,
                    symbol='star',
                    line=dict(color='black', width=2)
                ),
                name='Champion Profile',
                hoverinfo='name',
                hovertemplate='Champion Profile<br>AdjOE: %{x:.1f}<br>AdjDE: %{y:.1f}'
            )
        )
        
        # Reverse y-axis (lower defensive efficiency is better)
        fig_scatter.update_layout(yaxis_autorange="reversed")
        
        # Add reference lines for champion profile
        fig_scatter.add_shape(
            type="line",
            x0=champion_profile['AdjOE'],
            y0=predictions['AdjDE'].min(),
            x1=champion_profile['AdjOE'],
            y1=predictions['AdjDE'].max(),
            line=dict(color="gold", width=1, dash="dash"),
        )
        fig_scatter.add_shape(
            type="line",
            x0=predictions['AdjOE'].min(),
            y0=champion_profile['AdjDE'],
            x1=predictions['AdjOE'].max(),
            y1=champion_profile['AdjDE'],
            line=dict(color="gold", width=1, dash="dash"),
        )
        
        # Create a table of the top teams with actual similarity
        # Filter predictions to only include teams with similarity > 0
        similar_teams = predictions[predictions['SimilarityPct'] > 0].copy()
        
        # Get top 15 or all similar teams if less than 15
        top_teams_count = min(15, len(similar_teams))
        top_teams_df = similar_teams.head(top_teams_count)[['SimilarityRank', 'TeamName', 'SimilarityPct', 'ChampionPct', 'FinalFourPct', 
                                        'AdjEM', 'AdjOE', 'AdjDE', 'RankAdjEM']]
        
        # Format the columns
        top_teams_df['SimilarityPct'] = top_teams_df['SimilarityPct'].round(1)
        top_teams_df['ChampionPct'] = top_teams_df['ChampionPct'].round(1)
        top_teams_df['FinalFourPct'] = top_teams_df['FinalFourPct'].round(1)
        top_teams_df['AdjEM'] = top_teams_df['AdjEM'].round(1)
        top_teams_df['AdjOE'] = top_teams_df['AdjOE'].round(1)
        top_teams_df['AdjDE'] = top_teams_df['AdjDE'].round(1)
        
        # Rename columns for display
        top_teams_df.columns = ['Rank', 'Team', 'Similarity (%)', 'Champion (%)', 'Final Four (%)', 
                            'Adj EM', 'Off Eff', 'Def Eff', 'KenPom Rank']
        
        top_teams_table = dash_table.DataTable(
            id='top-teams-table',
            columns=[{"name": i, "id": i} for i in top_teams_df.columns],
            data=top_teams_df.to_dict('records'),
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
                    'if': {'column_id': 'Team'},
                    'textAlign': 'left'
                },
                {
                    'if': {'filter_query': '{Rank} <= 3'},
                    'backgroundColor': '#FFFFCC'
                }
            ],
            sort_action='native',
            filter_action='native',
            page_size=15,
        )
        
        # Create the layout
        content = [
            html.H1("Champion Profile Analysis", className="text-center mb-4"),
            html.P(
                "This page analyzes teams based on their similarity to the statistical profile of historical NCAA champions (2009-2024).",
                className="lead text-center mb-4"
            ),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Historical Champion Profile"),
                        dbc.CardBody([
                            html.P(
                                "The champion profile represents the average statistical profile of NCAA champions from 2009-2024. "
                                "Teams that closely match this profile have historically had greater success in the tournament."
                            ),
                            profile_table
                        ])
                    ], className="mb-4")
                ], md=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Top Teams by Similarity"),
                        dbc.CardBody([
                            dcc.Graph(figure=fig_similarity)
                        ])
                    ], className="mb-4")
                ], md=6)
            ]),
            
            dbc.Card([
                dbc.CardHeader("Offensive vs Defensive Efficiency"),
                dbc.CardBody([
                    html.P(
                        "This chart shows how teams compare to the champion profile in terms of offensive and defensive efficiency. "
                        "The gold star represents the champion profile. Teams closer to this point have a more balanced profile."
                    ),
                    dcc.Graph(figure=fig_scatter)
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Top Teams by Champion Profile Similarity"),
                dbc.CardBody([
                    html.P(
                        f"This table shows the top {top_teams_count} teams that have measurable similarity to the statistical profile of historical NCAA champions. "
                        "Only teams with similarity scores greater than 0% are included. "
                        "The similarity percentage indicates how closely each team matches the champion profile."
                    ),
                    top_teams_table
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Methodology"),
                dbc.CardBody([
                    html.P(
                        "The champion profile similarity model analyzes how closely each team's statistical profile matches "
                        "that of historical NCAA champions from 2009-2024. The model considers several key metrics:"
                    ),
                    html.Ul([
                        html.Li("Adjusted Efficiency Margin (AdjEM): Overall team strength"),
                        html.Li("National Ranking: Where the team ranks nationally"),
                        html.Li("Offensive Efficiency (AdjOE): Points scored per 100 possessions, adjusted for opponent"),
                        html.Li("Defensive Efficiency (AdjDE): Points allowed per 100 possessions, adjusted for opponent"),
                    ]),
                    html.P(
                        "Teams are ranked by their similarity to the champion profile, with higher percentages indicating "
                        "a closer match. Historical analysis shows that teams ranking in the top 10 by similarity have a "
                        "significantly higher probability of winning the championship."
                    ),
                ])
            ], className="mb-4"),
        ]
    
    except Exception as e:
        # Debug info for the exception
        print("\n=== CHAMPION PROFILE PAGE ERROR ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        print("=== END ERROR ===\n")
        
        # If data isn't available, show a message to run the models first
        content = [
            html.H1("Champion Profile Analysis", className="text-center mb-4"),
            
            dbc.Alert([
                html.H4("Data Not Available", className="alert-heading"),
                html.P(
                    "The champion profile data is not yet available. Please run the champion profile model first:"
                ),
                html.Pre(
                    "python march_madness_predictor/models/champion_profile/run_champion_profile_model.py",
                    className="bg-light p-2 border"
                ),
                html.Hr(),
                html.P(
                    "Once the model has been run, refresh this page to see the analysis.",
                    className="mb-0"
                ),
            ], color="warning", className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("About the Champion Profile Model"),
                dbc.CardBody([
                    html.P(
                        "The champion profile model identifies teams that most closely resemble the statistical profile "
                        "of historical NCAA champions based on key KenPom metrics. This approach is based on the observation "
                        "that championship teams tend to have similar statistical profiles, particularly in terms of "
                        "adjusted efficiency margin, offensive efficiency, and defensive efficiency."
                    ),
                    html.P(
                        "By comparing current teams to this historical profile, we can identify which teams have the "
                        "statistical makeup of a potential champion."
                    ),
                ])
            ], className="mb-4")
        ]
    
    return html.Div(content) 