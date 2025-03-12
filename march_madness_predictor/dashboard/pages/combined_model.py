import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def layout(data_loader):
    """
    Create the layout for the combined model page
    
    Parameters:
    -----------
    data_loader : DataLoader
        Instance of the DataLoader class
        
    Returns:
    --------
    html.Div
        Layout for the combined model page
    """
    try:
        # Load the combined predictions
        combined_predictions = data_loader.get_combined_predictions()
        
        if combined_predictions is None:
            raise ValueError("Combined predictions not available")
        
        # Create a bar chart of the top 20 teams by combined championship probability
        top_20 = combined_predictions.head(20)
        fig_championship = px.bar(
            top_20,
            y='TeamName',
            x='ChampionshipPct_Combined',
            orientation='h',
            title='Top 20 Championship Contenders (Combined Model)',
            labels={'TeamName': 'Team', 'ChampionshipPct_Combined': 'Championship Probability (%)'},
            color='ChampionshipPct_Combined',
            color_continuous_scale='Viridis',
        )
        fig_championship.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        # Create a scatter plot comparing the two models' championship probabilities
        fig_scatter = px.scatter(
            combined_predictions.head(50),
            x='ChampionshipPct_ChampProfile',
            y='ChampionshipPct_ExitRound',
            title='Comparison of Model Predictions (Top 50 Teams)',
            labels={
                'ChampionshipPct_ChampProfile': 'Champion Profile Model (%)',
                'ChampionshipPct_ExitRound': 'Exit Round Model (%)'
            },
            color='ChampionshipPct_Combined',
            size='AdjEM',
            hover_name='TeamName',
            hover_data=['Seed', 'SimilarityPct', 'PredictedExitRound'],
            color_continuous_scale='Viridis',
        )
        
        # Add a diagonal line for reference
        max_val = max(
            combined_predictions['ChampionshipPct_ChampProfile'].max(),
            combined_predictions['ChampionshipPct_ExitRound'].max()
        )
        
        fig_scatter.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max_val,
            y1=max_val,
            line=dict(color="gray", width=1, dash="dash"),
        )
        
        # Add annotations for teams above and below the line
        fig_scatter.add_annotation(
            x=max_val * 0.7,
            y=max_val * 0.3,
            text="Exit Round Model Favors",
            showarrow=False,
            font=dict(size=10, color="gray"),
        )
        
        fig_scatter.add_annotation(
            x=max_val * 0.3,
            y=max_val * 0.7,
            text="Champion Profile Model Favors",
            showarrow=False,
            font=dict(size=10, color="gray"),
        )
        
        # Create a table of the top 15 teams by combined championship probability
        top_15_df = combined_predictions.head(15)[
            ['TeamName', 'Seed', 'ChampionshipPct_Combined', 'FinalFourPct_Combined', 
             'ChampionshipPct_ChampProfile', 'ChampionshipPct_ExitRound',
             'SimilarityPct', 'PredictedExitRound', 'AdjEM', 'RankAdjEM']
        ]
        
        # Format the columns
        top_15_df['ChampionshipPct_Combined'] = top_15_df['ChampionshipPct_Combined'].round(1)
        top_15_df['FinalFourPct_Combined'] = top_15_df['FinalFourPct_Combined'].round(1)
        top_15_df['ChampionshipPct_ChampProfile'] = top_15_df['ChampionshipPct_ChampProfile'].round(1)
        top_15_df['ChampionshipPct_ExitRound'] = top_15_df['ChampionshipPct_ExitRound'].round(1)
        top_15_df['SimilarityPct'] = top_15_df['SimilarityPct'].round(1)
        top_15_df['PredictedExitRound'] = top_15_df['PredictedExitRound'].round(2)
        top_15_df['AdjEM'] = top_15_df['AdjEM'].round(1)
        
        # Rename columns for display
        top_15_df.columns = ['Team', 'Seed', 'Champion (%) Combined', 'Final Four (%) Combined', 
                            'Champion (%) Profile', 'Champion (%) Exit',
                            'Similarity (%)', 'Pred Exit Round', 'Adj EM', 'Nat Rank']
        
        top_15_table = dash_table.DataTable(
            id='top-15-combined-table',
            columns=[{"name": i, "id": i} for i in top_15_df.columns],
            data=top_15_df.to_dict('records'),
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
                    'if': {'filter_query': '{Champion (%) Combined} > 10'},
                    'backgroundColor': '#FFFFCC'
                }
            ],
            sort_action='native',
            filter_action='native',
            page_size=15,
        )
        
        # Create a heatmap of the top teams
        # Prepare data for heatmap
        heatmap_data = combined_predictions.head(20)[
            ['TeamName', 'SimilarityPct', 'PredictedExitRound', 'ChampionshipPct_Combined']
        ].copy()
        
        # Normalize PredictedExitRound to 0-100 scale for comparison
        max_exit = 7  # National Champion
        heatmap_data['PredictedExitRound_Norm'] = heatmap_data['PredictedExitRound'] / max_exit * 100
        
        # Create a long-format DataFrame for the heatmap
        heatmap_long = pd.DataFrame({
            'TeamName': heatmap_data['TeamName'].tolist() + heatmap_data['TeamName'].tolist(),
            'Model': ['Champion Profile'] * len(heatmap_data) + ['Exit Round'] * len(heatmap_data),
            'Value': heatmap_data['SimilarityPct'].tolist() + heatmap_data['PredictedExitRound_Norm'].tolist()
        })
        
        fig_heatmap = px.density_heatmap(
            heatmap_long,
            x='TeamName',
            y='Model',
            z='Value',
            title='Model Comparison: Champion Profile Similarity vs Exit Round Prediction (Top 20 Teams)',
            labels={'TeamName': 'Team', 'Value': 'Normalized Score (0-100)'},
            color_continuous_scale='Viridis',
        )
        
        fig_heatmap.update_layout(
            xaxis={'categoryorder': 'array', 'categoryarray': heatmap_data['TeamName'].tolist()},
            yaxis={'categoryorder': 'array', 'categoryarray': ['Champion Profile', 'Exit Round']}
        )
        
        # Create the layout
        content = [
            html.H1("Combined Model Predictions", className="text-center mb-4"),
            html.P(
                "This page combines predictions from both the Champion Profile and Exit Round models to provide a comprehensive view of tournament potential.",
                className="lead text-center mb-4"
            ),
            
            dbc.Card([
                dbc.CardHeader("Top Championship Contenders (Combined Model)"),
                dbc.CardBody([
                    dcc.Graph(figure=fig_championship)
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Model Comparison"),
                dbc.CardBody([
                    html.P(
                        "This chart compares championship probabilities from the Champion Profile model and the Exit Round model. "
                        "Teams above the diagonal line are favored more by the Exit Round model, while teams below the line are favored more by the Champion Profile model."
                    ),
                    dcc.Graph(figure=fig_scatter)
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Model Comparison Heatmap"),
                dbc.CardBody([
                    html.P(
                        "This heatmap shows how the top 20 teams perform in each model. Darker colors indicate higher scores. "
                        "The Exit Round predictions have been normalized to a 0-100 scale for comparison with the Champion Profile similarity percentages."
                    ),
                    dcc.Graph(figure=fig_heatmap)
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Top 15 Teams (Combined Model)"),
                dbc.CardBody([
                    html.P(
                        "This table shows the top 15 teams according to the combined model, which averages the championship probabilities "
                        "from both the Champion Profile and Exit Round models."
                    ),
                    top_15_table
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Methodology"),
                dbc.CardBody([
                    html.P(
                        "The combined model integrates predictions from two different approaches:"
                    ),
                    html.Ul([
                        html.Li([
                            html.Strong("Champion Profile Model: "),
                            "Identifies teams that most closely resemble the statistical profile of historical NCAA champions."
                        ]),
                        html.Li([
                            html.Strong("Exit Round Model: "),
                            "Uses deep learning to predict how far teams will advance in the tournament based on their statistical profile and estimated seed."
                        ]),
                    ]),
                    html.P(
                        "The combined model averages the championship and Final Four probabilities from both models to provide a more robust prediction. "
                        "This approach leverages the strengths of both models: the Champion Profile model's focus on historical champion characteristics "
                        "and the Exit Round model's consideration of tournament dynamics and seed performance."
                    ),
                    html.P(
                        "Teams that rank highly in both models are particularly strong contenders, as they both resemble historical champions "
                        "and are predicted to advance far in the tournament based on their statistical profile and seed."
                    ),
                ])
            ], className="mb-4"),
        ]
    
    except Exception as e:
        # If data isn't available, show a message to run the models first
        content = [
            html.H1("Combined Model Predictions", className="text-center mb-4"),
            
            dbc.Alert([
                html.H4("Data Not Available", className="alert-heading"),
                html.P(
                    "The combined model data is not yet available. Please run both prediction models first:"
                ),
                html.Ol([
                    html.Li("Run the Champion Profile model: python march_madness_predictor/models/champion_profile/run_champion_profile_model.py"),
                    html.Li("Run the Exit Round model: python march_madness_predictor/models/exit_round/run_exit_round_model.py"),
                ]),
                html.Hr(),
                html.P(
                    "Once both models have been run, refresh this page to see the combined predictions.",
                    className="mb-0"
                ),
            ], color="warning", className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("About the Combined Model"),
                dbc.CardBody([
                    html.P(
                        "The combined model integrates predictions from both the Champion Profile and Exit Round models "
                        "to provide a more comprehensive view of tournament potential. By combining these two approaches, "
                        "we can leverage the strengths of each model and provide more robust predictions."
                    ),
                    html.P(
                        "The Champion Profile model focuses on how closely teams resemble historical champions, while the "
                        "Exit Round model predicts how far teams will advance based on their statistical profile and seed. "
                        "Together, they provide a more complete picture of tournament potential."
                    ),
                ])
            ], className="mb-4")
        ]
    
    return html.Div(content) 