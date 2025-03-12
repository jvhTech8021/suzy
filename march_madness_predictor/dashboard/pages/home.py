import dash_bootstrap_components as dbc
from dash import html, dcc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def layout(data_loader):
    """
    Create the layout for the home page
    
    Parameters:
    -----------
    data_loader : DataLoader
        Instance of the DataLoader class
        
    Returns:
    --------
    html.Div
        Layout for the home page
    """
    # Try to load the combined predictions
    try:
        combined_predictions = data_loader.get_combined_predictions()
        top_teams = combined_predictions.head(10)
        
        # Create a bar chart of championship probabilities
        fig_championship = px.bar(
            top_teams,
            y='TeamName',
            x='ChampionshipPct_Combined',
            orientation='h',
            title='Top 10 Championship Contenders',
            labels={'TeamName': 'Team', 'ChampionshipPct_Combined': 'Championship Probability (%)'},
            color='ChampionshipPct_Combined',
            color_continuous_scale='Viridis',
        )
        fig_championship.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        # Create a bar chart of Final Four probabilities
        top_ff = combined_predictions.sort_values('FinalFourPct_Combined', ascending=False).head(10)
        fig_final_four = px.bar(
            top_ff,
            y='TeamName',
            x='FinalFourPct_Combined',
            orientation='h',
            title='Top 10 Final Four Contenders',
            labels={'TeamName': 'Team', 'FinalFourPct_Combined': 'Final Four Probability (%)'},
            color='FinalFourPct_Combined',
            color_continuous_scale='Plasma',
        )
        fig_final_four.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        # Create a scatter plot of champion profile similarity vs predicted exit round
        fig_scatter = px.scatter(
            combined_predictions.head(50),
            x='SimilarityPct',
            y='PredictedExitRound',
            title='Champion Profile Similarity vs Predicted Exit Round (Top 50 Teams)',
            labels={
                'SimilarityPct': 'Champion Profile Similarity (%)',
                'PredictedExitRound': 'Predicted Tournament Exit Round'
            },
            color='ChampionshipPct_Combined',
            size='AdjEM',
            hover_name='TeamName',
            hover_data=['Seed', 'AdjEM', 'AdjOE', 'AdjDE'],
            color_continuous_scale='Viridis',
        )
        
        # Add horizontal lines for tournament rounds
        round_names = {
            1: 'First Round',
            2: 'Second Round',
            3: 'Sweet 16',
            4: 'Elite 8',
            5: 'Final Four',
            6: 'Championship Game',
            7: 'National Champion'
        }
        
        for round_num, round_name in round_names.items():
            fig_scatter.add_shape(
                type="line",
                x0=0,
                y0=round_num,
                x1=100,
                y1=round_num,
                line=dict(color="gray", width=1, dash="dash"),
            )
            fig_scatter.add_annotation(
                x=0,
                y=round_num,
                text=round_name,
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                xshift=5,
                font=dict(size=10, color="gray"),
            )
        
        # Create the layout with the visualizations
        content = [
            html.H1("March Madness Predictor 2025", className="text-center mb-4"),
            html.P(
                "Welcome to the March Madness Predictor Dashboard for the 2025 NCAA Tournament. "
                "This dashboard combines multiple prediction models to help you analyze and predict "
                "tournament outcomes based on KenPom metrics and historical data.",
                className="lead text-center mb-4"
            ),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Top Championship Contenders"),
                        dbc.CardBody([
                            dcc.Graph(figure=fig_championship)
                        ])
                    ], className="mb-4")
                ], md=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Top Final Four Contenders"),
                        dbc.CardBody([
                            dcc.Graph(figure=fig_final_four)
                        ])
                    ], className="mb-4")
                ], md=6)
            ]),
            
            dbc.Card([
                dbc.CardHeader("Team Analysis"),
                dbc.CardBody([
                    dcc.Graph(figure=fig_scatter)
                ])
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Models Overview"),
                        dbc.CardBody([
                            html.H5("Champion Profile Model"),
                            html.P(
                                "Identifies teams that most closely resemble the statistical profile "
                                "of historical NCAA champions based on key KenPom metrics."
                            ),
                            html.H5("Exit Round Model"),
                            html.P(
                                "Uses deep learning to predict how far teams will advance in the tournament "
                                "based on their statistical profile and estimated seed."
                            ),
                            html.H5("Combined Model"),
                            html.P(
                                "Merges insights from both models to provide a comprehensive view of each team's "
                                "tournament potential."
                            ),
                            html.H5("Full Tournament Bracket"),
                            html.P(
                                "Simulates a complete 64-team NCAA tournament bracket based on the combined model predictions, "
                                "showing all matchups from the First Round through the Championship game."
                            ),
                            html.H5("Game Predictor"),
                            html.P(
                                "Uses KenPom statistics and a mathematical model to predict the outcome of individual "
                                "matchups between any two teams, with adjustments for tournament rounds."
                            ),
                        ])
                    ], className="mb-4")
                ], md=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Navigation"),
                        dbc.CardBody([
                            dbc.ListGroup([
                                dbc.ListGroupItem([
                                    html.H5("Champion Profile", className="mb-1"),
                                    html.P("View teams that most resemble historical champions", className="mb-1"),
                                    dbc.Button("Go", color="primary", size="sm", href="/champion-profile")
                                ]),
                                dbc.ListGroupItem([
                                    html.H5("Exit Round Predictions", className="mb-1"),
                                    html.P("See how far teams are predicted to advance", className="mb-1"),
                                    dbc.Button("Go", color="primary", size="sm", href="/exit-round")
                                ]),
                                dbc.ListGroupItem([
                                    html.H5("Combined Model", className="mb-1"),
                                    html.P("View comprehensive tournament predictions", className="mb-1"),
                                    dbc.Button("Go", color="primary", size="sm", href="/combined-model")
                                ]),
                                dbc.ListGroupItem([
                                    html.H5("Full Tournament Bracket", className="mb-1"),
                                    html.P("Explore a complete NCAA tournament bracket simulation", className="mb-1"),
                                    dbc.Button("Go", color="primary", size="sm", href="/full-bracket")
                                ]),
                                dbc.ListGroupItem([
                                    html.H5("Game Predictor", className="mb-1"),
                                    html.P("Predict outcomes of individual matchups between any two teams", className="mb-1"),
                                    dbc.Button("Go", color="primary", size="sm", href="/game-predictor")
                                ]),
                                dbc.ListGroupItem([
                                    html.H5("Team Explorer", className="mb-1"),
                                    html.P("Analyze individual teams in detail", className="mb-1"),
                                    dbc.Button("Go", color="primary", size="sm", href="/team-explorer")
                                ]),
                            ])
                        ])
                    ], className="mb-4")
                ], md=6)
            ])
        ]
    
    except Exception as e:
        # If data isn't available, show a message to run the models first
        content = [
            html.H1("March Madness Predictor 2025", className="text-center mb-4"),
            html.P(
                "Welcome to the March Madness Predictor Dashboard for the 2025 NCAA Tournament.",
                className="lead text-center mb-4"
            ),
            
            dbc.Alert([
                html.H4("Data Not Available", className="alert-heading"),
                html.P(
                    "The prediction data is not yet available. Please run the prediction models first:"
                ),
                html.Ol([
                    html.Li("Run the Champion Profile model: python march_madness_predictor/models/champion_profile/run_champion_profile_model.py"),
                    html.Li("Run the Exit Round model: python march_madness_predictor/models/exit_round/run_exit_round_model.py"),
                    html.Li("Generate the Full Bracket: python march_madness_predictor/full_bracket.py"),
                    html.Li("Use the Game Predictor to analyze individual matchups")
                ]),
                html.Hr(),
                html.P(
                    "Once the models have been run, refresh this page to see the predictions.",
                    className="mb-0"
                ),
            ], color="warning", className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Models Overview"),
                dbc.CardBody([
                    html.H5("Champion Profile Model"),
                    html.P(
                        "Identifies teams that most closely resemble the statistical profile "
                        "of historical NCAA champions based on key KenPom metrics."
                    ),
                    html.H5("Exit Round Model"),
                    html.P(
                        "Uses deep learning to predict how far teams will advance in the tournament "
                        "based on their statistical profile and estimated seed."
                    ),
                    html.H5("Combined Model"),
                    html.P(
                        "Merges insights from both models to provide a comprehensive view of each team's "
                        "tournament potential."
                    ),
                    html.H5("Full Tournament Bracket"),
                    html.P(
                        "Simulates a complete 64-team NCAA tournament bracket based on the combined model predictions, "
                        "showing all matchups from the First Round through the Championship game."
                    ),
                    html.H5("Game Predictor"),
                    html.P(
                        "Uses KenPom statistics and a mathematical model to predict the outcome of individual "
                        "matchups between any two teams, with adjustments for tournament rounds."
                    ),
                ])
            ], className="mb-4")
        ]
    
    return html.Div(content) 