import dash_bootstrap_components as dbc
from dash import html, dcc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from PIL import Image
import base64
import io

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
    # Check if bracket prediction image exists
    bracket_image_path = os.path.join('tournament_simulations', 'bracket_prediction.png')
    simulation_results_path = os.path.join('tournament_simulations', 'simulation_results.csv')
    championship_probs_path = os.path.join('tournament_simulations', 'championship_probabilities.csv')
    final_four_probs_path = os.path.join('tournament_simulations', 'final_four_probabilities.csv')
    
    # Try to load the bracket prediction and simulation results
    try:
        # Load the bracket image
        if os.path.exists(bracket_image_path):
            # Open the image and resize it to fit the dashboard
            img = Image.open(bracket_image_path)
            img = img.resize((1200, 900), Image.LANCZOS)
            
            # Convert the image to base64 for display
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            bracket_image = html.Img(
                src=f'data:image/png;base64,{encoded_image}',
                style={'width': '100%', 'height': 'auto'}
            )
        else:
            bracket_image = html.Div("Bracket image not found. Please run the tournament simulator.")
        
        # Load simulation results
        if os.path.exists(simulation_results_path):
            simulation_results = pd.read_csv(simulation_results_path)
            
            # Load championship probabilities
            championship_probs = pd.read_csv(championship_probs_path) if os.path.exists(championship_probs_path) else None
            
            # Load Final Four probabilities
            final_four_probs = pd.read_csv(final_four_probs_path) if os.path.exists(final_four_probs_path) else None
            
            # Create championship probability chart
            if championship_probs is not None:
                top_champions = championship_probs.head(10)
                fig_championship = px.bar(
                    top_champions,
                    y='Team',
                    x='Championship Probability',
                    orientation='h',
                    title='Top 10 Championship Contenders',
                    labels={'Team': 'Team', 'Championship Probability': 'Championship Probability (%)'},
                    color='Championship Probability',
                    color_continuous_scale='Viridis',
                )
                fig_championship.update_layout(yaxis={'categoryorder': 'total ascending'})
            else:
                # Use simulation results to create championship probability chart
                top_champions = simulation_results.sort_values('ChampionshipProbability', ascending=False).head(10)
                fig_championship = px.bar(
                    top_champions,
                    y='TeamName',
                    x='ChampionshipProbability',
                    orientation='h',
                    title='Top 10 Championship Contenders',
                    labels={'TeamName': 'Team', 'ChampionshipProbability': 'Championship Probability (%)'},
                    color='ChampionshipProbability',
                    color_continuous_scale='Viridis',
                )
                fig_championship.update_layout(yaxis={'categoryorder': 'total ascending'})
            
            # Create Final Four probability chart
            if final_four_probs is not None:
                top_ff = final_four_probs.head(10)
                fig_final_four = px.bar(
                    top_ff,
                    y='Team',
                    x='Final Four Probability',
                    orientation='h',
                    title='Top 10 Final Four Contenders',
                    labels={'Team': 'Team', 'Final Four Probability': 'Final Four Probability (%)'},
                    color='Final Four Probability',
                    color_continuous_scale='Plasma',
                )
                fig_final_four.update_layout(yaxis={'categoryorder': 'total ascending'})
            else:
                # Use simulation results to create Final Four probability chart
                top_ff = simulation_results.sort_values('FinalFourProbability', ascending=False).head(10)
                fig_final_four = px.bar(
                    top_ff,
                    y='TeamName',
                    x='FinalFourProbability',
                    orientation='h',
                    title='Top 10 Final Four Contenders',
                    labels={'TeamName': 'Team', 'FinalFourProbability': 'Final Four Probability (%)'},
                    color='FinalFourProbability',
                    color_continuous_scale='Plasma',
                )
                fig_final_four.update_layout(yaxis={'categoryorder': 'total ascending'})
            
            # Create a scatter plot comparing predicted vs simulated exit rounds
            fig_scatter = px.scatter(
                simulation_results.head(50),
                x='PredictedExitRound',
                y='SimulatedExitRound',
                title='Predicted vs Simulated Tournament Exit Rounds (Top 50 Teams)',
                labels={
                    'PredictedExitRound': 'Predicted Exit Round (Deep Learning)',
                    'SimulatedExitRound': 'Simulated Exit Round (Tournament Structure)'
                },
                color='ChampionshipProbability',
                size='AdjEM',
                hover_name='TeamName',
                hover_data=['Seed', 'AdjEM', 'PredictedExit', 'SimulatedExit'],
                color_continuous_scale='Viridis',
            )
            
            # Add diagonal line for reference
            fig_scatter.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=7,
                y1=7,
                line=dict(color="gray", width=1, dash="dash"),
            )
            
            # Add horizontal and vertical lines for tournament rounds
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
                # Horizontal line (simulated exit round)
                fig_scatter.add_shape(
                    type="line",
                    x0=0,
                    y0=round_num,
                    x1=7,
                    y1=round_num,
                    line=dict(color="lightgray", width=1, dash="dot"),
                )
                
                # Vertical line (predicted exit round)
                fig_scatter.add_shape(
                    type="line",
                    x0=round_num,
                    y0=0,
                    x1=round_num,
                    y1=7,
                    line=dict(color="lightgray", width=1, dash="dot"),
                )
                
                # Add annotations
                fig_scatter.add_annotation(
                    x=round_num,
                    y=0,
                    text=round_name,
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",
                    yshift=-5,
                    font=dict(size=8, color="gray"),
                    textangle=-45
                )
                
                fig_scatter.add_annotation(
                    x=0,
                    y=round_num,
                    text=round_name,
                    showarrow=False,
                    xanchor="right",
                    yanchor="middle",
                    xshift=-5,
                    font=dict(size=8, color="gray"),
                )
            
            # Create the layout with the visualizations
            content = [
                html.H1("NCAA Tournament Bracket Prediction 2025", className="text-center mb-4"),
                html.P(
                    "Welcome to the March Madness Predictor Dashboard for the 2025 NCAA Tournament. "
                    "Below is our predicted tournament bracket based on 1,000 simulations of the NCAA tournament "
                    "using our deep learning model and respecting the tournament structure.",
                    className="lead text-center mb-4"
                ),
                
                dbc.Card([
                    dbc.CardHeader("NCAA Tournament Bracket Prediction", className="text-center"),
                    dbc.CardBody([
                        bracket_image
                    ])
                ], className="mb-4"),
                
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
                    dbc.CardHeader("Deep Learning vs Tournament Simulation"),
                    dbc.CardBody([
                        html.P(
                            "This chart compares our deep learning model's exit round predictions with the "
                            "tournament simulation results that respect the bracket structure. Points above "
                            "the diagonal line indicate teams that perform better in the simulation than "
                            "predicted by the deep learning model alone.",
                            className="mb-3"
                        ),
                        dcc.Graph(figure=fig_scatter)
                    ])
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Simulation Methodology"),
                            dbc.CardBody([
                                html.H5("Tournament Structure"),
                                html.P(
                                    "Our simulation respects the NCAA tournament structure with 68 teams, "
                                    "including First Four play-in games, and ensures exactly 4 teams reach "
                                    "the Final Four, 2 teams reach the Championship, and 1 team becomes Champion."
                                ),
                                html.H5("Team Strength Metrics"),
                                html.P(
                                    "Teams are evaluated based on KenPom adjusted efficiency metrics, "
                                    "predicted exit rounds from our deep learning model, and seed projections."
                                ),
                                html.H5("Simulation Process"),
                                html.P(
                                    "We ran 1,000 simulations of the complete tournament to generate "
                                    "robust probability estimates for each team's championship and Final Four chances."
                                ),
                                html.H5("Bracket Generation"),
                                html.P(
                                    "The bracket shown represents our best prediction based on the simulation "
                                    "results, with teams assigned to regions based on seed projections and "
                                    "matchups determined by the tournament structure."
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
        else:
            # If simulation results aren't available, show a message to run the simulator
            content = [
                html.H1("NCAA Tournament Bracket Prediction 2025", className="text-center mb-4"),
                html.P(
                    "Welcome to the March Madness Predictor Dashboard for the 2025 NCAA Tournament.",
                    className="lead text-center mb-4"
                ),
                
                dbc.Alert([
                    html.H4("Tournament Simulation Not Available", className="alert-heading"),
                    html.P(
                        "The tournament simulation results are not yet available. Please run the tournament simulator first:"
                    ),
                    html.Ol([
                        html.Li("Run the Champion Profile model: python march_madness_predictor/models/champion_profile/run_champion_profile_model.py"),
                        html.Li("Run the Exit Round model: python march_madness_predictor/models/exit_round/run_exit_round_model.py"),
                        html.Li("Run the Tournament Simulator: python tournament_simulator.py"),
                        html.Li("Generate the Bracket: python generate_bracket.py")
                    ]),
                    html.Hr(),
                    html.P(
                        "Once the simulator has been run, refresh this page to see the tournament bracket prediction.",
                        className="mb-0"
                    ),
                ], color="warning", className="mb-4"),
                
                dbc.Card([
                    dbc.CardHeader("Tournament Simulator Overview"),
                    dbc.CardBody([
                        html.H5("Tournament Structure"),
                        html.P(
                            "Our simulation respects the NCAA tournament structure with 68 teams, "
                            "including First Four play-in games, and ensures exactly 4 teams reach "
                            "the Final Four, 2 teams reach the Championship, and 1 team becomes Champion."
                        ),
                        html.H5("Team Strength Metrics"),
                        html.P(
                            "Teams are evaluated based on KenPom adjusted efficiency metrics, "
                            "predicted exit rounds from our deep learning model, and seed projections."
                        ),
                        html.H5("Simulation Process"),
                        html.P(
                            "We run 1,000 simulations of the complete tournament to generate "
                            "robust probability estimates for each team's championship and Final Four chances."
                        ),
                        html.H5("Bracket Generation"),
                        html.P(
                            "The bracket represents our best prediction based on the simulation "
                            "results, with teams assigned to regions based on seed projections and "
                            "matchups determined by the tournament structure."
                        ),
                    ])
                ], className="mb-4"),
                
                # Navigation card (same as above)
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
            ]
    
    except Exception as e:
        # If there's an error, show a message
        content = [
            html.H1("NCAA Tournament Bracket Prediction 2025", className="text-center mb-4"),
            html.P(
                "Welcome to the March Madness Predictor Dashboard for the 2025 NCAA Tournament.",
                className="lead text-center mb-4"
            ),
            
            dbc.Alert([
                html.H4("Error Loading Tournament Simulation", className="alert-heading"),
                html.P(f"An error occurred while loading the tournament simulation results: {str(e)}"),
                html.P(
                    "Please make sure you have run the tournament simulator and generated the bracket:"
                ),
                html.Ol([
                    html.Li("Run the Champion Profile model: python march_madness_predictor/models/champion_profile/run_champion_profile_model.py"),
                    html.Li("Run the Exit Round model: python march_madness_predictor/models/exit_round/run_exit_round_model.py"),
                    html.Li("Run the Tournament Simulator: python tournament_simulator.py"),
                    html.Li("Generate the Bracket: python generate_bracket.py")
                ]),
                html.Hr(),
                html.P(
                    "Once the simulator has been run, refresh this page to see the tournament bracket prediction.",
                    className="mb-0"
                ),
            ], color="danger", className="mb-4"),
        ]
    
    return html.Div(content, className="container py-4") 