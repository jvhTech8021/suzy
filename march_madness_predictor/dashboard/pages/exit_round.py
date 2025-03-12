import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def layout(data_loader):
    """
    Create the layout for the exit round prediction page
    
    Parameters:
    -----------
    data_loader : DataLoader
        Instance of the DataLoader class
        
    Returns:
    --------
    html.Div
        Layout for the exit round prediction page
    """
    try:
        # Load the exit round predictions
        predictions = data_loader.get_exit_round_predictions()
        tournament_teams = predictions[predictions['Seed'].notnull()].copy()
        
        # Load seed performance data if available
        try:
            seed_performance = data_loader.get_seed_performance()
            has_seed_data = True
        except:
            has_seed_data = False
        
        # Create a bar chart of the top 20 teams by championship probability
        top_20 = tournament_teams.sort_values('ChampionshipPct', ascending=False).head(20)
        fig_championship = px.bar(
            top_20,
            y='TeamName',
            x='ChampionshipPct',
            orientation='h',
            title='Top 20 Championship Contenders',
            labels={'TeamName': 'Team', 'ChampionshipPct': 'Championship Probability (%)'},
            color='ChampionshipPct',
            color_continuous_scale='Viridis',
        )
        fig_championship.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        # Create a bar chart of the top 20 teams by Final Four probability
        top_20_ff = tournament_teams.sort_values('FinalFourPct', ascending=False).head(20)
        fig_final_four = px.bar(
            top_20_ff,
            y='TeamName',
            x='FinalFourPct',
            orientation='h',
            title='Top 20 Final Four Contenders',
            labels={'TeamName': 'Team', 'FinalFourPct': 'Final Four Probability (%)'},
            color='FinalFourPct',
            color_continuous_scale='Plasma',
        )
        fig_final_four.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        # Create a scatter plot of AdjEM vs predicted exit round
        fig_scatter = px.scatter(
            tournament_teams,
            x='AdjEM',
            y='PredictedExitRound',
            title='Team Strength vs Predicted Tournament Exit Round',
            labels={'AdjEM': 'Adjusted Efficiency Margin', 'PredictedExitRound': 'Predicted Exit Round'},
            color='Seed',
            size='ChampionshipPct',
            hover_name='TeamName',
            hover_data=['PredictedExit', 'ChampionshipPct', 'FinalFourPct'],
            color_continuous_scale='Viridis_r',  # Reversed so lower seeds (better) are at the red/yellow end
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
                x0=tournament_teams['AdjEM'].min(),
                y0=round_num,
                x1=tournament_teams['AdjEM'].max(),
                y1=round_num,
                line=dict(color="gray", width=1, dash="dash"),
            )
            fig_scatter.add_annotation(
                x=tournament_teams['AdjEM'].min(),
                y=round_num,
                text=round_name,
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                xshift=5,
                font=dict(size=10, color="gray"),
            )
        
        # Create a table of the top 100 teams (instead of top 15)
        top_100_df = tournament_teams.sort_values('ChampionshipPct', ascending=False).head(100)[
            ['TeamName', 'Seed', 'ChampionshipPct', 'FinalFourPct', 'PredictedExit', 
             'AdjEM', 'AdjOE', 'AdjDE', 'RankAdjEM']
        ]
        
        # Format the columns
        top_100_df['ChampionshipPct'] = top_100_df['ChampionshipPct'].round(1)
        top_100_df['FinalFourPct'] = top_100_df['FinalFourPct'].round(1)
        top_100_df['AdjEM'] = top_100_df['AdjEM'].round(1)
        top_100_df['AdjOE'] = top_100_df['AdjOE'].round(1)
        top_100_df['AdjDE'] = top_100_df['AdjDE'].round(1)
        
        # Rename columns for display
        top_100_df.columns = ['Team', 'Seed', 'Champion (%)', 'Final Four (%)', 'Predicted Exit', 
                            'Adj EM', 'Off Eff', 'Def Eff', 'Nat Rank']
        
        top_100_table = dash_table.DataTable(
            id='top-100-championship-table',
            columns=[{"name": i, "id": i} for i in top_100_df.columns],
            data=top_100_df.to_dict('records'),
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
                    'if': {'filter_query': '{Champion (%)} > 10'},
                    'backgroundColor': '#FFFFCC'
                }
            ],
            sort_action='native',
            filter_action='native',
            page_size=20,  # Increase page size for more teams per page
        )
        
        # Create a histogram of predicted exit rounds
        exit_counts = tournament_teams['PredictedExitRoundInt'].value_counts().sort_index()
        exit_df = pd.DataFrame({
            'Exit Round': [round_names[r] for r in exit_counts.index],
            'Count': exit_counts.values
        })
        
        fig_histogram = px.bar(
            exit_df,
            x='Exit Round',
            y='Count',
            title='Distribution of Predicted Tournament Exit Rounds',
            color='Count',
            color_continuous_scale='Viridis',
        )
        
        # Create seed performance visualization if data is available
        if has_seed_data:
            fig_seed = px.bar(
                seed_performance,
                x='Seed',
                y='AvgExitRound',
                title='Average Tournament Exit Round by Seed (Historical)',
                labels={'Seed': 'Seed', 'AvgExitRound': 'Average Exit Round'},
                color='AvgExitRound',
                color_continuous_scale='Viridis',
                text='Count'
            )
            
            # Add horizontal lines for tournament rounds
            for round_num, round_name in round_names.items():
                fig_seed.add_shape(
                    type="line",
                    x0=0,
                    y0=round_num,
                    x1=17,
                    y1=round_num,
                    line=dict(color="gray", width=1, dash="dash"),
                )
                fig_seed.add_annotation(
                    x=16.5,
                    y=round_num,
                    text=round_name,
                    showarrow=False,
                    xanchor="right",
                    yanchor="bottom",
                    font=dict(size=10, color="gray"),
                )
        
        # Create the layout
        content = [
            html.H1("Tournament Exit Round Predictions", className="text-center mb-4"),
            html.P(
                "This page shows predictions for how far teams will advance in the 2025 NCAA Tournament based on a deep learning model.",
                className="lead text-center mb-4"
            ),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Championship Contenders"),
                        dbc.CardBody([
                            dcc.Graph(figure=fig_championship)
                        ])
                    ], className="mb-4")
                ], md=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Final Four Contenders"),
                        dbc.CardBody([
                            dcc.Graph(figure=fig_final_four)
                        ])
                    ], className="mb-4")
                ], md=6)
            ]),
            
            dbc.Card([
                dbc.CardHeader("Team Strength vs Predicted Tournament Exit"),
                dbc.CardBody([
                    html.P(
                        "This chart shows the relationship between team strength (Adjusted Efficiency Margin) and predicted tournament exit round. "
                        "The size of each point represents championship probability, and the color represents seed (lighter colors = better seeds)."
                    ),
                    dcc.Graph(figure=fig_scatter)
                ])
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Distribution of Predicted Exit Rounds"),
                        dbc.CardBody([
                            dcc.Graph(figure=fig_histogram)
                        ])
                    ], className="mb-4")
                ], md=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Historical Seed Performance"),
                        dbc.CardBody([
                            dcc.Graph(figure=fig_seed) if has_seed_data else html.P("Seed performance data not available.")
                        ])
                    ], className="mb-4")
                ], md=6) if has_seed_data else None
            ]),
            
            dbc.Card([
                dbc.CardHeader("Top 100 Teams by Championship Probability"),
                dbc.CardBody([
                    html.P(
                        "This table shows the top 100 teams with the highest predicted championship probability. "
                        "The predictions are based on a deep learning model trained on historical tournament data."
                    ),
                    top_100_table
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Methodology"),
                dbc.CardBody([
                    html.P(
                        "The exit round prediction model uses a deep neural network trained on historical NCAA tournament data "
                        "from 2009-2024. The model considers several key factors:"
                    ),
                    html.Ul([
                        html.Li("Team strength metrics (AdjEM, AdjOE, AdjDE)"),
                        html.Li("National rankings"),
                        html.Li("Tournament seed"),
                        html.Li("Schedule strength"),
                        html.Li("Historical performance of similar teams"),
                    ]),
                    html.P(
                        "The model predicts how far each team is likely to advance in the tournament, from the First Round "
                        "to becoming the National Champion. These predictions are then converted to championship and Final Four "
                        "probabilities based on the predicted exit round."
                    ),
                    html.P(
                        "Note that tournament seeds for 2025 are estimated based on team rankings, as the actual tournament "
                        "seeding has not yet been determined."
                    ),
                ])
            ], className="mb-4"),
        ]
    
    except Exception as e:
        # If data isn't available, show a message to run the models first
        content = [
            html.H1("Tournament Exit Round Predictions", className="text-center mb-4"),
            
            dbc.Alert([
                html.H4("Data Not Available", className="alert-heading"),
                html.P(
                    "The exit round prediction data is not yet available. Please run the exit round model first:"
                ),
                html.Pre(
                    "python march_madness_predictor/models/exit_round/run_exit_round_model.py",
                    className="bg-light p-2 border"
                ),
                html.Hr(),
                html.P(
                    "Once the model has been run, refresh this page to see the predictions.",
                    className="mb-0"
                ),
            ], color="warning", className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("About the Exit Round Prediction Model"),
                dbc.CardBody([
                    html.P(
                        "The exit round prediction model uses deep learning to predict how far teams will advance in the "
                        "NCAA tournament based on their statistical profile and estimated seed. The model is trained on "
                        "historical tournament data from 2009-2024 and learns patterns that correlate with tournament success."
                    ),
                    html.P(
                        "By analyzing a team's current season metrics and comparing them to historical patterns, the model "
                        "can predict the most likely tournament outcome for each team."
                    ),
                ])
            ], className="mb-4")
        ]
    
    return html.Div(content) 