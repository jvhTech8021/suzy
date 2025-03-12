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
        
        # Calculate Sweet 16 and Elite 8 probabilities based on exit round predictions
        tournament_teams['SweetSixteenPct'] = 0.0
        tournament_teams['EliteEightPct'] = 0.0
        
        for idx, row in tournament_teams.iterrows():
            exit_round = row['PredictedExitRoundInt'] if pd.notna(row['PredictedExitRoundInt']) else 0
            seed = row['Seed'] if pd.notna(row['Seed']) else 16
            
            # Elite 8 probability calculation
            if exit_round >= 4:  # Predicted Elite Eight or better
                elite_eight_pct = 90.0
            elif exit_round == 3:  # Predicted Sweet Sixteen
                elite_eight_pct = 35.0
            elif seed <= 4:  # Top 4 seeds
                elite_eight_pct = 40.0
            elif seed <= 8:
                elite_eight_pct = 20.0
            else:
                elite_eight_pct = 10.0
            
            # Sweet 16 probability calculation
            if exit_round >= 3:  # Predicted Sweet Sixteen or better
                sweet_sixteen_pct = 90.0
            elif exit_round == 2:  # Predicted Round of 32
                sweet_sixteen_pct = 40.0
            elif seed <= 4:  # Top 4 seeds
                sweet_sixteen_pct = 65.0
            elif seed <= 8:
                sweet_sixteen_pct = 35.0
            else:
                sweet_sixteen_pct = 15.0
            
            tournament_teams.loc[idx, 'EliteEightPct'] = elite_eight_pct
            tournament_teams.loc[idx, 'SweetSixteenPct'] = sweet_sixteen_pct
        
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
        
        # Create a bar chart of the top 20 teams by Elite Eight probability
        top_20_e8 = tournament_teams.sort_values('EliteEightPct', ascending=False).head(20)
        fig_elite_eight = px.bar(
            top_20_e8,
            y='TeamName',
            x='EliteEightPct',
            orientation='h',
            title='Top 20 Elite Eight Contenders',
            labels={'TeamName': 'Team', 'EliteEightPct': 'Elite Eight Probability (%)'},
            color='EliteEightPct',
            color_continuous_scale='Teal',
        )
        fig_elite_eight.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        # Create a bar chart of the top 20 teams by Sweet Sixteen probability
        top_20_s16 = tournament_teams.sort_values('SweetSixteenPct', ascending=False).head(20)
        fig_sweet_sixteen = px.bar(
            top_20_s16,
            y='TeamName',
            x='SweetSixteenPct',
            orientation='h',
            title='Top 20 Sweet Sixteen Contenders',
            labels={'TeamName': 'Team', 'SweetSixteenPct': 'Sweet Sixteen Probability (%)'},
            color='SweetSixteenPct',
            color_continuous_scale='Cividis',
        )
        fig_sweet_sixteen.update_layout(yaxis={'categoryorder': 'total ascending'})
        
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
            hover_data=['PredictedExit', 'ChampionshipPct', 'FinalFourPct', 'EliteEightPct', 'SweetSixteenPct'],
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
        
        # Create a table of the top 200 teams (instead of 100)
        top_200_df = tournament_teams.sort_values('ChampionshipPct', ascending=False).head(200)[
            ['TeamName', 'Seed', 'ChampionshipPct', 'FinalFourPct', 'EliteEightPct', 'SweetSixteenPct', 'PredictedExit', 
             'AdjEM', 'AdjOE', 'AdjDE', 'RankAdjEM']
        ]
        
        # Format the columns
        top_200_df['ChampionshipPct'] = top_200_df['ChampionshipPct'].round(1)
        top_200_df['FinalFourPct'] = top_200_df['FinalFourPct'].round(1)
        top_200_df['EliteEightPct'] = top_200_df['EliteEightPct'].round(1)
        top_200_df['SweetSixteenPct'] = top_200_df['SweetSixteenPct'].round(1)
        top_200_df['AdjEM'] = top_200_df['AdjEM'].round(1)
        top_200_df['AdjOE'] = top_200_df['AdjOE'].round(1)
        top_200_df['AdjDE'] = top_200_df['AdjDE'].round(1)
        
        # Rename columns for display
        top_200_df.columns = ['Team', 'Seed', 'Champion (%)', 'Final Four (%)', 'Elite Eight (%)', 'Sweet 16 (%)', 'Predicted Exit', 
                            'Adj EM', 'Off Eff', 'Def Eff', 'Nat Rank']
        
        top_200_table = dash_table.DataTable(
            id='top-200-championship-table',
            columns=[{"name": i, "id": i} for i in top_200_df.columns],
            data=top_200_df.to_dict('records'),
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
            style_table={
                'height': '600px',  # Set a fixed height to enable scrolling
                'overflowY': 'auto'  # Enable vertical scrolling
            },
            sort_action='native',
            filter_action='native',
            page_size=200,  # Show all 200 teams on one page with scrolling
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
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Elite Eight Contenders"),
                        dbc.CardBody([
                            dcc.Graph(figure=fig_elite_eight)
                        ])
                    ], className="mb-4")
                ], md=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Sweet Sixteen Contenders"),
                        dbc.CardBody([
                            dcc.Graph(figure=fig_sweet_sixteen)
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
                dbc.CardHeader("Top 200 Teams by Tournament Round Probabilities"),
                dbc.CardBody([
                    html.P(
                        "This table shows the top 200 teams with their predicted probabilities of reaching various tournament rounds. "
                        "The predictions are based on a deep learning model trained on historical tournament data."
                    ),
                    top_200_table
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