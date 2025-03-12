import dash
from dash import dcc, html, Output, Input, State, callback, dash_table, clientside_callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import os
import traceback
from pathlib import Path
from dash.dependencies import Input, Output, State, ALL, MATCH
from dash.exceptions import PreventUpdate
from dash import no_update

def layout(data_loader):
    """
    Create the layout for the tournament success level analysis page
    
    Parameters:
    -----------
    data_loader : DataLoader
        Instance of the DataLoader class
        
    Returns:
    --------
    html.Div
        Layout for the tournament success level analysis page
    """
    try:
        # Debug info
        print("\n=== TOURNAMENT LEVEL ANALYSIS PAGE DEBUG ===")
        print(f"Champion profile dir: {data_loader.champion_profile_dir}")
        
        # Load current season data for detailed comparisons
        prediction_data = data_loader.get_current_season_data()
        
        # Define tournament round levels
        round_levels = {
            7: "National Champions",
            6: "Championship Game",
            5: "Final Four",
            4: "Elite Eight",
            3: "Sweet Sixteen",
            2: "Round of 32",
            1: "Tournament Qualifiers"
        }
        
        # Load tournament level analysis data
        round_data = {}
        for round_num, round_name in round_levels.items():
            json_path = os.path.join(
                data_loader.champion_profile_dir, 
                f"{round_name.lower().replace(' ', '_')}_analysis.json"
            )
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    round_data[round_num] = json.load(f)
                print(f"Loaded {round_name} analysis from {json_path}")
        
        # Check if we have data
        if not round_data:
            print("No tournament level analysis data found")
            print("=== END DEBUG ===\n")
            
            # Return error message
            return html.Div([
                html.H1("Tournament Success Level Analysis", className="text-center mb-4"),
                
                dbc.Alert([
                    html.H4("Data Not Available", className="alert-heading"),
                    html.P(
                        "The tournament success level analysis data is not available. Please run the champion profile model first:"
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
            ])
        
        # Create tabs for each round level
        tabs = []
        tab_contents = []
        
        for round_num in sorted(round_data.keys(), reverse=True):
            round_info = round_data[round_num]
            round_name = round_info['RoundName']
            
            # Tab
            tabs.append(
                dbc.Tab(
                    label=round_name,
                    tab_id=f"tab-{round_num}",
                )
            )
            
            # Create content for this round
            profile = round_info['Profile']
            
            # Add info note for National Champions tab
            info_note = None
            if round_name == "National Champions":
                info_note = dbc.Alert([
                    html.H5("Note on Champion Profile vs National Champions Analysis", className="alert-heading"),
                    html.P([
                        "While both the Champion Profile page and this National Champions tab analyze historical champions, there are key differences:"
                    ]),
                    html.Ul([
                        html.Li("The Champion Profile page uses a carefully calibrated profile with fixed weightings optimized over time"),
                        html.Li("This Tournament Level Analysis automatically creates profiles by averaging all historical teams at each level"),
                        html.Li("The Champion Profile page ranks teams based on overall similarity to the champion profile"),
                        html.Li("This view shows specific team-to-team comparisons with historical champions")
                    ]),
                    html.P([
                        "Both analyses provide valuable insights, but may highlight different teams due to these methodological differences."
                    ])
                ], color="info", className="mb-4")
            
            # Create profile table
            profile_table = dash_table.DataTable(
                id=f'profile-table-{round_num}',
                columns=[
                    {'name': 'Metric', 'id': 'Metric'},
                    {'name': 'Value', 'id': 'Value'},
                    {'name': 'Ranking', 'id': 'Ranking'}
                ],
                data=[
                    {'Metric': 'Adjusted Efficiency Margin (AdjEM)', 'Value': f"{profile['AdjEM']:.2f}", 'Ranking': f"{profile['RankAdjEM']:.1f}"},
                    {'Metric': 'Offensive Efficiency (AdjOE)', 'Value': f"{profile['AdjOE']:.1f}", 'Ranking': f"{profile.get('RankAdjOE', '-'):.1f}" if 'RankAdjOE' in profile else '-'},
                    {'Metric': 'Defensive Efficiency (AdjDE)', 'Value': f"{profile['AdjDE']:.1f}", 'Ranking': f"{profile.get('RankAdjDE', '-'):.1f}" if 'RankAdjDE' in profile else '-'},
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
            
            # Create table of top teams
            top_teams = round_info['CurrentTeams'][:20]  # Top 20 teams
            
            team_rows = []
            for i, team in enumerate(top_teams, 1):
                team_name = team['TeamName']
                similarity = team['Similarity']
                
                # Get similar teams text
                similar_teams_text = ""
                for similar in team['SimilarTeams'][:3]:  # Top 3 similar teams
                    similar_teams_text += f"{similar['TeamName']} ({similar['Season']}), "
                
                if similar_teams_text:
                    similar_teams_text = similar_teams_text[:-2]  # Remove trailing comma
                
                team_rows.append({
                    'Rank': i,
                    'Team': team_name,
                    'Similarity': f"{similarity:.1f}%",
                    'Similar Historical Teams': similar_teams_text
                })
            
            top_teams_table = dash_table.DataTable(
                id=f'top-teams-table-{round_num}',
                columns=[{"name": i, "id": i} for i in ['Rank', 'Team', 'Similarity', 'Similar Historical Teams']],
                data=team_rows,
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_cell={
                    'textAlign': 'center',
                    'padding': '10px',
                    'minWidth': '100px',
                    'maxWidth': '400px',
                    'whiteSpace': 'normal'
                },
                style_cell_conditional=[
                    {
                        'if': {'column_id': 'Team'},
                        'textAlign': 'left'
                    },
                    {
                        'if': {'column_id': 'Similar Historical Teams'},
                        'textAlign': 'left',
                        'width': '40%'
                    }
                ],
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    },
                    {
                        'if': {'filter_query': '{Rank} <= 3'},
                        'backgroundColor': '#FFFFCC'
                    }
                ],
                page_size=10,
            )
            
            # Create detailed comparison cards
            comparison_cards = []
            for i, team in enumerate(top_teams[:10], 1):  # Top 10 teams
                team_name = team['TeamName']
                similar_teams = team['SimilarTeams'][:5]  # Top 5 similar teams
                
                # Get current team stats from predictions
                current_team_stats = None
                for _, row in prediction_data.iterrows():
                    if row['TeamName'] == team_name:
                        current_team_stats = {
                            'AdjEM': row['AdjEM'],
                            'RankAdjEM': row['RankAdjEM'],
                            'AdjOE': row['AdjOE'],
                            'AdjDE': row['AdjDE']
                        }
                        
                        # Add rankings if available
                        if 'RankAdjOE' in row:
                            current_team_stats['RankAdjOE'] = row['RankAdjOE']
                        
                        if 'RankAdjDE' in row:
                            current_team_stats['RankAdjDE'] = row['RankAdjDE']
                        break
                
                # Similar teams list with detailed comparison
                similar_list = []
                for similar in similar_teams:
                    # Lookup historical team data to get complete stats
                    historical_team_stats = None
                    hist_season = similar['Season']
                    hist_team_name = similar['TeamName']
                    
                    # Get the exact same stats we used for similarity calculation
                    for year in range(2009, 2025):
                        try:
                            # Skip 2020 (COVID year)
                            if year == 2020:
                                continue
                                
                            # Only load the relevant year
                            if year != hist_season:
                                continue
                                
                            hist_df = None
                            try:
                                hist_df = data_loader.get_historical_data(year)
                            except:
                                continue
                            
                            if hist_df is not None:
                                # Find the team
                                for _, hist_row in hist_df.iterrows():
                                    if hist_row['TeamName'] == hist_team_name:
                                        historical_team_stats = {
                                            'AdjEM': hist_row['AdjEM'],
                                            'RankAdjEM': hist_row['RankAdjEM'],
                                            'AdjOE': hist_row['AdjOE'],
                                            'AdjDE': hist_row['AdjDE']
                                        }
                                        
                                        # Add rankings if available
                                        if 'RankAdjOE' in hist_row:
                                            historical_team_stats['RankAdjOE'] = hist_row['RankAdjOE']
                                        
                                        if 'RankAdjDE' in hist_row:
                                            historical_team_stats['RankAdjDE'] = hist_row['RankAdjDE']
                                        break
                                        
                        except Exception as e:
                            print(f"Error loading historical data for {hist_season} {hist_team_name}: {e}")
                    
                    exit_round_text = round_levels.get(similar['ExitRound'], "Unknown")
                    
                    # Create an accordion item for detailed comparison
                    if current_team_stats and historical_team_stats:
                        # Calculate differences
                        em_diff = current_team_stats['AdjEM'] - historical_team_stats['AdjEM']
                        em_rank_diff = current_team_stats['RankAdjEM'] - historical_team_stats['RankAdjEM']
                        oe_diff = current_team_stats['AdjOE'] - historical_team_stats['AdjOE']
                        de_diff = current_team_stats['AdjDE'] - historical_team_stats['AdjDE']
                        
                        # Calculate rank differences if available
                        oe_rank_diff = None
                        de_rank_diff = None
                        
                        if 'RankAdjOE' in current_team_stats and 'RankAdjOE' in historical_team_stats:
                            oe_rank_diff = current_team_stats['RankAdjOE'] - historical_team_stats['RankAdjOE']
                        
                        if 'RankAdjDE' in current_team_stats and 'RankAdjDE' in historical_team_stats:
                            de_rank_diff = current_team_stats['RankAdjDE'] - historical_team_stats['RankAdjDE']
                        
                        # Generate insights based on the differences
                        insights = []
                        
                        # Overall efficiency insight
                        if abs(em_diff) < 2:
                            insights.append(f"Nearly identical overall efficiency ({em_diff:.2f} points difference)")
                        elif em_diff > 0:
                            insights.append(f"{team_name} has a stronger overall efficiency by {em_diff:.2f} points")
                        else:
                            insights.append(f"{hist_team_name} had a stronger overall efficiency by {abs(em_diff):.2f} points")
                        
                        # National ranking insight
                        if abs(em_rank_diff) < 3:
                            insights.append(f"Very similar national rankings (within {abs(em_rank_diff)} positions)")
                        
                        # Offensive efficiency insight
                        if abs(oe_diff) < 2:
                            insights.append(f"Nearly identical offensive efficiency ({abs(oe_diff):.1f} points difference)")
                        elif oe_diff > 2:
                            insights.append(f"{team_name} has a more efficient offense by {oe_diff:.1f} points")
                        elif oe_diff < -2:
                            insights.append(f"{hist_team_name} had a more efficient offense by {abs(oe_diff):.1f} points")
                            
                        # Defensive efficiency insight
                        if abs(de_diff) < 2:
                            insights.append(f"Nearly identical defensive efficiency ({abs(de_diff):.1f} points difference)")
                        elif de_diff < -2:  # Lower is better for defense
                            insights.append(f"{team_name} has a more efficient defense by {abs(de_diff):.1f} points")
                        elif de_diff > 2:
                            insights.append(f"{hist_team_name} had a more efficient defense by {de_diff:.1f} points")
                        
                        # Add rank-based insights if available
                        if oe_rank_diff is not None:
                            if abs(oe_rank_diff) < 3:
                                insights.append(f"Very similar offensive rankings (within {abs(oe_rank_diff)} positions)")
                            
                        if de_rank_diff is not None:
                            if abs(de_rank_diff) < 3:
                                insights.append(f"Very similar defensive rankings (within {abs(de_rank_diff)} positions)")
                        
                        # Create the comparison table
                        comparison_table = html.Table([
                            html.Thead([
                                html.Tr([
                                    html.Th("Metric"),
                                    html.Th(team_name),
                                    html.Th(f"{hist_team_name} ({hist_season})"),
                                    html.Th("Difference")
                                ])
                            ]),
                            html.Tbody([
                                html.Tr([
                                    html.Td("Adjusted Efficiency Margin"),
                                    html.Td(f"{current_team_stats['AdjEM']:.2f}"),
                                    html.Td(f"{historical_team_stats['AdjEM']:.2f}"),
                                    html.Td(f"{em_diff:+.2f}", style={'color': 'green' if em_diff > 0 else 'red' if em_diff < 0 else 'black'})
                                ]),
                                html.Tr([
                                    html.Td("National Ranking"),
                                    html.Td(f"{current_team_stats['RankAdjEM']:.0f}"),
                                    html.Td(f"{historical_team_stats['RankAdjEM']:.0f}"),
                                    html.Td(f"{em_rank_diff:+.0f}", style={'color': 'red' if em_rank_diff > 0 else 'green' if em_rank_diff < 0 else 'black'})
                                ]),
                                html.Tr([
                                    html.Td("Offensive Efficiency"),
                                    html.Td(f"{current_team_stats['AdjOE']:.1f}"),
                                    html.Td(f"{historical_team_stats['AdjOE']:.1f}"),
                                    html.Td(f"{oe_diff:+.1f}", style={'color': 'green' if oe_diff > 0 else 'red' if oe_diff < 0 else 'black'})
                                ]),
                                html.Tr([
                                    html.Td("Defensive Efficiency"),
                                    html.Td(f"{current_team_stats['AdjDE']:.1f}"),
                                    html.Td(f"{historical_team_stats['AdjDE']:.1f}"),
                                    html.Td(f"{de_diff:+.1f}", style={'color': 'red' if de_diff > 0 else 'green' if de_diff < 0 else 'black'})
                                ])
                            ] + (
                                # Add offensive ranking row if available
                                [html.Tr([
                                    html.Td("Offensive Ranking"),
                                    html.Td(f"{current_team_stats['RankAdjOE']:.0f}"),
                                    html.Td(f"{historical_team_stats['RankAdjOE']:.0f}"),
                                    html.Td(f"{oe_rank_diff:+.0f}", style={'color': 'red' if oe_rank_diff > 0 else 'green' if oe_rank_diff < 0 else 'black'})
                                ])] if 'RankAdjOE' in current_team_stats and 'RankAdjOE' in historical_team_stats else []
                            ) + (
                                # Add defensive ranking row if available
                                [html.Tr([
                                    html.Td("Defensive Ranking"),
                                    html.Td(f"{current_team_stats['RankAdjDE']:.0f}"),
                                    html.Td(f"{historical_team_stats['RankAdjDE']:.0f}"),
                                    html.Td(f"{de_rank_diff:+.0f}", style={'color': 'red' if de_rank_diff > 0 else 'green' if de_rank_diff < 0 else 'black'})
                                ])] if 'RankAdjDE' in current_team_stats and 'RankAdjDE' in historical_team_stats else []
                            )
                        )], className="table table-striped table-bordered")
                        
                        comparison_details = [
                            html.H6(f"Statistical Comparison ({similar['Similarity']:.1f}% similarity)"),
                            comparison_table,
                            html.H6("Key Insights", className="mt-3"),
                            html.Ul([html.Li(insight) for insight in insights])
                        ]
                    else:
                        # Fallback if we don't have complete stats
                        comparison_details = html.P("Detailed statistics not available for complete comparison")
                    
                    # Create the collapsible item
                    similar_list.append(
                        html.Div([
                            html.Div([
                                html.Button([
                                    f"{hist_team_name} ({hist_season}): ",
                                    html.Span(f"{similar['Similarity']:.1f}% similarity", style={'fontWeight': 'bold'}),
                                    f", Exit: {exit_round_text}",
                                    html.I(id={'type': 'icon', 'id': f"{round_num}-{i}-{similar['Season']}-{hist_team_name.replace(' ', '-')}"}, 
                                          className="bi bi-chevron-down ms-2")
                                ], 
                                className="comparison-toggle btn btn-link text-dark text-decoration-none w-100 text-start",
                                id={'type': 'toggle', 'id': f"{round_num}-{i}-{similar['Season']}-{hist_team_name.replace(' ', '-')}"},
                                n_clicks=0
                                ),
                            ]),
                            html.Div([
                                html.Div(comparison_details, className="pt-3")
                            ], 
                            id={'type': 'collapse', 'id': f"{round_num}-{i}-{similar['Season']}-{hist_team_name.replace(' ', '-')}"},
                            className="comparison-details mt-2 ps-4",
                            style={'display': 'none'})
                        ], className="mb-2")
                    )
                
                # Build the card
                comparison_cards.append(
                    dbc.Card([
                        dbc.CardHeader(f"{i}. {team_name} - {team['Similarity']:.1f}% Similar to {round_name}"),
                        dbc.CardBody([
                            html.P("Teams that reached this level with similar statistical profiles:"),
                            html.Div(similar_list),
                            html.Div([
                                html.Hr(),
                                html.P([
                                    "The similarity score is based on a weighted formula that emphasizes rankings (65%) ",
                                    "over absolute statistical values (35%). This is because a team's relative position ",
                                    "within its season is more predictive of tournament success than raw metrics."
                                ], className="text-muted small")
                            ])
                        ])
                    ], className="mb-3")
                )
            
            # Create content for this tab
            tab_content = html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                f"{round_name} Profile Analysis"
                            ),
                            dbc.CardBody([
                                html.P(
                                    f"This analysis shows the average statistical profile of teams that reached the {round_name} round "
                                    f"in NCAA tournaments from 2009-2024. Analysis based on {round_info['TeamCount']} historical teams."
                                ),
                                info_note if info_note is not None else None,
                                profile_table
                            ])
                        ], className="mb-4"),
                    ], md=6),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(f"Top Teams Most Similar to {round_name} Profile"),
                            dbc.CardBody([
                                html.P(
                                    f"These current season teams have statistical profiles most similar to teams "
                                    f"that reached the {round_name} in previous tournaments."
                                ),
                                top_teams_table
                            ])
                        ], className="mb-4"),
                    ], md=6)
                ]),
                
                html.H4(f"Team Comparisons for {round_name}", className="mt-4 mb-3"),
                html.P(
                    "For each current team, we show the most similar historical teams that reached this tournament level. "
                    "Higher similarity percentages indicate stronger statistical resemblance."
                ),
                
                dbc.Row([
                    dbc.Col(
                        comparison_cards[:5], 
                        md=6
                    ),
                    dbc.Col(
                        comparison_cards[5:], 
                        md=6
                    )
                ])
            ])
            
            tab_contents.append(tab_content)
        
        # Create the tabs component
        tabs_component = dbc.Tabs(
            [
                dbc.Tab(
                    tab_contents[i],
                    label=tabs[i].label,
                    tab_id=tabs[i].tab_id,
                    className="p-4"
                )
                for i in range(len(tabs))
            ],
            id="tournament-level-tabs",
            active_tab="tab-7"  # Start with Champions tab
        )
        
        print("=== END DEBUG ===\n")
        
        # Create the layout
        content = [
            html.H1("Tournament Success Level Analysis", className="text-center mb-4"),
            html.P(
                "This analysis compares current teams to historical teams that reached specific rounds in the NCAA tournament.",
                className="lead text-center mb-4"
            ),
            
            dbc.Card([
                dbc.CardHeader("About This Analysis"),
                dbc.CardBody([
                    html.P([
                        "For each level of tournament success (from qualifying to winning a championship), ",
                        "we analyze the statistical profiles of historical teams that reached that level. ",
                        "Then we identify current teams that most closely resemble those historical performers."
                    ]),
                    html.P([
                        "This helps answer questions like: ",
                        html.Em("\"Which current teams have profiles similar to past Final Four teams?\" "),
                        "or ",
                        html.Em("\"Does Team X resemble historical champion-caliber teams?\"")
                    ]),
                    html.P([
                        "The analysis places significant weight (65%) on rankings rather than just raw statistical values, ",
                        "since a team's relative position within its season is more predictive of tournament success."
                    ])
                ])
            ], className="mb-4"),
            
            tabs_component,
            
            # Add some hidden divs for storing state
            html.Div(id='hidden-div', style={'display': 'none'})
        ]
        
        # Return the layout wrapped in a div
        return html.Div(content)

    except Exception as e:
        # Debug info for the exception
        print("\n=== TOURNAMENT LEVEL ANALYSIS PAGE ERROR ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        print("=== END ERROR ===\n")
        
        # If there's an error, show a message
        content = [
            html.H1("Tournament Success Level Analysis", className="text-center mb-4"),
            
            dbc.Alert([
                html.H4("An Error Occurred", className="alert-heading"),
                html.P(f"Error: {str(e)}"),
                html.Hr(),
                html.P(
                    "Please run the champion profile model with tournament level analysis:",
                    className="mb-0"
                ),
                html.Pre(
                    "python march_madness_predictor/models/champion_profile/run_champion_profile_model.py",
                    className="bg-light p-2 border"
                )
            ], color="danger", className="mb-4"),
        ]
    
    # Return the layout wrapped in a div
    return html.Div(content)

# Add this callback outside the layout function
@callback(
    Output({'type': 'collapse', 'id': MATCH}, 'style'),
    Input({'type': 'toggle', 'id': MATCH}, 'n_clicks'),
    State({'type': 'collapse', 'id': MATCH}, 'style'),
    prevent_initial_call=True
)
def toggle_collapse(n_clicks, current_style):
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate
    
    if current_style is None:
        current_style = {'display': 'none'}
    
    if current_style.get('display') == 'none':
        return {'display': 'block'}
    else:
        return {'display': 'none'}

# Add this callback after the toggle_collapse callback
@callback(
    Output({'type': 'icon', 'id': MATCH}, 'className'),
    Input({'type': 'toggle', 'id': MATCH}, 'n_clicks'),
    State({'type': 'collapse', 'id': MATCH}, 'style'),
    prevent_initial_call=True
)
def toggle_icon(n_clicks, current_style):
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate
    
    if current_style is None:
        current_style = {'display': 'none'}
    
    if current_style.get('display') == 'none':
        return "bi bi-chevron-up ms-2"
    else:
        return "bi bi-chevron-down ms-2" 