import json
import os
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

# Load historical picks from JSON file
def load_historical_picks(file_path='historical_picks.json'):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []

# Save updated historical picks to JSON file
def save_historical_picks(picks, file_path='historical_picks.json'):
    with open(file_path, 'w') as file:
        json.dump(picks, file, indent=4)

# Display historical picks and allow marking as win/loss
def display_historical_picks():
    picks = load_historical_picks()
    for i, pick in enumerate(picks):
        print(f"{i+1}. {pick['team1']['name']} vs {pick['team2']['name']} - Predicted Winner: {pick['team1']['name'] if pick['team1']['predicted_score'] > pick['team2']['predicted_score'] else pick['team2']['name']}")
        result = input("Enter 'w' for win, 'l' for loss, or 's' to skip: ").strip().lower()
        if result == 'w':
            pick['result'] = 'win'
        elif result == 'l':
            pick['result'] = 'loss'
    save_historical_picks(picks)

# Layout for displaying historical picks
def layout(data_loader=None):
    historical_picks = load_historical_picks()
    # Check if historical picks exist
    if not historical_picks:
        empty_message = html.Div([
            html.H4("No historical picks found", className="text-center text-muted my-4"),
            html.P("Make predictions in the matchup predictor to see them here.", className="text-center")
        ])
        return dbc.Container([
            html.H1("Historical Picks", className="mt-4 mb-4"),
            empty_message
        ])
    
    # Calculate statistics
    total_picks = len(historical_picks)
    wins = sum(1 for pick in historical_picks if pick.get('result') == 'win')
    losses = sum(1 for pick in historical_picks if pick.get('result') == 'loss')
    unmarked = total_picks - wins - losses
    
    # Calculate betting stats
    spread_bets = sum(1 for pick in historical_picks if 'spread' in pick and 'vegas_spread' in pick and 
                      pick['vegas_spread'] is not None and abs(pick['spread'] - pick['vegas_spread']) > 2)
    spread_wins = sum(1 for pick in historical_picks if pick.get('spread_bet_result') == 'win')
    spread_losses = sum(1 for pick in historical_picks if pick.get('spread_bet_result') == 'loss')
    
    total_bets = sum(1 for pick in historical_picks if 'total' in pick and 'vegas_total' in pick and 
                     pick['vegas_total'] is not None and abs(pick['total'] - pick['vegas_total']) > 3)
    total_wins = sum(1 for pick in historical_picks if pick.get('total_bet_result') == 'win')
    total_losses = sum(1 for pick in historical_picks if pick.get('total_bet_result') == 'loss')
    
    # Create statistics cards
    stats_cards = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{total_picks}", className="card-title text-center"),
                html.P("Total Picks", className="card-text text-center")
            ])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{wins}", className="card-title text-center text-success"),
                html.P("Wins", className="card-text text-center")
            ])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{losses}", className="card-title text-center text-danger"),
                html.P("Losses", className="card-text text-center")
            ])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{wins/(wins+losses)*100:.1f}%" if wins+losses > 0 else "N/A", 
                       className="card-title text-center"),
                html.P("Win Rate", className="card-text text-center")
            ])
        ]), width=3)
    ], className="mb-4")
    
    # Add betting stats cards
    betting_stats_cards = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{spread_bets}", className="card-title text-center"),
                html.P("Spread Bets", className="card-text text-center")
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{spread_wins}", className="card-title text-center text-success"),
                html.P("Spread Wins", className="card-text text-center")
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{spread_wins/(spread_wins+spread_losses)*100:.1f}%" if spread_wins+spread_losses > 0 else "N/A", 
                       className="card-title text-center"),
                html.P("Spread Win Rate", className="card-text text-center")
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{total_bets}", className="card-title text-center"),
                html.P("Total Bets", className="card-text text-center")
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{total_wins}", className="card-title text-center text-success"),
                html.P("Total Wins", className="card-text text-center")
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{total_wins/(total_wins+total_losses)*100:.1f}%" if total_wins+total_losses > 0 else "N/A", 
                       className="card-title text-center"),
                html.P("Total Win Rate", className="card-text text-center")
            ])
        ]), width=2)
    ], className="mb-4")
    
    # Create table rows with action buttons and predicted scores
    table_rows = []
    for i, pick in enumerate(historical_picks):
        team1_score = round(pick['team1']['predicted_score']) if 'predicted_score' in pick['team1'] else 'N/A'
        team2_score = round(pick['team2']['predicted_score']) if 'predicted_score' in pick['team2'] else 'N/A'
        predicted_winner = pick['team1']['name'] if pick['team1']['predicted_score'] > pick['team2']['predicted_score'] else pick['team2']['name']
        
        # Extract betting analysis information if available
        spread_buttons = None
        total_buttons = None
        betting_analysis_children = []
        
        if 'spread' in pick and 'vegas_spread' in pick and pick['vegas_spread'] is not None:
            model_spread = round(pick['spread'], 1)
            vegas_spread = pick['vegas_spread']
            spread_diff = model_spread - vegas_spread
            
            # Determine confidence level based on the absolute difference
            if abs(spread_diff) < 0.5:
                spread_confidence = "Very Low"
                spread_confidence_color = "secondary"
            elif abs(spread_diff) < 1.0:
                spread_confidence = "Low"
                spread_confidence_color = "warning"
            elif abs(spread_diff) < 1.5:
                spread_confidence = "Medium"
                spread_confidence_color = "primary"
            else:
                # Anything above 1.5 points difference is high confidence
                spread_confidence = "High"
                spread_confidence_color = "success"
            
            # Create spread analysis components
            betting_analysis_children.extend([
                html.Div([
                    html.Span(f"Model: {pick['team1']['name']} {'-' if model_spread > 0 else '+'}{abs(model_spread):.1f}, "),
                    html.Span(f"Vegas: {pick['team1']['name']} {'-' if vegas_spread > 0 else '+'}{abs(vegas_spread):.1f}")
                ]),
                html.Div(f"Diff: {abs(spread_diff):.1f} pts {'higher' if spread_diff > 0 else 'lower'}"),
            ])
            
            # Recommended bet based on spread
            if abs(spread_diff) > 2:
                team_to_bet = pick['team1']['name'] if spread_diff > 0 else pick['team2']['name']
                # Determine if the team to bet is getting or giving points
                if team_to_bet == pick['team1']['name']:
                    # Team1 is the reference team in the display
                    bet_spread = f"{'+' if vegas_spread < 0 else '-'}{abs(vegas_spread)}"
                else:
                    # Team2 is getting points if team1 is favored
                    bet_spread = f"{'+' if vegas_spread > 0 else '-'}{abs(vegas_spread)}"
                
                betting_analysis_children.append(html.Div([
                    html.Span("Spread "),
                    html.Strong(f"Bet: {team_to_bet} {bet_spread}"),
                    dbc.Badge(spread_confidence, color=spread_confidence_color, className="ml-2")
                ]))
                
                # Add buttons for tracking spread bet outcomes
                spread_result = pick.get('spread_bet_result', 'Not marked')
                if spread_result == 'Not marked' or not spread_result:
                    spread_buttons = html.Div([
                        dbc.Button("Win", id={"type": "spread-win-button", "index": i}, color="success", size="sm", className="me-1 mt-1", style={"fontSize": "0.7rem"}),
                        dbc.Button("Loss", id={"type": "spread-loss-button", "index": i}, color="danger", size="sm", className="mt-1", style={"fontSize": "0.7rem"})
                    ])
                else:
                    spread_buttons = html.Div(f"Spread bet: {spread_result}", className=f"text-{'success' if spread_result == 'win' else 'danger'}")
            else:
                betting_analysis_children.append(html.Div("Spread No edge"))
        
        # Add total analysis if available
        if 'total' in pick and 'vegas_total' in pick and pick['vegas_total'] is not None:
            model_total = round(pick['total'], 1)
            vegas_total = pick['vegas_total']
            total_diff = model_total - vegas_total
            
            # Add separator if we already have spread analysis
            if betting_analysis_children:
                betting_analysis_children.append(html.Hr(style={"margin": "8px 0"}))
            
            # Create total analysis components
            betting_analysis_children.extend([
                html.Div(f"Model Total: {model_total}, Vegas: {vegas_total}"),
                html.Div(f"Diff: {abs(total_diff):.1f} pts {'higher' if total_diff > 0 else 'lower'}"),
            ])
            
            # Recommended bet based on total
            if abs(total_diff) > 3:
                bet_direction = "Over" if total_diff > 0 else "Under"
                
                # Determine confidence level for total
                total_confidence = "Very Low"  # Default for no edge
                total_confidence_color = "secondary"  # Default for no edge
                if abs(total_diff) < 2:
                    total_confidence = "Very Low"
                    total_confidence_color = "secondary"
                elif abs(total_diff) < 4:
                    total_confidence = "Low"
                    total_confidence_color = "warning"
                elif abs(total_diff) < 8:
                    total_confidence = "Medium"
                    total_confidence_color = "primary"
                else:
                    # Anything above 8 points difference is high confidence
                    total_confidence = "High"
                    total_confidence_color = "success"
                
                betting_analysis_children.append(html.Div([
                    html.Span("Total "),
                    html.Strong(f"Bet: {bet_direction}"),
                    dbc.Badge(total_confidence, color=total_confidence_color, className="ml-2")
                ]))
                
                # Add buttons for tracking total bet outcomes
                total_result = pick.get('total_bet_result', 'Not marked')
                if total_result == 'Not marked' or not total_result:
                    total_buttons = html.Div([
                        dbc.Button("Win", id={"type": "total-win-button", "index": i}, color="success", size="sm", className="me-1 mt-1", style={"fontSize": "0.7rem"}),
                        dbc.Button("Loss", id={"type": "total-loss-button", "index": i}, color="danger", size="sm", className="mt-1", style={"fontSize": "0.7rem"})
                    ])
                else:
                    total_buttons = html.Div(f"Total bet: {total_result}", className=f"text-{'success' if total_result == 'win' else 'danger'}")
            else:
                betting_analysis_children.append(html.Div("Total No edge"))
                
        # Create a div for betting analysis with buttons
        if betting_analysis_children:
            # Create a betting recommendation summary card similar to game_predictor.py
            if 'spread' in pick and 'vegas_spread' in pick and pick['vegas_spread'] is not None:
                model_spread = round(pick['spread'], 1)
                vegas_spread = pick['vegas_spread']
                spread_diff = model_spread - vegas_spread
                
                # Determine which team is favored
                model_favored_team = pick['team1']['name'] if pick['team1']['predicted_score'] > pick['team2']['predicted_score'] else pick['team2']['name']
                vegas_favored_team = pick['team1']['name'] if vegas_spread > 0 else pick['team2']['name']
                
                # Determine team to bet
                if abs(spread_diff) > 1.5:
                    if model_favored_team == vegas_favored_team:
                        # If same team favored, bet on favorite if model spread is higher, underdog if model spread is lower
                        if spread_diff > 0:
                            team_to_bet = pick['team1']['name']
                            bet_spread = f"-{abs(vegas_spread)}"
                        else:
                            team_to_bet = pick['team2']['name']
                            bet_spread = f"+{abs(vegas_spread)}"
                    else:
                        # If different teams favored, bet on model's favorite
                        team_to_bet = model_favored_team
                        bet_spread = f"+{abs(vegas_spread)}" if team_to_bet == pick['team2']['name'] else f"-{abs(vegas_spread)}"
                    
                    bet_text = f"Take {team_to_bet} {bet_spread}"
                    has_spread_edge = True
                else:
                    bet_text = "No Strong Edge"
                    has_spread_edge = False
                
                # Total bet
                has_total_edge = False
                total_bet_text = "No Strong Edge"
                if 'total' in pick and 'vegas_total' in pick and pick['vegas_total'] is not None:
                    model_total = round(pick['total'], 1)
                    vegas_total = pick['vegas_total']
                    total_diff = model_total - vegas_total
                    
                    # Set confidence level for total
                    if abs(total_diff) < 2:
                        total_confidence = "Very Low"
                        total_confidence_color = "secondary"
                    elif abs(total_diff) < 4:
                        total_confidence = "Low"
                        total_confidence_color = "warning"
                    elif abs(total_diff) < 8:
                        total_confidence = "Medium"
                        total_confidence_color = "primary"
                    else:
                        # Anything above 8 points difference is high confidence
                        total_confidence = "High"
                        total_confidence_color = "success"
                    
                    if abs(total_diff) > 3:
                        bet_direction = "Over" if total_diff > 0 else "Under"
                        total_bet_text = f"Take {bet_direction} {vegas_total}"
                        has_total_edge = True
                
                # Create the betting recommendation card
                betting_div = dbc.Card([
                    dbc.CardHeader(html.H5("Betting Recommendation Summary", className="mb-0 text-center")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("Spread Bet", className="text-center"),
                                html.Div([
                                    html.H5([
                                        # Only show "No Strong Edge" when the difference is small
                                        not has_spread_edge
                                        and html.Span("No Strong Edge", className="text-muted")
                                        # Otherwise show a recommendation
                                        or dbc.Badge(
                                            bet_text,
                                            color=spread_confidence_color,
                                            className="p-2",
                                            style={"font-size": "0.9rem"}
                                        )
                                    ], className="text-center mt-2"),
                                    html.Div([
                                        html.Strong("Confidence: "),
                                        html.Span(spread_confidence)
                                    ], className="text-center mt-2 small"),
                                    html.Div([
                                        html.Strong("Difference: "),
                                        html.Span(f"{abs(spread_diff):.1f} points")
                                    ], className="text-center small")
                                ])
                            ], width=6, className="border-right"),
                            
                            dbc.Col([
                                html.H6("Total Bet", className="text-center"),
                                html.Div([
                                    html.H5([
                                        # Only show "No Strong Edge" when no edge
                                        not has_total_edge
                                        and html.Span("No Strong Edge", className="text-muted")
                                        # Otherwise show a recommendation
                                        or dbc.Badge(
                                            total_bet_text,
                                            color=total_confidence_color,
                                            className="p-2",
                                            style={"font-size": "0.9rem"}
                                        )
                                    ], className="text-center mt-2"),
                                    html.Div([
                                        html.Strong("Confidence: "),
                                        html.Span(total_confidence)
                                    ], className="text-center mt-2 small"),
                                    html.Div([
                                        html.Strong("Difference: "),
                                        html.Span(f"{abs(total_diff):.1f} points" if 'total_diff' in locals() else "0.0 points")
                                    ], className="text-center small")
                                ])
                            ], width=6)
                        ]),
                    ])
                ], className="mb-2")
                
                # Add tracking buttons below the summary card
                if spread_buttons or total_buttons:
                    betting_div = html.Div([
                        betting_div,
                        html.Div([
                            spread_buttons,
                            html.Div(className="mt-1") if spread_buttons and total_buttons else None,
                            total_buttons
                        ], className="mt-2")
                    ])
            else:
                # Fallback to original display if no spread/vegas data
                betting_div = html.Div([
                    html.Div(betting_analysis_children),
                    spread_buttons,
                    html.Div(className="mt-1") if spread_buttons and total_buttons else None,
                    total_buttons
                ])
        else:
            betting_div = html.Div("No betting data")
        
        # Create buttons for marking as win/loss
        action_buttons = html.Div([
            dbc.Button("Win", id={"type": "win-button", "index": i}, color="success", size="sm", className="me-2"),
            dbc.Button("Loss", id={"type": "loss-button", "index": i}, color="danger", size="sm")
        ]) if pick.get('result') == 'Not marked' or not pick.get('result') else html.Div(pick.get('result', 'Not marked'), className=f"text-{'success' if pick.get('result') == 'win' else 'danger'}")
        
        # Create management buttons (edit and delete)
        management_buttons = html.Div([
            dbc.Button("Edit", id={"type": "edit-button", "index": i}, color="primary", size="sm", className="me-2"),
            dbc.Button("Delete", id={"type": "delete-button", "index": i}, color="danger", size="sm")
        ])
        
        # Create table row
        table_rows.append(
            html.Tr([
                html.Td(f"{pick['team1']['name']} vs {pick['team2']['name']}", className="align-middle"),
                html.Td(f"{predicted_winner} ({team1_score}-{team2_score})", className="align-middle"),
                html.Td(betting_div, className="align-middle"),
                html.Td(action_buttons, className="align-middle"),
                html.Td(management_buttons, className="align-middle")
            ])
        )
    
    return dbc.Container([
        html.H1("Historical Picks", className="mt-4 mb-4"),
        stats_cards,
        betting_stats_cards,
        html.Div(id="result-update-status"),
        # Add modal for editing picks
        dbc.Modal([
            dbc.ModalHeader("Edit Pick"),
            dbc.ModalBody([
                dbc.Label("Vegas Spread:"),
                dbc.Input(id="edit-vegas-spread", type="number", step=0.5, placeholder="Enter new Vegas spread"),
                dbc.FormText("Enter a positive number. The spread is applied to the first team listed."),
                html.Div(id="edit-pick-info", className="mt-3")
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="edit-cancel", className="me-2"),
                dbc.Button("Save Changes", id="edit-save", color="primary")
            ])
        ], id="edit-modal"),
        dbc.Table(
            # Table header
            [html.Thead(html.Tr([html.Th("Game"), html.Th("Predicted Winner"), html.Th("Betting Analysis"), html.Th("Result"), html.Th("Actions")]))] +
            # Table body
            [html.Tbody(table_rows)],
            bordered=True,
            hover=True,
            striped=True
        ),
        # Store the index of the pick being edited
        dcc.Store(id="edited-pick-index")
    ])

# Callback to mark predictions as wins or losses
@callback(
    Output("result-update-status", "children"),
    [Input({"type": "win-button", "index": dash.ALL}, "n_clicks"),
     Input({"type": "loss-button", "index": dash.ALL}, "n_clicks"),
     Input({"type": "spread-win-button", "index": dash.ALL}, "n_clicks"),
     Input({"type": "spread-loss-button", "index": dash.ALL}, "n_clicks"),
     Input({"type": "total-win-button", "index": dash.ALL}, "n_clicks"),
     Input({"type": "total-loss-button", "index": dash.ALL}, "n_clicks"),
     Input({"type": "delete-button", "index": dash.ALL}, "n_clicks")],
    prevent_initial_call=True
)
def update_result(win_clicks, loss_clicks, spread_win_clicks, spread_loss_clicks, total_win_clicks, total_loss_clicks, delete_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    
    # Get the triggered button's id
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # Parse the JSON id
    try:
        button_dict = json.loads(button_id)
        index = button_dict["index"]
        button_type = button_dict["type"]
        
        # Determine what kind of result we're updating
        if button_type == "win-button" or button_type == "loss-button":
            result_type = "result"
            result = "win" if button_type == "win-button" else "loss"
            display_text = f"Prediction #{index+1} marked as {result}!"
            
        elif button_type == "spread-win-button" or button_type == "spread-loss-button":
            result_type = "spread_bet_result"
            result = "win" if button_type == "spread-win-button" else "loss"
            display_text = f"Spread bet for prediction #{index+1} marked as {result}!"
            
        elif button_type == "total-win-button" or button_type == "total-loss-button":
            result_type = "total_bet_result"
            result = "win" if button_type == "total-win-button" else "loss"
            display_text = f"Total bet for prediction #{index+1} marked as {result}!"
            
        elif button_type == "delete-button":
            # Handle deletion
            picks = load_historical_picks()
            if index < len(picks):
                deleted_pick = picks.pop(index)  # Remove the pick at the specified index
                save_historical_picks(picks)
                return html.P(f"Deleted prediction: {deleted_pick['team1']['name']} vs {deleted_pick['team2']['name']}", 
                             className="text-danger alert alert-danger")
            return html.P("Error deleting prediction", className="text-danger")
            
        else:
            return html.P("Unknown button type", className="text-danger")
            
    except:
        return html.P("Error identifying button", className="text-danger")
    
    # Load picks, update the result, and save
    picks = load_historical_picks()
    if index < len(picks):
        picks[index][result_type] = result
        save_historical_picks(picks)
        return html.P(display_text, className="text-success")
    
    return html.P("Error updating prediction result", className="text-danger")

# Callbacks for the edit modal
@callback(
    [Output("edit-modal", "is_open"),
     Output("edited-pick-index", "data"),
     Output("edit-vegas-spread", "value"),
     Output("edit-pick-info", "children")],
    [Input({"type": "edit-button", "index": dash.ALL}, "n_clicks"),
     Input("edit-cancel", "n_clicks"),
     Input("edit-save", "n_clicks")],
    [State("edit-modal", "is_open"),
     State("edited-pick-index", "data"),
     State("edit-vegas-spread", "value")],
    prevent_initial_call=True
)
def toggle_edit_modal(edit_clicks, cancel_clicks, save_clicks, is_open, current_index, new_vegas_spread):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open, current_index, None, ""
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if "edit-button" in trigger_id:
        # Parse the button ID to get the index
        button_dict = json.loads(trigger_id)
        index = button_dict["index"]
        
        # Load the current pick data
        picks = load_historical_picks()
        if index < len(picks):
            pick = picks[index]
            current_vegas_spread = pick.get('vegas_spread')
            team1_name = pick['team1']['name']
            team2_name = pick['team2']['name']
            info_text = f"Editing: {team1_name} vs {team2_name}"
            return True, index, current_vegas_spread, info_text
    
    elif trigger_id == "edit-save" and current_index is not None:
        # Save the changes
        picks = load_historical_picks()
        if current_index < len(picks) and new_vegas_spread is not None:
            picks[current_index]['vegas_spread'] = new_vegas_spread
            save_historical_picks(picks)
        return False, None, None, ""
    
    elif trigger_id == "edit-cancel":
        return False, None, None, ""
    
    return is_open, current_index, None, "" 