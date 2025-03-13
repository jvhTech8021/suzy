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
            
            # Create spread analysis components
            betting_analysis_children.extend([
                html.Div([
                    html.Span(f"Model: {pick['team1']['name']} {model_spread:+.1f}, "),
                    html.Span(f"Vegas: {pick['team1']['name']} {vegas_spread:+.1f}")
                ]),
                html.Div(f"Diff: {abs(spread_diff):.1f} pts {'higher' if spread_diff > 0 else 'lower'}"),
            ])
            
            # Recommended bet based on spread
            if abs(spread_diff) > 2:
                team_to_bet = pick['team1']['name'] if spread_diff > 0 else pick['team2']['name']
                betting_analysis_children.append(html.Div([
                    html.Span("Spread "),
                    html.Strong(f"Bet: {team_to_bet}")
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
                betting_analysis_children.append(html.Div([
                    html.Span("Total "),
                    html.Strong(f"Bet: {bet_direction}")
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
        
        # Create table row
        table_rows.append(
            html.Tr([
                html.Td(f"{pick['team1']['name']} vs {pick['team2']['name']}", className="align-middle"),
                html.Td(f"{predicted_winner} ({team1_score}-{team2_score})", className="align-middle"),
                html.Td(betting_div, className="align-middle"),
                html.Td(action_buttons, className="align-middle")
            ])
        )
    
    return dbc.Container([
        html.H1("Historical Picks", className="mt-4 mb-4"),
        stats_cards,
        betting_stats_cards,
        html.Div(id="result-update-status"),
        dbc.Table(
            # Table header
            [html.Thead(html.Tr([html.Th("Game"), html.Th("Predicted Winner"), html.Th("Betting Analysis"), html.Th("Result")]))] +
            # Table body
            [html.Tbody(table_rows)],
            bordered=True,
            hover=True,
            striped=True
        )
    ])

# Callback to mark predictions as wins or losses
@callback(
    Output("result-update-status", "children"),
    [Input({"type": "win-button", "index": dash.ALL}, "n_clicks"),
     Input({"type": "loss-button", "index": dash.ALL}, "n_clicks"),
     Input({"type": "spread-win-button", "index": dash.ALL}, "n_clicks"),
     Input({"type": "spread-loss-button", "index": dash.ALL}, "n_clicks"),
     Input({"type": "total-win-button", "index": dash.ALL}, "n_clicks"),
     Input({"type": "total-loss-button", "index": dash.ALL}, "n_clicks")],
    prevent_initial_call=True
)
def update_result(win_clicks, loss_clicks, spread_win_clicks, spread_loss_clicks, total_win_clicks, total_loss_clicks):
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