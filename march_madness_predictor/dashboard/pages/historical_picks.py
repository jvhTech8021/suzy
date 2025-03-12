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
    
    # Create table rows with action buttons and predicted scores
    table_rows = []
    for i, pick in enumerate(historical_picks):
        team1_score = round(pick['team1']['predicted_score']) if 'predicted_score' in pick['team1'] else 'N/A'
        team2_score = round(pick['team2']['predicted_score']) if 'predicted_score' in pick['team2'] else 'N/A'
        predicted_winner = pick['team1']['name'] if pick['team1']['predicted_score'] > pick['team2']['predicted_score'] else pick['team2']['name']
        
        # Create buttons for marking as win/loss
        action_buttons = html.Div([
            dbc.Button("Win", id={"type": "win-button", "index": i}, color="success", size="sm", className="me-2"),
            dbc.Button("Loss", id={"type": "loss-button", "index": i}, color="danger", size="sm")
        ]) if pick.get('result') == 'Not marked' or not pick.get('result') else pick.get('result', 'Not marked')
        
        # Create table row
        table_rows.append(
            html.Tr([
                html.Td(f"{pick['team1']['name']} vs {pick['team2']['name']}", className="align-middle"),
                html.Td(f"{predicted_winner} ({team1_score}-{team2_score})", className="align-middle"),
                html.Td(action_buttons, className="align-middle")
            ])
        )
    
    return dbc.Container([
        html.H1("Historical Picks", className="mt-4 mb-4"),
        stats_cards,
        html.Div(id="result-update-status"),
        dbc.Table(
            # Table header
            [html.Thead(html.Tr([html.Th("Game"), html.Th("Predicted Winner"), html.Th("Result")]))] +
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
     Input({"type": "loss-button", "index": dash.ALL}, "n_clicks")],
    prevent_initial_call=True
)
def update_result(win_clicks, loss_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    
    # Get the triggered button's id
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # Parse the JSON id
    try:
        button_dict = json.loads(button_id)
        index = button_dict["index"]
        result = "win" if button_dict["type"] == "win-button" else "loss"
    except:
        return html.P("Error identifying button", className="text-danger")
    
    # Load picks, update the result, and save
    picks = load_historical_picks()
    if index < len(picks):
        picks[index]['result'] = result
        save_historical_picks(picks)
        return html.P(f"Prediction #{index+1} marked as {result}!", className="text-success")
    
    return html.P("Error updating prediction result", className="text-danger") 