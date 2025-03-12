import json
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Load saved predictions from JSON file
def load_saved_predictions(file_path='historical_picks.json'):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []

# Layout for displaying saved predictions
def layout():
    saved_predictions = load_saved_predictions()
    return dbc.Container([
        html.H1("Saved Predictions", className="mt-4 mb-4"),
        html.Div([
            dbc.Table(
                # Table header
                [html.Thead(html.Tr([html.Th("Game"), html.Th("Predicted Winner"), html.Th("Result")]))] +
                # Table body
                [html.Tbody([
                    html.Tr([
                        html.Td(f"{pred['team1']['name']} vs {pred['team2']['name']}", className="align-middle"),
                        html.Td(pred['team1']['name'] if pred['team1']['predicted_score'] > pred['team2']['predicted_score'] else pred['team2']['name'], className="align-middle"),
                        html.Td(pred.get('result', 'Not marked'), className="align-middle")
                    ]) for pred in saved_predictions
                ])],
                bordered=True,
                hover=True,
                striped=True
            )
        ])
    ]) 