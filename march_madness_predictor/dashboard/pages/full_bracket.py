import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

from dashboard.tabs import full_bracket_tab

def layout(data_loader=None):
    """Create the layout for the full bracket page."""
    return html.Div([
        html.H1("NCAA Tournament Bracket Simulation", className="mb-4"),
        html.P([
            "This page displays a complete NCAA tournament bracket simulation based on the combined results from the ",
            html.Strong("Champion Profile"), " and ", html.Strong("Exit Round"), " prediction models."
        ], className="lead"),
        
        html.Hr(),
        
        # Full bracket tab content
        full_bracket_tab.get_layout(),
        
        html.Hr(),
        
        html.Div([
            html.H4("About This Simulation", className="mt-4"),
            html.P([
                "The bracket is generated using a combination of metrics from both prediction models:",
                html.Ul([
                    html.Li("Champion Profile similarity percentages"),
                    html.Li("Exit round predictions"),
                    html.Li("Championship probabilities"),
                    html.Li("Team seeding (estimated based on rankings)")
                ])
            ]),
            html.P([
                "The simulation includes some randomness to account for the unpredictability of tournament games. ",
                "Each time you regenerate the bracket, you may get slightly different results."
            ]),
            html.P([
                "To generate a new bracket, click the 'Regenerate Bracket' button above."
            ])
        ], className="mt-4")
    ]) 