import os
import pandas as pd
import json
import dash
import subprocess
import sys
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

def get_bracket_data():
    """Load bracket data from files."""
    try:
        # Define paths
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        bracket_file = os.path.join(base_path, "models/full_bracket/model/full_bracket.txt")
        results_file = os.path.join(base_path, "models/full_bracket/model/tournament_results.csv")
        
        # Check if files exist
        if not os.path.exists(bracket_file):
            return None, None
        
        # Load bracket text
        with open(bracket_file, 'r') as f:
            bracket_text = f.read()
        
        # Load results if available
        results_df = None
        if os.path.exists(results_file):
            results_df = pd.read_csv(results_file)
        
        return bracket_text, results_df
    
    except Exception as e:
        print(f"Error loading bracket data: {e}")
        return None, None

def create_layout():
    """Create the layout for the full bracket tab."""
    bracket_text, results_df = get_bracket_data()
    
    if bracket_text is None:
        return html.Div([
            html.H2("Full Tournament Bracket", className="mt-4"),
            html.P("No bracket data available. Please run the full bracket generator first."),
            html.Button(
                "Generate Full Bracket", 
                id="generate-bracket-btn",
                className="btn btn-primary mt-3"
            ),
            html.Div(id="generate-bracket-output")
        ])
    
    # Create champion card if results are available
    champion_card = html.Div()
    if results_df is not None and not results_df.empty:
        champion = results_df[results_df['Region'] == 'Champion']
        if not champion.empty:
            champion_row = champion.iloc[0]
            champion_card = dbc.Card(
                dbc.CardBody([
                    html.H3("Tournament Champion", className="card-title text-center"),
                    html.H4(champion_row['TeamName'], className="text-center text-primary"),
                    html.P([
                        html.Strong("Seed: "), f"{champion_row['Seed']}"
                    ], className="mb-1"),
                    html.P([
                        html.Strong("Champion Profile Similarity: "), f"{champion_row['SimilarityPct']:.1f}%"
                    ], className="mb-1"),
                    html.P([
                        html.Strong("Championship Probability: "), f"{champion_row['ChampionshipPct']:.1f}%"
                    ], className="mb-1"),
                    html.P([
                        html.Strong("Combined Score: "), f"{champion_row['CombinedScore']:.1f}"
                    ], className="mb-1"),
                ]),
                className="mb-4"
            )
    
    # Create Final Four table if results are available
    final_four_table = html.Div()
    if results_df is not None and not results_df.empty:
        final_four = results_df[results_df['Region'] == 'Final Four']
        if not final_four.empty:
            final_four_table = html.Div([
                html.H4("Final Four Teams", className="mt-4"),
                dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Team"),
                        html.Th("Seed"),
                        html.Th("Champion Profile Similarity"),
                        html.Th("Championship Probability"),
                        html.Th("Combined Score")
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td(row['TeamName']),
                            html.Td(row['Seed']),
                            html.Td(f"{row['SimilarityPct']:.1f}%"),
                            html.Td(f"{row['ChampionshipPct']:.1f}%"),
                            html.Td(f"{row['CombinedScore']:.1f}")
                        ]) for _, row in final_four.iterrows()
                    ])
                ], bordered=True, hover=True, striped=True, className="mb-4")
            ])
    
    # Format bracket text for display
    bracket_display = html.Pre(
        bracket_text,
        style={
            'whiteSpace': 'pre-wrap',
            'fontFamily': 'monospace',
            'fontSize': '0.9rem',
            'backgroundColor': '#f8f9fa',
            'padding': '15px',
            'border': '1px solid #dee2e6',
            'borderRadius': '5px'
        }
    )
    
    return html.Div([
        html.H2("Full Tournament Bracket", className="mt-4"),
        html.P([
            "This tab displays a simulated NCAA tournament bracket based on the combined results from the ",
            html.Strong("Champion Profile"), " and ", html.Strong("Exit Round"), " prediction models."
        ]),
        html.Hr(),
        
        dbc.Row([
            dbc.Col(champion_card, width=12, lg=4),
            dbc.Col(final_four_table, width=12, lg=8)
        ]),
        
        html.H4("Complete Bracket", className="mt-4"),
        html.P("Below is the complete simulated NCAA tournament bracket with all rounds and matchups:"),
        bracket_display,
        
        html.Button(
            "Regenerate Bracket", 
            id="regenerate-bracket-btn",
            className="btn btn-primary mt-3"
        ),
        html.Div(id="regenerate-bracket-output"),
        
        html.Hr(),
        html.P([
            "Note: This bracket is a simulation based on the prediction models and includes some randomness. ",
            "The actual tournament results may vary significantly."
        ], className="text-muted")
    ])

# Callback to run the bracket generator
@callback(
    Output("generate-bracket-output", "children"),
    Input("generate-bracket-btn", "n_clicks"),
    prevent_initial_call=True
)
def run_bracket_generator(n_clicks):
    """Run the bracket generator when the button is clicked."""
    if n_clicks:
        try:
            # Run the full bracket generator
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            bracket_script = os.path.join(base_path, "full_bracket.py")
            
            # Run the script
            process = subprocess.Popen(
                [sys.executable, bracket_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for the process to complete
            stdout, stderr = process.communicate()
            
            # Check the return code
            if process.returncode == 0:
                return html.Div([
                    html.P("Bracket generated successfully! Refreshing page...", className="text-success"),
                    dcc.Location(id="refresh-location", pathname="/full-bracket", refresh=True)
                ])
            else:
                return html.Div([
                    html.P("Error generating bracket. See console for details.", className="text-danger"),
                    html.Pre(stderr, className="text-danger")
                ])
                
        except Exception as e:
            return html.Div([
                html.P(f"Error: {str(e)}", className="text-danger")
            ])
    
    return html.Div()

# Callback to regenerate the bracket
@callback(
    Output("regenerate-bracket-output", "children"),
    Input("regenerate-bracket-btn", "n_clicks"),
    prevent_initial_call=True
)
def regenerate_bracket(n_clicks):
    """Regenerate the bracket when the button is clicked."""
    if n_clicks:
        try:
            # Run the full bracket generator
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            bracket_script = os.path.join(base_path, "full_bracket.py")
            
            # Run the script
            process = subprocess.Popen(
                [sys.executable, bracket_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for the process to complete
            stdout, stderr = process.communicate()
            
            # Check the return code
            if process.returncode == 0:
                return html.Div([
                    html.P("Bracket regenerated successfully! Refreshing page...", className="text-success"),
                    dcc.Location(id="refresh-location", pathname="/full-bracket", refresh=True)
                ])
            else:
                return html.Div([
                    html.P("Error regenerating bracket. See console for details.", className="text-danger"),
                    html.Pre(stderr, className="text-danger")
                ])
                
        except Exception as e:
            return html.Div([
                html.P(f"Error: {str(e)}", className="text-danger")
            ])
    
    return html.Div()

# Layout function for the tab
def get_layout():
    """Return the layout for the full bracket tab."""
    return html.Div(
        id="full-bracket-tab-content",
        children=create_layout()
    ) 