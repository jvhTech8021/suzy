import os
import sys
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import traceback

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the pages and components
from dashboard.pages import home, champion_profile, exit_round, combined_model, team_explorer, about, full_bracket, game_predictor, historical_picks, tournament_level_analysis, saved_predictions
from dashboard.components import navbar
from utils.data_loader import DataLoader

# Initialize the data loader
try:
    print("Initializing DataLoader...")
    data_loader = DataLoader()
    print("DataLoader initialized successfully")
except Exception as e:
    print(f"ERROR initializing DataLoader: {str(e)}")
    traceback.print_exc()
    # Create a minimal DataLoader
    data_loader = DataLoader()

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True
)

# Set server to app.server for Gunicorn to use
server = app.server

# Store data_loader in app.server for access in callbacks
app.server.data_loader = data_loader

app.title = "March Madness Predictor 2025"

# Define the layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar.create_navbar(),
    html.Div(id='page-content', className='container mt-4 mb-5')
])

# Callback to render the correct page based on the URL
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    try:
        if pathname == '/champion-profile':
            return champion_profile.layout(data_loader)
        elif pathname == '/exit-round':
            return exit_round.layout(data_loader)
        elif pathname == '/combined-model':
            return combined_model.layout(data_loader)
        elif pathname == '/team-explorer':
            return team_explorer.layout(data_loader)
        elif pathname == '/full-bracket':
            return full_bracket.layout(data_loader)
        elif pathname == '/game-predictor':
            return game_predictor.layout(data_loader)
        elif pathname == '/tournament-level-analysis':
            return tournament_level_analysis.layout(data_loader)
        elif pathname == '/about':
            return about.layout()
        elif pathname == '/historical_picks':
            return historical_picks.layout()
        else:
            return home.layout(data_loader)
    except Exception as e:
        print(f"ERROR in display_page for {pathname}: {str(e)}")
        traceback.print_exc()
        # Return a generic error message
        return html.Div([
            html.H1("Error Loading Page", className="text-danger"),
            html.P(f"There was an error loading the {pathname} page:"),
            html.Pre(str(e), className="bg-light p-3 border"),
            html.P("Please try running the reset_cache.py script to fix this issue:"),
            html.Pre("python3 reset_cache.py", className="bg-light p-2 border"),
            html.Hr(),
            html.A("Return to Home", href="/", className="btn btn-primary")
        ])

# Run the app
if __name__ == '__main__':
    print("Starting March Madness Predictor Dashboard...")
    # Check if required data files exist
    try:
        # Load some data to verify it's available
        try:
            current_data = data_loader.get_current_season_data()
            print(f"Loaded current season data with {len(current_data)} teams")
        except Exception as e:
            print(f"WARNING: Could not load current season data: {str(e)}")
        
        # Try loading model predictions
        try:
            champion_data = data_loader.get_champion_profile_predictions()
            print(f"Loaded champion profile predictions for {len(champion_data)} teams")
        except Exception as e:
            print(f"WARNING: Could not load champion profile predictions: {str(e)}")
        
        try:
            exit_data = data_loader.get_exit_round_predictions()
            print(f"Loaded exit round predictions for {len(exit_data)} teams")
        except Exception as e:
            print(f"WARNING: Could not load exit round predictions: {str(e)}")
        
        # Try loading full bracket data
        try:
            import os
            bracket_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                       "models/full_bracket/model/full_bracket.txt")
            if os.path.exists(bracket_file):
                print("Full bracket data is available")
            else:
                print("WARNING: Full bracket data not found. Run the full bracket generator first.")
        except Exception as e:
            print(f"WARNING: Could not check for full bracket data: {e}")
        
        # Start the app
        print("Dashboard ready! Visit http://127.0.0.1:8050/ in your browser")
        app.run_server(debug=True)
        
    except Exception as e:
        print(f"ERROR starting app: {str(e)}")
        traceback.print_exc()
        print("Please make sure the required data files are available.")
        print("Run the models first to generate the necessary prediction data.")
        # Try to start the app anyway
        try:
            app.run_server(debug=True)
        except Exception as e2:
            print(f"FATAL ERROR: Could not start app: {str(e2)}")
            sys.exit(1) 