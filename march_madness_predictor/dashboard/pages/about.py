import dash_bootstrap_components as dbc
from dash import html

def layout():
    """
    Create the layout for the about page
    
    Returns:
    --------
    html.Div
        Layout for the about page
    """
    content = [
        html.H1("About the March Madness Predictor", className="text-center mb-4"),
        html.P(
            "This dashboard provides predictions and analysis for the 2025 NCAA March Madness tournament using advanced statistical models.",
            className="lead text-center mb-4"
        ),
        
        dbc.Card([
            dbc.CardHeader("Project Overview"),
            dbc.CardBody([
                html.P(
                    "The March Madness Predictor is a comprehensive tool for analyzing and predicting NCAA tournament outcomes. "
                    "It combines multiple prediction models to provide insights into which teams are most likely to succeed in the tournament."
                ),
                html.P(
                    "The predictions are based on KenPom efficiency metrics, which have proven to be strong indicators of tournament success. "
                    "By analyzing historical data and applying machine learning techniques, we can identify patterns that correlate with tournament performance."
                ),
            ])
        ], className="mb-4"),
        
        dbc.Card([
            dbc.CardHeader("Models"),
            dbc.CardBody([
                html.H5("Champion Profile Model"),
                html.P(
                    "This model identifies teams that most closely resemble the statistical profile of historical NCAA champions. "
                    "It analyzes key metrics such as Adjusted Efficiency Margin, Offensive Efficiency, Defensive Efficiency, and National Ranking "
                    "to determine how similar each team is to past champions."
                ),
                html.P(
                    "Teams with higher similarity scores have historically had a greater probability of winning the championship."
                ),
                
                html.H5("Exit Round Model"),
                html.P(
                    "This deep learning model predicts how far each team will advance in the tournament based on their statistical profile and estimated seed. "
                    "It is trained on historical tournament data from 2009-2024 and learns patterns that correlate with tournament success."
                ),
                html.P(
                    "The model considers factors such as team strength metrics, national rankings, tournament seed, and schedule strength "
                    "to predict the most likely exit round for each team."
                ),
                
                html.H5("Combined Model"),
                html.P(
                    "The combined model integrates predictions from both the Champion Profile and Exit Round models to provide a more comprehensive view of tournament potential. "
                    "By averaging the championship and Final Four probabilities from both models, we can leverage the strengths of each approach and provide more robust predictions."
                ),
            ])
        ], className="mb-4"),
        
        dbc.Card([
            dbc.CardHeader("Data Sources"),
            dbc.CardBody([
                html.P(
                    "The predictions are based on KenPom efficiency metrics, which are widely regarded as the gold standard for college basketball analytics. "
                    "These metrics adjust for pace of play and strength of schedule to provide a more accurate assessment of team strength."
                ),
                html.P(
                    "The historical data used for training the models includes tournament results from 2009-2024 (excluding 2020 due to COVID-19 cancellation). "
                    "This data is combined with KenPom metrics to identify patterns that correlate with tournament success."
                ),
                html.P(
                    "Note that tournament seeds for 2025 are estimated based on team rankings, as the actual tournament seeding has not yet been determined."
                ),
            ])
        ], className="mb-4"),
        
        dbc.Card([
            dbc.CardHeader("Limitations"),
            dbc.CardBody([
                html.P(
                    "While the models provide valuable insights, it's important to recognize their limitations:"
                ),
                html.Ul([
                    html.Li(
                        "The NCAA tournament is inherently unpredictable, with upsets and unexpected outcomes being a defining characteristic."
                    ),
                    html.Li(
                        "The models are based on historical data and may not account for unique circumstances or changes in the tournament format."
                    ),
                    html.Li(
                        "Team dynamics, injuries, and other intangible factors can significantly impact tournament performance but are difficult to quantify."
                    ),
                    html.Li(
                        "The estimated seeds are based on current team rankings and may differ from the actual tournament seeding."
                    ),
                ]),
                html.P(
                    "These predictions should be viewed as a starting point for analysis rather than definitive forecasts."
                ),
            ])
        ], className="mb-4"),
        
        dbc.Card([
            dbc.CardHeader("How to Use This Dashboard"),
            dbc.CardBody([
                html.P(
                    "The dashboard is organized into several sections:"
                ),
                html.Ul([
                    html.Li([
                        html.Strong("Home: "),
                        "Provides an overview of the top championship and Final Four contenders based on the combined model."
                    ]),
                    html.Li([
                        html.Strong("Champion Profile: "),
                        "Shows teams that most closely resemble historical champions and their similarity scores."
                    ]),
                    html.Li([
                        html.Strong("Exit Round: "),
                        "Displays predictions for how far teams will advance in the tournament based on the deep learning model."
                    ]),
                    html.Li([
                        html.Strong("Combined Model: "),
                        "Integrates predictions from both models to provide a comprehensive view of tournament potential."
                    ]),
                    html.Li([
                        html.Strong("Team Explorer: "),
                        "Allows you to select a specific team and view detailed statistics and predictions."
                    ]),
                ]),
                html.P(
                    "Use the navigation bar at the top of the page to switch between these sections."
                ),
            ])
        ], className="mb-4"),
        
        dbc.Card([
            dbc.CardHeader("Contact"),
            dbc.CardBody([
                html.P(
                    "For questions, feedback, or suggestions, please contact the development team."
                ),
                html.P(
                    "This project was developed as part of an advanced analytics initiative for NCAA basketball prediction."
                ),
            ])
        ], className="mb-4"),
    ]
    
    return html.Div(content) 