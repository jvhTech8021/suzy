import dash_bootstrap_components as dbc
from dash import html

def tournament_adjustment_breakdown(prediction_result):
    """
    Create a component to display the detailed tournament round adjustments for both teams.
    
    Parameters:
    -----------
    prediction_result : dict
        The prediction result from GamePredictor
    
    Returns:
    --------
    html.Div
        A Dash component displaying the tournament adjustment details
    """
    if not prediction_result:
        return html.Div()
    
    team1_name = prediction_result["team1"]["name"]
    team2_name = prediction_result["team2"]["name"]
    
    # Round names for display
    round_names = {
        "championship_pct": "Championship",
        "final_four_pct": "Final Four",
        "elite_eight_pct": "Elite Eight",
        "sweet_sixteen_pct": "Sweet Sixteen",
        "round_32_pct": "Round of 32"
    }
    
    # Create breakdown for team 1
    team1_breakdown_rows = []
    if "tournament_adjustment_detail" in prediction_result["team1"]:
        team1_details = prediction_result["team1"]["tournament_adjustment_detail"]
        for round_key, round_name in round_names.items():
            if round_key in team1_details:
                detail = team1_details[round_key]
                
                # Check if the adjustment is a bonus or penalty
                adjustment_type = detail.get('type', 'bonus')  # Default to bonus for backward compatibility
                points = detail['points']
                is_bonus = adjustment_type == 'bonus' or points >= 0
                
                # Format the points display with appropriate sign and color
                points_style = {'color': 'green'} if is_bonus else {'color': 'red'}
                points_display = f"+{points:.2f}" if is_bonus else f"{points:.2f}"
                
                # Add baseline expectation column if available
                baseline = detail.get('baseline', None)
                if baseline is not None:
                    team1_breakdown_rows.append(
                        html.Tr([
                            html.Td(round_name),
                            html.Td(f"{detail['percentage']:.1f}%"),
                            html.Td(f"{baseline:.1f}%"),
                            html.Td(f"{detail['factor']:.3f}"),
                            html.Td(points_display, style=points_style)
                        ])
                    )
                else:
                    team1_breakdown_rows.append(
                        html.Tr([
                            html.Td(round_name),
                            html.Td(f"{detail['percentage']:.1f}%"),
                            html.Td("N/A"),
                            html.Td(f"{detail['factor']:.3f}"),
                            html.Td(points_display, style=points_style)
                        ])
                    )
        
        # Add total row
        if team1_breakdown_rows:
            total_adjustment = prediction_result['tournament_adjustment_team1']
            total_style = {'color': 'green'} if total_adjustment >= 0 else {'color': 'red'}
            total_display = f"+{total_adjustment:.2f}" if total_adjustment >= 0 else f"{total_adjustment:.2f}"
            
            team1_breakdown_rows.append(
                html.Tr([
                    html.Td(html.Strong("Total")),
                    html.Td(""),
                    html.Td(""),
                    html.Td(""),
                    html.Td(html.Strong(total_display), style=total_style)
                ], className="table-active")
            )
    
    # Create breakdown for team 2
    team2_breakdown_rows = []
    if "tournament_adjustment_detail" in prediction_result["team2"]:
        team2_details = prediction_result["team2"]["tournament_adjustment_detail"]
        for round_key, round_name in round_names.items():
            if round_key in team2_details:
                detail = team2_details[round_key]
                
                # Check if the adjustment is a bonus or penalty
                adjustment_type = detail.get('type', 'bonus')  # Default to bonus for backward compatibility
                points = detail['points']
                is_bonus = adjustment_type == 'bonus' or points >= 0
                
                # Format the points display with appropriate sign and color
                points_style = {'color': 'green'} if is_bonus else {'color': 'red'}
                points_display = f"+{points:.2f}" if is_bonus else f"{points:.2f}"
                
                # Add baseline expectation column if available
                baseline = detail.get('baseline', None)
                if baseline is not None:
                    team2_breakdown_rows.append(
                        html.Tr([
                            html.Td(round_name),
                            html.Td(f"{detail['percentage']:.1f}%"),
                            html.Td(f"{baseline:.1f}%"),
                            html.Td(f"{detail['factor']:.3f}"),
                            html.Td(points_display, style=points_style)
                        ])
                    )
                else:
                    team2_breakdown_rows.append(
                        html.Tr([
                            html.Td(round_name),
                            html.Td(f"{detail['percentage']:.1f}%"),
                            html.Td("N/A"),
                            html.Td(f"{detail['factor']:.3f}"),
                            html.Td(points_display, style=points_style)
                        ])
                    )
        
        # Add total row
        if team2_breakdown_rows:
            total_adjustment = prediction_result['tournament_adjustment_team2']
            total_style = {'color': 'green'} if total_adjustment >= 0 else {'color': 'red'}
            total_display = f"+{total_adjustment:.2f}" if total_adjustment >= 0 else f"{total_adjustment:.2f}"
            
            team2_breakdown_rows.append(
                html.Tr([
                    html.Td(html.Strong("Total")),
                    html.Td(""),
                    html.Td(""),
                    html.Td(""),
                    html.Td(html.Strong(total_display), style=total_style)
                ], className="table-active")
            )
    
    # Create the component
    return html.Div([
        html.H4("Tournament Adjustment Breakdown", className="mt-4"),
        html.P([
            "Points added to or subtracted from each team's score based on tournament round probabilities. ",
            html.Span("Teams above baseline expectations receive bonuses (green)", style={'color': 'green'}),
            " while ",
            html.Span("teams below baseline expectations receive penalties (red)", style={'color': 'red'}),
            "."
        ]),
        
        # Team 1 breakdown
        html.H5(f"{team1_name} Adjustments", className="mt-3"),
        dbc.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("Tournament Round"),
                        html.Th("Probability"),
                        html.Th("Baseline"),
                        html.Th("Factor"),
                        html.Th("Points")
                    ])
                ),
                html.Tbody(team1_breakdown_rows if team1_breakdown_rows else 
                          [html.Tr([html.Td("No tournament data available", colSpan=5)])])
            ],
            bordered=True,
            hover=True,
            striped=True,
            size="sm",
            className="mb-4"
        ),
        
        # Team 2 breakdown
        html.H5(f"{team2_name} Adjustments", className="mt-3"),
        dbc.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("Tournament Round"),
                        html.Th("Probability"),
                        html.Th("Baseline"),
                        html.Th("Factor"),
                        html.Th("Points")
                    ])
                ),
                html.Tbody(team2_breakdown_rows if team2_breakdown_rows else 
                          [html.Tr([html.Td("No tournament data available", colSpan=5)])])
            ],
            bordered=True,
            hover=True,
            striped=True,
            size="sm"
        ),
        
        # Net impact on spread
        html.Div([
            html.H5("Net Impact on Spread"),
            html.P([
                html.Strong(f"{abs(prediction_result['tournament_adjustment']):.2f} points "),
                f"in favor of {team1_name if prediction_result['tournament_adjustment'] > 0 else team2_name}" 
                if prediction_result['tournament_adjustment'] != 0 else "No impact on spread"
            ])
        ], className="mt-4 alert alert-info")
    ], className="mt-4") 