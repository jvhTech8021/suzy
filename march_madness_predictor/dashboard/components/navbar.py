import dash_bootstrap_components as dbc
from dash import html

def create_navbar():
    """
    Create a navigation bar for the dashboard
    
    Returns:
    --------
    dbc.Navbar
        Navbar component with links to different pages
    """
    navbar = dbc.Navbar(
        dbc.Container(
            [
                # Logo and Title
                dbc.NavbarBrand(
                    [
                        html.I(className="bi bi-clipboard-data me-2"),
                        "March Madness Predictor 2025"
                    ],
                    href="/",
                    className="ms-2"
                ),
                
                # Toggle button for mobile view
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                
                # Navigation links
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink("Home", href="/", active="exact")),
                            dbc.NavItem(dbc.NavLink("Champion Profile", href="/champion-profile", active="exact")),
                            dbc.NavItem(dbc.NavLink("Tournament Level Analysis", href="/tournament-level-analysis", active="exact")),
                            dbc.NavItem(dbc.NavLink("Exit Round", href="/exit-round", active="exact")),
                            dbc.NavItem(dbc.NavLink("Combined Model", href="/combined-model", active="exact")),
                            dbc.NavItem(dbc.NavLink("Full Bracket", href="/full-bracket", active="exact")),
                            dbc.NavItem(dbc.NavLink("Game Predictor", href="/game-predictor", active="exact")),
                            dbc.NavItem(dbc.NavLink("Team Explorer", href="/team-explorer", active="exact")),
                            dbc.NavItem(dbc.NavLink("About", href="/about", active="exact")),
                        ],
                        className="ms-auto",
                        navbar=True
                    ),
                    id="navbar-collapse",
                    navbar=True,
                    is_open=False,
                ),
            ]
        ),
        color="primary",
        dark=True,
        className="mb-4",
    )
    
    return navbar 