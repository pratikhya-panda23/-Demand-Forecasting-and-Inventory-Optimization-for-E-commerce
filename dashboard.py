import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import json
from datetime import datetime, timedelta

# Add parent directory to path to import from src/models and src/simulation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models and simulation
try:
    from models.demand import DemandEstimator
    from models.pricing import DynamicPricingModel
    from simulation.simulation import PricingSimulation
except ImportError as e:
    print(f"Error importing modules: {e}")

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Needed for deployment

# Set title
app.title = "Dynamic Pricing Dashboard for Local Events"

# Initialize models
demand_estimator = DemandEstimator()
pricing_model = DynamicPricingModel()
simulation = PricingSimulation()

# Try to load models
try:
    demand_estimator.load_model()
    pricing_model.load_model()
except Exception as e:
    print(f"Warning: Could not load models: {e}")
    print("Dashboard will use default models or sample data.")

# Load sample data or simulation results if available
try:
    # Try to load simulation results
    results_path = os.path.join('../../data/processed', 'pricing_simulation_results.csv')
    if os.path.exists(results_path):
        simulation_results = pd.read_csv(results_path)
    else:
        # Create sample data if no results available
        sample_data = {
            'title': ['Concert in the Park', 'Tech Workshop', 'Food Festival', 'Art Exhibition'],
            'price_value': [25.0, 15.0, 10.0, 5.0],
            'event_date': ['2023-07-15', '2023-07-20', '2023-07-25', '2023-07-30'],
            'attendance': [150, 75, 300, 100],
            'days_until_event': [30, 35, 40, 45],
            'is_weekend': [1, 0, 1, 0],
            'strategy': ['Fixed Price'] * 4,
            'optimized_price': [25.0, 15.0, 10.0, 5.0],
            'expected_demand': [150, 75, 300, 100],
            'revenue': [3750, 1125, 3000, 500],
            'profit': [2250, 675, 1800, 300]
        }
        simulation_results = pd.DataFrame(sample_data)
        
        # Generate more data with different strategies
        strategies = ['Revenue Maximizing', 'Profit Maximizing', 'Early Bird Discount', 'Demand Based']
        for strategy in strategies:
            strategy_data = sample_data.copy()
            strategy_data['strategy'] = [strategy] * 4
            
            # Adjust prices and demand based on strategy
            if strategy == 'Revenue Maximizing':
                strategy_data['optimized_price'] = [30.0, 18.0, 12.0, 6.0]
                strategy_data['expected_demand'] = [130, 65, 280, 90]
            elif strategy == 'Profit Maximizing':
                strategy_data['optimized_price'] = [35.0, 20.0, 15.0, 7.0]
                strategy_data['expected_demand'] = [120, 60, 250, 85]
            elif strategy == 'Early Bird Discount':
                strategy_data['optimized_price'] = [20.0, 12.0, 8.0, 4.0]
                strategy_data['expected_demand'] = [170, 85, 330, 110]
            elif strategy == 'Demand Based':
                strategy_data['optimized_price'] = [28.0, 13.0, 12.0, 4.5]
                strategy_data['expected_demand'] = [140, 80, 270, 105]
            
            # Calculate revenue and profit
            strategy_data['revenue'] = [p * d for p, d in zip(strategy_data['optimized_price'], strategy_data['expected_demand'])]
            strategy_data['profit'] = [r * 0.6 for r in strategy_data['revenue']]  # Assume 60% profit margin
            
            # Append to simulation results
            simulation_results = pd.concat([simulation_results, pd.DataFrame(strategy_data)])
    
    # Try to load strategy summary
    summary_path = os.path.join('../../data/processed', 'pricing_strategy_summary.csv')
    if os.path.exists(summary_path):
        strategy_summary = pd.read_csv(summary_path)
    else:
        # Create summary from simulation results
        strategy_summary = simulation_results.groupby('strategy').agg({
            'revenue': ['mean', 'sum'],
            'profit': ['mean', 'sum'],
            'expected_demand': ['mean', 'sum']
        })
        strategy_summary.columns = ['_'.join(col).strip() for col in strategy_summary.columns.values]
        strategy_summary = strategy_summary.reset_index()
    
    # Try to load analysis results
    analysis_path = os.path.join('../../data/processed', 'pricing_analysis.json')
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            analysis_results = json.load(f)
    else:
        # Create simple analysis
        best_revenue_strategy = strategy_summary.loc[strategy_summary['revenue_sum'].idxmax()]['strategy']
        best_profit_strategy = strategy_summary.loc[strategy_summary['profit_sum'].idxmax()]['strategy']
        
        analysis_results = {
            'best_revenue_strategy': best_revenue_strategy,
            'best_profit_strategy': best_profit_strategy
        }

except Exception as e:
    print(f"Error loading data: {e}")
    print("Using minimal sample data.")
    
    # Create minimal sample data
    simulation_results = pd.DataFrame({
        'title': ['Sample Event'],
        'strategy': ['Fixed Price'],
        'optimized_price': [25.0],
        'expected_demand': [100],
        'revenue': [2500],
        'profit': [1500]
    })
    
    strategy_summary = pd.DataFrame({
        'strategy': ['Fixed Price'],
        'revenue_sum': [2500],
        'profit_sum': [1500]
    })
    
    analysis_results = {
        'best_revenue_strategy': 'Fixed Price',
        'best_profit_strategy': 'Fixed Price'
    }

# Define the layout of the dashboard
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Dynamic Pricing Dashboard for Local Events", className="text-center my-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Pricing Strategy Performance", className="card-title")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Best Revenue Strategy", className="card-title"),
                                    html.H3(id="best-revenue-strategy", className="card-text text-primary")
                                ])
                            ], className="mb-4")
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Best Profit Strategy", className="card-title"),
                                    html.H3(id="best-profit-strategy", className="card-text text-success")
                                ])
                            ], className="mb-4")
                        ], width=6)
                    ]),
                    dcc.Graph(id="strategy-comparison-chart")
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Price Optimization Tool", className="card-title")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Event Title"),
                            dbc.Input(id="event-title", type="text", placeholder="Enter event title", value="New Event")
                        ], width=6),
                        dbc.Col([
                            html.Label("Base Price ($)"),
                            dbc.Input(id="base-price", type="number", placeholder="Enter base price", value=25)
                        ], width=6)
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Expected Attendance"),
                            dbc.Input(id="expected-attendance", type="number", placeholder="Enter expected attendance", value=100)
                        ], width=6),
                        dbc.Col([
                            html.Label("Days Until Event"),
                            dbc.Input(id="days-until-event", type="number", placeholder="Enter days until event", value=30)
                        ], width=6)
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Is Weekend?"),
                            dbc.Checklist(
                                options=[{"label": "Yes", "value": 1}],
                                value=[1],
                                id="is-weekend",
                                switch=True
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Price Elasticity"),
                            dcc.Slider(
                                id="price-elasticity",
                                min=-2.0,
                                max=0,
                                step=0.1,
                                value=-1.2,
                                marks={-2.0: "Elastic (-2.0)", -1.0: "Unit (-1.0)", 0: "Inelastic (0)"},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=6)
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Optimization Objective"),
                            dbc.RadioItems(
                                options=[
                                    {"label": "Maximize Revenue", "value": "revenue"},
                                    {"label": "Maximize Profit", "value": "profit"}
                                ],
                                value="revenue",
                                id="optimization-objective",
                                inline=True
                            )
                        ], width=12)
                    ], className="mb-3"),
                    
                    dbc.Button("Calculate Optimal Price", id="calculate-button", color="primary", className="mt-3")
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Price Optimization Results", className="card-title")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Optimal Price", className="card-title"),
                                    html.H3(id="optimal-price", className="card-text text-primary")
                                ])
                            ])
                        ], width=4),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Expected Demand", className="card-title"),
                                    html.H3(id="expected-demand", className="card-text text-info")
                                ])
                            ])
                        ], width=4),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Projected Revenue", className="card-title"),
                                    html.H3(id="projected-revenue", className="card-text text-success")
                                ])
                            ])
                        ], width=4)
                    ], className="mb-4"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.H5("Price Tiers", className="mb-3"),
                            html.Div(id="price-tiers-table")
                        ], width=6),
                        dbc.Col([
                            dcc.Graph(id="price-demand-curve")
                        ], width=6)
                    ])
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Event Data Explorer", className="card-title")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Filter by Strategy"),
                            dcc.Dropdown(
                                id="strategy-filter",
                                options=[{"label": strategy, "value": strategy} for strategy in simulation_results['strategy'].unique()],
                                value=simulation_results['strategy'].unique().tolist(),
                                multi=True
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Sort By"),
                            dcc.Dropdown(
                                id="sort-by",
                                options=[
                                    {"label": "Revenue (High to Low)", "value": "revenue_desc"},
                                    {"label": "Revenue (Low to High)", "value": "revenue_asc"},
                                    {"label": "Profit (High to Low)", "value": "profit_desc"},
                                    {"label": "Profit (Low to High)", "value": "profit_asc"},
                                    {"label": "Price (High to Low)", "value": "price_desc"},
                                    {"label": "Price (Low to High)", "value": "price_asc"}
                                ],
                                value="revenue_desc"
                            )
                        ], width=6)
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id="events-scatter-plot")
                        ], width=12)
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.H5("Event Data Table", className="mt-4 mb-3"),
                            html.Div(id="events-data-table")
                        ], width=12)
                    ])
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Footer([
                html.P("Dynamic Pricing Model for Local Events & Experiences", className="text-center"),
                html.P("© 2023 - Data Science Project", className="text-center")
            ], className="mt-4 mb-4")
        ], width=12)
    ])
], fluid=True)

# Define callbacks
@app.callback(
    [Output("best-revenue-strategy", "children"),
     Output("best-profit-strategy", "children"),
     Output("strategy-comparison-chart", "figure")],
    [Input("strategy-filter", "value")]
)
def update_strategy_metrics(selected_strategies):
    # Filter strategy summary based on selected strategies
    filtered_summary = strategy_summary[strategy_summary['strategy'].isin(selected_strategies)]
    
    # Get best strategies
    best_revenue = analysis_results.get('best_revenue_strategy', 'N/A')
    best_profit = analysis_results.get('best_profit_strategy', 'N/A')
    
    # Create comparison chart
    fig = go.Figure()
    
    # Add revenue bars
    fig.add_trace(go.Bar(
        x=filtered_summary['strategy'],
        y=filtered_summary['revenue_sum'] if 'revenue_sum' in filtered_summary.columns else filtered_summary['revenue_mean'],
        name='Revenue',
        marker_color='royalblue'
    ))
    
    # Add profit bars
    fig.add_trace(go.Bar(
        x=filtered_summary['strategy'],
        y=filtered_summary['profit_sum'] if 'profit_sum' in filtered_summary.columns else filtered_summary['profit_mean'],
        name='Profit',
        marker_color='green'
    ))
    
    # Update layout
    fig.update_layout(
        title='Revenue and Profit by Pricing Strategy',
        xaxis_title='Strategy',
        yaxis_title='Amount ($)',
        barmode='group',
        legend=dict(x=0, y=1.0),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return best_revenue, best_profit, fig

@app.callback(
    [Output("optimal-price", "children"),
     Output("expected-demand", "children"),
     Output("projected-revenue", "children"),
     Output("price-tiers-table", "children"),
     Output("price-demand-curve", "figure")],
    [Input("calculate-button", "n_clicks")],
    [State("event-title", "value"),
     State("base-price", "value"),
     State("expected-attendance", "value"),
     State("days-until-event", "value"),
     State("is-weekend", "value"),
     State("price-elasticity", "value"),
     State("optimization-objective", "value")]
)
def calculate_optimal_price(n_clicks, title, base_price, attendance, days_until, is_weekend, elasticity, objective):
    if n_clicks is None:
        # Initial load - use default values
        optimal_price = "$25.00"
        expected_demand = "100"
        projected_revenue = "$2,500.00"
        
        # Default price tiers
        tiers_table = dbc.Table([
            html.Thead(html.Tr([html.Th("Tier"), html.Th("Price"), html.Th("Features")])),
            html.Tbody([
                html.Tr([html.Td("Basic"), html.Td("$20.00"), html.Td("Standard admission")]),
                html.Tr([html.Td("Standard"), html.Td("$25.00"), html.Td("Standard admission, Premium seating")]),
                html.Tr([html.Td("Premium"), html.Td("$32.50"), html.Td("Standard admission, Premium seating, VIP access")])
            ])
        ], bordered=True, hover=True, striped=True, className="mb-0")
        
        # Default price-demand curve
        fig = go.Figure()
        prices = np.linspace(10, 40, 100)
        demands = 100 * (prices / 25) ** -1.2
        revenues = prices * demands
        
        fig.add_trace(go.Scatter(x=prices, y=demands, mode='lines', name='Demand', line=dict(color='royalblue')))
        fig.add_trace(go.Scatter(x=prices, y=revenues/100, mode='lines', name='Revenue (÷100)', line=dict(color='green')))
        
        fig.update_layout(
            title='Price-Demand Curve',
            xaxis_title='Price ($)',
            yaxis_title='Demand / Revenue (÷100)',
            legend=dict(x=0, y=1.0),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return optimal_price, expected_demand, projected_revenue, tiers_table, fig
    
    # Convert inputs to appropriate types
    base_price = float(base_price) if base_price else 25.0
    attendance = int(attendance) if attendance else 100
    days_until = int(days_until) if days_until else 30
    is_weekend_val = 1 if is_weekend and 1 in is_weekend else 0
    
    # Set price elasticity in the pricing model
    pricing_model.price_elasticity = elasticity
    
    # Create event data
    event_data = {
        'title': title,
        'price_value': base_price,
        'attendance': attendance,
        'days_until_event': days_until,
        'is_weekend': is_weekend_val,
        'event_date': (datetime.now() + timedelta(days=days_until)).strftime('%Y-%m-%d')
    }
    
    # Convert to DataFrame for the model
    event_df = pd.DataFrame([event_data])
    
    try:
        # Calculate optimal price
        optimal_price_val = pricing_model.optimize_price(event_df, objective=objective)
        
        if optimal_price_val is None:
            optimal_price_val = base_price
        
        # Calculate expected demand at optimal price
        price_ratio = optimal_price_val / base_price
        expected_demand_val = attendance * (price_ratio ** elasticity)
        
        # Calculate projected revenue
        projected_revenue_val = optimal_price_val * expected_demand_val
        
        # Generate price tiers
        tiers = pricing_model.generate_price_tiers(event_df, num_tiers=3)
        
        # If tiers generation failed, create default tiers
        if not tiers:
            tiers = [
                {'name': 'Basic', 'price': optimal_price_val * 0.8, 'features': ['Standard admission']},
                {'name': 'Standard', 'price': optimal_price_val, 'features': ['Standard admission', 'Premium seating']},
                {'name': 'Premium', 'price': optimal_price_val * 1.3, 'features': ['Standard admission', 'Premium seating', 'VIP access']}
            ]
        
        # Create price tiers table
        tiers_table = dbc.Table([
            html.Thead(html.Tr([html.Th("Tier"), html.Th("Price"), html.Th("Features")])),
            html.Tbody([
                html.Tr([html.Td(tier['name']), html.Td(f"${tier['price']:.2f}"), html.Td(", ".join(tier['features']))]) for tier in tiers
            ])
        ], bordered=True, hover=True, striped=True, className="mb-0")
        
        # Create price-demand curve
        fig = go.Figure()
        
        # Define price range around optimal price
        min_price = max(1, optimal_price_val * 0.5)
        max_price = optimal_price_val * 2
        prices = np.linspace(min_price, max_price, 100)
        
        # Calculate demand and revenue at each price point
        demands = []
        revenues = []
        
        for price in prices:
            price_ratio = price / base_price
            demand = attendance * (price_ratio ** elasticity)
            revenue = price * demand
            
            demands.append(demand)
            revenues.append(revenue)
        
        # Add demand curve
        fig.add_trace(go.Scatter(x=prices, y=demands, mode='lines', name='Demand', line=dict(color='royalblue')))
        
        # Add revenue curve (scaled down to fit on same axis)
        revenue_scale = max(demands) / max(revenues) if max(revenues) > 0 else 0.01
        scaled_revenues = [r * revenue_scale for r in revenues]
        
        fig.add_trace(go.Scatter(
            x=prices, 
            y=scaled_revenues, 
            mode='lines', 
            name=f'Revenue (×{1/revenue_scale:.0f})', 
            line=dict(color='green')
        ))
        
        # Mark optimal price
        fig.add_vline(x=optimal_price_val, line_dash="dash", line_color="red")
        fig.add_annotation(
            x=optimal_price_val,
            y=max(demands) * 0.9,
            text=f"Optimal: ${optimal_price_val:.2f}",
            showarrow=True,
            arrowhead=1
        )
        
        # Update layout
        fig.update_layout(
            title='Price-Demand Curve',
            xaxis_title='Price ($)',
            yaxis_title='Demand / Scaled Revenue',
            legend=dict(x=0, y=1.0),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        # Format output values
        optimal_price_str = f"${optimal_price_val:.2f}"
        expected_demand_str = f"{int(expected_demand_val)}"
        projected_revenue_str = f"${projected_revenue_val:,.2f}"
        
    except Exception as e:
        print(f"Error calculating optimal price: {e}")
        
        # Use default values on error
        optimal_price_str = f"${base_price:.2f}"
        expected_demand_str = f"{attendance}"
        projected_revenue_str = f"${base_price * attendance:,.2f}"
        
        # Default price tiers
        tiers_table = dbc.Table([
            html.Thead(html.Tr([html.Th("Tier"), html.Th("Price"), html.Th("Features")]))
        ], bordered=True, hover=True, striped=True, className="mb-0")
        
        # Default price-demand curve
        fig = go.Figure()
    
    return optimal_price_str, expected_demand_str, projected_revenue_str, tiers_table, fig

@app.callback(
    [Output("events-scatter-plot", "figure"),
     Output("events-data-table", "children")],
    [Input("strategy-filter", "value"),
     Input("sort-by", "value")]
)
def update_events_explorer(selected_strategies, sort_by):
    # Filter data based on selected strategies
    filtered_data = simulation_results[simulation_results['strategy'].isin(selected_strategies)]
    
    # Sort data based on selected option
    if sort_by == "revenue_desc":
        filtered_data = filtered_data.sort_values(by="revenue", ascending=False)
    elif sort_by == "revenue_asc":
        filtered_data = filtered_data.sort_values(by="revenue", ascending=True)
    elif sort_by == "profit_desc":
        filtered_data = filtered_data.sort_values(by="profit", ascending=False)
    elif sort_by == "profit_asc":
        filtered_data = filtered_data.sort_values(by="profit", ascending=True)
    elif sort_by == "price_desc":
        filtered_data = filtered_data.sort_values(by="optimized_price", ascending=False)
    elif sort_by == "price_asc":
        filtered_data = filtered_data.sort_values(by="optimized_price", ascending=True)
    
    # Create scatter plot
    fig = px.scatter(
        filtered_data,
        x="optimized_price",
        y="expected_demand",
        size="revenue",
        color="strategy",
        hover_name="title" if "title" in filtered_data.columns else None,
        hover_data=["optimized_price", "expected_demand", "revenue", "profit"],
        labels={
            "optimized_price": "Price ($)",
            "expected_demand": "Expected Demand",
            "revenue": "Revenue ($)",
            "profit": "Profit ($)",
            "strategy": "Strategy"
        },
        title="Price vs. Demand by Strategy"
    )
    
    fig.update_layout(
        xaxis_title="Price ($)",
        yaxis_title="Expected Demand",
        legend=dict(x=0, y=1.0),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    # Create data table
    # Select columns to display
    display_cols = ["title", "strategy", "optimized_price", "expected_demand", "revenue", "profit"]
    display_cols = [col for col in display_cols if col in filtered_data.columns]
    
    # Format the data for display
    table_data = filtered_data[display_cols].head(10)  # Limit to 10 rows for readability
    
    # Create table header
    header = html.Tr([html.Th(col.replace("_", " ").title()) for col in display_cols])
    
    # Create table rows
    rows = []
    for i, row in table_data.iterrows():
        cells = []
        for col in display_cols:
            value = row[col]
            # Format numeric values
            if col in ["optimized_price", "revenue", "profit"]:
                cells.append(html.Td(f"${value:,.2f}"))
            elif col in ["expected_demand"]:
                cells.append(html.Td(f"{int(value):,}"))
            else:
                cells.append(html.Td(value))
        rows.append(html.Tr(cells))
    
    # Create table
    table = dbc.Table(
        [html.Thead(header), html.Tbody(rows)],
        bordered=True,
        hover=True,
        striped=True,
        responsive=True
    )
    
    return fig, table

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)