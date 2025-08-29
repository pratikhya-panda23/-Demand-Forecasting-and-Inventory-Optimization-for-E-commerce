import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json

def main():
    """Run the pricing simulation and generate analysis results."""
    print("Starting Dynamic Pricing Simulation...")
    
    # Add the project root directory to the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root)
    
    # Import custom modules
    try:
        from src.data.preprocessor import EventDataPreprocessor
        from src.models.demand import DemandEstimator
        from src.models.pricing import DynamicPricingModel
        from src.simulation.simulation import PricingSimulation
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure you're running this script from the project root directory.")
        return
    
    # Check if real data exists
    raw_data_path = os.path.join(project_root, 'data', 'raw')
    processed_data_path = os.path.join(project_root, 'data', 'processed')
    eventbrite_path = os.path.join(raw_data_path, 'eventbrite_events.csv')
    meetup_path = os.path.join(raw_data_path, 'meetup_events.csv')
    
    # Create directories if they don't exist
    os.makedirs(raw_data_path, exist_ok=True)
    os.makedirs(processed_data_path, exist_ok=True)
    
    if os.path.exists(eventbrite_path) or os.path.exists(meetup_path):
        print("Loading real event data...")
        
        # Initialize preprocessor
        preprocessor = EventDataPreprocessor()
        
        # Load and preprocess data
        if os.path.exists(eventbrite_path):
            eventbrite_df = preprocessor.load_data(eventbrite_path)
            eventbrite_df = preprocessor.preprocess_eventbrite_data(eventbrite_df)
            print(f"Loaded {len(eventbrite_df)} events from Eventbrite")
        else:
            eventbrite_df = pd.DataFrame()
        
        if os.path.exists(meetup_path):
            meetup_df = preprocessor.load_data(meetup_path)
            meetup_df = preprocessor.preprocess_meetup_data(meetup_df)
            print(f"Loaded {len(meetup_df)} events from Meetup")
        else:
            meetup_df = pd.DataFrame()
        
        # Merge data if both sources are available
        if not eventbrite_df.empty and not meetup_df.empty:
            events_df = preprocessor.merge_data([eventbrite_df, meetup_df])
        elif not eventbrite_df.empty:
            events_df = eventbrite_df
        elif not meetup_df.empty:
            events_df = meetup_df
        else:
            events_df = pd.DataFrame()
        
        # Add features
        if not events_df.empty:
            events_df = preprocessor.add_features(events_df)
            print(f"Final dataset contains {len(events_df)} events with {len(events_df.columns)} features")
    else:
        print("No real data found. Creating synthetic dataset for demonstration...")
        
        # Create synthetic data
        np.random.seed(42)  # For reproducibility
        
        # Define event categories
        categories = ['Concert', 'Workshop', 'Conference', 'Festival', 'Exhibition', 'Networking', 'Sports']
        locations = ['Downtown', 'Convention Center', 'Park', 'University', 'Theater', 'Stadium', 'Gallery']
        
        # Generate random events
        n_events = 100
        
        # Generate dates between now and 3 months from now
        start_date = datetime.now()
        end_date = pd.Timestamp(start_date) + pd.Timedelta(days=90)
        dates = [start_date + pd.Timedelta(days=np.random.randint(1, 91)) for _ in range(n_events)]
        
        # Generate synthetic data
        events_data = {
            'title': [f"{np.random.choice(categories)} Event {i}" for i in range(1, n_events+1)],
            'category': np.random.choice(categories, n_events),
            'location': np.random.choice(locations, n_events),
            'event_date': dates,
            'price_value': np.random.uniform(5, 100, n_events),  # Random prices between $5 and $100
            'attendance': np.random.randint(20, 500, n_events),  # Random attendance between 20 and 500
            'organizer': [f"Organizer {i}" for i in range(1, n_events+1)],
            'description_length': np.random.randint(100, 1000, n_events),  # Length of description in characters
            'source': np.random.choice(['Eventbrite', 'Meetup'], n_events)
        }
        
        # Create DataFrame
        events_df = pd.DataFrame(events_data)
        
        # Add derived features
        events_df['days_until_event'] = (events_df['event_date'] - datetime.now()).dt.days
        events_df['day_of_week'] = events_df['event_date'].dt.dayofweek
        events_df['is_weekend'] = events_df['day_of_week'].isin([5, 6]).astype(int)
        events_df['month'] = events_df['event_date'].dt.month
        
        # Add price category
        events_df['price_category'] = pd.cut(
            events_df['price_value'], 
            bins=[0, 15, 30, 50, 100, float('inf')],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        # Add trend score (simulated)
        events_df['trend_score'] = np.random.uniform(0, 100, n_events)
        
        print(f"Created synthetic dataset with {len(events_df)} events")
        
        # Save synthetic data
        synthetic_data_path = os.path.join(raw_data_path, 'synthetic_events.csv')
        events_df.to_csv(synthetic_data_path, index=False)
        print(f"Saved synthetic data to {synthetic_data_path}")
    
    # Split data into training and testing sets
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(events_df, test_size=0.2, random_state=42)
    
    # Train demand estimation model
    print("\nTraining demand estimation model...")
    demand_estimator = DemandEstimator()
    demand_model = demand_estimator.train_model(train_df, demand_col='attendance')
    
    if demand_model is not None:
        # Save demand model
        model_dir = os.path.join(project_root, 'src', 'models', 'saved_models')
        os.makedirs(model_dir, exist_ok=True)
        demand_model_path = os.path.join(model_dir, 'demand_model.pkl')
        demand_estimator.save_model(demand_model_path)
        print(f"Saved demand model to {demand_model_path}")
        
        # Analyze demand factors
        demand_analysis = demand_estimator.analyze_demand_factors(train_df)
        
        # Save demand analysis
        analysis_dir = os.path.join(project_root, 'data', 'processed', 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        demand_analysis_path = os.path.join(analysis_dir, 'demand_analysis.json')
        
        # Convert analysis to JSON-serializable format
        serializable_analysis = {}
        for key, value in demand_analysis.items():
            if key == 'feature_importance':
                serializable_analysis[key] = value.to_dict('records')
            elif isinstance(value, np.ndarray):
                serializable_analysis[key] = value.tolist()
            elif isinstance(value, np.float64):
                serializable_analysis[key] = float(value)
            else:
                serializable_analysis[key] = value
        
        with open(demand_analysis_path, 'w') as f:
            json.dump(serializable_analysis, f, indent=4)
        print(f"Saved demand analysis to {demand_analysis_path}")
    else:
        print("Failed to train demand estimation model")
        return
    
    # Train pricing model
    print("\nTraining pricing model...")
    pricing_model = DynamicPricingModel()
    price_model = pricing_model.train_model(train_df, price_col='price_value')
    
    if price_model is not None:
        # Save pricing model
        pricing_model_path = os.path.join(model_dir, 'pricing_model.pkl')
        pricing_model.save_model(pricing_model_path)
        print(f"Saved pricing model to {pricing_model_path}")
    else:
        print("Failed to train pricing model")
        return
    
    # Run simulation
    print("\nRunning pricing simulation...")
    simulation = PricingSimulation()
    simulation.load_models(demand_model_path, pricing_model_path)
    
    # Generate scenarios
    scenarios_df = simulation.generate_event_scenarios(test_df, num_scenarios=5)
    print(f"Generated {len(scenarios_df)} event scenarios")
    
    # Define pricing strategies
    strategies = [
        {'name': 'Fixed Price', 'type': 'fixed', 'params': {}},
        {'name': 'Revenue Maximizing', 'type': 'optimize', 'params': {'objective': 'revenue'}},
        {'name': 'Profit Maximizing', 'type': 'optimize', 'params': {'objective': 'profit'}},
        {'name': 'Early Bird Discount', 'type': 'time_based', 'params': {'early_discount': 0.2, 'regular_price': 1.0, 'late_premium': 0.1}},
        {'name': 'Demand Based', 'type': 'demand_based', 'params': {'low_demand_discount': 0.15, 'high_demand_premium': 0.15}}
    ]
    
    # Simulate pricing strategies
    results_df = simulation.simulate_pricing_strategies(scenarios_df, strategies=strategies)
    print(f"Simulated {len(strategies)} pricing strategies for {len(scenarios_df)} events")
    
    # Analyze simulation results
    analysis = simulation.analyze_simulation_results(results_df)
    
    # Save simulation results
    results_path = os.path.join(processed_data_path, 'simulation_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Saved simulation results to {results_path}")
    
    # Save analysis results
    analysis_path = os.path.join(analysis_dir, 'simulation_analysis.json')
    
    # Convert analysis to JSON-serializable format
    serializable_analysis = {}
    for key, value in analysis.items():
        if key in ['strategy_summary', 'revenue_improvement', 'profit_improvement']:
            if isinstance(value, pd.DataFrame):
                serializable_analysis[key] = value.to_dict('records')
            else:
                serializable_analysis[key] = value
        elif isinstance(value, np.ndarray):
            serializable_analysis[key] = value.tolist()
        elif isinstance(value, np.float64):
            serializable_analysis[key] = float(value)
        else:
            serializable_analysis[key] = value
    
    with open(analysis_path, 'w') as f:
        json.dump(serializable_analysis, f, indent=4)
    print(f"Saved analysis results to {analysis_path}")
    
    # Create comparison charts
    charts_dir = os.path.join(project_root, 'data', 'processed', 'charts')
    os.makedirs(charts_dir, exist_ok=True)
    
    simulation.create_comparison_charts(results_df, analysis, save_dir=charts_dir)
    print(f"Created comparison charts in {charts_dir}")
    
    print("\nSimulation completed successfully!")
    print(f"Best Revenue Strategy: {analysis['best_revenue_strategy']}")
    print(f"Revenue Improvement: {analysis['revenue_improvement']['improvement_pct']:.2f}%")
    print(f"Best Profit Strategy: {analysis['best_profit_strategy']}")
    print(f"Profit Improvement: {analysis['profit_improvement']['improvement_pct']:.2f}%")
    print("\nRun the dashboard to visualize the results: python run_dashboard.py")

if __name__ == "__main__":
    main()