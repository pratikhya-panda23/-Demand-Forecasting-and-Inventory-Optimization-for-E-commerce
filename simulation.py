import pandas as pd
import numpy as np
import logging
import os
import sys
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import from src/models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models
try:
    from models.demand import DemandEstimator
    from models.pricing import DynamicPricingModel
except ImportError as e:
    print(f"Error importing models: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pricing_simulation')

class PricingSimulation:
    """Class for simulating dynamic pricing strategies for events"""
    
    def __init__(self, output_dir="../../data/processed"):
        self.output_dir = output_dir
        self.demand_estimator = DemandEstimator()
        self.pricing_model = DynamicPricingModel()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created directory: {output_dir}")
    
    def load_models(self, demand_model_path=None, pricing_model_path=None):
        """Load trained demand and pricing models"""
        demand_loaded = self.demand_estimator.load_model(demand_model_path)
        pricing_loaded = self.pricing_model.load_model(pricing_model_path)
        
        if demand_loaded and pricing_loaded:
            logger.info("Successfully loaded demand and pricing models")
            return True
        else:
            logger.warning("Failed to load one or both models")
            return False
    
    def generate_event_scenarios(self, base_events_df, num_scenarios=5):
        """Generate multiple event scenarios from base events for simulation"""
        if base_events_df.empty:
            logger.warning("Empty DataFrame provided for scenario generation")
            return pd.DataFrame()
        
        logger.info(f"Generating {num_scenarios} scenarios for each event")
        
        scenarios = []
        
        for _, event in base_events_df.iterrows():
            # Create variations of the base event
            for i in range(num_scenarios):
                scenario = event.copy()
                
                # Vary the event date (if present)
                if 'event_date' in scenario:
                    # Randomly shift the date by -30 to +60 days
                    date_shift = np.random.randint(-30, 61)
                    scenario['event_date'] = scenario['event_date'] + pd.Timedelta(days=date_shift)
                    
                    # Update days_until_event if present
                    if 'days_until_event' in scenario:
                        scenario['days_until_event'] = (scenario['event_date'] - pd.Timestamp.now()).days
                    
                    # Update day_of_week and is_weekend if present
                    if 'day_of_week' in scenario:
                        scenario['day_of_week'] = scenario['event_date'].dayofweek
                    if 'is_weekend' in scenario:
                        scenario['is_weekend'] = int(scenario['event_date'].dayofweek >= 5)
                
                # Vary the price (if present)
                if 'price_value' in scenario:
                    # Randomly adjust price by ±20%
                    price_factor = np.random.uniform(0.8, 1.2)
                    scenario['price_value'] = scenario['price_value'] * price_factor
                
                # Vary the demand/attendance (if present)
                for col in ['attendance', 'predicted_demand']:
                    if col in scenario:
                        # Randomly adjust demand by ±30%
                        demand_factor = np.random.uniform(0.7, 1.3)
                        scenario[col] = scenario[col] * demand_factor
                
                # Add scenario ID
                scenario['scenario_id'] = f"{event.name}_{i+1}"
                
                scenarios.append(scenario)
        
        # Convert list of scenarios to DataFrame
        scenarios_df = pd.DataFrame(scenarios)
        logger.info(f"Generated {len(scenarios_df)} total scenarios")
        
        return scenarios_df
    
    def simulate_pricing_strategies(self, events_df, strategies=None):
        """Simulate different pricing strategies for events"""
        if events_df.empty:
            logger.warning("Empty DataFrame provided for simulation")
            return pd.DataFrame()
        
        # Define default strategies if not provided
        if strategies is None:
            strategies = [
                {'name': 'Fixed Price', 'type': 'fixed', 'params': {}},
                {'name': 'Revenue Maximizing', 'type': 'optimize', 'params': {'objective': 'revenue'}},
                {'name': 'Profit Maximizing', 'type': 'optimize', 'params': {'objective': 'profit'}},
                {'name': 'Early Bird Discount', 'type': 'time_based', 'params': {'early_discount': 0.2, 'regular_price': 1.0, 'late_premium': 0.1}},
                {'name': 'Demand Based', 'type': 'demand_based', 'params': {'low_demand_discount': 0.15, 'high_demand_premium': 0.15}}
            ]
        
        logger.info(f"Simulating {len(strategies)} pricing strategies for {len(events_df)} events")
        
        # Ensure we have demand predictions
        if 'predicted_demand' not in events_df.columns and self.demand_estimator.model is not None:
            logger.info("Generating demand predictions for simulation")
            events_with_demand = self.demand_estimator.predict_demand(events_df)
            if not events_with_demand.empty:
                events_df = events_with_demand
        
        # Initialize results list
        results = []
        
        # Simulate each strategy for each event
        for _, event in events_df.iterrows():
            event_dict = event.to_dict()
            
            # Get base price and demand
            base_price = event['price_value'] if 'price_value' in event else 0
            base_demand = event['predicted_demand'] if 'predicted_demand' in event else (
                event['attendance'] if 'attendance' in event else 100)
            
            # Calculate elasticity if not already set
            if self.pricing_model.price_elasticity is None:
                # Default to moderately elastic demand
                self.pricing_model.price_elasticity = -1.2
            
            elasticity = self.pricing_model.price_elasticity
            
            for strategy in strategies:
                strategy_name = strategy['name']
                strategy_type = strategy['type']
                params = strategy['params']
                
                # Apply the pricing strategy
                if strategy_type == 'fixed':
                    # Use the original price
                    price = base_price
                    
                elif strategy_type == 'optimize':
                    # Use the pricing model to optimize
                    objective = params.get('objective', 'revenue')
                    price = self.pricing_model.optimize_price(event_dict, objective=objective)
                    if price is None:
                        price = base_price
                    
                elif strategy_type == 'time_based':
                    # Apply time-based pricing
                    if 'days_until_event' in event:
                        days_until = event['days_until_event']
                        early_discount = params.get('early_discount', 0.2)
                        late_premium = params.get('late_premium', 0.1)
                        
                        if days_until > 30:  # Early bird
                            price = base_price * (1 - early_discount)
                        elif days_until < 7:  # Last minute
                            price = base_price * (1 + late_premium)
                        else:  # Regular
                            price = base_price
                    else:
                        price = base_price
                    
                elif strategy_type == 'demand_based':
                    # Apply demand-based pricing
                    low_demand_threshold = 0.7  # 70% of average demand
                    high_demand_threshold = 1.3  # 130% of average demand
                    
                    avg_demand = events_df['predicted_demand'].mean() if 'predicted_demand' in events_df.columns else (
                        events_df['attendance'].mean() if 'attendance' in events_df.columns else base_demand)
                    
                    low_demand_discount = params.get('low_demand_discount', 0.15)
                    high_demand_premium = params.get('high_demand_premium', 0.15)
                    
                    if base_demand < (avg_demand * low_demand_threshold):  # Low demand
                        price = base_price * (1 - low_demand_discount)
                    elif base_demand > (avg_demand * high_demand_threshold):  # High demand
                        price = base_price * (1 + high_demand_premium)
                    else:  # Normal demand
                        price = base_price
                    
                else:  # Default to base price
                    price = base_price
                
                # Calculate expected demand at this price using elasticity
                if price > 0 and base_price > 0:
                    price_ratio = price / base_price
                    demand = base_demand * (price_ratio ** elasticity)
                else:
                    demand = base_demand
                
                # Calculate revenue and profit
                revenue = price * demand
                
                # Assume costs are 30% of base price per attendee plus a fixed cost
                variable_cost = base_price * 0.3 * demand
                fixed_cost = base_price * 10  # Arbitrary fixed cost
                total_cost = variable_cost + fixed_cost
                profit = revenue - total_cost
                
                # Store the result
                result = {
                    'event_id': event.name,
                    'scenario_id': event.get('scenario_id', 'base'),
                    'strategy': strategy_name,
                    'original_price': base_price,
                    'optimized_price': price,
                    'price_change_pct': ((price - base_price) / base_price * 100) if base_price > 0 else 0,
                    'original_demand': base_demand,
                    'expected_demand': demand,
                    'demand_change_pct': ((demand - base_demand) / base_demand * 100) if base_demand > 0 else 0,
                    'revenue': revenue,
                    'cost': total_cost,
                    'profit': profit,
                    'elasticity': elasticity
                }
                
                # Add event details
                for key, value in event_dict.items():
                    if key not in result and key not in ['price_value', 'attendance', 'predicted_demand']:
                        result[key] = value
                
                results.append(result)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        output_path = os.path.join(self.output_dir, 'pricing_simulation_results.csv')
        results_df.to_csv(output_path, index=False)
        logger.info(f"Simulation results saved to {output_path}")
        
        return results_df
    
    def analyze_simulation_results(self, results_df):
        """Analyze the results of the pricing simulation"""
        if results_df.empty:
            logger.warning("Empty DataFrame provided for analysis")
            return {}
        
        logger.info("Analyzing simulation results")
        
        # Group by strategy and calculate aggregate metrics
        strategy_summary = results_df.groupby('strategy').agg({
            'revenue': ['mean', 'sum', 'std'],
            'profit': ['mean', 'sum', 'std'],
            'expected_demand': ['mean', 'sum', 'std'],
            'price_change_pct': ['mean', 'min', 'max'],
            'demand_change_pct': ['mean', 'min', 'max']
        })
        
        # Flatten the column hierarchy
        strategy_summary.columns = ['_'.join(col).strip() for col in strategy_summary.columns.values]
        
        # Reset index for easier access
        strategy_summary = strategy_summary.reset_index()
        
        # Find the best strategy for revenue and profit
        best_revenue_strategy = strategy_summary.loc[strategy_summary['revenue_sum'].idxmax()]['strategy']
        best_profit_strategy = strategy_summary.loc[strategy_summary['profit_sum'].idxmax()]['strategy']
        
        # Calculate improvement over fixed price strategy
        fixed_price_revenue = strategy_summary.loc[strategy_summary['strategy'] == 'Fixed Price', 'revenue_sum'].values[0]
        fixed_price_profit = strategy_summary.loc[strategy_summary['strategy'] == 'Fixed Price', 'profit_sum'].values[0]
        
        strategy_summary['revenue_improvement_pct'] = ((strategy_summary['revenue_sum'] - fixed_price_revenue) / fixed_price_revenue * 100)
        strategy_summary['profit_improvement_pct'] = ((strategy_summary['profit_sum'] - fixed_price_profit) / fixed_price_profit * 100)
        
        # Save strategy summary
        summary_path = os.path.join(self.output_dir, 'pricing_strategy_summary.csv')
        strategy_summary.to_csv(summary_path, index=False)
        logger.info(f"Strategy summary saved to {summary_path}")
        
        # Create visualizations
        self._create_strategy_comparison_charts(results_df, strategy_summary)
        
        # Prepare analysis results
        analysis = {
            'best_revenue_strategy': best_revenue_strategy,
            'best_profit_strategy': best_profit_strategy,
            'strategy_summary': strategy_summary.to_dict('records'),
            'revenue_improvement': {
                'strategy': best_revenue_strategy,
                'improvement_pct': strategy_summary.loc[strategy_summary['strategy'] == best_revenue_strategy, 'revenue_improvement_pct'].values[0]
            },
            'profit_improvement': {
                'strategy': best_profit_strategy,
                'improvement_pct': strategy_summary.loc[strategy_summary['strategy'] == best_profit_strategy, 'profit_improvement_pct'].values[0]
            }
        }
        
        # Save analysis as JSON
        analysis_path = os.path.join(self.output_dir, 'pricing_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=4)
        logger.info(f"Analysis results saved to {analysis_path}")
        
        return analysis
    
    def _create_strategy_comparison_charts(self, results_df, strategy_summary):
        """Create charts comparing different pricing strategies"""
        # Set the style
        sns.set(style="whitegrid")
        
        # Create directory for charts if it doesn't exist
        charts_dir = os.path.join(self.output_dir, 'charts')
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)
        
        # 1. Revenue by Strategy
        plt.figure(figsize=(10, 6))
        chart = sns.barplot(x='strategy', y='revenue_sum', data=strategy_summary)
        chart.set_title('Total Revenue by Pricing Strategy')
        chart.set_xlabel('Strategy')
        chart.set_ylabel('Total Revenue ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'revenue_by_strategy.png'))
        plt.close()
        
        # 2. Profit by Strategy
        plt.figure(figsize=(10, 6))
        chart = sns.barplot(x='strategy', y='profit_sum', data=strategy_summary)
        chart.set_title('Total Profit by Pricing Strategy')
        chart.set_xlabel('Strategy')
        chart.set_ylabel('Total Profit ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'profit_by_strategy.png'))
        plt.close()
        
        # 3. Demand by Strategy
        plt.figure(figsize=(10, 6))
        chart = sns.barplot(x='strategy', y='expected_demand_sum', data=strategy_summary)
        chart.set_title('Total Expected Demand by Pricing Strategy')
        chart.set_xlabel('Strategy')
        chart.set_ylabel('Total Expected Demand')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'demand_by_strategy.png'))
        plt.close()
        
        # 4. Price Change by Strategy
        plt.figure(figsize=(10, 6))
        chart = sns.boxplot(x='strategy', y='price_change_pct', data=results_df)
        chart.set_title('Price Change Percentage by Strategy')
        chart.set_xlabel('Strategy')
        chart.set_ylabel('Price Change (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'price_change_by_strategy.png'))
        plt.close()
        
        # 5. Revenue vs. Demand Scatter Plot
        plt.figure(figsize=(10, 6))
        chart = sns.scatterplot(x='expected_demand', y='revenue', hue='strategy', data=results_df)
        chart.set_title('Revenue vs. Expected Demand by Strategy')
        chart.set_xlabel('Expected Demand')
        chart.set_ylabel('Revenue ($)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'revenue_vs_demand.png'))
        plt.close()
        
        # 6. Price vs. Demand Scatter Plot
        plt.figure(figsize=(10, 6))
        chart = sns.scatterplot(x='optimized_price', y='expected_demand', hue='strategy', data=results_df)
        chart.set_title('Expected Demand vs. Price by Strategy')
        chart.set_xlabel('Price ($)')
        chart.set_ylabel('Expected Demand')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'demand_vs_price.png'))
        plt.close()
        
        logger.info(f"Created strategy comparison charts in {charts_dir}")


if __name__ == "__main__":
    # Example usage
    # This would typically use real data from the processed events
    
    # Create a sample dataset
    sample_data = {
        'title': ['Concert in the Park', 'Tech Workshop', 'Food Festival', 'Art Exhibition'],
        'price_value': [25.0, 15.0, 10.0, 5.0],
        'event_date': pd.to_datetime(['2023-07-15', '2023-07-20', '2023-07-25', '2023-07-30']),
        'attendance': [150, 75, 300, 100],  # Simulated actual attendance
        'days_until_event': [30, 35, 40, 45],
        'is_weekend': [1, 0, 1, 0]
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    # Initialize simulation
    simulation = PricingSimulation()
    
    # Generate scenarios
    scenarios_df = simulation.generate_event_scenarios(sample_df, num_scenarios=3)
    
    # Simulate pricing strategies
    results_df = simulation.simulate_pricing_strategies(scenarios_df)
    
    # Analyze results
    analysis = simulation.analyze_simulation_results(results_df)
    
    print("\nSimulation Analysis:")
    print(f"Best Revenue Strategy: {analysis['best_revenue_strategy']}")
    print(f"Revenue Improvement: {analysis['revenue_improvement']['improvement_pct']:.2f}%")
    print(f"Best Profit Strategy: {analysis['best_profit_strategy']}")
    print(f"Profit Improvement: {analysis['profit_improvement']['improvement_pct']:.2f}%")