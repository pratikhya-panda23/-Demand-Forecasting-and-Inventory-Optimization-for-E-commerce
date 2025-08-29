import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pricing_model')

class DynamicPricingModel:
    """Class for dynamic pricing model for local events and experiences"""
    
    def __init__(self, model_dir="../../models"):
        self.model_dir = model_dir
        self.model = None
        self.price_elasticity = None
        self.price_bounds = {'min': 0, 'max': 1000}  # Default price bounds
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logger.info(f"Created directory: {model_dir}")
    
    def prepare_features(self, events_df):
        """Prepare features for price optimization"""
        if events_df.empty:
            logger.warning("Empty DataFrame provided for feature preparation")
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        df = events_df.copy()
        
        # Basic features from event data
        logger.info("Preparing features for price optimization")
        
        # Handle missing values
        for col in ['attendance', 'predicted_demand']:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Create time-based features if not already present
        if 'event_date' in df.columns:
            if 'days_until_event' not in df.columns:
                df['days_until_event'] = (df['event_date'] - pd.Timestamp.now()).dt.days
            if 'month' not in df.columns:
                df['month'] = df['event_date'].dt.month
            if 'day_of_week' not in df.columns:
                df['day_of_week'] = df['event_date'].dt.dayofweek
            if 'is_weekend' not in df.columns:
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Select and return features
        feature_cols = ['days_until_event', 'month', 'day_of_week', 'is_weekend']
        
        # Add demand features if available
        if 'predicted_demand' in df.columns:
            feature_cols.append('predicted_demand')
        elif 'attendance' in df.columns:
            feature_cols.append('attendance')
        
        # Add trend score if available
        if 'trend_score' in df.columns:
            feature_cols.append('trend_score')
        
        # Ensure all selected columns exist in the DataFrame
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        return df[feature_cols]
    
    def train_model(self, events_df, price_col='price_value', test_size=0.2, random_state=42):
        """Train a pricing optimization model"""
        if events_df.empty or price_col not in events_df.columns:
            logger.error(f"Cannot train model: DataFrame is empty or missing '{price_col}' column")
            return None
        
        logger.info("Training pricing optimization model")
        
        # Prepare features
        X = self.prepare_features(events_df)
        y = events_df[price_col]
        
        # Set price bounds based on data
        self.price_bounds['min'] = max(0, y.min() * 0.5)  # Minimum price (at least 0)
        self.price_bounds['max'] = y.max() * 1.5  # Maximum price (50% above current max)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Define preprocessing pipeline
        preprocessor = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Define models to try
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0, random_state=random_state),
            'lasso': Lasso(alpha=0.1, random_state=random_state),
            'polynomial': Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression())
            ]),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=random_state)
        }
        
        best_model = None
        best_score = -float('inf')
        best_model_name = None
        
        # Train and evaluate each model
        for name, model in models.items():
            logger.info(f"Training {name} model")
            
            # Create pipeline with preprocessing and model
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = pipeline.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"{name.capitalize()} model - MSE: {mse:.2f}, R²: {r2:.2f}")
            
            # Keep track of the best model
            if r2 > best_score:
                best_score = r2
                best_model = pipeline
                best_model_name = name
        
        if best_model is not None:
            logger.info(f"Best model: {best_model_name.capitalize()} (R²: {best_score:.2f})")
            
            # Save the best model
            model_path = os.path.join(self.model_dir, 'pricing_model.joblib')
            joblib.dump(best_model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            self.model = best_model
            
            # Calculate price elasticity
            self._calculate_price_elasticity(events_df)
            
            return best_model
        else:
            logger.error("Failed to train any model")
            return None
    
    def load_model(self, model_path=None):
        """Load a trained pricing model"""
        if model_path is None:
            model_path = os.path.join(self.model_dir, 'pricing_model.joblib')
        
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return self.model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return None
    
    def _calculate_price_elasticity(self, events_df, price_col='price_value', demand_col='attendance'):
        """Calculate price elasticity of demand"""
        if price_col not in events_df.columns or demand_col not in events_df.columns:
            logger.warning(f"Cannot calculate price elasticity: missing '{price_col}' or '{demand_col}' column")
            return None
        
        # Filter out events with zero price or demand
        df = events_df[(events_df[price_col] > 0) & (events_df[demand_col] > 0)].copy()
        
        if len(df) < 10:  # Need enough data points for meaningful calculation
            logger.warning("Not enough data points to calculate price elasticity")
            self.price_elasticity = -1.0  # Default to unit elastic
            return self.price_elasticity
        
        # Calculate log values
        df['log_price'] = np.log(df[price_col])
        df['log_demand'] = np.log(df[demand_col])
        
        # Fit linear regression to log values
        model = LinearRegression()
        model.fit(df[['log_price']], df['log_demand'])
        
        # Coefficient is the elasticity
        elasticity = model.coef_[0]
        self.price_elasticity = elasticity
        
        logger.info(f"Calculated price elasticity: {elasticity:.2f}")
        return elasticity
    
    def optimize_price(self, event_data, objective='revenue', constraints=None):
        """Optimize price for an event based on the trained model and objective"""
        if self.model is None:
            logger.warning("No model loaded. Attempting to load default model.")
            if not self.load_model():
                logger.error("No model available for price optimization")
                return None
        
        # Convert single event to DataFrame if necessary
        if not isinstance(event_data, pd.DataFrame):
            event_data = pd.DataFrame([event_data])
        
        # Prepare features
        X = self.prepare_features(event_data)
        
        # Get base price prediction from model
        try:
            base_price = self.model.predict(X)[0]
            logger.info(f"Base price prediction: ${base_price:.2f}")
        except Exception as e:
            logger.error(f"Error predicting base price: {e}")
            return None
        
        # Apply constraints if provided
        if constraints is None:
            constraints = {}
        
        min_price = constraints.get('min_price', self.price_bounds['min'])
        max_price = constraints.get('max_price', self.price_bounds['max'])
        
        # Get demand prediction if available
        demand = None
        if 'predicted_demand' in event_data.columns:
            demand = event_data['predicted_demand'].iloc[0]
        elif 'attendance' in event_data.columns:
            demand = event_data['attendance'].iloc[0]
        
        # If we have demand and elasticity, we can optimize for revenue or profit
        if demand is not None and self.price_elasticity is not None:
            logger.info(f"Optimizing price with demand: {demand}, elasticity: {self.price_elasticity}")
            
            # If elastic (e < -1), lower price to increase revenue
            # If inelastic (e > -1), raise price to increase revenue
            # If unit elastic (e = -1), price doesn't affect revenue
            
            if objective == 'revenue':
                if self.price_elasticity < -1:  # Elastic demand
                    # For elastic demand, optimal price for revenue is where elasticity = -1
                    # We'll approximate by reducing the base price
                    optimal_price = base_price * 0.9
                elif self.price_elasticity > -1:  # Inelastic demand
                    # For inelastic demand, higher price means more revenue
                    optimal_price = base_price * 1.1
                else:  # Unit elastic
                    optimal_price = base_price
            elif objective == 'profit':
                # For profit maximization, we need cost information
                # As a simplification, we'll assume costs are 30% of the base price
                cost = base_price * 0.3
                
                # Optimal price for profit is where MR = MC
                # For linear demand, optimal price = (cost + base_price) / 2
                optimal_price = (cost + base_price) / 2
            else:  # Default to base price
                optimal_price = base_price
        else:
            # Without demand or elasticity, just use the base price
            optimal_price = base_price
        
        # Apply constraints
        optimal_price = max(min_price, min(optimal_price, max_price))
        
        # Apply time-based adjustments
        if 'days_until_event' in event_data.columns:
            days_until_event = event_data['days_until_event'].iloc[0]
            
            # Early bird discount (more than 30 days before event)
            if days_until_event > 30:
                optimal_price *= 0.9
                logger.info(f"Applied early bird discount: ${optimal_price:.2f}")
            # Last minute premium (less than 7 days before event)
            elif days_until_event < 7:
                optimal_price *= 1.1
                logger.info(f"Applied last minute premium: ${optimal_price:.2f}")
        
        # Apply weekend premium if applicable
        if 'is_weekend' in event_data.columns and event_data['is_weekend'].iloc[0] == 1:
            optimal_price *= 1.05
            logger.info(f"Applied weekend premium: ${optimal_price:.2f}")
        
        # Round to nearest dollar or appropriate increment
        optimal_price = round(optimal_price, 2)
        
        logger.info(f"Optimized price: ${optimal_price:.2f}")
        return optimal_price
    
    def generate_price_tiers(self, event_data, num_tiers=3):
        """Generate price tiers for an event"""
        # Get optimized price
        optimal_price = self.optimize_price(event_data)
        
        if optimal_price is None:
            return []
        
        # Generate tiers around the optimal price
        tiers = []
        
        if num_tiers == 3:
            # Basic tier (lower price)
            basic_price = optimal_price * 0.8
            basic_tier = {
                'name': 'Basic',
                'price': round(basic_price, 2),
                'features': ['Standard admission', 'Basic amenities']
            }
            
            # Standard tier (optimal price)
            standard_tier = {
                'name': 'Standard',
                'price': round(optimal_price, 2),
                'features': ['Standard admission', 'Basic amenities', 'Premium seating']
            }
            
            # Premium tier (higher price)
            premium_price = optimal_price * 1.3
            premium_tier = {
                'name': 'Premium',
                'price': round(premium_price, 2),
                'features': ['Standard admission', 'Basic amenities', 'Premium seating', 'VIP access', 'Complimentary refreshments']
            }
            
            tiers = [basic_tier, standard_tier, premium_tier]
        elif num_tiers == 5:
            # Super Early Bird
            super_early_price = optimal_price * 0.7
            super_early_tier = {
                'name': 'Super Early Bird',
                'price': round(super_early_price, 2),
                'features': ['Standard admission', 'Limited availability']
            }
            
            # Early Bird
            early_price = optimal_price * 0.85
            early_tier = {
                'name': 'Early Bird',
                'price': round(early_price, 2),
                'features': ['Standard admission', 'Basic amenities']
            }
            
            # Regular
            regular_tier = {
                'name': 'Regular',
                'price': round(optimal_price, 2),
                'features': ['Standard admission', 'Basic amenities', 'Regular seating']
            }
            
            # VIP
            vip_price = optimal_price * 1.25
            vip_tier = {
                'name': 'VIP',
                'price': round(vip_price, 2),
                'features': ['Standard admission', 'Basic amenities', 'Premium seating', 'VIP access']
            }
            
            # Premium VIP
            premium_vip_price = optimal_price * 1.5
            premium_vip_tier = {
                'name': 'Premium VIP',
                'price': round(premium_vip_price, 2),
                'features': ['Standard admission', 'Basic amenities', 'Premium seating', 'VIP access', 'Complimentary refreshments', 'Exclusive merchandise']
            }
            
            tiers = [super_early_tier, early_tier, regular_tier, vip_tier, premium_vip_tier]
        else:
            # Default to single tier
            standard_tier = {
                'name': 'Standard',
                'price': round(optimal_price, 2),
                'features': ['Standard admission']
            }
            
            tiers = [standard_tier]
        
        logger.info(f"Generated {len(tiers)} price tiers")
        return tiers
    
    def visualize_price_demand_curve(self, event_data, price_range=None, elasticity=None):
        """Visualize the price-demand curve for an event"""
        if elasticity is None:
            elasticity = self.price_elasticity
            
        if elasticity is None:
            logger.warning("No price elasticity available for visualization")
            return None
        
        # Get base price and demand
        base_price = self.optimize_price(event_data)
        
        if base_price is None:
            return None
        
        # Get demand prediction if available
        base_demand = None
        if isinstance(event_data, pd.DataFrame):
            if 'predicted_demand' in event_data.columns:
                base_demand = event_data['predicted_demand'].iloc[0]
            elif 'attendance' in event_data.columns:
                base_demand = event_data['attendance'].iloc[0]
        
        if base_demand is None:
            base_demand = 100  # Default value if no demand data available
        
        # Define price range to visualize
        if price_range is None:
            min_price = max(1, base_price * 0.5)
            max_price = base_price * 2
            price_range = np.linspace(min_price, max_price, 100)
        
        # Calculate demand at each price point using elasticity
        demands = []
        revenues = []
        
        for price in price_range:
            # Calculate demand using elasticity formula: %change in demand = elasticity * %change in price
            price_ratio = price / base_price
            demand_ratio = price_ratio ** elasticity
            demand = base_demand * demand_ratio
            revenue = price * demand
            
            demands.append(demand)
            revenues.append(revenue)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot price-demand curve
        ax1.plot(price_range, demands, 'b-')
        ax1.set_title('Price-Demand Curve')
        ax1.set_xlabel('Price ($)')
        ax1.set_ylabel('Demand (Attendance)')
        ax1.grid(True)
        
        # Mark optimal price for demand
        ax1.axvline(x=base_price, color='r', linestyle='--')
        ax1.text(base_price, max(demands) * 0.9, f'Base Price: ${base_price:.2f}', 
                 color='r', ha='right', va='top')
        
        # Plot price-revenue curve
        ax2.plot(price_range, revenues, 'g-')
        ax2.set_title('Price-Revenue Curve')
        ax2.set_xlabel('Price ($)')
        ax2.set_ylabel('Revenue ($)')
        ax2.grid(True)
        
        # Find revenue-maximizing price
        max_revenue_idx = np.argmax(revenues)
        max_revenue_price = price_range[max_revenue_idx]
        max_revenue = revenues[max_revenue_idx]
        
        # Mark revenue-maximizing price
        ax2.axvline(x=max_revenue_price, color='r', linestyle='--')
        ax2.text(max_revenue_price, max_revenue * 0.9, 
                 f'Revenue Max: ${max_revenue_price:.2f}', 
                 color='r', ha='right', va='top')
        
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(self.model_dir, 'price_demand_curve.png')
        plt.savefig(fig_path)
        logger.info(f"Price-demand curve saved to {fig_path}")
        
        return fig


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
    
    # Initialize pricing model
    pricing_model = DynamicPricingModel()
    
    # Train model on sample data
    model = pricing_model.train_model(sample_df)
    
    if model is not None:
        # Optimize price for an event
        event = sample_df.iloc[0].to_dict()
        optimal_price = pricing_model.optimize_price(event, objective='revenue')
        print(f"\nOptimal price for '{event['title']}': ${optimal_price:.2f}")
        
        # Generate price tiers
        tiers = pricing_model.generate_price_tiers(event, num_tiers=3)
        print("\nPrice Tiers:")
        for tier in tiers:
            print(f"{tier['name']}: ${tier['price']:.2f} - {', '.join(tier['features'])}")
        
        # Visualize price-demand curve
        pricing_model.visualize_price_demand_curve(event)