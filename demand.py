import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import requests
import json
from pytrends.request import TrendReq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('demand_model')

class GoogleTrendsAnalyzer:
    """Class for analyzing Google Trends data for event demand estimation"""
    
    def __init__(self):
        self.pytrends = TrendReq(hl='en-US', tz=360)
        
    def get_interest_over_time(self, keywords, timeframe='today 3-m'):
        """Get interest over time for specified keywords"""
        try:
            # Build payload
            self.pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='', gprop='')
            
            # Get interest over time
            interest_df = self.pytrends.interest_over_time()
            
            if not interest_df.empty:
                logger.info(f"Retrieved Google Trends data for keywords: {keywords}")
                return interest_df
            else:
                logger.warning(f"No Google Trends data found for keywords: {keywords}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error retrieving Google Trends data: {e}")
            return pd.DataFrame()
    
    def get_related_queries(self, keywords, timeframe='today 3-m'):
        """Get related queries for specified keywords"""
        try:
            # Build payload
            self.pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='', gprop='')
            
            # Get related queries
            related_queries = self.pytrends.related_queries()
            
            if related_queries:
                logger.info(f"Retrieved related queries for keywords: {keywords}")
                return related_queries
            else:
                logger.warning(f"No related queries found for keywords: {keywords}")
                return {}
        except Exception as e:
            logger.error(f"Error retrieving related queries: {e}")
            return {}
    
    def get_interest_by_region(self, keywords, resolution='COUNTRY', timeframe='today 3-m'):
        """Get interest by region for specified keywords"""
        try:
            # Build payload
            self.pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='', gprop='')
            
            # Get interest by region
            region_df = self.pytrends.interest_by_region(resolution=resolution, inc_low_vol=True, inc_geo_code=False)
            
            if not region_df.empty:
                logger.info(f"Retrieved regional interest data for keywords: {keywords}")
                return region_df
            else:
                logger.warning(f"No regional interest data found for keywords: {keywords}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error retrieving regional interest data: {e}")
            return pd.DataFrame()


class DemandEstimator:
    """Class for estimating demand for events based on various signals"""
    
    def __init__(self, model_dir="../../models"):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.trends_analyzer = GoogleTrendsAnalyzer()
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logger.info(f"Created directory: {model_dir}")
    
    def prepare_features(self, events_df, include_trends=True):
        """Prepare features for demand estimation"""
        if events_df.empty:
            logger.warning("Empty DataFrame provided for feature preparation")
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        df = events_df.copy()
        
        # Basic features from event data
        logger.info("Preparing basic features from event data")
        
        # Handle missing values
        df['price_value'] = df['price_value'].fillna(0)
        
        # Create time-based features
        if 'event_date' in df.columns:
            df['days_until_event'] = (df['event_date'] - pd.Timestamp.now()).dt.days
            df['month'] = df['event_date'].dt.month
            df['day_of_week'] = df['event_date'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Add Google Trends data if requested
        if include_trends:
            logger.info("Adding Google Trends data")
            
            # Create a new column for trend score
            df['trend_score'] = 0
            
            # Get trends data for each event title
            for idx, row in df.iterrows():
                title = row['title']
                keywords = [title]
                
                # Get interest over time
                trends_df = self.trends_analyzer.get_interest_over_time(keywords)
                
                if not trends_df.empty and title in trends_df.columns:
                    # Use the average interest over the past week as the trend score
                    recent_trends = trends_df[title].tail(7)
                    trend_score = recent_trends.mean()
                    df.at[idx, 'trend_score'] = trend_score
        
        # Select and return features
        feature_cols = ['price_value', 'days_until_event', 'month', 'day_of_week', 'is_weekend']
        if include_trends:
            feature_cols.append('trend_score')
        
        return df[feature_cols]
    
    def train_model(self, events_df, demand_col='attendance', test_size=0.2, random_state=42):
        """Train a demand estimation model"""
        if events_df.empty or demand_col not in events_df.columns:
            logger.error(f"Cannot train model: DataFrame is empty or missing '{demand_col}' column")
            return None
        
        logger.info("Training demand estimation model")
        
        # Prepare features
        X = self.prepare_features(events_df)
        y = events_df[demand_col]
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Define preprocessing for numeric features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ])
        
        # Define models to try
        models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(random_state=random_state),
            'gradient_boosting': GradientBoostingRegressor(random_state=random_state)
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
            model_path = os.path.join(self.model_dir, 'demand_model.joblib')
            joblib.dump(best_model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            self.model = best_model
            return best_model
        else:
            logger.error("Failed to train any model")
            return None
    
    def load_model(self, model_path=None):
        """Load a trained demand estimation model"""
        if model_path is None:
            model_path = os.path.join(self.model_dir, 'demand_model.joblib')
        
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return self.model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return None
    
    def predict_demand(self, events_df):
        """Predict demand for events"""
        if events_df.empty:
            logger.warning("Empty DataFrame provided for prediction")
            return pd.DataFrame()
        
        if self.model is None:
            logger.warning("No model loaded. Attempting to load default model.")
            if not self.load_model():
                logger.error("No model available for prediction")
                return pd.DataFrame()
        
        # Prepare features
        X = self.prepare_features(events_df)
        
        # Make predictions
        try:
            predictions = self.model.predict(X)
            
            # Add predictions to the DataFrame
            result_df = events_df.copy()
            result_df['predicted_demand'] = predictions
            
            logger.info(f"Generated demand predictions for {len(result_df)} events")
            return result_df
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return pd.DataFrame()
    
    def analyze_demand_factors(self, events_df, demand_col='attendance'):
        """Analyze factors affecting demand"""
        if events_df.empty or demand_col not in events_df.columns:
            logger.error(f"Cannot analyze demand factors: DataFrame is empty or missing '{demand_col}' column")
            return {}
        
        logger.info("Analyzing demand factors")
        
        # Prepare features
        X = self.prepare_features(events_df)
        y = events_df[demand_col]
        
        # Train a Random Forest model for feature importance
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Get feature importances
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        logger.info(f"Feature importance:\n{feature_importance}")
        
        # Analyze price elasticity
        price_elasticity = self._calculate_price_elasticity(events_df, demand_col)
        
        # Analyze day of week effect
        day_effect = events_df.groupby('day_of_week')[demand_col].mean().to_dict()
        
        # Analyze trend score effect
        if 'trend_score' in X.columns:
            trend_correlation = events_df['trend_score'].corr(events_df[demand_col])
        else:
            trend_correlation = None
        
        # Return analysis results
        analysis = {
            'feature_importance': feature_importance.to_dict('records'),
            'price_elasticity': price_elasticity,
            'day_of_week_effect': day_effect,
            'trend_correlation': trend_correlation
        }
        
        return analysis
    
    def _calculate_price_elasticity(self, events_df, demand_col='attendance'):
        """Calculate price elasticity of demand"""
        if 'price_value' not in events_df.columns or demand_col not in events_df.columns:
            return None
        
        # Filter out events with zero price or demand
        df = events_df[(events_df['price_value'] > 0) & (events_df[demand_col] > 0)].copy()
        
        if len(df) < 10:  # Need enough data points for meaningful calculation
            return None
        
        # Calculate log values
        df['log_price'] = np.log(df['price_value'])
        df['log_demand'] = np.log(df[demand_col])
        
        # Fit linear regression to log values
        model = LinearRegression()
        model.fit(df[['log_price']], df['log_demand'])
        
        # Coefficient is the elasticity
        elasticity = model.coef_[0]
        
        return elasticity


if __name__ == "__main__":
    # Example usage
    # This would typically use real data from the processed events
    
    # Create a sample dataset
    sample_data = {
        'title': ['Concert in the Park', 'Tech Workshop', 'Food Festival', 'Art Exhibition'],
        'price_value': [25.0, 15.0, 10.0, 5.0],
        'event_date': pd.to_datetime(['2023-07-15', '2023-07-20', '2023-07-25', '2023-07-30']),
        'attendance': [150, 75, 300, 100]  # Simulated actual attendance
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    # Initialize demand estimator
    estimator = DemandEstimator()
    
    # Train model on sample data
    model = estimator.train_model(sample_df)
    
    if model is not None:
        # Make predictions
        predictions_df = estimator.predict_demand(sample_df)
        print("\nDemand Predictions:")
        print(predictions_df[['title', 'price_value', 'attendance', 'predicted_demand']])
        
        # Analyze demand factors
        analysis = estimator.analyze_demand_factors(sample_df)
        print("\nDemand Factor Analysis:")
        print(f"Feature Importance: {analysis['feature_importance']}")
        print(f"Price Elasticity: {analysis['price_elasticity']}")