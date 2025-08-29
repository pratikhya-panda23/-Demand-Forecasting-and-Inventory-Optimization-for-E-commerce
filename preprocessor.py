import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_preprocessor')

class EventDataPreprocessor:
    """Class for preprocessing event data from various sources"""
    
    def __init__(self, raw_data_dir="../../data/raw", processed_data_dir="../../data/processed"):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        
        # Create processed data directory if it doesn't exist
        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)
            logger.info(f"Created directory: {processed_data_dir}")
    
    def load_data(self, filename):
        """Load data from CSV file"""
        file_path = os.path.join(self.raw_data_dir, filename)
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return pd.DataFrame()
    
    def clean_eventbrite_data(self, df):
        """Clean and preprocess Eventbrite event data"""
        if df.empty:
            logger.warning("Empty DataFrame provided for cleaning")
            return df
        
        logger.info("Cleaning Eventbrite data...")
        
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = df_clean.fillna({
            'title': 'Unknown Event',
            'date': 'Date not specified',
            'price': 'Price not specified'
        })
        
        # Extract price information
        def extract_price(price_str):
            if price_str == 'Price not specified' or price_str == 'Free':
                return 0.0
            
            # Extract numeric price using regex
            price_match = re.search(r'\$([\d.]+)', price_str)
            if price_match:
                return float(price_match.group(1))
            return np.nan
        
        df_clean['price_value'] = df_clean['price'].apply(extract_price)
        
        # Extract date information
        def parse_date(date_str):
            if date_str == 'Date not specified':
                return pd.NaT
            
            try:
                # Try different date formats
                for fmt in ['%a, %b %d, %Y %I:%M %p', '%b %d, %Y', '%Y-%m-%d']:
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except:
                        continue
                
                # If specific formats fail, try general parsing
                return pd.to_datetime(date_str)
            except:
                return pd.NaT
        
        df_clean['event_date'] = df_clean['date'].apply(parse_date)
        
        # Add source column
        df_clean['source'] = 'Eventbrite'
        
        # Add timestamp for when the data was processed
        df_clean['processed_at'] = datetime.now()
        
        logger.info(f"Cleaned {len(df_clean)} Eventbrite records")
        return df_clean
    
    def clean_meetup_data(self, df):
        """Clean and preprocess Meetup event data"""
        if df.empty:
            logger.warning("Empty DataFrame provided for cleaning")
            return df
        
        logger.info("Cleaning Meetup data...")
        
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = df_clean.fillna({
            'title': 'Unknown Event',
            'date': 'Date not specified',
            'group': 'Group not specified'
        })
        
        # Extract date information
        def parse_date(date_str):
            if date_str == 'Date not specified':
                return pd.NaT
            
            try:
                # Try different date formats
                for fmt in ['%a, %b %d, %Y %I:%M %p', '%b %d, %Y', '%Y-%m-%d']:
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except:
                        continue
                
                # If specific formats fail, try general parsing
                return pd.to_datetime(date_str)
            except:
                return pd.NaT
        
        df_clean['event_date'] = df_clean['date'].apply(parse_date)
        
        # Add price column (Meetup doesn't typically show prices in listings)
        df_clean['price_value'] = np.nan
        
        # Add source column
        df_clean['source'] = 'Meetup'
        
        # Add timestamp for when the data was processed
        df_clean['processed_at'] = datetime.now()
        
        logger.info(f"Cleaned {len(df_clean)} Meetup records")
        return df_clean
    
    def merge_event_data(self, dfs):
        """Merge event data from multiple sources"""
        if not dfs:
            logger.warning("No DataFrames provided for merging")
            return pd.DataFrame()
        
        # Ensure all DataFrames have the same columns
        required_columns = ['title', 'url', 'event_date', 'price_value', 'source', 'processed_at']
        
        merged_df = pd.concat([df[required_columns] for df in dfs if not df.empty], ignore_index=True)
        logger.info(f"Merged {len(merged_df)} records from {len(dfs)} sources")
        
        return merged_df
    
    def add_features(self, df):
        """Add additional features to the processed data"""
        if df.empty:
            logger.warning("Empty DataFrame provided for feature addition")
            return df
        
        # Make a copy to avoid modifying the original
        df_features = df.copy()
        
        # Add day of week
        df_features['day_of_week'] = df_features['event_date'].dt.day_name()
        
        # Add weekend flag
        df_features['is_weekend'] = df_features['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
        
        # Add days until event
        today = pd.Timestamp(datetime.now().date())
        df_features['days_until_event'] = (df_features['event_date'].dt.date - today).dt.days
        
        # Add price category
        def categorize_price(price):
            if pd.isna(price) or price == 0:
                return 'Free'
            elif price < 20:
                return 'Low'
            elif price < 50:
                return 'Medium'
            else:
                return 'High'
        
        df_features['price_category'] = df_features['price_value'].apply(categorize_price)
        
        logger.info(f"Added features to {len(df_features)} records")
        return df_features
    
    def process_all_data(self):
        """Process all available event data"""
        # Process Eventbrite data
        eventbrite_df = self.load_data('eventbrite_events.csv')
        if not eventbrite_df.empty:
            eventbrite_clean = self.clean_eventbrite_data(eventbrite_df)
        else:
            eventbrite_clean = pd.DataFrame()
        
        # Process Meetup data
        meetup_df = self.load_data('meetup_events.csv')
        if not meetup_df.empty:
            meetup_clean = self.clean_meetup_data(meetup_df)
        else:
            meetup_clean = pd.DataFrame()
        
        # Merge data from different sources
        merged_df = self.merge_event_data([eventbrite_clean, meetup_clean])
        
        # Add features
        if not merged_df.empty:
            processed_df = self.add_features(merged_df)
            
            # Save processed data
            output_path = os.path.join(self.processed_data_dir, 'processed_events.csv')
            processed_df.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
            
            return processed_df
        else:
            logger.warning("No data to process")
            return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    preprocessor = EventDataPreprocessor()
    processed_data = preprocessor.process_all_data()
    
    if not processed_data.empty:
        print(f"Processed {len(processed_data)} events")
        print(processed_data.head())
    else:
        print("No data was processed")