import os
import sys
import argparse
import pandas as pd
import json
from datetime import datetime, timedelta

def main():
    """Run Google Trends analysis for event categories."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fetch and analyze Google Trends data for event categories')
    parser.add_argument('--categories', type=str, nargs='+', 
                        default=['concerts', 'workshops', 'conferences', 'festivals', 'exhibitions', 'networking events', 'sports events'],
                        help='Event categories to analyze')
    parser.add_argument('--location', type=str, default='US',
                        help='Location for trends data (country code)')
    parser.add_argument('--timeframe', type=str, default='today 3-m',
                        choices=['today 1-m', 'today 3-m', 'today 12-m', 'today 5-y'],
                        help='Timeframe for trends data')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save trends data (default: data/raw/trends)')
    args = parser.parse_args()
    
    print(f"Starting Google Trends Analysis...")
    print(f"Categories: {args.categories}")
    print(f"Location: {args.location}")
    print(f"Timeframe: {args.timeframe}")
    
    # Add the project root directory to the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root)
    
    # Set default output directory if not provided
    output_dir = args.output_dir if args.output_dir else os.path.join(project_root, 'data', 'raw', 'trends')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Import custom modules
    try:
        from src.models.demand import GoogleTrendsAnalyzer
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure you're running this script from the project root directory.")
        return
    
    # Initialize Google Trends analyzer
    trends_analyzer = GoogleTrendsAnalyzer()
    
    # Get current date for filename
    current_date = datetime.now().strftime('%Y%m%d')
    
    # Fetch trends data for each category
    all_trends_data = {}
    for category in args.categories:
        print(f"\nFetching trends data for '{category}'...")
        try:
            # Fetch interest over time
            interest_over_time = trends_analyzer.fetch_interest_over_time(
                keyword=category,
                geo=args.location,
                timeframe=args.timeframe
            )
            
            if interest_over_time is not None:
                print(f"Got interest over time data with {len(interest_over_time)} data points")
                
                # Convert to list of dictionaries for JSON serialization
                interest_over_time_list = []
                for date, row in interest_over_time.iterrows():
                    interest_over_time_list.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'interest': int(row[category])
                    })
                
                # Fetch related queries
                related_queries = trends_analyzer.fetch_related_queries(
                    keyword=category,
                    geo=args.location,
                    timeframe=args.timeframe
                )
                
                if related_queries is not None:
                    print(f"Got related queries data")
                    
                    # Convert to list of dictionaries for JSON serialization
                    top_queries = []
                    rising_queries = []
                    
                    if 'top' in related_queries and related_queries['top'] is not None:
                        for _, row in related_queries['top'].iterrows():
                            top_queries.append({
                                'query': row['query'],
                                'value': int(row['value'])
                            })
                    
                    if 'rising' in related_queries and related_queries['rising'] is not None:
                        for _, row in related_queries['rising'].iterrows():
                            rising_queries.append({
                                'query': row['query'],
                                'value': row['value']
                            })
                    
                    related_queries_dict = {
                        'top': top_queries,
                        'rising': rising_queries
                    }
                else:
                    related_queries_dict = None
                
                # Fetch related topics
                related_topics = trends_analyzer.fetch_related_topics(
                    keyword=category,
                    geo=args.location,
                    timeframe=args.timeframe
                )
                
                if related_topics is not None:
                    print(f"Got related topics data")
                    
                    # Convert to list of dictionaries for JSON serialization
                    top_topics = []
                    rising_topics = []
                    
                    if 'top' in related_topics and related_topics['top'] is not None:
                        for _, row in related_topics['top'].iterrows():
                            top_topics.append({
                                'topic': row['value'],
                                'value': int(row['formattedValue'].replace('%', ''))
                            })
                    
                    if 'rising' in related_topics and related_topics['rising'] is not None:
                        for _, row in related_topics['rising'].iterrows():
                            rising_topics.append({
                                'topic': row['value'],
                                'value': row['formattedValue']
                            })
                    
                    related_topics_dict = {
                        'top': top_topics,
                        'rising': rising_topics
                    }
                else:
                    related_topics_dict = None
                
                # Store all data for this category
                all_trends_data[category] = {
                    'interest_over_time': interest_over_time_list,
                    'related_queries': related_queries_dict,
                    'related_topics': related_topics_dict
                }
            else:
                print(f"Failed to get trends data for '{category}'")
        except Exception as e:
            print(f"Error fetching trends data for '{category}': {e}")
    
    if all_trends_data:
        # Add metadata
        trends_data = {
            'metadata': {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'location': args.location,
                'timeframe': args.timeframe,
                'categories': args.categories
            },
            'data': all_trends_data
        }
        
        # Save trends data to JSON file
        output_path = os.path.join(output_dir, f'trends_data_{current_date}.json')
        with open(output_path, 'w') as f:
            json.dump(trends_data, f, indent=4)
        print(f"\nSaved trends data to {output_path}")
        
        # Create a symlink or copy to the standard filename
        standard_output_path = os.path.join(output_dir, 'trends_data.json')
        if os.path.exists(standard_output_path):
            os.remove(standard_output_path)
        
        # On Windows, we need to copy the file instead of creating a symlink
        import shutil
        shutil.copy2(output_path, standard_output_path)
        print(f"Created copy at {standard_output_path}")
        
        print("\nTrends analysis completed successfully!")
        print("Run the simulation to incorporate trends data: python run_simulation.py")
    else:
        print("\nNo trends data collected. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()