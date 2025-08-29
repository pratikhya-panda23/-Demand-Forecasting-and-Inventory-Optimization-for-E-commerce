import os
import sys
import argparse
from datetime import datetime

def main():
    """Run the event data preprocessor to clean and prepare data for analysis."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Preprocess event data from Eventbrite and Meetup')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Directory containing input CSV files (default: data/raw)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save processed data (default: data/processed)')
    parser.add_argument('--eventbrite_file', type=str, default=None,
                        help='Eventbrite CSV file name (default: eventbrite_events.csv)')
    parser.add_argument('--meetup_file', type=str, default=None,
                        help='Meetup CSV file name (default: meetup_events.csv)')
    args = parser.parse_args()
    
    print(f"Starting Event Data Preprocessor...")
    
    # Add the project root directory to the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root)
    
    # Set default directories if not provided
    input_dir = args.input_dir if args.input_dir else os.path.join(project_root, 'data', 'raw')
    output_dir = args.output_dir if args.output_dir else os.path.join(project_root, 'data', 'processed')
    
    # Set default filenames if not provided
    eventbrite_file = args.eventbrite_file if args.eventbrite_file else 'eventbrite_events.csv'
    meetup_file = args.meetup_file if args.meetup_file else 'meetup_events.csv'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Import custom modules
    try:
        from src.data.preprocessor import EventDataPreprocessor
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure you're running this script from the project root directory.")
        return
    
    # Initialize preprocessor
    preprocessor = EventDataPreprocessor()
    
    # Check if input files exist
    eventbrite_path = os.path.join(input_dir, eventbrite_file)
    meetup_path = os.path.join(input_dir, meetup_file)
    
    eventbrite_exists = os.path.exists(eventbrite_path)
    meetup_exists = os.path.exists(meetup_path)
    
    if not eventbrite_exists and not meetup_exists:
        print(f"No input files found in {input_dir}")
        print(f"Expected files: {eventbrite_file}, {meetup_file}")
        print("Please run the scraper first: python run_scraper.py")
        return
    
    # Process Eventbrite data
    if eventbrite_exists:
        print(f"\nProcessing Eventbrite data from {eventbrite_path}...")
        eventbrite_df = preprocessor.load_data(eventbrite_path)
        print(f"Loaded {len(eventbrite_df)} events from Eventbrite")
        
        eventbrite_df = preprocessor.preprocess_eventbrite_data(eventbrite_df)
        print(f"Preprocessed Eventbrite data: {len(eventbrite_df)} events with {len(eventbrite_df.columns)} features")
    else:
        print(f"Eventbrite data file not found: {eventbrite_path}")
        eventbrite_df = None
    
    # Process Meetup data
    if meetup_exists:
        print(f"\nProcessing Meetup data from {meetup_path}...")
        meetup_df = preprocessor.load_data(meetup_path)
        print(f"Loaded {len(meetup_df)} events from Meetup")
        
        meetup_df = preprocessor.preprocess_meetup_data(meetup_df)
        print(f"Preprocessed Meetup data: {len(meetup_df)} events with {len(meetup_df.columns)} features")
    else:
        print(f"Meetup data file not found: {meetup_path}")
        meetup_df = None
    
    # Merge data if both sources are available
    if eventbrite_df is not None and meetup_df is not None:
        print("\nMerging data from both sources...")
        events_df = preprocessor.merge_data([eventbrite_df, meetup_df])
        print(f"Merged data: {len(events_df)} events")
    elif eventbrite_df is not None:
        events_df = eventbrite_df
    elif meetup_df is not None:
        events_df = meetup_df
    else:
        print("No data to process")
        return
    
    # Add features
    print("\nAdding features...")
    events_df = preprocessor.add_features(events_df)
    print(f"Added features: {len(events_df)} events with {len(events_df.columns)} features")
    
    # Get current date for filename
    current_date = datetime.now().strftime('%Y%m%d')
    
    # Save processed data
    output_path = os.path.join(output_dir, f'processed_events_{current_date}.csv')
    events_df.to_csv(output_path, index=False)
    print(f"\nSaved processed data to {output_path}")
    
    # Create a symlink or copy to the standard filename
    standard_output_path = os.path.join(output_dir, 'processed_events.csv')
    if os.path.exists(standard_output_path):
        os.remove(standard_output_path)
    
    # On Windows, we need to copy the file instead of creating a symlink
    import shutil
    shutil.copy2(output_path, standard_output_path)
    print(f"Created copy at {standard_output_path}")
    
    print("\nPreprocessing completed successfully!")
    print("Run the simulation to analyze pricing strategies: python run_simulation.py")

if __name__ == "__main__":
    main()