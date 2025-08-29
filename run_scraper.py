import os
import sys
import argparse
from datetime import datetime

def main():
    """Run the event data scraper to collect data from Eventbrite and Meetup."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Scrape event data from Eventbrite and Meetup')
    parser.add_argument('--source', type=str, choices=['eventbrite', 'meetup', 'all'], default='all',
                        help='Source to scrape data from (eventbrite, meetup, or all)')
    parser.add_argument('--location', type=str, default='New York',
                        help='Location to search for events')
    parser.add_argument('--categories', type=str, nargs='+', 
                        default=['music', 'business', 'food', 'arts', 'sports'],
                        help='Categories to search for events')
    parser.add_argument('--num_events', type=int, default=50,
                        help='Number of events to scrape per category')
    args = parser.parse_args()
    
    print(f"Starting Event Data Scraper...")
    print(f"Source: {args.source}")
    print(f"Location: {args.location}")
    print(f"Categories: {args.categories}")
    print(f"Number of events per category: {args.num_events}")
    
    # Add the project root directory to the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root)
    
    # Import custom modules
    try:
        from src.data.scraper import EventbriteScraper, MeetupScraper
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure you're running this script from the project root directory.")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(project_root, 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current date for filename
    current_date = datetime.now().strftime('%Y%m%d')
    
    # Scrape Eventbrite data
    if args.source in ['eventbrite', 'all']:
        print("\nScraping Eventbrite data...")
        eventbrite_scraper = EventbriteScraper()
        
        eventbrite_events = []
        for category in args.categories:
            print(f"Searching for {category} events in {args.location}...")
            category_events = eventbrite_scraper.search_events(
                location=args.location,
                category=category,
                num_events=args.num_events
            )
            
            if category_events:
                print(f"Found {len(category_events)} {category} events")
                eventbrite_events.extend(category_events)
            else:
                print(f"No {category} events found")
        
        if eventbrite_events:
            # Get detailed information for each event
            detailed_events = []
            for i, event in enumerate(eventbrite_events):
                print(f"Getting details for event {i+1}/{len(eventbrite_events)}: {event['title']}")
                event_details = eventbrite_scraper.get_event_details(event['url'])
                if event_details:
                    detailed_events.append({**event, **event_details})
            
            # Save data to CSV
            eventbrite_output_path = os.path.join(output_dir, f'eventbrite_events_{current_date}.csv')
            eventbrite_scraper.save_to_csv(detailed_events, eventbrite_output_path)
            print(f"Saved {len(detailed_events)} Eventbrite events to {eventbrite_output_path}")
            
            # Create a symlink or copy to the standard filename
            standard_eventbrite_path = os.path.join(output_dir, 'eventbrite_events.csv')
            if os.path.exists(standard_eventbrite_path):
                os.remove(standard_eventbrite_path)
            
            # On Windows, we need to copy the file instead of creating a symlink
            import shutil
            shutil.copy2(eventbrite_output_path, standard_eventbrite_path)
            print(f"Created copy at {standard_eventbrite_path}")
        else:
            print("No Eventbrite events found")
    
    # Scrape Meetup data
    if args.source in ['meetup', 'all']:
        print("\nScraping Meetup data...")
        meetup_scraper = MeetupScraper()
        
        meetup_events = []
        for category in args.categories:
            print(f"Searching for {category} events in {args.location}...")
            category_events = meetup_scraper.search_events(
                location=args.location,
                category=category,
                num_events=args.num_events
            )
            
            if category_events:
                print(f"Found {len(category_events)} {category} events")
                meetup_events.extend(category_events)
            else:
                print(f"No {category} events found")
        
        if meetup_events:
            # Get detailed information for each event
            detailed_events = []
            for i, event in enumerate(meetup_events):
                print(f"Getting details for event {i+1}/{len(meetup_events)}: {event['title']}")
                event_details = meetup_scraper.get_event_details(event['url'])
                if event_details:
                    detailed_events.append({**event, **event_details})
            
            # Save data to CSV
            meetup_output_path = os.path.join(output_dir, f'meetup_events_{current_date}.csv')
            meetup_scraper.save_to_csv(detailed_events, meetup_output_path)
            print(f"Saved {len(detailed_events)} Meetup events to {meetup_output_path}")
            
            # Create a symlink or copy to the standard filename
            standard_meetup_path = os.path.join(output_dir, 'meetup_events.csv')
            if os.path.exists(standard_meetup_path):
                os.remove(standard_meetup_path)
            
            # On Windows, we need to copy the file instead of creating a symlink
            import shutil
            shutil.copy2(meetup_output_path, standard_meetup_path)
            print(f"Created copy at {standard_meetup_path}")
        else:
            print("No Meetup events found")
    
    print("\nScraping completed successfully!")
    print("Run the preprocessor to prepare the data for analysis: python run_preprocessor.py")

if __name__ == "__main__":
    main()