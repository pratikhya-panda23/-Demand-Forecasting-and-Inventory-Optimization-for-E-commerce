import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('event_scraper')

class EventbriteScraper:
    """Scraper for Eventbrite events and ticket information"""
    
    def __init__(self, headless=True):
        self.base_url = "https://www.eventbrite.com"
        self.headless = headless
        self.driver = None
        
    def _initialize_driver(self):
        """Initialize Selenium WebDriver"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise
    
    def search_events(self, location, category=None, date_range=None, max_pages=3):
        """Search for events based on location, category, and date range"""
        if not self.driver:
            self._initialize_driver()
            
        events = []
        search_url = f"{self.base_url}/d/{location}/events/"
        if category:
            search_url += f"{category}/"
            
        logger.info(f"Searching events at: {search_url}")
        self.driver.get(search_url)
        
        # Wait for events to load
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".search-event-card-square"))
            )
        except Exception as e:
            logger.warning(f"Timeout waiting for events to load: {e}")
        
        # Scrape events from multiple pages
        for page in range(max_pages):
            logger.info(f"Scraping page {page+1} of {max_pages}")
            
            # Extract event cards
            event_cards = self.driver.find_elements(By.CSS_SELECTOR, ".search-event-card-square")
            
            for card in event_cards:
                try:
                    event_data = {}
                    
                    # Extract event details
                    event_data['title'] = card.find_element(By.CSS_SELECTOR, ".event-card__title").text
                    event_data['url'] = card.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                    
                    # Extract date and time if available
                    try:
                        event_data['date'] = card.find_element(By.CSS_SELECTOR, ".event-card__date").text
                    except:
                        event_data['date'] = "Not specified"
                    
                    # Extract price if available
                    try:
                        event_data['price'] = card.find_element(By.CSS_SELECTOR, ".event-card__price").text
                    except:
                        event_data['price'] = "Not specified"
                    
                    events.append(event_data)
                except Exception as e:
                    logger.error(f"Error extracting event data: {e}")
            
            # Go to next page if available
            try:
                next_button = self.driver.find_element(By.CSS_SELECTOR, ".pagination__next")
                if "disabled" not in next_button.get_attribute("class"):
                    next_button.click()
                    time.sleep(random.uniform(2, 4))  # Random delay to avoid detection
                    
                    # Wait for next page to load
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".search-event-card-square"))
                    )
                else:
                    logger.info("No more pages available")
                    break
            except Exception as e:
                logger.warning(f"Could not navigate to next page: {e}")
                break
        
        logger.info(f"Found {len(events)} events")
        return events
    
    def get_event_details(self, event_url):
        """Get detailed information about a specific event"""
        if not self.driver:
            self._initialize_driver()
            
        logger.info(f"Getting details for event: {event_url}")
        self.driver.get(event_url)
        
        # Wait for event details to load
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".event-title"))
            )
        except Exception as e:
            logger.warning(f"Timeout waiting for event details to load: {e}")
        
        event_details = {}
        
        try:
            # Basic event information
            event_details['title'] = self.driver.find_element(By.CSS_SELECTOR, ".event-title").text
            event_details['organizer'] = self.driver.find_element(By.CSS_SELECTOR, ".organizer-name").text
            
            # Ticket information
            ticket_elements = self.driver.find_elements(By.CSS_SELECTOR, ".ticket-box")
            tickets = []
            
            for ticket_elem in ticket_elements:
                ticket = {}
                try:
                    ticket['name'] = ticket_elem.find_element(By.CSS_SELECTOR, ".ticket-name").text
                    ticket['price'] = ticket_elem.find_element(By.CSS_SELECTOR, ".ticket-price").text
                    ticket['availability'] = ticket_elem.find_element(By.CSS_SELECTOR, ".ticket-quantity-info").text
                    tickets.append(ticket)
                except Exception as e:
                    logger.error(f"Error extracting ticket data: {e}")
            
            event_details['tickets'] = tickets
            
            # Additional event information
            try:
                event_details['description'] = self.driver.find_element(By.CSS_SELECTOR, ".event-description").text
            except:
                event_details['description'] = "No description available"
                
            try:
                event_details['location'] = self.driver.find_element(By.CSS_SELECTOR, ".event-location").text
            except:
                event_details['location'] = "Location not specified"
                
            try:
                event_details['date_time'] = self.driver.find_element(By.CSS_SELECTOR, ".event-date-time").text
            except:
                event_details['date_time'] = "Date and time not specified"
                
        except Exception as e:
            logger.error(f"Error extracting event details: {e}")
        
        return event_details
    
    def close(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
            logger.info("WebDriver closed")


class MeetupScraper:
    """Scraper for Meetup events and ticket information"""
    
    def __init__(self, headless=True):
        self.base_url = "https://www.meetup.com"
        self.headless = headless
        self.driver = None
        
    def _initialize_driver(self):
        """Initialize Selenium WebDriver"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise
    
    def search_events(self, location, category=None, max_pages=3):
        """Search for events based on location and category"""
        if not self.driver:
            self._initialize_driver()
            
        events = []
        search_url = f"{self.base_url}/find/?location={location}"
        if category:
            search_url += f"&category={category}"
            
        logger.info(f"Searching events at: {search_url}")
        self.driver.get(search_url)
        
        # Wait for events to load
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".event-card"))
            )
        except Exception as e:
            logger.warning(f"Timeout waiting for events to load: {e}")
        
        # Scrape events from multiple pages
        for page in range(max_pages):
            logger.info(f"Scraping page {page+1} of {max_pages}")
            
            # Extract event cards
            event_cards = self.driver.find_elements(By.CSS_SELECTOR, ".event-card")
            
            for card in event_cards:
                try:
                    event_data = {}
                    
                    # Extract event details
                    event_data['title'] = card.find_element(By.CSS_SELECTOR, ".event-card-title").text
                    event_data['url'] = card.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                    
                    # Extract date and time if available
                    try:
                        event_data['date'] = card.find_element(By.CSS_SELECTOR, ".event-card-date").text
                    except:
                        event_data['date'] = "Not specified"
                    
                    # Extract group name if available
                    try:
                        event_data['group'] = card.find_element(By.CSS_SELECTOR, ".event-card-group").text
                    except:
                        event_data['group'] = "Not specified"
                    
                    events.append(event_data)
                except Exception as e:
                    logger.error(f"Error extracting event data: {e}")
            
            # Go to next page if available
            try:
                next_button = self.driver.find_element(By.CSS_SELECTOR, ".pagination-next")
                if not next_button.get_attribute("disabled"):
                    next_button.click()
                    time.sleep(random.uniform(2, 4))  # Random delay to avoid detection
                    
                    # Wait for next page to load
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".event-card"))
                    )
                else:
                    logger.info("No more pages available")
                    break
            except Exception as e:
                logger.warning(f"Could not navigate to next page: {e}")
                break
        
        logger.info(f"Found {len(events)} events")
        return events
    
    def get_event_details(self, event_url):
        """Get detailed information about a specific event"""
        if not self.driver:
            self._initialize_driver()
            
        logger.info(f"Getting details for event: {event_url}")
        self.driver.get(event_url)
        
        # Wait for event details to load
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".event-title"))
            )
        except Exception as e:
            logger.warning(f"Timeout waiting for event details to load: {e}")
        
        event_details = {}
        
        try:
            # Basic event information
            event_details['title'] = self.driver.find_element(By.CSS_SELECTOR, ".event-title").text
            event_details['group'] = self.driver.find_element(By.CSS_SELECTOR, ".group-name").text
            
            # Event details
            try:
                event_details['description'] = self.driver.find_element(By.CSS_SELECTOR, ".event-description").text
            except:
                event_details['description'] = "No description available"
                
            try:
                event_details['location'] = self.driver.find_element(By.CSS_SELECTOR, ".event-location").text
            except:
                event_details['location'] = "Location not specified"
                
            try:
                event_details['date_time'] = self.driver.find_element(By.CSS_SELECTOR, ".event-date-time").text
            except:
                event_details['date_time'] = "Date and time not specified"
                
            # Attendee information
            try:
                event_details['attendees'] = self.driver.find_element(By.CSS_SELECTOR, ".attendee-count").text
            except:
                event_details['attendees'] = "Attendee count not available"
                
        except Exception as e:
            logger.error(f"Error extracting event details: {e}")
        
        return event_details
    
    def close(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
            logger.info("WebDriver closed")


def save_to_csv(data, filename):
    """Save scraped data to CSV file"""
    try:
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Data saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving data to CSV: {e}")


if __name__ == "__main__":
    # Example usage
    eventbrite_scraper = EventbriteScraper(headless=True)
    try:
        # Search for events in New York
        events = eventbrite_scraper.search_events(location="new-york", category="music", max_pages=2)
        
        # Save events to CSV
        save_to_csv(events, "../../data/raw/eventbrite_events.csv")
        
        # Get details for the first event
        if events:
            event_details = eventbrite_scraper.get_event_details(events[0]['url'])
            print(f"Event details: {event_details}")
    finally:
        eventbrite_scraper.close()
    
    # Example with Meetup
    meetup_scraper = MeetupScraper(headless=True)
    try:
        # Search for events in New York
        events = meetup_scraper.search_events(location="new-york", category="tech", max_pages=2)
        
        # Save events to CSV
        save_to_csv(events, "../../data/raw/meetup_events.csv")
        
        # Get details for the first event
        if events:
            event_details = meetup_scraper.get_event_details(events[0]['url'])
            print(f"Event details: {event_details}")
    finally:
        meetup_scraper.close()