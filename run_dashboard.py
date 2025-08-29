import os
import sys
from src.dashboard.dashboard import create_dashboard

def main():
    """Run the dynamic pricing dashboard application."""
    print("Starting Dynamic Pricing Dashboard...")
    
    # Create and run the dashboard
    app = create_dashboard()
    
    # Run the app
    app.run_server(debug=True, port=8050)
    
    print("Dashboard is running at http://localhost:8050")

if __name__ == "__main__":
    # Add the project root directory to the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root)
    
    main()