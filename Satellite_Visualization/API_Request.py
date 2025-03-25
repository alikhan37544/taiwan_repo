import pandas as pd
import requests
import matplotlib.pyplot as plt

# Function to fetch Landsat imagery using NASA's API
def fetch_landsat_image(latitude, longitude, date):
    # NASA Landsat API base URL
    base_url = "https://api.nasa.gov/planetary/earth/imagery"
    
    # API key (you need to sign up for a free API key on NASA's website)
    api_key = "vqwrQNBtFErq5ChPWIQdJ2BjMckMFgzyXSM4m6gO"
    
    # Construct the API request URL
    request_url = f"{base_url}?lon={longitude}&lat={latitude}&date={date}&dim=0.1&api_key={api_key}"
    
    try:
        # Send GET request to NASA's API
        response = requests.get(request_url)
        response.raise_for_status()  # Raise an exception for any HTTP error
        
        # Plot the image
        image_data = plt.imread(response.content)
        plt.imshow(image_data)
        plt.axis('off')
        plt.title(f"Landsat Image: {date}")
        plt.show()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Landsat image: {e}")

# Example row from CSV file
example_row = {
    'latitude': 40.7128,
    'longitude': -74.0060,
    'date': '2014-02-01',
    'disastertype': 'Hurricane',
    'location': 'New York City'
}

# Extract information from the example row
latitude = example_row['latitude']
longitude = example_row['longitude']
date = example_row['date']
disaster_type = example_row['disastertype']
place = example_row['location']

# Fetch Landsat image
fetch_landsat_image(latitude, longitude, date)
