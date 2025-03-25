import pandas as pd
import requests
import matplotlib.pyplot as plt
import random
import io

# Function to fetch Landsat imagery using NASA's API
def fetch_landsat_image(latitude, longitude, date):
    print("Fetching Landsat image...")
    # NASA Landsat API base URL
    base_url = "https://api.nasa.gov/planetary/earth/imagery"
    
    # API key (you need to sign up for a free API key on NASA's website)
    api_key = "vqwrQNBtFErq5ChPWIQdJ2BjMckMFgzyXSM4m6gO"
    
    # Construct the API request URL
    request_url = f"{base_url}?lon={longitude}&lat={latitude}&date={date}-01-01&dim=0.1&api_key={api_key}"
    print("API request URL:", request_url)
    
    try:
        # Send GET request to NASA's API
        response = requests.get(request_url)
        response.raise_for_status()  # Raise an exception for any HTTP error
        print("API response status code:", response.status_code)
        
        # Decode the response content as a byte stream
        image_data = io.BytesIO(response.content)
        
        # Plot the image
        plt.imshow(plt.imread(image_data))
        plt.axis('off')
        plt.title(f"Landsat Image: {date}")
        plt.show()
        print("Landsat image fetched successfully!")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Landsat image: {e}")

# Load the dataset
print("Loading dataset...")
df = pd.read_csv("/home/akz-portable/programming/NASA-Hackathon/merged.csv")  # Update with your dataset filename

# Randomly choose an index
random_index = random.randint(0, len(df) - 1)
print("Randomly chosen index:", random_index)

# Extract information from the randomly chosen row
row = df.iloc[random_index]
latitude = row['latitude']
longitude = row['longitude']
date = row['year']  # Assuming the 'year' column contains the year information
disaster_type = row['disastertype']
place = row['location']

print("Latitude:", latitude)
print("Longitude:", longitude)
print("Date:", date)
print("Disaster Type:", disaster_type)
print("Place:", place)

# Fetch Landsat image
fetch_landsat_image(latitude, longitude, date)
