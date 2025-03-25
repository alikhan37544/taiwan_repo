import pandas as pd

import pandas as pd 
import numpy as np

df = pd.read_csv('gee_catalog.csv')

def display_thumbnail(df, column_name):
  """Displays the thumbnail URL from the specified column.

  Args:
      df (pandas.DataFrame): The DataFrame containing the data.
      column_name (str): The name of the column containing the thumbnail URLs.
  """

  if column_name not in df.columns:
    print(f"Error: Column '{column_name}' not found in the DataFrame.")
    return

  # Check if there are any non-null thumbnail URLs in the column
  if not df[column_name].notna().any():
    print("No thumbnail URLs found in the selected column.")
    return

  # Get the first non-null URL (you can modify this to get multiple or random)
  thumbnail_url = df[column_name].dropna().iloc[0]
  print(f"Opening thumbnail URL: {thumbnail_url}")

  # Open the thumbnail URL using an appropriate library based on your environment
  # You might need to install an additional library like webbrowser
  # import webbrowser
  # webbrowser.open(thumbnail_url)

# Display DataFrame column names for clarity
print("Available columns:")
for col in df.columns:
  print(col)

# Get user input for column selection (handle potential errors)
while True:
  column_name = input("Enter the column name containing thumbnail URLs (or 'q' to quit): ")
  if column_name.lower() == 'q':
    break
  try:
    display_thumbnail(df.copy(), column_name)  # Use a copy to avoid modifying original DataFrame
    break
  except Exception as e:
    print(f"An error occurred: {e}")
    print("Please try again or enter 'q' to quit.")

print("Done!")
