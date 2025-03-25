import pandas as pd

def display_thumbnail(df):
    """Displays the thumbnail URL from the 'thumbnail_url' column after filtering the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
    """
    if 'thumbnail_url' not in df.columns:
        print("Error: Column 'thumbnail_url' not found in the DataFrame.")
        return

    if df.empty:
        print("No data available after filtering.")
        return

    thumbnail_urls = df['thumbnail_url'].tolist()
    for thumbnail_url in thumbnail_urls:
        print(f"Opening thumbnail URL: {thumbnail_url}")
        # Code to open the thumbnail URL using an appropriate library

# Load DataFrame from CSV
df = pd.read_csv('gee_catalog.csv')

# Display DataFrame column names for clarity
print("Available columns:")
for col in df.columns:
    print(col)

# Get user input for filtering
while True:
    filter_column = input("Enter the column name to filter by (or 'q' to quit): ")
    if filter_column.lower() == 'q':
        break
    elif filter_column not in df.columns:
        print("Invalid column name. Please enter a valid column name.")
        continue
    else:
        filter_value = input(f"Enter the value to filter '{filter_column}' by: ")
        try:
            # Filter the DataFrame based on user input
            filtered_df = df[df[filter_column] == filter_value].copy()
            display_thumbnail(filtered_df)
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again or enter 'q' to quit.")

print("Done!")
