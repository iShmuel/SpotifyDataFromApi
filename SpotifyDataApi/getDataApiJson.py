import os
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sqlalchemy import create_engine
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

# Spotify API credentials
client_id = '5dec95d9975d4511ab3c764b8b896646'
client_secret = '21a2a02d661b4ccc98a08b6d145bb45f'

# Set up Spotify API client
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Fetch multiple tracks
track_ids = ['3n3Ppam7vgaVa1iaRUc9Lp', '7ouMYWpwJ422jRcDASZB7P', '4VqPOruhp5EdPBeR92t6lQ', '2takcwOaAZWiXQijPHIx7B']
tracks_data = []

for track_id in track_ids:
    try:
        track = sp.track(track_id)
        tracks_data.append(track)
    except spotipy.SpotifyException as e:
        print(f"Error fetching track {track_id}: {e}")

if not tracks_data:
    print("No valid track data found. Exiting.")
    exit()

# Normalize and flatten the JSON data
tracks_df = pd.json_normalize(tracks_data)

# Convert dictionaries and lists to JSON strings
def convert_to_json(value):
    if isinstance(value, dict) or isinstance(value, list):
        return json.dumps(value)
    return value

# Apply conversion to all columns
for col in tracks_df.columns:
    tracks_df[col] = tracks_df[col].apply(convert_to_json)

# Extract relevant fields from dictionaries and lists
tracks_df['album_type'] = tracks_df['album.album_type']
tracks_df['album_name'] = tracks_df['album.name']
tracks_df['artist_names'] = tracks_df['album.artists'].apply(
    lambda x: ', '.join([artist['name'] for artist in json.loads(x)] if isinstance(x, str) else [artist['name'] for artist in x])
)

# Drop columns if they exist
columns_to_drop = ['album', 'external_ids', 'external_urls']
existing_columns_to_drop = [col for col in columns_to_drop if col in tracks_df.columns]

# Drop existing columns
tracks_flat = tracks_df.drop(columns=existing_columns_to_drop)

# Connect to the PostgreSQL database
engine = create_engine('postgresql://postgres:273322@localhost/spotify_db')

# Insert data into the 'tracks' table
tracks_flat.to_sql('tracks', engine, if_exists='replace', index=False)

# Basic Analysis and Visualization
# Plot distribution of track popularity
plt.figure(figsize=(10, 6))
sns.histplot(tracks_flat['popularity'].astype(float), bins=20, kde=True)
plt.title('Track Popularity Distribution')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.show()

# Plot popularity vs. track duration
plt.figure(figsize=(10, 6))
sns.scatterplot(x='duration_ms', y='popularity', data=tracks_flat)
plt.title('Popularity vs. Track Duration')
plt.xlabel('Duration (ms)')
plt.ylabel('Popularity')
plt.show()

# Feature Engineering
# Add a feature for track duration in minutes
tracks_flat['duration_min'] = tracks_flat['duration_ms'].astype(float) / 60000

# Convert album_type to a categorical variable
tracks_flat['album_type'] = tracks_flat['album_type'].astype('category').cat.codes

# Prepare Data for Machine Learning
# Define features and target variable
features = ['duration_min', 'album_type']
target = 'popularity'

X = tracks_flat[features]
y = tracks_flat[target].astype(float)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Visualize predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Popularity')
plt.ylabel('Predicted Popularity')
plt.title('Actual vs. Predicted Popularity')
plt.show()

# Advanced Analysis
# Apply KMeans clustering
X_clustering = tracks_flat[features]
kmeans = KMeans(n_clusters=3, random_state=42)
tracks_flat['cluster'] = kmeans.fit_predict(X_clustering)

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='duration_min', y='popularity', hue='cluster', data=tracks_flat, palette='viridis')
plt.title('Track Clusters')
plt.xlabel('Duration (min)')
plt.ylabel('Popularity')
plt.show()

# Generating a summary report
summary = tracks_flat[['name', 'popularity', 'duration_ms']].describe()
summary.to_csv('tracks_summary_report.csv')

print("Data processing complete and summary report generated.")
