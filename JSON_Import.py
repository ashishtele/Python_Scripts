import json
import pandas as pd
import streamlit as st
from datetime import datetime


# Set page layout
st.set_page_config(
    page_title="Travel",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache(allow_output_mutation=True)
def read_json(json_file):
    # read the json file
    with open(json_file, 'r') as f:
        data = json.loads(f.read())
    # Flattern dataframe
    df = pd.json_normalize(data, record_path =['locations'])
    # create a dataframe from the json file
    #df = pd.DataFrame(data)
    # return the dataframe
    return df

# create a function to read timestampms as a column and convert to datetime 
def convert_timestampms(df):
    # convert the timestampms column to datetime
    df['timestampMs'] = pd.to_datetime(df['timestampMs'], unit='ms')
    df['date'] = df['timestampMs'].dt.date
    df['Time'] = df['timestampMs'].dt.time
    # return the dataframe
    return df


# create a function to convert latitudeE7 and longitudeE7 to decimal degrees
def convert_latlong(df):
    # convert the latitudeE7 and longitudeE7 to decimal degrees
    df['latitude'] = df['latitudeE7'] / 10000000
    df['longitude'] = df['longitudeE7'] / 10000000
    # return the dataframe
    return df

# Variable/ path declaration
json_file = 'C:/Users/ashis/OneDrive/Desktop/BI/Takeout/Location History/'
json_file = json_file + 'Location_History.json'

df = read_json(json_file)
df = convert_timestampms(df)
df = convert_latlong(df)
print(f'The shape of the imported data is {df.shape}')
print(df.head())

# filtering the data for WIFI

df = df[df['source'] != 'WIFI']

# Keeping required columns
columns = ['date', 'Time', 'latitude', 'longitude', 'accuracy', 'source', 'velocity', 'altitude', 'platform', 'platformType']
df_2016 = df.loc[df['date'] > pd.to_datetime('2016-01-01'), columns].reset_index(drop=True)

print(f'The shape of the imported data is {df_2016.shape}')

st.title("🌍 Travels Exploration")

# Calculate the timerange for the slider
min_ts = min(df_2016["date"])
max_ts = max(df_2016["date"])

st.sidebar.subheader("Inputs")
min_selection, max_selection = st.sidebar.slider(
    "Timeline", min_value=min_ts, max_value=max_ts, value=[min_ts, max_ts]
)

# Toggles for the feature selection in sidebar
show_detailed_months = st.sidebar.checkbox("Show Detailed Split per Year")
show_code = st.sidebar.checkbox("Show Code")

# Filter Data based on selection
st.write(f"Filtering between {min_selection} & {max_selection}")
travel_data = df_2016[
    (df_2016["date"] >= min_selection) & (df_2016["date"] <= max_selection)
]
st.write(f"Data Points: {len(df)}")

# Plot the GPS coordinates on the map
st.map(travel_data)