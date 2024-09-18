import streamlit as st
import requests
import folium
from streamlit_folium import folium_static

# Function to get coordinates using OpenCage Geocoder
def get_coordinates(location, api_key):
    """Fetches coordinates for a given location (city or country) using OpenCage Geocoder."""
    url = f"https://api.opencagedata.com/geocode/v1/json?q={location}&key={api_key}&limit=1"
   
    try:
        # Send the request
        response = requests.get(url)
 
        # Check if the response is valid (status code 200 means success)
        if response.status_code != 200:
            raise ValueError(f"Error: Received status code {response.status_code} for location '{location}'")
 
        # Try to parse the response as JSON
        data = response.json()
       
        # Check if data contains results
        if not data['results']:
            raise ValueError(f"Error: No data found for location '{location}'")
       
        # Extract coordinates
        return [float(data['results'][0]['geometry']['lat']), float(data['results'][0]['geometry']['lng'])]
 
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Request failed: {e}")
 
    except ValueError as ve:
        raise ValueError(f"Error: {ve}")

# Function to create and display the map in Streamlit
def mapper(locations, api_key):
    # Create a base map centered around a default location
    map_center = [20, 0]  # Center of the world
    my_map = folium.Map(location=map_center, zoom_start=2)
 
    coordinates = []
 
    # Get coordinates for the provided locations, handle errors individually
    for location in locations:
        try:
            coord = get_coordinates(location, api_key)
            coordinates.append(coord)
        except ValueError as e:
            st.error(e)
            continue  # Skip to the next location if there’s an error
 
    # Add the polyline (the route) if there are valid coordinates
    if coordinates:
        folium.PolyLine(coordinates, color="red", weight=2.5, opacity=1).add_to(my_map)
 
    # Add numbered markers for each valid location
    for i, coord in enumerate(coordinates):
        folium.Marker(
            location=coord,
            popup=f"{locations[i]}",
            icon=folium.DivIcon(html=f"""<div style="font-size: 12px; color: black"><b>{i+1}</b></div>""")
        ).add_to(my_map)
    
    # Display the map in Streamlit
    folium_static(my_map)

