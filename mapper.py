import folium
from streamlit_folium import st_folium
import requests
import streamlit as st
from langchain.tools import tool
 
@st.cache_data
def generate_map(coordinates_list, locations):
    map_center = [20, 0]
    my_map = folium.Map(location=map_center, zoom_start=2)
    create_map(coordinates_list, locations, my_map)
    return my_map
 
def get_coordinates(location):

    api_key = "c819d2cf3ada4f94ad7fcb694f67deed" 
    # Fetches coordinates for a given location (city or country) using Nominatim.
    url = f"https://api.opencagedata.com/geocode/v1/json?q={location}&key={api_key}&limit=1"
   
    try:
        # Send the request
        response = requests.get(url)
 
        # Check if the response is valid (status code 200 means success)
        if response.status_code != 200:
            raise ValueError(f"Error: Received status code {response.status_code} for location '{location}'")
 
        # Try to parse the response as JSON
        data = response.json()
       
        # Check if data is empty
        if not data:
            raise ValueError(f"Error: No data found for location '{location}'")
       
        # Extract coordinates
        return [float(data[0]['lat']), float(data[0]['lon'])]
 
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Request failed: {e}")
 
    except ValueError as ve:
        raise ValueError(f"Error: {ve}")
 
def create_map(coordinates_list, locations, my_map):
    # Add route to the map
    if coordinates_list:
        folium.PolyLine(coordinates_list, color="red", weight=2.5, opacity=1).add_to(my_map)
 
    # Add numbered markers for each valid location
    for i, coord in enumerate(coordinates_list):
        folium.Marker(
            location=coord,
            popup=f"{locations[i]}",
            icon=folium.DivIcon(html=f"""<div style="font-size: 12px; color: black"><b>{i+1}</b></div>""")
        ).add_to(my_map)
 
    try:
        st_folium(my_map, width=700, height=500)
    except:
        return st.error("Error drawing map.")
 
@tool
def get_locations(query: str) -> str:
    """Draw a map by extracting locations from a query and map them with numbered markers in Streamlit.
    """
    locations = query.split(",")
 
    if not locations:
        return "No locations were found in the query. Please try again."
   
    # Initialize the map with a default location (center of the world)
    map_center = [20, 0]
    my_map = folium.Map(location=map_center, zoom_start=2)
 
    # Store coordinates as list
    coordinates_list = []
 
    # Loop through each location and fetch the coordinates
    for location in locations:
        coordinates = get_coordinates(location)
        if coordinates:
            coordinates_list.append(coordinates)
        else:
            st.error(f"Coordinates for '{location}' could not be found.")
            break
   
    # Generate and cache the map
    my_map = generate_map(coordinates_list, locations)
   
    st_folium(my_map, width=700, height=500)


