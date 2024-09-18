import folium
from streamlit_folium import st_folium
import requests
import streamlit as st
from langchain.tools import tool

def generate_map(coordinates_list, locations):
    map_center = [20, 0]
    my_map = folium.Map(location=map_center, zoom_start=2)
    create_map(coordinates_list, locations, my_map)
    return my_map

def get_coordinates(location):
    api_key = "c819d2cf3ada4f94ad7fcb694f67deed"
    url = f"https://api.opencagedata.com/geocode/v1/json?q={location}&key={api_key}&limit=1"

    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Error: Received status code {response.status_code} for location '{location}'")

        data = response.json()

        if not data or not data['results']:
            raise ValueError(f"Error: No data found for location '{location}'")

        # Extract coordinates
        coordinates = data['results'][0]['geometry']
        return [coordinates['lat'], coordinates['lng']]

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
    """Draw a map by extracting locations from a query and map them with numbered markers in Streamlit."""
    locations = query.split(",")

    if not locations:
        return "No locations were found in the query. Please try again."

    coordinates_list = []

    for location in locations:
        try:
            coordinates = get_coordinates(location.strip())
            coordinates_list.append(coordinates)
        except ValueError as e:
            return str(e)

    # Generate and display the map
    my_map = generate_map(coordinates_list, locations)
    folium_static(my_map)




