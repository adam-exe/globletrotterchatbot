import folium
import requests
import streamlit as st
from langchain.tools import tool
from streamlit_folium import folium_static
from io import StringIO

# Function to get coordinates using OpenCage Geocoder
def get_coordinates(location, api_key):
    """Fetches coordinates for a given location (city or country) using OpenCage Geocoder."""
    url = f"https://api.opencagedata.com/geocode/v1/json?q={location}&key={api_key}&limit=1"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Error: Received status code {response.status_code} for location '{location}'")
        data = response.json()
        if not data['results']:
            raise ValueError(f"Error: No data found for location '{location}'")
        return [float(data['results'][0]['geometry']['lat']), float(data['results'][0]['geometry']['lng'])]
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Request failed: {e}")
    except ValueError as ve:
        raise ValueError(f"Error: {ve}")

# Function to create and display the map in Streamlit
def mapper(locations, api_key):
    """Creates a map with markers for given locations."""
    map_center = [20, 0]  # Center of the world
    my_map = folium.Map(location=map_center, zoom_start=2)
    coordinates = []

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

    return my_map

def generate_map(coordinates_list, locations):
    """Generates an interactive map and provides a download link."""
    api_key = "c819d2cf3ada4f94ad7fcb694f67deed"
    my_map = mapper(locations, api_key)
    folium_static(my_map)

    # Convert map to HTML
    map_html = my_map._repr_html_()
    
    # Create an in-memory buffer for the HTML
    html_buffer = StringIO()
    html_buffer.write(map_html)
    html_buffer.seek(0)
    
    # Add download button for the HTML file
    st.download_button(
        label="Download Map as HTML",
        data=html_buffer.getvalue(),
        file_name="map.html",
        mime="text/html"
    )

@tool
def get_locations(query: str) -> str:
    """Draw a map by extracting locations from a query and map them with numbered markers in Streamlit."""
    locations = query.split(",")

    if not locations:
        return "No locations were found in the query. Please try again."

    # Generate and display the map
    generate_map(None, [location.strip() for location in locations])

    




