from flask import Flask,request,jsonify, render_template
import pandas as pd
import numpy as np
import itertools
import requests
import flexpolyline as fp
import geopandas as gpd
from sklearn.neighbors import BallTree
import json
import os


app = Flask(__name__)
app.jinja_env.cache = {}

print("Template Folder:", app.template_folder)
print("Current Directory:", os.getcwd())

def AIroute():

    test_df_aligned = pd.read_csv(os.getcwd() + '\\test_data_with_pred.csv')




    # Create GeoDataFrame from test_df_aligned
    test_gdf = gpd.GeoDataFrame(
        test_df_aligned,
        geometry=gpd.points_from_xy(test_df_aligned.longitude, test_df_aligned.latitude),
        crs='EPSG:4326'
    )

    # Define the coordinates for the points you want to visit (must be within the boundry box)
    origin = (24.734047,46.755066)  # Start point
    point_coords_1 = (24.718416,46.778369)  # Point 1
    point_coords_2 = (24.701730,46.753650)  # Point 2
    point_coords_3 = (24.721690,46.751590)  # Point 3

    # Create a list of the waypoints (excluding the start point)
    waypoints = [point_coords_1, point_coords_2, point_coords_3]

    # Generate all possible orders for the waypoints
    waypoint_permutations = list(itertools.permutations(waypoints))

    # HERE API key
    API_KEY = 'sFR8fGF9tv_-oQTgOPCZe4hcUH4kurAUCLc-zRmxcsA'

    # Define the destination (since it's a round trip, destination is the origin)
    destination = origin

    # Prepare routes list with different waypoint orders
    routes_list = []
    for idx, permutation in enumerate(waypoint_permutations):
        # Build the full coordinate list
        coords = [origin] + list(permutation) + [destination]
        routes_list.append({
            'route_index': idx,
            'coordinates': coords
        })


    # Request routes from the HERE API
    for route in routes_list:
        try:
            # Construct the base parameters
            params = [
                ('transportMode', 'car'),
                ('origin', f'{origin[0]},{origin[1]}'),
                ('destination', f'{destination[0]},{destination[1]}'),
                ('apikey', API_KEY),
                ('return', 'polyline,summary')
            ]
            # Add via parameters without indices
            for waypoint in route['coordinates'][1:-1]:
                params.append(('via', f'{waypoint[0]},{waypoint[1]}'))

            # Send GET request
            response = requests.get('https://router.hereapi.com/v8/routes', params=params)

            if response.status_code == 200:
                route_data = response.json()
                route['route_data'] = route_data
            else:
                print(f"Error fetching route {route['route_index']}: {response.status_code} {response.text}")
        except Exception as e:
            print(f"Exception fetching route {route['route_index']}: {e}")




    for route in routes_list[2:]:
        route_data = route.get('route_data')
        print(f"\nRoute {route['route_index']} data:")
        print(json.dumps(route_data, indent=2))



    # Process each route
    # Initialize list to store route metrics
    route_metrics = []

    for route in routes_list:
        print(f"\nProcessing Route {route['route_index']}...")
        route_data = route.get('route_data')
        if not route_data:
            print(f"No route data for route {route['route_index']}")
            continue

        # Check if routes are available
        if not route_data.get('routes'):
            print(f"No routes found for route {route['route_index']}")
            continue

        # Extract and decode polylines from all sections
        try:
            coordinates = []
            for section in route_data['routes'][0]['sections']:
                polyline = section['polyline']
                # Decode the polyline using HERE's flexible polyline decoder
                section_coords = fp.decode(polyline)
                coordinates.extend(section_coords)
        except Exception as e:
            print(f"Error processing route {route['route_index']}: {e}")
            continue

        # Convert coordinates to DataFrame
        route_coords = pd.DataFrame(coordinates, columns=['lat', 'lng'])

        # Create GeoDataFrame for route points
        route_gdf = gpd.GeoDataFrame(
            route_coords,
            geometry=gpd.points_from_xy(route_coords.lng, route_coords.lat),
            crs='EPSG:4326'
        )

        # Prepare coordinate arrays for BallTree
        # Convert lat/lng to radians
        route_points_rad = np.deg2rad(route_gdf[['lat', 'lng']].values)
        test_points_rad = np.deg2rad(test_gdf[['latitude', 'longitude']].values)

        # Build BallTree using haversine metric
        tree = BallTree(test_points_rad, metric='haversine')

        # Query for the nearest neighbor
        distances, indices = tree.query(route_points_rad, k=1)

        # Get the matched points
        matched_points = test_gdf.iloc[indices.flatten()].reset_index(drop=True)

        # Add predicted jam factors to route_gdf
        route_gdf['predicted_jamFactor'] = matched_points['predicted_jamFactor'].values

        # Handle any missing values
        route_gdf.fillna({'predicted_jamFactor': route_gdf['predicted_jamFactor'].mean()}, inplace=True)

        # Calculate total and average predicted jam factors
        total_jam_factor = route_gdf['predicted_jamFactor'].sum()
        average_jam_factor = route_gdf['predicted_jamFactor'].mean()

        print(f"Route {route['route_index']} -  Predicted Jam Factor: {average_jam_factor}")

        # Store route metrics
        route_metrics.append({
            'route_index': route['route_index'],
            'total_jam_factor': total_jam_factor,
            'average_jam_factor': average_jam_factor,
            'route_data': route_data,
            'route_gdf': route_gdf,
            'coordinates': route['coordinates']
        })

    # Determine the best route based on the minimum total jam factor
    if route_metrics:
        best_route = min(route_metrics, key=lambda x: x['average_jam_factor'])
        best_route_data = best_route['route_data']
        print(f"\nBest Route Index: {best_route['route_index']}")
        print(f"Waypoints Order: {best_route['coordinates'][1:-1]}")
        print(f"Predicted Jam Factor: {best_route['average_jam_factor']}")
        
        coordinates = []
        for section in best_route_data['routes'][0]['sections']:
            polyline = section['polyline']
            # Decode the polyline using HERE's flexible polyline decoder
            section_coords = fp.decode(polyline)
            coordinates.extend(section_coords)  

        realCords = []
        for section in best_route_data['routes'][0]['sections']:
            location = section['arrival']['place']['location']
            latLon = [[location['lat'], location['lng']]]
            realCords.extend(latLon)
     
        return realCords    
    else:
        print("No routes were successfully fetched.")
        return []
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/best-route')
def bestRoute():
    return jsonify(AIroute()), 200



if __name__ == '__main__':
    app.run(debug=True)