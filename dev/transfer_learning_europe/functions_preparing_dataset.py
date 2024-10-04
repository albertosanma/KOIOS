import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
import cartopy
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature
from shapely.geometry import Point

def set_column_formats(df, format_dict):
    """
    Function to format DataFrame columns based on a dictionary of formats.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to format
    format_dict (dict): A dictionary where keys are column names and values are formats (e.g., 'int8', 'float32', formatting functions)
    
    Returns:
    pd.DataFrame: DataFrame with formatted columns
    """
    for column, fmt in format_dict.items():
        # Check if the format is a numpy data type for int or float
        if fmt in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
            # Convert to the corresponding numpy data type
            df[column] = df[column].astype(fmt)
        elif fmt == 'int':
            df[column] = df[column].astype(int)  # Default integer conversion
        elif fmt == 'float':
            df[column] = df[column].astype(float)  # Default float conversion
        elif callable(fmt):
            df[column] = df[column].apply(fmt)  # Apply callable function if given
        else:
            # If a specific format string (like "%.2f") is passed
            df[column] = df[column].map(lambda x: fmt % x if pd.notnull(x) else x)
    
    return df

def get_unique_lat_lon(df, lat_col='latitude', lon_col='longitude'):
    # Get unique combinations of latitude and longitude
    unique_lat_lon = df[[lat_col, lon_col]].drop_duplicates()
    return unique_lat_lon


# Function to check if a single point (latitude, longitude) is in the sea
def is_point_in_sea(lat, lon):
    land = cfeature.NaturalEarthFeature('physical', 'land', '110m')
    point = Point(lon, lat)

    # Load land geometries
    land_geometries = list(land.geometries())

    # Check if the point is on land by seeing if it is contained within any land polygon
    for geom in land_geometries:
        if geom.contains(point):
            return False  # It's on land
    return True  # It's in the sea

# Function to filter out all rows in the DataFrame where the point is in the sea
def filter_out_sea_points(df, lat_col='latitude', lon_col='longitude'):
    df['is_in_sea'] = df.apply(lambda row: is_point_in_sea(row[lat_col], row[lon_col]), axis=1)
    filtered_df = df[~df['is_in_sea']].drop(columns=['is_in_sea'])
    return filtered_df

# Main function to filter out sea points based on unique lat/lon combinations
def filter_sea_from_full_df(full_df, lat_col='latitude', lon_col='longitude'):
    # Step 1: Get unique lat/lon combinations
    unique_lat_lon = get_unique_lat_lon(full_df, lat_col, lon_col)

    # Step 2: Filter out sea points
    filtered_lat_lon = filter_out_sea_points(unique_lat_lon, lat_col, lon_col)

    # Step 3: Merge back with the original dataset
    filtered_df = full_df.merge(filtered_lat_lon, on=[lat_col, lon_col], how='inner')

    return filtered_df