# Importing functions
import netCDF4 as Dataset
import glob
import wrf
import xarray as xr
import cartopy.crs as crs
import cartopy
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
import imageio
import warnings
import pandas as pd
import datetime
from PIL import Image
from shapely.geometry import Point
import geopandas as gpd

#data['day']=data['day'].astype(int)
#data['month']=data['month'].astype(int)
#data['year']=data['year'].astype(int)

def extract_date(data,year,month,day,variable="probability"):
    """
    Given the output of Georg-AI of probability and convective events, it extracts a single date.
    
    Args:
        - data: DataFrame with output of Georg-AI in csv (full_predictions.csv)
        - year: int containing the year that needs to be extracted
        - month: int containing the month that needs to be extracted
        - day: int containing the day that needs to be extracted
        - variable: variable to extract (probability or Convective_Event)
        
    Returns:
        - data_extracted: output of Georg-AI for one single date and variable
        - convective_day_flag: 0 if data is empty (no convective day). 1 otherwise
        - date: entered date, in datetime format
    """
    
    # Convert date columns to datetime
    data["date"]=pd.to_datetime(data[["year","month","day"]]).dt.date
    
    # Set chosen date as datetime
    date = datetime.date(year, month, day)
    
    # Extract data for the chosen date
    data_extracted=data[data["date"]==date][["latitude","longitude",variable]].reset_index(drop=True)
    
    # See if data exists for the chosen date
    if len(data_extracted)==0:
        convective_day_flag=0
        warnings.warn("The chosen day is not a convective day")
    else:
        convective_day_flag=1
        
    return data_extracted, convective_day_flag, date

def average_data(data,variable):
    """
    Gets Georg-AI output data and calculate its mean, median and maximum across time. 
    
    Args:
        - data: DataFrame containing the Georg-AI output
        - variable: str with the variable to extract (probability or Convective_Event)
        
    Returns:
        - data_mean, data_max, data_median, data_sum: mean, max, median, sum across times outputs
    """
    
    # Convert date columns to datetime
    data["date"]=pd.to_datetime(data[["year","month","day"]]).dt.date
    
    # Leave only necessary columns
    data=data[["latitude","longitude","date",variable]]
    
    # Perform calculations
    data_mean=data.groupby(["latitude","longitude"])[variable].mean().reset_index()
    data_max=data.groupby(["latitude","longitude"])[variable].max().reset_index()
    data_median=data.groupby(["latitude","longitude"])[variable].median().reset_index()  
    data_sum=data.groupby(["latitude","longitude"])[variable].sum().reset_index()  
    
    return data_mean,data_max, data_median, data_sum

def average_data_yearly(data,variable):
    """
    Gets Georg-AI output data and calculate its mean, median and maximum across time for each year. 
    
    Args:
        - data: DataFrame containing the Georg-AI output
        - variable: str with the variable to extract (probability or Convective_Event)
        
    Returns:
        - data_mean, data_max, data_median, data_sum: mean, max, median, sum across times outputs
    """
    data_average=data.copy()
    
    # Convert date columns to datetime
    data_average["date"]=pd.to_datetime(data_average[["year","month","day"]]).dt.date
    data_average["year"]=pd.to_datetime(data_average['year'].astype(int).astype(str), format='%Y')

    
    # Leave only necessary columns
    data_average=data_average[["latitude","longitude","date","year",variable]]
    
    # Perform calculations
    data_mean=data_average.groupby(["year","latitude","longitude"])[variable].mean().reset_index()
    data_max=data_average.groupby(["year","latitude","longitude"])[variable].max().reset_index()
    data_median=data_average.groupby(["year","latitude","longitude"])[variable].median().reset_index()  
    data_sum=data_average.groupby(["year","latitude","longitude"])[variable].sum().reset_index()  
    
    return data_mean,data_max, data_median, data_sum

           
def show_duplicates(data,variable="probability"):
    """
    Explores the duplicates (same pair of latitude and longitude) across the Georg-AI output. 
    If there are duplicates that contain different values the code will send a printed warning.
    This is done by observing the stds of the values of the duplicates.
    
    Args:
        - data: DataFrame with output of Georg-AI
        - variable: variable to extract (probability or Convective_Event)
    
    Returns: 
        - number_non_unique: number of lat-lon duplicates that have different values
    
    """
    
    data['Label'] = data.groupby(['latitude', 'longitude']).cumcount()
    
    data_duplicate=data[data["Label"]>0]
    
    #Calculate standard deviation of each group of duplicated lat-lons. 
    # If these are different to zero or nan, it means they have different values
    data_duplicate_std=data_duplicate.groupby(['latitude', 'longitude'])[variable].std()
    
    # See how many data duplicates for the same lat-lon pair have a positive standard deviation. 
    # If more than 0, it means some duplicates contain different data for the same location
    number_non_unique=np.array([data_duplicate_std>0]).sum()
    if number_non_unique>0:
        print(date+' '+variable+": There are some lat-lon duplicates which contain different values")

    return number_non_unique

def visualise_data(
    data,
    date,
    variable="probability",
    binary=False,
    vmin=0.0,
    vmax=1,
    figsize=(7,5),
    logarithmic=False,
    cmap_steps=5,
    colorbar_name=False,
    mask_georgia=False
):
    """
    Plots a map with the convective events or the probability of convective events.
    
    Args:
        - data: DataFrame containing the output of Georg-AI for a single date (or averaged acorss all dates)
        - date: 
            - if single date, this is the datetime containig the date of the data
            - if this is an average or a maximum, add a str with the description (e.g: "Average" or "Max")
        - variable: str with the name of the variable to plot (probability or Convective_Event)
        - binary: if True it will asume that you are plotting convective days and remove the colorbar
        - colorbar parameters: (vmin=0.01, vmax=1)
        - figsize: two numbers containing the figure size (e.g: figsize=(7,5))
        - cmap_steps: int with the nuber of discretisations of the cmap (if set to a very large number, the cmap will look sequential)
        - colorbar_name: If False, the colorbar label will be same as variable. If an str is provided, this will be the colorbar label
        - mask_georgia: if true, gris outside Georgia will be masked out.
        
    
    Returns:
        - fig,ax: figure
        - data: plotted data
    """
    
    # Checking if there are non unique data duplicates for all lat-lon
    number_non_unique=show_duplicates(data,variable=variable)

    # Drop duplicates for the same lat-lon
    data=data.drop_duplicates(subset=['latitude', 'longitude'])
    
    # Convert data into a grid
    data = data.pivot(index='latitude', columns='longitude', values=variable)
    
    # Masking outside of Georgia
    if mask_georgia==True:
        data=masking_georgia(data)
    
    # Extrac lat lons and grid 
    longitude=np.array(data.columns)
    latitude=np.array(data.index)
    data_grid=data.values

    data_grid[data_grid==0]=np.nan
    
    # Create a figure and axis with a map projection
    fig, ax = plt.subplots(subplot_kw={'projection': crs.PlateCarree()},figsize=figsize)


    # Borders
    borders = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_0_boundary_lines_land",
        scale="10m",
        facecolor="none",
    )
    ax.add_feature(borders, edgecolor="black", lw=0.8,linestyle='-')

    # Add coastlines and gridlines
    ax.coastlines()
    #ax.gridlines()

    """    
    # Set range
    lon_min=-110
    lon_max=-85
    lat_min=28
    lat_max=50
    #ax.set_xlim(lon_min, lon_max)
    #ax.set_ylim(lat_min, lat_max)
    """
    #Creating a discretised colormap
    cmap=discretised_cmap('Reds',cmap_steps)
    
    # Add logarithmic scale if requested
    if logarithmic:
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)  # No normalization (linear)
    
    # Plot the data on the map
    c = plt.pcolormesh(longitude, latitude, data_grid, transform=crs.PlateCarree(), cmap=cmap,norm=norm)

    if binary==False:
        cbar = plt.colorbar(c, ax=ax, orientation='vertical', format='%.2f')
        if colorbar_name!= False:
            cbar.set_label(colorbar_name)
        if colorbar_name == False:
            cbar.set_label(variable)
        
    # Adding date or average
    if type(date)==datetime.date:
        plt.title(datetime.datetime.strftime(date,format="%Y/%B/%d"))
    else: 
        plt.title(date)
        
    
    return fig,ax,data
    
def add_US_features(fig,ax):
    """
    Given a figure (fig,ax), it adds the US features for the map (i.e: US border states)
    """
    
    # US states
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none'
    )
    ax.add_feature(states_provinces, edgecolor='gray', linewidth=0.5)
    
    return fig,ax

def format_coordinates(dd,coordinate):
    """
    Given a coordinate in decimal format, it returns its minutes and seconds as a str.
    
    Args:
        - dd: coordinates in decimal format (float)
        - coordinate: str containig the coordinate (lat or lon)
        
    Returns:
        - formated: str containing the location in DD°mm' format    
    """
    
    # Calculate degrees and minutes
    is_positive = dd >= 0
    dd = abs(dd)
    minutes = dd*60
    degrees,minutes = divmod(minutes,60)
    degrees = degrees if is_positive else - degrees
    
    # Round degrees and minutes
    degrees=round(degrees)
    minutes=round(minutes)
    
    
    # Add 0 before degrees and minutes if smaller than 10 and 00 if degrees smaller than 100 for lons
    if coordinate=="lon":
        if (degrees<100) and (degrees>9):
            degrees="0"+str(degrees)
        elif degrees<10:
            degrees="00"+str(degrees)
        if minutes<10:
            minutes="0"+str(minutes)
            
    # Add 0 before degrees and minutes if smaller than 10 for lats
    if coordinate=="lat":
        if degrees<10:
            degrees="0"+str(degrees)
        if minutes<10:
            minutes="0"+str(minutes)

    # Construct final str 
    formated=str(degrees)+"°"+str(minutes)+"'"
    
    return formated

def dms_to_decimal(dms_str):
    """
    Transforms a coordinate originally in "DD°mm'" to decimal
    
    Args:
        - dms_str: str containing the coordinate in "DD°mm'" format
        
    Returns: 
        - 
    """
    # Split the input string into degrees and minutes
    parts = dms_str.split('°')
    
    # Extract degrees and minutes as strings
    degrees_str = parts[0].strip()
    minutes_str = parts[1].strip("'")

    # Convert degrees and minutes to integers
    degrees = int(degrees_str)
    minutes = int(minutes_str)

    # Calculate the decimal representation
    decimal = degrees + minutes / 60.0

    return decimal

def time_to_str(time):
    """
    Transforms a datetime time object into a str with hh:mm 
    """
    # Empty output
    time_str=np.nan
    
    # Perform operation if input is datetime, otherwise, the output will be nan
    if type(time)==datetime.time:
        # Extract hour and minute
        hour=str(time.hour)
        minute=str(time.minute)
        
        #Add 0 if values below 10
        if float(hour)<10:
            hour="0"+hour
        if float(minute)<10:
            minute="0"+minute
            
        # Construct str
        time_str=str(hour)+":"+str(minute)
        
    return time_str


def add_features_georgia(fig,ax,georgian=False):
    """
    Given a WRF output over the region of Georgia, it adds the names of some cities, as well as the country borders and the urban areas
    
    Args:
        - fig, ax: of a WRF output map of the region of Georgia
        - georgian: set to True or False in order to activate names of the cities in georgian
    
    Returns:
        - fig, ax
    """
    
    # Define cities cities
    if georgian==True:
        cities = ['Tbilisi \n თბილისი','Batumi \n ბათუმი','Kutaisi \n ქუთაისი']
    if georgian==False:
        cities = ['Tbilisi','Batumi','Kutaisi']
        
    loc = [[41.72250, 44.79250],
          [41.64583, 41.64167],
          [42.266243, 42.718002],
          ]
    
    # Getting size of figure to calculate appropiate fontsizes
    fig_width, fig_height = plt.gcf().get_size_inches()

    # Add cities and markers
    for city in zip(cities,loc):
        plt.plot(city[1][1],city[1][0],"ro",markersize=2,transform=crs.PlateCarree(),color="k")
        plt.text(city[1][1]*1.0025,city[1][0]*0.995,city[0],
                 horizontalalignment='left',transform=crs.PlateCarree(),fontsize=fig_width+1)

    # Add state borders
    borders = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_0_boundary_lines_land",
        scale="10m",
        facecolor="none",
    )
    ax.add_feature(borders, edgecolor="black", lw=0.8,linestyle='--')
    
    #Add urban areas
    states = NaturalEarthFeature(category="cultural", scale="10m",
                                 facecolor="none",
                                 name='urban_areas')
    ax.add_feature(states, linewidth=0.2, edgecolor="black",facecolor="grey",hatch='....',alpha=0.25)

    #Add legend for urban areas
    circ1 = matplotlib.patches.Patch(color="grey",alpha=0.25,hatch='....',label="Urban areas")
    ax.legend(handles = [circ1],loc=2)

    
    return fig, ax

def discretised_cmap(cmap,steps):
    """
    Discretizes a given colormap.
    
    Args: 
        - cmap: str containing a valid colormap from matplotlib (e.g: "Reds")
        - steps: int containing the number of discretizations
        
    Returns:
        - custom_cmap: discretised colormap
    """
    
    # Create the original "Reds" colormap
    original_cmap = plt.get_cmap(cmap)

    # Create an array of equally spaced values from 0 to 1 to sample the colormap
    color_positions = np.linspace(0, 1, steps)
    color_positions = np.logspace(-1,0,6)

    # Sample the "Reds" colormap at the specified positions
    sampled_colors = original_cmap(color_positions)

    # Create a custom colormap with the sampled colors
    custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("discretized_"+str(cmap), sampled_colors, N=steps)

    return custom_cmap

def create_location_graph(df,figsize=(5.5,5)):
    """
    Plots the globe of the Earth including the area of interested, as defined by a df containing a set of "latitudes" and "longitudes"
   
    Args: 
        - df: df containing a set of  "latitudes" and "longitudes"
    """
    # Extract range of lats and lons from df
    max_lat=df["latitude"].max()
    max_lon=df["longitude"].max()
    min_lat=df["latitude"].min()
    min_lon=df["longitude"].min()

    # Create a map with an Orthographic projection centered around the specific area
    fig, ax = plt.subplots(subplot_kw={'projection': crs.Orthographic(
        central_latitude=(min_lat + max_lat) / 2,
        central_longitude=(min_lon + max_lon) / 2)},figsize=figsize)

    # Add map features (e.g., coastlines, countries)
    ax.coastlines(resolution='110m', linewidth=1)
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1)
    #ax.add_feature(cartopy.feature.STATES, linestyle='-', linewidth=0.1)

    # Draw a rectangle to highlight the specific area
    ax.plot([min_lon, max_lon, max_lon, min_lon, min_lon],
            [min_lat, min_lat, max_lat, max_lat, min_lat],
            color='red', linewidth=2, transform=crs.PlateCarree())

    # Set the map extent to show the entire globe
    ax.set_global()

    # Add parallels and meridians
    ax.gridlines(color="white",alpha=0.5)

    # Color the land and water
    ax.add_feature(cfeature.LAND, color='tan')
    ax.add_feature(cfeature.OCEAN, color='lightblue')
    
    return fig, ax
    
def combine_images(image_filenames, output_name, grid_size = (2, 3)):
    """
    Combines several images into a single one.
    
    Args:
        - image_filenames: list containing str with the path to each of the images to combine
        - output_name: str with the name of the final combined image
        - grid_size: arrangement of the images (e.g 2,3 means three columns and two rows)
        
    Returns:
        - saves the combined image and deletes the individual ones
    """
    # Load the images
    images = [Image.open(filename) for filename in image_filenames]

    # Calculate the total size of the combined image
    combined_width = max(image.width for image in images) * grid_size[1] 
    combined_height = max(image.height for image in images) * grid_size[0] 

    # Create a new blank image with the calculated size
    combined_image = Image.new('RGB', (combined_width, combined_height), color='white')

    # Paste each image into the combined image
    for i, image in enumerate(images):
        x = i % grid_size[1] * image.width
        y = i // grid_size[1] * image.height
        combined_image.paste(image, (x, y))

    # Save the combined image
    combined_image.save(output_name,dpi=(300, 300),quality=200)

    # Delete the temporary images
    for filename in image_filenames:
        os.remove(filename)

    # Show the map
    plt.show()
    
def plot_time_series(data,legend,fig,axs):
    # Convert days and months into datetime 
    data['day']=data['day'].astype(int)
    data['month']=data['month'].astype(int)
    
    data=data[(data["month"]>2) & (data["month"]<10)]

    # Groupby month and day to create multi year averages
    data_year_avg=data.groupby(['month','day'],as_index=False)[["Convective_Event","probability","Convective_Event"]].mean()
    data_year_std=data.groupby(['month','day'],as_index=False)[["Convective_Event","probability","Convective_Event"]].std()

    # Create date and concert them inot month-day format
    data_year_avg["date"] = data_year_avg["month"].astype(str) + data_year_avg["day"].astype(str)
    data_year_avg["date"] = data_year_avg["date"].apply(lambda x: datetime.datetime.strptime(x, "%m%d") if x else pd.NaT)

    
    axs=np.array(axs)

    #Adding plots
    axs[0].plot(data_year_avg["date"].values,data_year_avg["Convective_Event"].values,label=legend,alpha=0.8)
    axs[1].plot(data_year_avg["date"].values,data_year_avg["probability"].values,label=legend,alpha=0.8)

    axs[1].fill_between(
        data_year_avg["date"].values,
        data_year_avg["probability"].values-data_year_std["probability"].values,
        data_year_avg["probability"].values+data_year_std["probability"].values,
        alpha=0.2
    )
    
    
    # First plot settings
    axs[0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b/%d"))
    axs[0].set_xlim([datetime.date(1900, 1, 1), datetime.date(1900, 12, 1)])
    axs[0].tick_params(axis='x', labelrotation=30)
    axs[0].grid(alpha=0.25)
    axs[0].set_ylim([0,1.025])
    axs[0].set_ylabel("Fraction of convective days")
    axs[0].grid(True)

    # Second plot settings
    axs[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b/%d"))
    axs[1].set_xlim([datetime.date(1900, 1, 1), datetime.date(1900, 12, 1)])
    axs[1].tick_params(axis='x', labelrotation=30)
    axs[1].grid(alpha=0.25)
    axs[1].set_ylim([0,1.025])
    axs[1].set_ylabel("Probability")
    axs[1].legend(frameon=False)
    axs[1].grid(True)

def plot_US_capitals():
    # Define a dictionary of U.S. state capitals with their coordinates (longitude, latitude)
    us_cities_coordinates = {
        "Houston": (-95.3698, 29.7604),
        "San Antonio": (-98.4936, 29.4241),
        "Dallas": (-96.7969, 32.7767),
        "Austin": (-97.7431, 30.2672),
        "Fort Worth": (-97.3301, 32.7555),
        "Denver": (-104.9840, 39.7392),
        "El Paso": (-106.4850, 31.7619),
        "Oklahoma City": (-97.5164, 35.4676),
        "Albuquerque": (-106.6056, 35.0844),
        "Kansas City": (-94.5786, 39.0997),
        "Wichita": (-97.3375, 37.6872),
        "Des Moines": (-93.6091, 41.5868),
        "Minneapolis": (-93.2650, 44.9778),
        "Tulsa, Oklahoma": (-95.9928, 36.1540),
    }
    
    # Plot state capitals in the Central Time Zone on the map
    for state, (lon, lat) in us_cities_coordinates.items():
        plt.plot(lon, lat, 'ro', markersize=5,color="green",alpha=0.75)
        plt.text(lon + 1, lat, state, transform=crs.PlateCarree(), fontsize=8, fontweight='bold')
   
def masking_georgia(data):
    # Download the shapefile for the country of Georgia
    gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    georgia_shape = gdf[gdf['name'] == 'Georgia']

    #
    georgia_mask = georgia_shape.geometry.unary_union

    # Apply the mask to your data
    buffer_distance=0.1
    masked_data = data.copy()  # Create a copy of the original data
    for lat in data.index:
        for lon in data.columns:
            point = Point(lon, lat)
            if not (georgia_mask.contains(point) or (georgia_mask.distance(point) < buffer_distance)):
                masked_data.loc[lat, lon] = None
               
    return masked_data
    