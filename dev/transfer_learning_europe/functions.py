import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
import cartopy
import cartopy.io.img_tiles as cimgt

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

def process_cdd_data(data,variable,process_type):
    """
    Converts a CDD predicted probability outpu into mean, min, max, and sum for the whol period or for each month.
    """
    
    process_type_list=["mean","min","max","sum","median","monthly_mean","monthly_max","monthly_sum"]
    variables=["probability","Convective_Event"]
    
    if variable not in variables:
        raise Exception("The selected variable is not valid. Select one from the following: ",variables)
    
    if process_type not in process_type_list:
        raise Exception("The selected processing is not a valid operation. Select one from the following: ",process_type_list)
    
    data_process=data.copy()[["latitude","longitude","year","month","day",variable]]
    data_process["date"]=pd.to_datetime(data_process[["year","month","day"]]).dt.date
    data_process["year"]=pd.to_datetime(data_process['year'].astype(int).astype(str), format='%Y')
    
    if process_type=="mean":
        return data_process.groupby(["latitude","longitude"])[variable].mean().reset_index()
    if process_type=="min":
        return data_process.groupby(["latitude","longitude"])[variable].min().reset_index()   
    if process_type=="max":
        return data_process.groupby(["latitude","longitude"])[variable].max().reset_index()
    if process_type=="sum":
        return data_process.groupby(["latitude","longitude"])[variable].sum().reset_index()
    if process_type=="median":
        return data_process.groupby(["latitude","longitude"])[variable].median().reset_index()
    if process_type=="monthly_mean":
        return data_process.groupby(["month","latitude","longitude"])[variable].mean().reset_index()
    if process_type=="monthly_max":
        return data_process.groupby(["month","latitude","longitude"])[variable].max().reset_index()
    if process_type=="monthly_sum":
        return data_process.groupby(["month","latitude","longitude"])[variable].sum().reset_index()
        
def plot_processed_cdd_data(ax,data,variable,cmap_steps=10,logarithmic=False,vmin=0,vmax=1,alpha=0.5,zoom=7,colorbar_name=""):
    
    # Convert data into a grid
    data = data.pivot(index='latitude', columns='longitude', values=variable)
    
    # Extrac lat lons and grid 
    longitude=np.array(data.columns)
    latitude=np.array(data.index)
    data_grid=data.values

    data_grid[data_grid==0]=np.nan
    
    #Creating a discretised colormap
    cmap=discretised_cmap('Reds',cmap_steps)
    
    # Add logarithmic scale if requested
    if logarithmic:
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)  # No normalization (linear)
        
    c = ax.pcolormesh(longitude, latitude, data_grid, transform=crs.PlateCarree(), cmap=cmap,norm=norm,alpha=alpha)
    cbar = plt.colorbar(c, ax=ax, orientation='vertical', format='%.2f',shrink=0.75, pad=0.04)
    cbar.set_label(colorbar_name)
    
    ax.set_extent([min(longitude),max(longitude),min(latitude),max(latitude)], crs=cartopy.crs.PlateCarree())
        
    ax.add_image(cimgt.OSM(), zoom)
    
    # Borders
    borders = cartopy.feature.NaturalEarthFeature(
        category="cultural",
        name="admin_0_boundary_lines_land",
        scale="10m",
        facecolor="none",
    )
    ax.add_feature(borders, edgecolor="black", lw=0.8,linestyle='-',alpha=0.75)
    ax.coastlines(alpha=0.75)
    
    return ax

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