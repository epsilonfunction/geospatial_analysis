"""
Weather Visualization with GeoPandas and Open Source Satellite Imagery
A comprehensive tutorial for creating weather visualizations
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import contextily as ctx
from datetime import datetime

# =============================================================================
# PART 1: BASIC SETUP - Creating Sample Weather Data
# =============================================================================

def create_sample_weather_data(n_stations=50):
    """
    Create sample weather station data for demonstration
    In practice, you'd load this from APIs or files
    """
    np.random.seed(42)
    
    # Generate random coordinates (e.g., covering a region)
    lons = np.random.uniform(-125, -65, n_stations)  # US longitude range
    lats = np.random.uniform(25, 50, n_stations)     # US latitude range
    
    # Generate weather parameters
    temps = np.random.normal(20, 10, n_stations)  # Temperature in Celsius
    precipitation = np.random.exponential(5, n_stations)  # mm of rain
    humidity = np.random.uniform(30, 90, n_stations)  # Percentage
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'station_id': [f'STN_{i:03d}' for i in range(n_stations)],
        'temperature': temps,
        'precipitation': precipitation,
        'humidity': humidity,
        'geometry': gpd.points_from_xy(lons, lats)
    }, crs='EPSG:4326')  # WGS84 coordinate system
    
    return gdf

# =============================================================================
# PART 2: BASIC WEATHER MAP VISUALIZATION
# =============================================================================

def plot_basic_weather_map(weather_gdf, column='temperature'):
    """
    Create a basic weather map with point data
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot weather stations colored by temperature
    weather_gdf.plot(
        column=column,
        cmap='RdYlBu_r',  # Red-Yellow-Blue reversed (hot to cold)
        legend=True,
        markersize=100,
        ax=ax,
        alpha=0.7,
        edgecolor='black',
        legend_kwds={'label': f'{column.capitalize()} (°C)'}
    )
    
    # Add basemap
    ctx.add_basemap(
        ax,
        crs=weather_gdf.crs.to_string(),
        source=ctx.providers.OpenStreetMap.Mapnik,
        alpha=0.5
    )
    
    ax.set_title(f'Weather Station {column.capitalize()}', fontsize=16, pad=20)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    plt.tight_layout()
    return fig, ax

# =============================================================================
# PART 3: INTERPOLATED WEATHER SURFACE (Heatmap Style)
# =============================================================================

def create_interpolated_weather_map(weather_gdf, column='temperature', 
                                   resolution=100):
    """
    Create an interpolated weather surface using griddata
    This creates a smooth heatmap effect
    """
    from scipy.interpolate import griddata
    
    # Extract coordinates and values
    points = np.array([[p.x, p.y] for p in weather_gdf.geometry])
    values = weather_gdf[column].values
    
    # Create grid
    lon_min, lon_max = points[:, 0].min(), points[:, 0].max()
    lat_min, lat_max = points[:, 1].min(), points[:, 1].max()
    
    grid_lon = np.linspace(lon_min, lon_max, resolution)
    grid_lat = np.linspace(lat_min, lat_max, resolution)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
    
    # Interpolate
    grid_values = griddata(points, values, (grid_lon, grid_lat), method='cubic')
    
    # Plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Create contour plot
    contour = ax.contourf(
        grid_lon, grid_lat, grid_values,
        levels=15,
        cmap='RdYlBu_r',
        alpha=0.6
    )
    
    # Add weather stations
    weather_gdf.plot(ax=ax, color='black', markersize=20, alpha=0.8)
    
    # Add basemap
    ctx.add_basemap(
        ax,
        crs='EPSG:4326',
        source=ctx.providers.OpenStreetMap.Mapnik,
        alpha=0.3
    )
    
    cbar = plt.colorbar(contour, ax=ax, label=f'{column.capitalize()} (°C)')
    ax.set_title(f'Interpolated {column.capitalize()} Map', fontsize=16)
    
    plt.tight_layout()
    return fig, ax

# =============================================================================
# PART 4: OVERLAY WITH ADMINISTRATIVE BOUNDARIES
# =============================================================================

def plot_weather_with_boundaries(weather_gdf, column='temperature'):
    """
    Overlay weather data on administrative boundaries
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Load country boundaries directly from Natural Earth
    world_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(world_url)
    usa = world[world.NAME == 'United States of America']
    
    # Plot USA boundary
    usa.boundary.plot(ax=ax, linewidth=2, edgecolor='black')
    usa.plot(ax=ax, color='lightgray', alpha=0.3)
    
    # Plot weather data
    weather_gdf.plot(
        column=column,
        cmap='RdYlBu_r',
        legend=True,
        markersize=150,
        ax=ax,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
        legend_kwds={'label': f'{column.capitalize()} (°C)', 'shrink': 0.8}
    )
    
    # Set bounds to focus on data
    bounds = weather_gdf.total_bounds
    margin = 2  # degrees
    ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
    ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
    
    ax.set_title('Weather Data with Administrative Boundaries', fontsize=16)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

# =============================================================================
# PART 5: MULTI-PARAMETER VISUALIZATION
# =============================================================================

def plot_multi_parameter_weather(weather_gdf):
    """
    Create a multi-panel visualization showing different weather parameters
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    parameters = [
        ('temperature', 'RdYlBu_r', '°C'),
        ('precipitation', 'Blues', 'mm'),
        ('humidity', 'Greens', '%'),
        ('temperature', 'viridis', '°C')  # Alternative colormap
    ]
    
    world_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(world_url)
    usa = world[world.NAME == 'United States of America']
    
    for idx, (param, cmap, unit) in enumerate(parameters):
        ax = axes[idx]
        
        # Plot country boundary
        usa.boundary.plot(ax=ax, linewidth=1.5, edgecolor='black', alpha=0.5)
        
        # Plot weather data
        weather_gdf.plot(
            column=param,
            cmap=cmap,
            legend=True,
            markersize=100,
            ax=ax,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.3,
            legend_kwds={'label': f'{param.capitalize()} ({unit})'}
        )
        
        # Set bounds
        bounds = weather_gdf.total_bounds
        margin = 2
        ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
        ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
        
        ax.set_title(f'{param.capitalize()} Distribution', fontsize=12, pad=10)
        ax.grid(True, alpha=0.2)
    
    plt.suptitle('Multi-Parameter Weather Analysis', fontsize=18, y=0.995)
    plt.tight_layout()
    return fig, axes

# =============================================================================
# PART 6: WORKING WITH REAL SATELLITE IMAGERY
# =============================================================================

def example_satellite_integration():
    """
    Example code for integrating satellite imagery
    Note: This requires additional packages and API keys
    """
    example_code = """
    # OPTION 1: Using NASA GIBS (requires requests package)
    import requests
    from PIL import Image
    from io import BytesIO
    
    def get_nasa_gibs_image(date, bbox, layer='MODIS_Terra_CorrectedReflectance_TrueColor'):
        base_url = "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/"
        url = f"{base_url}{layer}/default/{date}/250m/8/0/0.jpg"
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    
    # OPTION 2: Using Sentinel-2 (requires sentinelsat package)
    from sentinelsat import SentinelAPI
    
    api = SentinelAPI('username', 'password', 'https://scihub.copernicus.eu/dhus')
    footprint = 'POLYGON((lon1 lat1, lon2 lat2, ...))'
    products = api.query(footprint, date=('20240101', '20240131'), 
                        platformname='Sentinel-2')
    
    # OPTION 3: Using contextily for satellite basemaps
    import contextily as ctx
    
    # Add satellite imagery as basemap
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
    """
    return example_code

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Weather Visualization with GeoPandas Tutorial")
    print("=" * 60)
    
    # Create sample data
    print("\n1. Creating sample weather data...")
    weather_data = create_sample_weather_data(n_stations=50)
    print(f"   Created {len(weather_data)} weather stations")
    print(weather_data.head())
    
    # Basic map
    print("\n2. Creating basic weather map...")
    fig1, ax1 = plot_basic_weather_map(weather_data, 'temperature')
    plt.savefig('./sample_img/weather_basic.png', dpi=150, bbox_inches='tight')
    print("   Saved: weather_basic.png")
    
    # Interpolated map
    print("\n3. Creating interpolated weather map...")
    fig2, ax2 = create_interpolated_weather_map(weather_data, 'temperature')
    plt.savefig('./sample_img/weather_interpolated.png', dpi=150, bbox_inches='tight')
    print("   Saved: weather_interpolated.png")
    
    # Map with boundaries
    print("\n4. Creating map with administrative boundaries...")
    fig3, ax3 = plot_weather_with_boundaries(weather_data, 'temperature')
    plt.savefig('./sample_img/weather_boundaries.png', dpi=150, bbox_inches='tight')
    print("   Saved: weather_boundaries.png")
    
    # Multi-parameter map
    print("\n5. Creating multi-parameter visualization...")
    fig4, axes4 = plot_multi_parameter_weather(weather_data)
    plt.savefig('./sample_img/weather_multiparameter.png', dpi=150, bbox_inches='tight')
    print("   Saved: weather_multiparameter.png")
    
    print("\n" + "=" * 60)
    print("Tutorial complete! Check the generated PNG files.")
    print("\nNext steps:")
    print("- Install satellite data packages: pip install sentinelsat rasterio")
    print("- Get NASA EARTHDATA account: https://urs.earthdata.nasa.gov/")
    print("- Explore Copernicus Open Access Hub for Sentinel data")
    
    plt.show()
