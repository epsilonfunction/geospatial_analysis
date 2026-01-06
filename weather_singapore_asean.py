"""
Weather Visualization for Singapore and ASEAN Region
Using NEA (National Environment Agency) and Global Meteorological Data
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import contextily as ctx
from datetime import datetime, timedelta
import requests
from io import StringIO

# =============================================================================
# PART 1: NEA DATA ACCESS (Singapore)
# =============================================================================

class NEAWeatherAPI:
    """
    Interface for accessing NEA (National Environment Agency) Singapore data
    API Documentation: https://beta.data.gov.sg/
    """
    
    BASE_URL = "https://api.data.gov.sg/v1/environment"
    
    @staticmethod
    def get_realtime_weather():
        """
        Get real-time weather readings from NEA stations
        """
        try:
            # 2-hour weather forecast
            url = f"{NEAWeatherAPI.BASE_URL}/2-hour-weather-forecast"
            response = requests.get(url)
            print(f"response: {response}")
            data = response.json()
            
            forecasts = []
            if 'items' in data and len(data['items']) > 0:
                for forecast in data['items'][0]['forecasts']:
                    forecasts.append({
                        'area': forecast['area'],
                        'forecast': forecast['forecast']
                    })
            
            return pd.DataFrame(forecasts)
        except Exception as e:
            print(f"Error fetching NEA data: {e}")
            return None
    
    @staticmethod
    def get_air_temperature():
        """
        Get air temperature readings from NEA stations
        """
        try:
            url = f"{NEAWeatherAPI.BASE_URL}/air-temperature"
            response = requests.get(url)
            print(f"response: {response}")
            data = response.json()
            
            stations = []
            if 'items' in data and len(data['items']) > 0:
                metadata = data['metadata']['stations']
                readings = data['items'][0]['readings']
                
                for reading in readings:
                    station_info = next((s for s in metadata if s['id'] == reading['station_id']), None)
                    if station_info:
                        stations.append({
                            'station_id': reading['station_id'],
                            'station_name': station_info['name'],
                            'latitude': station_info['location']['latitude'],
                            'longitude': station_info['location']['longitude'],
                            'temperature': reading['value']
                        })
            
            return pd.DataFrame(stations)
        except Exception as e:
            print(f"Error fetching temperature data: {e}")
            return None
    
    @staticmethod
    def get_rainfall():
        """
        Get rainfall readings from NEA stations
        """
        try:
            url = f"{NEAWeatherAPI.BASE_URL}/rainfall"
            response = requests.get(url)
            print(f"response: {response}")

            data = response.json()
            
            stations = []
            if 'items' in data and len(data['items']) > 0:
                metadata = data['metadata']['stations']
                readings = data['items'][0]['readings']
                
                for reading in readings:
                    station_info = next((s for s in metadata if s['id'] == reading['station_id']), None)
                    if station_info:
                        stations.append({
                            'station_id': reading['station_id'],
                            'station_name': station_info['name'],
                            'latitude': station_info['location']['latitude'],
                            'longitude': station_info['location']['longitude'],
                            'rainfall': reading['value']
                        })
            
            return pd.DataFrame(stations)
        except Exception as e:
            print(f"Error fetching rainfall data: {e}")
            return None

# =============================================================================
# PART 2: SAMPLE DATA GENERATION FOR ASEAN REGION
# =============================================================================

def create_asean_weather_data():
    """
    Create sample weather data for ASEAN countries
    In production, this would come from various national meteorological agencies
    """
    
    # ASEAN capital cities and major weather stations
    stations = {
        'Singapore': {'lat': 1.3521, 'lon': 103.8198, 'country': 'Singapore'},
        'Jakarta': {'lat': -6.2088, 'lon': 106.8456, 'country': 'Indonesia'},
        'Surabaya': {'lat': -7.2575, 'lon': 112.7521, 'country': 'Indonesia'},
        'Kuala Lumpur': {'lat': 3.1390, 'lon': 101.6869, 'country': 'Malaysia'},
        'Bangkok': {'lat': 13.7563, 'lon': 100.5018, 'country': 'Thailand'},
        'Chiang Mai': {'lat': 18.7883, 'lon': 98.9853, 'country': 'Thailand'},
        'Manila': {'lat': 14.5995, 'lon': 120.9842, 'country': 'Philippines'},
        'Hanoi': {'lat': 21.0285, 'lon': 105.8542, 'country': 'Vietnam'},
        'Ho Chi Minh': {'lat': 10.8231, 'lon': 106.6297, 'country': 'Vietnam'},
        'Phnom Penh': {'lat': 11.5564, 'lon': 104.9282, 'country': 'Cambodia'},
        'Vientiane': {'lat': 17.9757, 'lon': 102.6331, 'country': 'Laos'},
        'Yangon': {'lat': 16.8661, 'lon': 96.1951, 'country': 'Myanmar'},
        'Naypyidaw': {'lat': 19.7633, 'lon': 96.0785, 'country': 'Myanmar'},
        'Bandar Seri Begawan': {'lat': 4.9031, 'lon': 114.9398, 'country': 'Brunei'},
    }
    
    # Generate realistic tropical weather data
    np.random.seed(42)
    data = []
    
    for city, info in stations.items():
        # Tropical climate parameters
        base_temp = np.random.uniform(26, 32)  # Typical tropical temperatures
        temp_variation = np.random.normal(0, 2)
        
        data.append({
            'city': city,
            'country': info['country'],
            'latitude': info['lat'],
            'longitude': info['lon'],
            'temperature': base_temp + temp_variation,
            'humidity': np.random.uniform(60, 90),  # High humidity in tropics
            'rainfall_24h': np.random.exponential(15),  # mm in last 24h
            'wind_speed': np.random.uniform(5, 25),  # km/h
            'pressure': np.random.normal(1010, 5)  # hPa
        })
    
    df = pd.DataFrame(data)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs='EPSG:4326'
    )
    
    return gdf

def combine_nea_and_regional_data():
    """
    Combine NEA Singapore data with regional ASEAN data
    """
    # Get NEA data
    nea_temp = NEAWeatherAPI.get_air_temperature()
    nea_rainfall = NEAWeatherAPI.get_rainfall()
    
    # Create regional data
    asean_data = create_asean_weather_data()
    
    if nea_temp is not None and not nea_temp.empty:
        # Convert NEA data to GeoDataFrame
        nea_gdf = gpd.GeoDataFrame(
            nea_temp,
            geometry=gpd.points_from_xy(nea_temp.longitude, nea_temp.latitude),
            crs='EPSG:4326'
        )
        
        # Add rainfall data if available
        if nea_rainfall is not None and not nea_rainfall.empty:
            rainfall_dict = dict(zip(nea_rainfall.station_id, nea_rainfall.rainfall))
            nea_gdf['rainfall_24h'] = nea_gdf['station_id'].map(rainfall_dict).fillna(0)
        else:
            nea_gdf['rainfall_24h'] = 0
        
        # Standardize columns
        nea_gdf['city'] = nea_gdf['station_name']
        nea_gdf['country'] = 'Singapore'
        nea_gdf['humidity'] = np.random.uniform(70, 85, len(nea_gdf))  # Estimated
        nea_gdf['wind_speed'] = np.random.uniform(5, 15, len(nea_gdf))
        nea_gdf['pressure'] = 1010  # Standard
        
        # Combine datasets
        combined = pd.concat([
            asean_data[['city', 'country', 'latitude', 'longitude', 'temperature', 
                       'humidity', 'rainfall_24h', 'wind_speed', 'pressure', 'geometry']],
            nea_gdf[['city', 'country', 'latitude', 'longitude', 'temperature',
                    'humidity', 'rainfall_24h', 'wind_speed', 'pressure', 'geometry']]
        ], ignore_index=True)
        
        return gpd.GeoDataFrame(combined, crs='EPSG:4326')
    else:
        print("Using sample ASEAN data only (NEA API not available)")
        return asean_data

# =============================================================================
# PART 3: ASEAN REGION VISUALIZATION
# =============================================================================

def plot_asean_temperature_map(weather_gdf):
    """
    Create temperature map focused on ASEAN region
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Load ASEAN country boundaries
    world_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(world_url)
    
    asean_countries = ['Singapore', 'Indonesia', 'Malaysia', 'Thailand', 'Philippines',
                       'Vietnam', 'Myanmar', 'Cambodia', 'Laos', 'Brunei']
    asean = world[world.NAME.isin(asean_countries)]
    
    # Plot country boundaries
    asean.boundary.plot(ax=ax, linewidth=1.5, edgecolor='black', alpha=0.7)
    asean.plot(ax=ax, color='lightgray', alpha=0.2)
    
    # Plot weather data
    weather_gdf.plot(
        column='temperature',
        cmap='RdYlBu_r',
        legend=True,
        markersize=200,
        ax=ax,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.8,
        legend_kwds={
            'label': 'Temperature (°C)',
            'shrink': 0.8,
            'pad': 0.05
        }
    )
    
    # Add city labels
    for idx, row in weather_gdf.iterrows():
        if row['country'] in asean_countries:  # Only label major cities
            ax.annotate(
                row['city'],
                xy=(row.geometry.x, row.geometry.y),
                xytext=(3, 3),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
    
    # Set bounds to ASEAN region
    ax.set_xlim(95, 125)
    ax.set_ylim(-10, 25)
    
    ax.set_title('ASEAN Region Temperature Map', fontsize=18, pad=20, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M SGT')
    ax.text(0.02, 0.02, f'Data as of: {timestamp}', 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig, ax

# =============================================================================
# PART 4: SINGAPORE DETAILED VIEW
# =============================================================================

def plot_singapore_detailed(weather_gdf):
    """
    Create detailed weather map focused on Singapore
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Filter Singapore data
    sg_data = weather_gdf[weather_gdf['country'] == 'Singapore'].copy()
    
    if len(sg_data) == 0:
        print("No Singapore data available")
        return None, None
    
    # Plot temperature with larger markers
    sg_data.plot(
        column='temperature',
        cmap='RdYlBu_r',
        legend=True,
        markersize=300,
        ax=ax,
        alpha=0.85,
        edgecolor='darkblue',
        linewidth=1.5,
        legend_kwds={'label': 'Temperature (°C)', 'shrink': 0.9}
    )
    
    # Add station labels
    for idx, row in sg_data.iterrows():
        ax.annotate(
            f"{row['city']}\n{row['temperature']:.1f}°C",
            xy=(row.geometry.x, row.geometry.y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7, edgecolor='black')
        )
    
    # Add basemap
    try:
        ctx.add_basemap(
            ax,
            crs=sg_data.crs.to_string(),
            source=ctx.providers.OpenStreetMap.Mapnik,
            alpha=0.6
        )
    except:
        print("Basemap not available, continuing without it")
    
    # Zoom to Singapore
    bounds = sg_data.total_bounds
    margin = 0.05
    ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
    ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
    
    ax.set_title('Singapore Weather Stations - Detailed View', 
                 fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add info box
    avg_temp = sg_data['temperature'].mean()
    max_temp = sg_data['temperature'].max()
    min_temp = sg_data['temperature'].min()
    
    info_text = f"Average: {avg_temp:.1f}°C\nMax: {max_temp:.1f}°C\nMin: {min_temp:.1f}°C"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig, ax

# =============================================================================
# PART 5: RAINFALL ANALYSIS
# =============================================================================

def plot_asean_rainfall(weather_gdf):
    """
    Create rainfall map for ASEAN region
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Load boundaries
    world_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(world_url)
    
    asean_countries = ['Singapore', 'Indonesia', 'Malaysia', 'Thailand', 'Philippines',
                       'Vietnam', 'Myanmar', 'Cambodia', 'Laos', 'Brunei']
    asean = world[world.NAME.isin(asean_countries)]
    
    # Plot boundaries
    asean.boundary.plot(ax=ax, linewidth=1.5, edgecolor='navy', alpha=0.7)
    asean.plot(ax=ax, color='lightblue', alpha=0.1)
    
    # Plot rainfall data
    weather_gdf.plot(
        column='rainfall_24h',
        cmap='Blues',
        legend=True,
        markersize=200,
        ax=ax,
        alpha=0.8,
        edgecolor='darkblue',
        linewidth=0.8,
        legend_kwds={'label': '24h Rainfall (mm)', 'shrink': 0.8}
    )
    
    # Add labels for high rainfall areas
    high_rainfall = weather_gdf[weather_gdf['rainfall_24h'] > 20]
    for idx, row in high_rainfall.iterrows():
        ax.annotate(
            f"{row['city']}\n{row['rainfall_24h']:.1f}mm",
            xy=(row.geometry.x, row.geometry.y),
            xytext=(3, 3),
            textcoords='offset points',
            fontsize=8,
            color='darkblue',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
    
    # Set bounds
    ax.set_xlim(95, 125)
    ax.set_ylim(-10, 25)
    
    ax.set_title('ASEAN Region - 24 Hour Rainfall', fontsize=18, pad=20, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

# =============================================================================
# PART 6: MULTI-PARAMETER DASHBOARD
# =============================================================================

def create_asean_weather_dashboard(weather_gdf):
    """
    Create comprehensive weather dashboard for ASEAN
    """
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Load boundaries
    world_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(world_url)
    asean_countries = ['Singapore', 'Indonesia', 'Malaysia', 'Thailand', 'Philippines',
                       'Vietnam', 'Myanmar', 'Cambodia', 'Laos', 'Brunei']
    asean = world[world.NAME.isin(asean_countries)]
    
    # 1. Temperature Map
    ax1 = fig.add_subplot(gs[0, 0])
    asean.boundary.plot(ax=ax1, linewidth=1, edgecolor='black', alpha=0.5)
    weather_gdf.plot(column='temperature', cmap='RdYlBu_r', legend=True,
                     markersize=100, ax=ax1, alpha=0.7, edgecolor='black',
                     legend_kwds={'label': 'Temperature (°C)', 'shrink': 0.7})
    ax1.set_xlim(95, 125)
    ax1.set_ylim(-10, 25)
    ax1.set_title('Temperature', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Rainfall Map
    ax2 = fig.add_subplot(gs[0, 1])
    asean.boundary.plot(ax=ax2, linewidth=1, edgecolor='black', alpha=0.5)
    weather_gdf.plot(column='rainfall_24h', cmap='Blues', legend=True,
                     markersize=100, ax=ax2, alpha=0.7, edgecolor='darkblue',
                     legend_kwds={'label': 'Rainfall (mm)', 'shrink': 0.7})
    ax2.set_xlim(95, 125)
    ax2.set_ylim(-10, 25)
    ax2.set_title('24-Hour Rainfall', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Humidity Map
    ax3 = fig.add_subplot(gs[1, 0])
    asean.boundary.plot(ax=ax3, linewidth=1, edgecolor='black', alpha=0.5)
    weather_gdf.plot(column='humidity', cmap='Greens', legend=True,
                     markersize=100, ax=ax3, alpha=0.7, edgecolor='darkgreen',
                     legend_kwds={'label': 'Humidity (%)', 'shrink': 0.7})
    ax3.set_xlim(95, 125)
    ax3.set_ylim(-10, 25)
    ax3.set_title('Humidity', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Wind Speed Map
    ax4 = fig.add_subplot(gs[1, 1])
    asean.boundary.plot(ax=ax4, linewidth=1, edgecolor='black', alpha=0.5)
    weather_gdf.plot(column='wind_speed', cmap='YlOrRd', legend=True,
                     markersize=100, ax=ax4, alpha=0.7, edgecolor='darkred',
                     legend_kwds={'label': 'Wind Speed (km/h)', 'shrink': 0.7})
    ax4.set_xlim(95, 125)
    ax4.set_ylim(-10, 25)
    ax4.set_title('Wind Speed', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Statistical Summary
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    summary_data = []
    for country in weather_gdf['country'].unique():
        country_data = weather_gdf[weather_gdf['country'] == country]
        summary_data.append({
            'Country': country,
            'Stations': len(country_data),
            'Avg Temp': f"{country_data['temperature'].mean():.1f}°C",
            'Avg Rainfall': f"{country_data['rainfall_24h'].mean():.1f}mm",
            'Avg Humidity': f"{country_data['humidity'].mean():.1f}%",
            'Avg Wind': f"{country_data['wind_speed'].mean():.1f}km/h"
        })
    
    summary_df = pd.DataFrame(summary_data)
    table = ax5.table(cellText=summary_df.values, colLabels=summary_df.columns,
                     cellLoc='center', loc='center', bbox=[0.1, 0.3, 0.8, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax5.set_title('Statistical Summary by Country', fontsize=14, 
                  fontweight='bold', pad=20)
    
    # Main title
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M SGT')
    fig.suptitle(f'ASEAN Weather Dashboard\n{timestamp}', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ASEAN & Singapore Weather Visualization")
    print("=" * 70)
    
    # Fetch and combine data
    print("\n1. Fetching weather data from NEA and regional sources...")
    weather_data = combine_nea_and_regional_data()
    print(f"   Retrieved data for {len(weather_data)} stations")
    print(f"   Countries covered: {', '.join(weather_data['country'].unique())}")
    
    # ASEAN Temperature Map
    print("\n2. Creating ASEAN temperature map...")
    fig1, ax1 = plot_asean_temperature_map(weather_data)
    plt.savefig('asean_temperature.png', dpi=300, bbox_inches='tight')
    print("   Saved: asean_temperature.png")
    
    # Singapore Detailed View
    print("\n3. Creating Singapore detailed view...")
    fig2, ax2 = plot_singapore_detailed(weather_data)
    if fig2:
        plt.savefig('singapore_detailed.png', dpi=300, bbox_inches='tight')
        print("   Saved: singapore_detailed.png")
    
    # Rainfall Map
    print("\n4. Creating ASEAN rainfall map...")
    fig3, ax3 = plot_asean_rainfall(weather_data)
    plt.savefig('asean_rainfall.png', dpi=300, bbox_inches='tight')
    print("   Saved: asean_rainfall.png")
    
    # Weather Dashboard
    print("\n5. Creating comprehensive weather dashboard...")
    fig4 = create_asean_weather_dashboard(weather_data)
    plt.savefig('asean_dashboard.png', dpi=300, bbox_inches='tight')
    print("   Saved: asean_dashboard.png")
    
    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("\nData Sources:")
    print("- NEA Singapore: https://api.data.gov.sg/v1/environment")
    print("- Regional data: Sample/Simulated (integrate real APIs as needed)")
    print("\nNext Steps:")
    print("- Integrate APIs from other ASEAN meteorological agencies")
    print("- Add real-time satellite imagery overlay")
    print("- Create time-series animations")
    print("- Add weather alerts and warnings")
    
    plt.show()
