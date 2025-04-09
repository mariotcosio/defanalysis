"""
Global Deforestation and Climate Change Analysis
This script analyzes the relationship between deforestation and climate change,
creating visualizations to support academic research.

Author: mariotcosio
Created: 2025-04-09 04:57:36 UTC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime, timezone
import getpass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Set basic style parameters
plt.rcParams.update({
    'figure.figsize': [10.0, 6.0],
    'figure.dpi': 100,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#cccccc',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.axisbelow': True,
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'text.color': 'black',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14
})

# Configure seaborn without using style sheets
sns.set_context("paper", font_scale=1.2)

class DeforestationData:
    def __init__(self):
        self.years = range(1990, 2023)
        self.regions = [
            'Amazon', 'Congo Basin', 'Southeast Asia', 
            'Boreal Forest', 'North America', 'Europe'
        ]
        self.tropical_countries = {
            'BRA': 'Brazil',
            'IDN': 'Indonesia',
            'COD': 'Democratic Republic of the Congo',
            'MYS': 'Malaysia',
            'BOL': 'Bolivia',
            'COL': 'Colombia',
            'PER': 'Peru'
        }
        self.current_time = datetime.now(timezone.utc)
        self.ne_countries_path = Path("ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")

    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Load and prepare deforestation and climate data.
        Returns:
            pd.DataFrame: Prepared dataset with deforestation and climate metrics
        """
        logger.info("Loading and preparing data...")
        
        try:
            # Generate sample time series data
            data = {
                'Year': list(self.years),
                'Global_Forest_Area_MHa': [4168 - i*5.2 for i in range(len(self.years))],
                'Temperature_Anomaly_C': [
                    0.3 + i*0.02 + np.random.normal(0, 0.05) 
                    for i in range(len(self.years))
                ],
                'CO2_PPM': [
                    350 + i*2.1 + np.random.normal(0, 0.5) 
                    for i in range(len(self.years))
                ]
            }
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Calculate derived metrics
            df['Annual_Forest_Loss'] = (
                df['Global_Forest_Area_MHa'].diff().fillna(0) * -1
            )
            df['Cumulative_Forest_Loss'] = df['Annual_Forest_Loss'].cumsum()
            
            # Add regional data
            self._add_regional_data(df)
            
            logger.info(f"Data prepared with {len(df)} records from "
                       f"{df['Year'].min()} to {df['Year'].max()}.")
            return df
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise

    def _add_regional_data(self, df: pd.DataFrame) -> None:
        """
        Add regional deforestation data to the DataFrame.
        Args:
            df: DataFrame to add regional data to
        """
        # Different deforestation rates for different regions
        rates = {
            'Amazon': 7.5,
            'Congo Basin': 5.2,
            'Southeast Asia': 8.1,
            'Boreal Forest': 2.8,
            'North America': 1.5,
            'Europe': 0.9
        }
        
        np.random.seed(42)  # For reproducibility
        for region, base_rate in rates.items():
            values = []
            for i in range(len(self.years)):
                # Create region-specific trends with some randomness
                value = base_rate + i * base_rate/30 + np.random.normal(0, base_rate/10)
                values.append(max(0.1, value))  # Ensure non-negative values
            df[f'{region}_Loss'] = values

    def load_geospatial_data(self) -> gpd.GeoDataFrame:
        """
        Load and prepare world geospatial data for map visualizations.
        Returns:
            gpd.GeoDataFrame: Prepared geospatial data with deforestation rates
        """
        logger.info("Loading geospatial data...")
        
        try:
            if not self.ne_countries_path.exists():
                raise FileNotFoundError(
                    f"Natural Earth countries shapefile not found at {self.ne_countries_path}"
                )
            
            # Load Natural Earth countries
            world = gpd.read_file(self.ne_countries_path)
            
            # Create sample deforestation data
            forest_loss = {}
            np.random.seed(42)  # For reproducibility
            
            for idx, row in world.iterrows():
                country_code = row['ADM0_A3']  # Natural Earth uses ADM0_A3 for country codes
                # Assign higher deforestation to tropical countries
                if country_code in self.tropical_countries:
                    forest_loss[country_code] = np.random.uniform(1.5, 4.5)
                else:
                    forest_loss[country_code] = np.random.uniform(0, 1.5)
            
            # Add forest loss data to the GeoDataFrame
            world['forest_loss_percent'] = world['ADM0_A3'].map(forest_loss)
            
            logger.info(f"Geospatial data prepared with {len(world)} countries.")
            return world
            
        except Exception as e:
            logger.error(f"Error in geospatial data loading: {str(e)}")
            raise

class DeforestationVisualizer:
    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = output_dir
        self.timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def _get_output_path(self, filename: str) -> Path:
        """Create a timestamped output path."""
        base_name = filename.rsplit('.', 1)[0]
        extension = filename.rsplit('.', 1)[1]
        return self.output_dir / f"{base_name}_{self.timestamp}.{extension}"

    def create_visualizations(self, df: pd.DataFrame, world_data: gpd.GeoDataFrame) -> None:
        """Create all visualizations."""
        logger.info("Creating visualizations...")
        
        try:
            self._create_forest_temperature_plot(df)
            self._create_regional_loss_plot(df)
            self._create_cumulative_co2_plot(df)
            self._create_deforestation_map(world_data)
            self._create_interactive_visualizations(df)
            
            logger.info("All visualizations created successfully.")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            raise

    def _create_forest_temperature_plot(self, df: pd.DataFrame) -> None:
        """Create forest area and temperature anomaly plot."""
        try:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Forest area plot
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Global Forest Area (Million Hectares)', color='darkgreen')
            line1 = ax1.plot(df['Year'], df['Global_Forest_Area_MHa'], 
                           color='darkgreen', linewidth=2.5, label='Forest Area')
            ax1.tick_params(axis='y', labelcolor='darkgreen')
            
            # Temperature anomaly plot
            ax2 = ax1.twinx()
            ax2.set_ylabel('Temperature Anomaly (°C)', color='darkred')
            line2 = ax2.plot(df['Year'], df['Temperature_Anomaly_C'], 
                           color='darkred', linewidth=2.5, label='Temperature')
            ax2.tick_params(axis='y', labelcolor='darkred')
            
            # Title and legend
            plt.title('Global Forest Area and Temperature Anomalies (1990-2022)', 
                     pad=20, fontweight='bold')
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right')
            
            plt.tight_layout()
            plt.savefig(self._get_output_path('forest_temperature.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating forest-temperature plot: {str(e)}")
            plt.close()
            raise

    def _create_regional_loss_plot(self, df: pd.DataFrame) -> None:
        """Create regional forest loss plot."""
        try:
            plt.figure(figsize=(14, 8))
            
            colors = sns.color_palette("husl", n_colors=6)
            regions = ['Amazon', 'Congo Basin', 'Southeast Asia',
                      'Boreal Forest', 'North America', 'Europe']
            
            for region, color in zip(regions, colors):
                plt.plot(df['Year'], df[f'{region}_Loss'],
                        linewidth=2.5, label=region, color=color)
            
            plt.xlabel('Year')
            plt.ylabel('Annual Forest Loss (Million Hectares)')
            plt.title('Annual Forest Loss by Region (1990-2022)',
                     pad=20, fontweight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self._get_output_path('regional_loss.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating regional loss plot: {str(e)}")
            plt.close()
            raise

    def _create_cumulative_co2_plot(self, df: pd.DataFrame) -> None:
        """Create cumulative forest loss and CO2 plot."""
        try:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Cumulative forest loss plot
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Cumulative Forest Loss (Million Hectares)', 
                         color='saddlebrown')
            line1 = ax1.plot(df['Year'], df['Cumulative_Forest_Loss'],
                           color='saddlebrown', linewidth=2.5, 
                           label='Cumulative Forest Loss')
            ax1.tick_params(axis='y', labelcolor='saddlebrown')
            
            # CO2 concentration plot
            ax2 = ax1.twinx()
            ax2.set_ylabel('CO₂ Concentration (PPM)', color='navy')
            line2 = ax2.plot(df['Year'], df['CO2_PPM'],
                           color='navy', linewidth=2.5, 
                           label='CO₂ Concentration')
            ax2.tick_params(axis='y', labelcolor='navy')
            
            plt.title('Cumulative Forest Loss and CO₂ Concentration (1990-2022)',
                     pad=20, fontweight='bold')
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
            
            plt.tight_layout()
            plt.savefig(self._get_output_path('cumulative_co2.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating cumulative CO2 plot: {str(e)}")
            plt.close()
            raise

    def _create_deforestation_map(self, world_data: gpd.GeoDataFrame) -> None:
        """Create world map of deforestation rates."""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            
            cmap = LinearSegmentedColormap.from_list(
                'custom_cmap', 
                ['#1a9850', '#ffffbf', '#d73027']
            )
            
            world_data.plot(
                column='forest_loss_percent',
                ax=ax,
                legend=True,
                cmap=cmap,
                legend_kwds={
                    'label': 'Annual Forest Loss (%)',
                    'orientation': 'horizontal',
                    'shrink': 0.6
                }
            )
            
            ax.set_title('Global Deforestation Rates by Country',
                        fontsize=16, fontweight='bold')
            ax.set_axis_off()
            
            plt.savefig(self._get_output_path('deforestation_map.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating deforestation map: {str(e)}")
            plt.close()
            raise

    def _create_interactive_visualizations(self, df: pd.DataFrame) -> None:
        """Create interactive Plotly visualizations."""
        try:
            # Time series visualization
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(
                    x=df['Year'],
                    y=df['Global_Forest_Area_MHa'],
                    name="Forest Area",
                    line=dict(color='green', width=3)
                ),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['Year'],
                    y=df['Temperature_Anomaly_C'],
                    name="Temperature Anomaly",
                    line=dict(color='red', width=3)
                ),
                secondary_y=True,
            )
            
            fig.update_layout(
                title_text="Forest Area and Temperature Anomalies Over Time",
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    type="linear"
                )
            )
            
            fig.update_xaxes(title_text="Year")
            fig.update_yaxes(title_text="Forest Area (Million Hectares)",
                           secondary_y=False)
            fig.update_yaxes(title_text="Temperature Anomaly (°C)",
                           secondary_y=True)
            
            fig.write_html(str(self._get_output_path("interactive_forest_temperature.html")))
            
        except Exception as e:
            logger.error(f"Error creating interactive visualizations: {str(e)}")
            raise

def main():
    """Main function to run the deforestation analysis."""
    try:
        logger.info(
            f"Starting global deforestation analysis - "
            f"User: {getpass.getuser()}, "
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        
        # Initialize classes
        data_handler = DeforestationData()
        visualizer = DeforestationVisualizer()
        
        # Load and prepare data
        df = data_handler.load_and_prepare_data()
        world_data = data_handler.load_geospatial_data()
        
        # Create visualizations
        visualizer.create_visualizations(df, world_data)
        
        logger.info("\nAnalysis complete! All visualizations have been saved.")
        logger.info(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()