"""
Global Deforestation and Climate Change Analysis
This script analyzes the relationship between deforestation and climate change,
creating visualizations to support academic research.
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

# Set plotting style
plt.style.use('seaborn-whitegrid')
sns.set_palette('viridis')
sns.set_context("paper", font_scale=1.2)

def load_and_prepare_data():
    """
    Load and prepare deforestation and climate data.
    In a real implementation, you would load these from CSV files or APIs.
    Here we'll generate sample data for demonstration purposes.
    """
    print("Loading and preparing data...")
    
    # Generate sample time series data (1990-2022)
    years = range(1990, 2023)
    
    # Sample data - in reality you would load this from actual datasets
    data = {
        'Year': list(years),
        'Global_Forest_Area_MHa': [4168 - i*5.2 for i in range(len(years))],  # Declining trend
        'Temperature_Anomaly_C': [0.3 + i*0.02 + np.random.normal(0, 0.05) for i in range(len(years))],  # Rising trend with noise
        'CO2_PPM': [350 + i*2.1 + np.random.normal(0, 0.5) for i in range(len(years))],  # Rising trend with noise
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Calculate derived metrics
    df['Annual_Forest_Loss'] = df['Global_Forest_Area_MHa'].diff().fillna(0) * -1  # Convert to positive for loss
    df['Cumulative_Forest_Loss'] = df['Annual_Forest_Loss'].cumsum()
    
    # Generate regional data 
    regions = ['Amazon', 'Congo Basin', 'Southeast Asia', 'Boreal Forest', 'North America', 'Europe']
    
    # Different deforestation rates for different regions
    rates = [7.5, 5.2, 8.1, 2.8, 1.5, 0.9]
    regional_data = {region: [] for region in regions}
    
    for i in range(len(years)):
        for j, region in enumerate(regions):
            # Create region-specific trends with some randomness
            value = rates[j] + i * rates[j]/30 + np.random.normal(0, rates[j]/10)
            if value < 0:  # Prevent negative values
                value = 0.1
            regional_data[region].append(value)
    
    for region in regions:
        df[f'{region}_Loss'] = regional_data[region]
    
    print(f"Data prepared with {len(df)} records from {df['Year'].min()} to {df['Year'].max()}.")
    return df

def load_geospatial_data():
    """
    Load world geospatial data for map visualizations.
    """
    print("Loading geospatial data...")
    
    # Use geopandas built-in dataset for world countries
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    
    # Create sample deforestation data by country
    countries = world['name'].tolist()
    
    # Create dictionary with sample deforestation rates
    # In reality, you would join with actual data from a source like Global Forest Watch
    np.random.seed(42)  # For reproducibility
    forest_loss = {}
    for country in countries:
        # Assign higher deforestation to tropical countries (simplified)
        is_tropical = country in ['Brazil', 'Indonesia', 'Democratic Republic of the Congo', 
                                  'Malaysia', 'Bolivia', 'Colombia', 'Peru']
        
        if is_tropical:
            forest_loss[country] = np.random.uniform(1.5, 4.5)
        else:
            forest_loss[country] = np.random.uniform(0, 1.5)
    
    # Add forest loss data to the GeoDataFrame
    world['forest_loss_percent'] = world['name'].map(forest_loss)
    
    print(f"Geospatial data prepared with {len(world)} countries.")
    return world

def create_time_series_plots(df):
    """
    Create time series charts showing deforestation trends and climate indicators.
    """
    print("Creating time series plots...")
    
    # Figure 1: Forest Area and Temperature Anomaly Over Time
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:green'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Global Forest Area (Million Hectares)', color=color)
    ax1.plot(df['Year'], df['Global_Forest_Area_MHa'], color=color, linewidth=2.5)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Temperature Anomaly (°C)', color=color)
    ax2.plot(df['Year'], df['Temperature_Anomaly_C'], color=color, linewidth=2.5)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Global Forest Area and Temperature Anomalies (1990-2022)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig('forest_and_temperature_trends.png', dpi=300, bbox_inches='tight')
    
    # Figure 2: Annual Forest Loss by Region
    plt.figure(figsize=(14, 8))
    
    regions = ['Amazon', 'Congo Basin', 'Southeast Asia', 'Boreal Forest', 'North America', 'Europe']
    for region in regions:
        plt.plot(df['Year'], df[f'{region}_Loss'], linewidth=2, label=region)
    
    plt.xlabel('Year')
    plt.ylabel('Annual Forest Loss (Million Hectares)')
    plt.title('Annual Forest Loss by Region (1990-2022)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig('regional_forest_loss.png', dpi=300, bbox_inches='tight')
    
    # Figure 3: Cumulative Forest Loss and CO2 Concentration
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:brown'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Cumulative Forest Loss (Million Hectares)', color=color)
    ax1.plot(df['Year'], df['Cumulative_Forest_Loss'], color=color, linewidth=2.5)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('CO₂ Concentration (PPM)', color=color)
    ax2.plot(df['Year'], df['CO2_PPM'], color=color, linewidth=2.5)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Cumulative Forest Loss and CO₂ Concentration (1990-2022)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig('forest_loss_and_co2.png', dpi=300, bbox_inches='tight')
    
    print("Time series plots created successfully.")

def create_map_visualizations(world_data):
    """
    Create map visualizations showing global deforestation patterns.
    """
    print("Creating map visualizations...")
    
    # World map of deforestation rates
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    # Create a custom colormap from green to red
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#1a9850', '#ffffbf', '#d73027'])
    
    # Plot the map with deforestation data
    world_data.plot(column='forest_loss_percent', 
                    ax=ax,
                    legend=True,
                    cmap=cmap,
                    legend_kwds={'label': 'Annual Forest Loss (%)',
                                'orientation': 'horizontal',
                                'shrink': 0.6})
    
    ax.set_title('Global Deforestation Rates by Country', fontsize=16, fontweight='bold')
    ax.set_axis_off()
    
    plt.savefig('global_deforestation_map.png', dpi=300, bbox_inches='tight')
    
    print("Map visualizations created successfully.")

def analyze_correlations(df):
    """
    Analyze and visualize correlations between deforestation and climate indicators.
    """
    print("Analyzing correlations...")
    
    # Select relevant columns for correlation analysis
    correlation_data = df[['Global_Forest_Area_MHa', 'Annual_Forest_Loss', 
                         'Cumulative_Forest_Loss', 'Temperature_Anomaly_C', 'CO2_PPM']]
    
    # Calculate correlation matrix
    corr_matrix = correlation_data.corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title('Correlation Between Deforestation and Climate Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('deforestation_climate_correlation.png', dpi=300, bbox_inches='tight')
    
    # Scatter plot with regression line: Forest Loss vs Temperature
    plt.figure(figsize=(10, 6))
    sns.regplot(x='Cumulative_Forest_Loss', y='Temperature_Anomaly_C', data=df, 
                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    
    # Calculate and display correlation coefficient
    corr, p = pearsonr(df['Cumulative_Forest_Loss'], df['Temperature_Anomaly_C'])
    plt.annotate(f'Correlation: r = {corr:.2f} (p = {p:.4f})', 
                 xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12)
    
    plt.title('Relationship Between Cumulative Forest Loss and Temperature Anomalies', fontsize=14, fontweight='bold')
    plt.xlabel('Cumulative Forest Loss (Million Hectares)')
    plt.ylabel('Temperature Anomaly (°C)')
    plt.tight_layout()
    plt.savefig('forest_loss_temperature_relationship.png', dpi=300, bbox_inches='tight')
    
    print("Correlation analysis completed successfully.")

def create_advanced_visualizations(df):
    """
    Create additional insightful visualizations.
    """
    print("Creating advanced visualizations...")
    
    # 1. Area chart showing composition of forest loss by region over time
    plt.figure(figsize=(14, 8))
    
    regions = ['Amazon', 'Congo Basin', 'Southeast Asia', 'Boreal Forest', 'North America', 'Europe']
    region_data = df[[f'{region}_Loss' for region in regions]].values.T
    
    plt.stackplot(df['Year'], region_data, labels=regions, alpha=0.8)
    
    plt.xlabel('Year')
    plt.ylabel('Forest Loss (Million Hectares)')
    plt.title('Composition of Global Forest Loss by Region', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig('forest_loss_composition.png', dpi=300, bbox_inches='tight')
    
    # 2. Create a dashboard-style multi-panel figure
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Forest area trend
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['Year'], df['Global_Forest_Area_MHa'], 'g-', linewidth=3)
    ax1.set_title('Global Forest Area', fontsize=14)
    ax1.set_ylabel('Million Hectares')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Temperature anomaly trend
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['Year'], df['Temperature_Anomaly_C'], 'r-', linewidth=3)
    ax2.set_title('Global Temperature Anomaly', fontsize=14)
    ax2.set_ylabel('°C')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Annual deforestation trend
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(df['Year'], df['Annual_Forest_Loss'], color='brown', alpha=0.7)
    ax3.set_title('Annual Global Forest Loss', fontsize=14)
    ax3.set_ylabel('Million Hectares')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: CO2 concentration trend
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df['Year'], df['CO2_PPM'], 'b-', linewidth=3)
    ax4.set_title('Atmospheric CO₂ Concentration', fontsize=14) 
    ax4.set_ylabel('Parts Per Million (PPM)')
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Regional comparison (most recent year)
    ax5 = fig.add_subplot(gs[2, 0])
    most_recent_year = df.iloc[-1]
    regions = ['Amazon', 'Congo Basin', 'Southeast Asia', 'Boreal Forest', 'North America', 'Europe']
    regional_values = [most_recent_year[f'{region}_Loss'] for region in regions]
    
    colors = sns.color_palette('viridis', len(regions))
    ax5.bar(regions, regional_values, color=colors)
    ax5.set_title(f'Regional Forest Loss in {int(most_recent_year["Year"])}', fontsize=14)
    ax5.set_ylabel('Million Hectares')
    plt.xticks(rotation=45, ha='right')
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Scatter plot of forest loss vs temperature
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.scatter(df['Cumulative_Forest_Loss'], df['Temperature_Anomaly_C'], 
               alpha=0.7, s=80, c=df['Year'], cmap='viridis')
    
    # Add regression line
    x = sm.add_constant(df['Cumulative_Forest_Loss'])
    model = sm.OLS(df['Temperature_Anomaly_C'], x).fit()
    x_pred = np.linspace(df['Cumulative_Forest_Loss'].min(), df['Cumulative_Forest_Loss'].max(), 100)
    x_pred2 = sm.add_constant(x_pred)
    y_pred = model.predict(x_pred2)
    ax6.plot(x_pred, y_pred, 'r--', linewidth=2)
    
    ax6.set_title('Forest Loss vs Temperature Relationship', fontsize=14)
    ax6.set_xlabel('Cumulative Forest Loss (Million Hectares)')
    ax6.set_ylabel('Temperature Anomaly (°C)')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Global Deforestation and Climate Change Dashboard', fontsize=20, fontweight='bold', y=0.98)
    plt.savefig('deforestation_climate_dashboard.png', dpi=300, bbox_inches='tight')
    
    print("Advanced visualizations created successfully.")

def create_interactive_visualizations(df):
    """
    Create interactive visualizations using Plotly.
    These can be saved as HTML files and included in digital versions of your paper.
    """
    print("Creating interactive visualizations...")
    
    # 1. Interactive time series with multiple traces and rangeslider
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=df['Year'], y=df['Global_Forest_Area_MHa'], name="Forest Area",
                  line=dict(color='green', width=3)),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=df['Year'], y=df['Temperature_Anomaly_C'], name="Temperature Anomaly",
                  line=dict(color='red', width=3)),
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
    fig.update_yaxes(title_text="Forest Area (Million Hectares)", secondary_y=False)
    fig.update_yaxes(title_text="Temperature Anomaly (°C)", secondary_y=True)
    
    fig.write_html("interactive_forest_temperature.html")
    
    # 2. Interactive choropleth map of deforestation (using sample data)
    # In a real implementation, you would use actual country-level data
    
    # Create sample country data
    countries = ['USA', 'CAN', 'MEX', 'BRA', 'ARG', 'GBR', 'FRA', 'DEU', 'RUS', 'CHN', 
                'IND', 'AUS', 'ZAF', 'EGY', 'NGA', 'IDN', 'COD']
    
    # Sample deforestation rates - in reality, use actual data
    np.random.seed(42)
    forest_loss_rates = np.random.uniform(0, 5, size=len(countries))
    
    # Higher rates for countries with known high deforestation
    forest_loss_rates[countries.index('BRA')] = 4.8  # Brazil
    forest_loss_rates[countries.index('IDN')] = 4.5  # Indonesia
    forest_loss_rates[countries.index('COD')] = 4.2  # DR Congo
    
    map_fig = px.choropleth(
        locations=countries,
        locationmode="ISO-3",
        color=forest_loss_rates,
        color_continuous_scale="RdYlGn_r",
        range_color=(0, 5),
        title="Global Deforestation Rates by Country",
        labels={'color': 'Annual Forest Loss (%)'}
    )
    
    map_fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        )
    )
    
    map_fig.write_html("interactive_deforestation_map.html")
    
    # 3. Interactive regional comparison with animation over time
    years = df['Year'].tolist()
    regions = ['Amazon', 'Congo Basin', 'Southeast Asia', 'Boreal Forest', 'North America', 'Europe']
    
    # Create data for animation
    animation_data = []
    for year_idx, year in enumerate(years):
        for region in regions:
            animation_data.append({
                'Year': year,
                'Region': region,
                'Forest_Loss': df.iloc[year_idx][f'{region}_Loss']
            })
    
    animation_df = pd.DataFrame(animation_data)
    
    animated_fig = px.bar(
        animation_df,
        x="Region",
        y="Forest_Loss",
        animation_frame="Year",
        color="Region",
        range_y=[0, animation_df['Forest_Loss'].max() * 1.1],
        title="Annual Forest Loss by Region Over Time",
        labels={'Forest_Loss': 'Forest Loss (Million Hectares)'}
    )
    
    animated_fig.update_layout(
        xaxis_title="Region",
        yaxis_title="Forest Loss (Million Hectares)",
        legend_title="Region"
    )
    
    animated_fig.write_html("interactive_regional_comparison.html")
    
    print("Interactive visualizations created successfully.")

def run_deforestation_analysis():
    """
    Main function to run the deforestation analysis.
    """
    print("Starting global deforestation and climate change analysis...")
    
    # Load and prepare data
    df = load_and_prepare_data()
    world_data = load_geospatial_data()
    
    # Create visualizations
    create_time_series_plots(df)
    create_map_visualizations(world_data)
    analyze_correlations(df)
    create_advanced_visualizations(df)
    create_interactive_visualizations(df)
    
    print("\nAnalysis complete! All visualizations have been saved.")
    print("\nSuggested uses in your paper:")
    print("1. Use 'forest_and_temperature_trends.png' to show the inverse relationship between forest area and rising temperatures")
    print("2. Include 'global_deforestation_map.png' to highlight geographic patterns of forest loss")
    print("3. Use 'deforestation_climate_correlation.png' to demonstrate statistical relationships")
    print("4. Include 'forest_loss_composition.png' to show where most deforestation is occurring")
    print("5. The dashboard ('deforestation_climate_dashboard.png') provides a comprehensive overview")
    print("6. For digital publications, include the interactive HTML visualizations")

if __name__ == "__main__":
    run_deforestation_analysis()
