import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import numpy as np

# Enhanced forest-inspired color palette
COLORS = {
    'background': '#E8F3EC',      # Light mint background
    'panel': '#FFFFFF',           # White panel background
    'text': '#1B4332',           # Deep forest green text
    'primary': '#2D6A4F',        # Dark forest green
    'secondary': '#40916C',      # Medium forest green
    'accent1': '#52B788',        # Bright forest green
    'accent2': '#74C69D',        # Light forest green
    'accent3': '#B7E4C7',        # Pale green
    'warning': '#BC6C25',        # Earth brown (for deforestation)
    'grid': '#DDE5B6',           # Light sage for grid
    'chart_colors': [
        '#2D6A4F',  # Deep forest green
        '#40916C',  # Medium forest green
        '#52B788',  # Bright forest green
        '#74C69D',  # Light forest green
        '#95D5B2',  # Pale forest green
        '#B7E4C7'   # Very pale green
    ]
}

# Common chart layout settings
CHART_LAYOUT = {
    'paper_bgcolor': COLORS['panel'],
    'plot_bgcolor': COLORS['background'],
    'font': {
        'color': COLORS['text'],
        'family': 'Helvetica, Arial, sans-serif'
    },
    'title': {
        'font': {
            'color': COLORS['text'],
            'size': 20,
            'family': 'Helvetica, Arial, sans-serif'
        },
        'y': 0.95
    },
    'margin': {'t': 60, 'r': 30, 'l': 30, 'b': 30},
    'xaxis': {
        'gridcolor': COLORS['grid'],
        'showgrid': True,
        'zeroline': False,
        'linecolor': COLORS['text'],
        'title': {'font': {'size': 14}}
    },
    'yaxis': {
        'gridcolor': COLORS['grid'],
        'showgrid': True,
        'zeroline': False,
        'linecolor': COLORS['text'],
        'title': {'font': {'size': 14}}
    }
}

# Initialize the Dash app
app = dash.Dash(__name__)

# Load the data
try:
    df = pd.read_csv('deforestation_climate_data.csv')
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading data: {str(e)}")
    raise

# Dashboard layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1('Forest Conservation Dashboard',
                style={
                    'textAlign': 'center',
                    'color': COLORS['text'],
                    'fontFamily': 'Helvetica, Arial, sans-serif',
                    'fontSize': '2.5em',
                    'padding': '20px',
                    'marginBottom': '10px',
                    'borderBottom': f'3px solid {COLORS["secondary"]}'
                }),
        html.P('Tracking Global Deforestation and Climate Change Patterns',
               style={
                   'textAlign': 'center',
                   'color': COLORS['secondary'],
                   'fontSize': '1.2em',
                   'marginBottom': '30px'
               })
    ], style={'backgroundColor': COLORS['panel'], 'borderRadius': '10px', 'margin': '20px 0'}),
    
    # Timestamp and user info with specified format
    html.Div([
        dcc.Interval(
            id='interval-component',
            interval=1*1000,  # updates every second
            n_intervals=0
        ),
        html.P([
            html.Strong("Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): "),
            html.Span(id='live-timestamp', style={'fontFamily': 'monospace'})
        ], style={'color': COLORS['text'], 'fontSize': '0.9em', 'margin': '5px'}),
        html.P([
            html.Strong("Current User's Login: "),
            html.Span("mariotcosio")
        ], style={'color': COLORS['text'], 'fontSize': '0.9em', 'margin': '5px'})
    ], style={
        'textAlign': 'right',
        'padding': '15px',
        'backgroundColor': COLORS['panel'],
        'borderRadius': '10px',
        'marginBottom': '20px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),
    
    # Year Range Slider
    html.Div([
        html.Label('Select Time Period:',
                  style={
                      'color': COLORS['text'],
                      'fontSize': '1.1em',
                      'fontWeight': 'bold'
                  }),
        dcc.RangeSlider(
            id='year-slider',
            min=df['Year'].min(),
            max=df['Year'].max(),
            value=[df['Year'].min(), df['Year'].max()],
            marks={str(year): {'label': str(year),
                             'style': {'color': COLORS['text']}}
                   for year in df['Year'].unique()[::5]},
            step=1
        )
    ], style={
        'width': '90%',
        'margin': '20px auto',
        'padding': '20px',
        'backgroundColor': COLORS['panel'],
        'borderRadius': '10px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),
    
    # First row - Forest Area and Temperature
    html.Div([
        html.Div([
            dcc.Graph(id='forest-area-trend')
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='temperature-anomaly')
        ], style={'width': '48%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px 0'}),
    
    # Second row - Forest Loss and CO2
    html.Div([
        html.Div([
            dcc.Graph(id='annual-forest-loss')
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='co2-concentration')
        ], style={'width': '48%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px 0'}),
    
    # Third row - Regional Comparison and Correlation
    html.Div([
        html.Div([
            dcc.Graph(id='regional-loss-comparison')
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='loss-temp-correlation')
        ], style={'width': '48%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px 0'})
    
], style={'backgroundColor': COLORS['background'], 'padding': '20px', 'minHeight': '100vh'})

# Callback for updating timestamp
@app.callback(
    Output('live-timestamp', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_time(n):
    return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

# Callback for forest area trend
@app.callback(
    Output('forest-area-trend', 'figure'),
    [Input('year-slider', 'value')]
)
def update_forest_area(years):
    filtered_df = df[(df['Year'] >= years[0]) & (df['Year'] <= years[1])]
    
    y_values = filtered_df['Global_Forest_Area_MHa']
    y_min = y_values.min()
    y_max = y_values.max()
    
    # Calculate nice round numbers for y-axis intervals
    y_range = y_max - y_min
    interval = np.ceil(y_range / 10 / 100) * 100
    y_min_plot = np.floor(y_min / 100) * 100
    y_max_plot = np.ceil(y_max / 100) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_df['Year'],
        y=filtered_df['Global_Forest_Area_MHa'],
        mode='lines',
        name='Global Forest Area',
        line=dict(color=COLORS['primary'], width=3),
        fill='tonexty',
        fillcolor=COLORS['accent3']
    ))
    
    fig.update_layout(
        title={
            'text': 'Global Forest Area Trend',
            'font': {'color': COLORS['text'], 'size': 24}
        },
        paper_bgcolor=COLORS['panel'],
        plot_bgcolor=COLORS['background'],
        xaxis_title='Year',
        yaxis_title='Forest Area (Million Hectares)',
        yaxis=dict(
            range=[y_min_plot, y_max_plot],
            dtick=interval,
            gridcolor=COLORS['grid'],
            showgrid=True,
            zeroline=False,
            linecolor=COLORS['text'],
            tickformat=',.0f'
        ),
        hovermode='x',
        margin={'t': 60, 'r': 30, 'l': 30, 'b': 30}
    )
    
    return fig

# Callback for temperature anomaly
@app.callback(
    Output('temperature-anomaly', 'figure'),
    [Input('year-slider', 'value')]
)
def update_temperature(years):
    filtered_df = df[(df['Year'] >= years[0]) & (df['Year'] <= years[1])]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_df['Year'],
        y=filtered_df['Temperature_Anomaly_C'],
        mode='lines',
        name='Temperature Anomaly',
        line=dict(color=COLORS['warning'], width=3)
    ))
    
    fig.update_layout(
        title={
            'text': 'Global Temperature Anomaly',
            'font': {'color': COLORS['text'], 'size': 24}
        },
        paper_bgcolor=COLORS['panel'],
        plot_bgcolor=COLORS['background'],
        xaxis_title='Year',
        yaxis_title='Temperature Anomaly (°C)',
        hovermode='x',
        margin={'t': 60, 'r': 30, 'l': 30, 'b': 30}
    )
    
    return fig

# Callback for annual forest loss
@app.callback(
    Output('annual-forest-loss', 'figure'),
    [Input('year-slider', 'value')]
)
def update_annual_loss(years):
    filtered_df = df[(df['Year'] >= years[0]) & (df['Year'] <= years[1])].copy()
    
    fig = go.Figure()
    
    # Add the bar chart with values from CSV
    fig.add_trace(go.Bar(
        x=filtered_df['Year'],
        y=filtered_df['Annual_Forest_Loss'],
        name='Annual Forest Loss',
        marker_color=COLORS['warning'],
        opacity=0.7
    ))
    
    # Calculate and add the trend line
    trend_data = filtered_df['Annual_Forest_Loss'].rolling(window=3, center=True).mean()
    
    # Calculate and add the trend line
    trend_data = filtered_df['Annual_Forest_Loss'].rolling(window=3, center=True).mean()
    fig.add_trace(go.Scatter(
        x=filtered_df['Year'],
        y=trend_data,
        name='3-Year Moving Average',
        line=dict(color=COLORS['primary'], width=2, dash='dash'),
        opacity=0.8
    ))
    
    fig.update_layout(
        title={
            'text': 'Annual Global Forest Loss',
            'font': {'color': COLORS['text'], 'size': 24}
        },
        paper_bgcolor=COLORS['panel'],
        plot_bgcolor=COLORS['background'],
        xaxis_title='Year',
        yaxis_title='Forest Loss (Million Hectares)',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        hovermode='x unified',
        margin={'t': 60, 'r': 30, 'l': 30, 'b': 30}
    )
    
    # Update hover templates
    fig.update_traces(
        hovertemplate="<br>".join([
            "Year: %{x}",
            "Forest Loss: %{y:.2f} Million Hectares",
            "<extra></extra>"
        ]),
        selector=dict(type='bar')
    )
    
    fig.update_traces(
        hovertemplate="<br>".join([
            "Year: %{x}",
            "Average Loss: %{y:.2f} Million Hectares",
            "<extra></extra>"
        ]),
        selector=dict(type='scatter')
    )
    
    return fig

# Callback for CO2 concentration
@app.callback(
    Output('co2-concentration', 'figure'),
    [Input('year-slider', 'value')]
)
def update_co2(years):
    filtered_df = df[(df['Year'] >= years[0]) & (df['Year'] <= years[1])]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_df['Year'],
        y=filtered_df['CO2_PPM'],
        mode='lines',
        name='CO₂ Concentration',
        line=dict(color=COLORS['secondary'], width=3)
    ))
    
    fig.update_layout(
        title={
            'text': 'Atmospheric CO₂ Concentration',
            'font': {'color': COLORS['text'], 'size': 24}
        },
        paper_bgcolor=COLORS['panel'],
        plot_bgcolor=COLORS['background'],
        xaxis_title='Year',
        yaxis_title='CO₂ (Parts Per Million)',
        hovermode='x',
        margin={'t': 60, 'r': 30, 'l': 30, 'b': 30}
    )
    
    return fig

# Callback for regional comparison
@app.callback(
    Output('regional-loss-comparison', 'figure'),
    [Input('year-slider', 'value')]
)
def update_regional_comparison(years):
    filtered_df = df[(df['Year'] >= years[0]) & (df['Year'] <= years[1])]
    latest_year = filtered_df['Year'].max()
    latest_data = filtered_df[filtered_df['Year'] == latest_year]
    
    regions = ['Amazon_Loss', 'Congo_Basin_Loss', 'Southeast_Asia_Loss', 
               'Boreal_Forest_Loss', 'North_America_Loss', 'Europe_Loss']
    region_names = [r.replace('_Loss', '').replace('_', ' ') for r in regions]
    values = [latest_data[r].iloc[0] for r in regions]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=region_names,
        y=values,
        marker_color=COLORS['chart_colors']
    ))
    
    fig.update_layout(
        title={
            'text': f'Regional Forest Loss Comparison ({int(latest_year)})',
            'font': {'color': COLORS['text'], 'size': 24}
        },
        paper_bgcolor=COLORS['panel'],
        plot_bgcolor=COLORS['background'],
        xaxis_title='Region',
        yaxis_title='Forest Loss (Million Hectares)',
        showlegend=False,
        margin={'t': 60, 'r': 30, 'l': 30, 'b': 30}
    )
    
    fig.update_xaxes(tickangle=45)
    return fig

# Callback for loss-temperature correlation
@app.callback(
    Output('loss-temp-correlation', 'figure'),
    [Input('year-slider', 'value')]
)
def update_correlation(years):
    filtered_df = df[(df['Year'] >= years[0]) & (df['Year'] <= years[1])]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_df['Cumulative_Forest_Loss'],
        y=filtered_df['Temperature_Anomaly_C'],
        mode='markers',
        marker=dict(
            size=10,
            color=filtered_df['Year'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Year'),
            opacity=0.7
        )
    ))
    
    fig.update_layout(
        title={
            'text': 'Forest Loss vs Temperature Relationship',
            'font': {'color': COLORS['text'], 'size': 24}
        },
        paper_bgcolor=COLORS['panel'],
        plot_bgcolor=COLORS['background'],
        xaxis_title='Cumulative Forest Loss (Million Hectares)',
        yaxis_title='Temperature Anomaly (°C)',
        margin={'t': 60, 'r': 30, 'l': 30, 'b': 30}
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)