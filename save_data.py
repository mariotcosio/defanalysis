import pandas as pd
import numpy as np

def prepare_and_save_data():
    """
    Prepare the data and save it as a CSV file with the required columns.
    """
    print("Preparing and saving data...")
    
    # Generate time series data (1990-2022)
    years = range(1990, 2023)
    
    # Create the base dataset
    data = {
        'Year': list(years),
        'Global_Forest_Area_MHa': [4168 - i*5.2 for i in range(len(years))],
        'Temperature_Anomaly_C': [0.3 + i*0.02 + np.random.normal(0, 0.05) for i in range(len(years))],
        'CO2_PPM': [350 + i*2.1 + np.random.normal(0, 0.5) for i in range(len(years))],
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    try:
        # Define regions with consistent underscore naming
        regions_data = {
            'Amazon': 7.5,
            'Congo_Basin': 5.2,
            'Southeast_Asia': 8.1,
            'Boreal_Forest': 2.8,
            'North_America': 1.5,
            'Europe': 0.9
        }
        
        # Add each region's loss data
        for region, base_rate in regions_data.items():
            column_name = f'{region}_Loss'  # Create consistent column names
            df[column_name] = [
                max(0.1, base_rate + i * base_rate/30 + np.random.normal(0, base_rate/10))
                for i in range(len(years))
            ]
        
        # Calculate derived metrics
        regional_loss_columns = [col for col in df.columns if col.endswith('_Loss')]
        df['Annual_Forest_Loss'] = df[regional_loss_columns].sum(axis=1)
        df['Cumulative_Forest_Loss'] = df['Annual_Forest_Loss'].cumsum()
        
        # Save to CSV
        output_file = 'deforestation_climate_data.csv'
        df.to_csv(output_file, index=False)
        print(f"\nData saved successfully to {output_file}")
        
        # Display information about the saved data
        print("\nColumns in the dataset:")
        for col in df.columns:
            print(f"- {col}")
            
        print("\nFirst few rows of the data:")
        print(df.head())
        
        # Verify specific columns exist
        required_columns = [
            'Year',
            'Global_Forest_Area_MHa',
            'Temperature_Anomaly_C',
            'CO2_PPM',
            'Amazon_Loss',
            'Congo_Basin_Loss',
            'Southeast_Asia_Loss',
            'Boreal_Forest_Loss',
            'North_America_Loss',
            'Europe_Loss'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print("\nWarning: The following required columns are missing:")
            for col in missing_columns:
                print(f"- {col}")
        else:
            print("\nAll required columns are present in the dataset.")
        
        return df
        
    except Exception as e:
        print(f"\nError while preparing data: {str(e)}")
        print("Please check the column names and data structure.")
        raise

if __name__ == "__main__":
    try:
        df = prepare_and_save_data()
    except Exception as e:
        print(f"\nScript execution failed: {str(e)}")