import pandas as pd
import numpy as np

def validate_data(df):
    """
    Validate the uploaded data format
    
    Args:
        df (pandas.DataFrame): The uploaded dataframe
    
    Returns:
        dict: A dictionary with validation results
    """
    result = {'valid': False, 'message': '', 'data': None}
    
    # Check for required columns
    required_cols = ['Date', 'Sales']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        result['message'] = f"Missing required columns: {', '.join(missing_cols)}"
        return result
    
    # Convert Date column to datetime
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception as e:
        result['message'] = f"Error converting Date column to datetime: {e}"
        return result
    
    # Check for numeric Sales column
    if not pd.api.types.is_numeric_dtype(df['Sales']):
        result['message'] = "Sales column must contain numeric values"
        return result
    
    # Data is valid
    result['valid'] = True
    result['data'] = df
    return result

def create_example_template():
    """
    Create an example CSV template for download
    
    Returns:
        bytes: CSV data as bytes
    """
    example_dates = pd.date_range(start='2020-01-01', periods=100)
    example_sales = np.random.normal(loc=1000, scale=200, size=100) + np.arange(100) * 5
    example_df = pd.DataFrame({
        'Date': example_dates.strftime('%Y-%m-%d'),
        'Sales': example_sales.round(2)
    })
    return example_df.to_csv(index=False).encode('utf-8')