import pandas as pd
import numpy as np
from prophet import Prophet
import datetime

def create_forecast_model(df, settings):
    """
    Create and fit a Prophet forecasting model
    
    Args:
        df (pandas.DataFrame): Historical data
        settings (dict): Model settings
    
    Returns:
        tuple: (model, prophet_df, forecast) - The fitted model, prepared data, and forecast
    """
    # Prepare data for Prophet
    prophet_df = df[['Date', 'Sales']].copy()
    prophet_df.columns = ['ds', 'y']
    
    # Configure the model based on settings
    model = Prophet(
        changepoint_prior_scale=settings['changepoint_prior_scale'],
        seasonality_prior_scale=settings['seasonality_prior_scale'],
        interval_width=settings['confidence_interval'],
        yearly_seasonality=settings['enable_yearly'],
        weekly_seasonality=settings['enable_weekly'],
        daily_seasonality=settings['enable_daily']
    )
    
    # Add additional regressors if available in data
    for col in df.columns:
        if col not in ['Date', 'Sales'] and df[col].dtype in [np.float64, np.int64]:
            prophet_df[col] = df[col]
            model.add_regressor(col)
    
    # Fit the model
    model.fit(prophet_df)
    
    # Predict the future
    future = model.make_future_dataframe(periods=settings['forecast_period'])
    
    # Add regressors to future dataframe if applicable
    for col in df.columns:
        if col not in ['Date', 'Sales'] and df[col].dtype in [np.float64, np.int64]:
            # For simplicity, use the last value
            future[col] = df[col].iloc[-1]
    
    # Generate forecast
    forecast = model.predict(future)
    
    return model, prophet_df, forecast

def calculate_forecast_metrics(df, forecast):
    """
    Calculate key forecast metrics
    
    Args:
        df (pandas.DataFrame): Historical data
        forecast (pandas.DataFrame): Forecast data
    
    Returns:
        dict: Dictionary of forecast metrics
    """
    # Extract future-only data
    forecast_start_date = df['Date'].max()
    future_forecast = forecast[forecast['ds'] > forecast_start_date]
    
    # Calculate metrics
    metrics = {
        'total_forecast_sales': future_forecast['yhat'].sum(),
        'average_daily_sales': future_forecast['yhat'].mean(),
        'forecast_start_date': forecast_start_date + datetime.timedelta(days=1),
        'peak_date': future_forecast.loc[future_forecast['yhat'].idxmax(), 'ds'],
        'peak_value': future_forecast['yhat'].max(),
        'trough_date': future_forecast.loc[future_forecast['yhat'].idxmin(), 'ds'],
        'trough_value': future_forecast['yhat'].min()
    }

    print(f"Metrics calculated: {metrics}")
    
    return metrics