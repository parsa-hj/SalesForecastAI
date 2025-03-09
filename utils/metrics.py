import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_error_metrics(y_true, y_pred):
    """
    Calculate error metrics for model evaluation
    
    Args:
        y_true (array-like): Actual values
        y_pred (array-like): Predicted values
    
    Returns:
        dict: Dictionary of error metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Calculate MAPE with handling for zeros
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, np.nan))) * 100
    
    if np.isnan(mape):
        mape = 0  # Handle case where all y_true values are zero
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape
    }

def create_residuals_plot(dates, actual, predicted):
    """Create a plot of the model residuals"""
    import plotly.graph_objects as go
    import numpy as np
    
    residuals = actual - predicted
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=np.zeros(len(dates)),
        mode='lines',
        name='Zero Line',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='Model Residuals (Actual - Predicted)',
        xaxis_title='Date',
        yaxis_title='Residual Value',
        template='plotly_white'
    )
    
    return fig