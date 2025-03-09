import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def create_forecast_plot(prophet_df, forecast, forecast_start):
    """
    Create a plot for the forecast visualization
    
    Args:
        prophet_df (pandas.DataFrame): Prophet input data
        forecast (pandas.DataFrame): Forecast output
        forecast_start (datetime): Start date of the forecast
    
    Returns:
        plotly.graph_objects.Figure: The plotly figure
    """
    # Create a more sophisticated plot
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       subplot_titles=("Sales Forecast", "Trend Components"),
                       vertical_spacing=0.12, row_heights=[0.7, 0.3])
    
    # Add actual sales
    fig.add_trace(
        go.Scatter(
            x=prophet_df['ds'], 
            y=prophet_df['y'], 
            mode='markers', 
            name='Historical Sales',
            marker=dict(color='rgba(0, 0, 255, 0.8)', size=4)
        ),
        row=1, col=1
    )
    
    # Add forecast line
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'], 
            y=forecast['yhat'], 
            mode='lines', 
            name='Forecast',
            line=dict(color='rgba(31, 119, 180, 1)', width=2)
        ),
        row=1, col=1
    )
    
    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'], 
            y=forecast['yhat_upper'], 
            mode='lines', 
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'], 
            y=forecast['yhat_lower'], 
            mode='lines', 
            name='Lower Bound',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.2)',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add vertical line to indicate forecast start
    fig.add_vline(
        x=forecast_start, 
        line_width=2, 
        line_dash="dash", 
        line_color="red",
        row=1, col=1
    )
    
    # Add annotation for forecast start
    fig.add_annotation(
        x=forecast_start, 
        y=forecast['yhat'].max(),
        text="Forecast Start",
        showarrow=True,
        arrowhead=1,
        row=1, col=1
    )
    
    # Add trend component
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'], 
            y=forecast['trend'], 
            mode='lines', 
            name='Trend',
            line=dict(color='rgba(255, 127, 14, 1)', width=2)
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Sales", row=1, col=1)
    fig.update_yaxes(title_text="Trend Value", row=2, col=1)
    
    return fig

def create_residuals_plot(dates, y_true, y_pred):
    """
    Create a plot of the residuals
    
    Args:
        dates (array-like): The dates
        y_true (array-like): Actual values
        y_pred (array-like): Predicted values
    
    Returns:
        plotly.graph_objects.Figure: The plotly figure
    """
    residuals = y_true - y_pred
    residual_df = pd.DataFrame({
        'Date': dates,
        'Residual': residuals
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=residual_df['Date'], 
        y=residual_df['Residual'],
        mode='markers',
        name='Residuals'
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(
        title='Residuals Plot',
        xaxis_title='Date',
        yaxis_title='Residual (Actual - Predicted)'
    )
    
    return fig