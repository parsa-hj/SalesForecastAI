import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Page configuration
st.set_page_config(
    page_title="Sales Prediction Tool",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem !important;
        color: #424242;
    }
    .info-text {
        color: #555555;
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #1E88E5;
    }
    .metric-card {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://www.example.com/logo.png", width=100)
    st.markdown("### Settings")
    
    # Forecast settings
    forecast_period = st.slider("Forecast Period (days)", min_value=30, max_value=730, value=365, step=30)
    confidence_interval = st.slider("Confidence Interval", min_value=0.5, max_value=0.99, value=0.9, step=0.05)
    
    # Seasonality settings
    st.markdown("### Seasonality Settings")
    enable_yearly = st.checkbox("Yearly Seasonality", value=True)
    enable_weekly = st.checkbox("Weekly Seasonality", value=True)
    enable_daily = st.checkbox("Daily Seasonality", value=False)
    
    # Advanced settings (collapsible)
    with st.expander("Advanced Settings"):
        changepoint_prior_scale = st.slider(
            "Changepoint Prior Scale (flexibility)", 
            min_value=0.001, 
            max_value=0.5, 
            value=0.05, 
            step=0.001,
            format="%.3f"
        )
        seasonality_prior_scale = st.slider(
            "Seasonality Prior Scale", 
            min_value=0.01, 
            max_value=10.0, 
            value=1.0, 
            step=0.1
        )

# Main content
st.markdown('<p class="main-header">📊 Sales and Revenue Prediction Tool</p>', unsafe_allow_html=True)
st.markdown('<p class="info-text">Upload your historical sales data to get insights and predictions for future sales trends.</p>', unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["Data Upload", "Forecast", "Analysis", "Help"])

with tab1:
    st.markdown('<p class="sub-header">Data Upload</p>', unsafe_allow_html=True)
    
    # File upload with example template
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    with col2:
        # Create example template for download
        example_dates = pd.date_range(start='2020-01-01', periods=100)
        example_sales = np.random.normal(loc=1000, scale=200, size=100) + np.arange(100) * 5
        example_df = pd.DataFrame({
            'Date': example_dates.strftime('%Y-%m-%d'),
            'Sales': example_sales.round(2)
        })
        example_csv = example_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Template",
            data=example_csv,
            file_name='sales_template.csv',
            mime='text/csv',
        )
    
    st.markdown('<div>Your CSV should have at least two columns: "Date" (in YYYY-MM-DD format) and "Sales" (numeric values)</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Load the data
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check for required columns
            required_cols = ['Date', 'Sales']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.stop()
            
            # Data preview with option to see more
            st.write("### Preview of Uploaded Data")
            st.dataframe(df.head(10))
            
            with st.expander("Data Statistics"):
                st.write("#### Summary Statistics")
                st.dataframe(df.describe())
                
                # Date range info
                df['Date'] = pd.to_datetime(df['Date'])
                min_date = df['Date'].min().strftime('%Y-%m-%d')
                max_date = df['Date'].max().strftime('%Y-%m-%d')
                date_range = (df['Date'].max() - df['Date'].min()).days
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Start Date", min_date)
                col2.metric("End Date", max_date)
                col3.metric("Date Range", f"{date_range} days")
                
                # Quick data visualization
                st.write("#### Data Visualization")
                fig = px.line(df, x='Date', y='Sales', title='Historical Sales Data')
                fig.update_layout(xaxis_title="Date", yaxis_title="Sales")
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

with tab2:
    st.markdown('<p class="sub-header">Sales Forecast</p>', unsafe_allow_html=True)
    
    if 'df' in locals():
        try:
            # Prepare data for Prophet
            prophet_df = df[['Date', 'Sales']].copy()
            prophet_df.columns = ['ds', 'y']
            
            # Show progress bar during model training
            with st.spinner("Training forecast model..."):
                # Configure the model based on sidebar settings
                model = Prophet(
                    changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale,
                    interval_width=confidence_interval,
                    yearly_seasonality=enable_yearly,
                    weekly_seasonality=enable_weekly,
                    daily_seasonality=enable_daily
                )
                
                # Add additional regressors if available in data
                for col in df.columns:
                    if col not in ['Date', 'Sales'] and df[col].dtype in [np.float64, np.int64]:
                        prophet_df[col] = df[col]
                        model.add_regressor(col)
                
                # Fit the model
                model.fit(prophet_df)
                
                # Predict the future
                future = model.make_future_dataframe(periods=forecast_period)
                
                # Add regressors to future dataframe if applicable
                for col in df.columns:
                    if col not in ['Date', 'Sales'] and df[col].dtype in [np.float64, np.int64]:
                        # For simplicity, we'll use the last value, but you might want to improve this
                        future[col] = df[col].iloc[-1]
                
                forecast = model.predict(future)
                
                # Key metrics for the entire forecast period
                forecast_start_date = df['Date'].max() + datetime.timedelta(days=1)
                total_forecast_sales = forecast[forecast['ds'] > df['Date'].max()]['yhat'].sum()
                average_daily_sales = forecast[forecast['ds'] > df['Date'].max()]['yhat'].mean()
                
                # Display key metrics
                st.markdown("### Forecast Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Forecast Period", 
                    f"{forecast_period} days", 
                    help="Number of days in the future to forecast"
                )
                col2.metric(
                    "Total Forecasted Sales", 
                    f"{total_forecast_sales:,.2f}", 
                    help="Sum of predicted sales for the forecast period"
                )
                col3.metric(
                    "Average Daily Sales", 
                    f"{average_daily_sales:,.2f}", 
                    help="Average daily sales for the forecast period"
                )
                
                # Forecast visualization
                st.markdown("### Forecast Visualization")
                
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
                forecast_start = df['Date'].max()
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
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast components
                with st.expander("Forecast Components"):
                    components_fig = model.plot_components(forecast)
                    st.pyplot(components_fig)
                
                # Model performance metrics
                with st.expander("Model Performance"):
                    # Calculate error metrics on training data
                    y_true = prophet_df['y'].values
                    y_pred = forecast[:len(y_true)]['yhat'].values
                    
                    mae = mean_absolute_error(y_true, y_pred)
                    mse = mean_squared_error(y_true, y_pred)
                    rmse = np.sqrt(mse)
                    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("MAE", f"{mae:.2f}")
                    col2.metric("RMSE", f"{rmse:.2f}")
                    col3.metric("MAPE", f"{mape:.2f}%")
                    col4.metric("MSE", f"{mse:.2f}")
                    
                    # Residuals plot
                    residuals = y_true - y_pred
                    residual_df = pd.DataFrame({
                        'Date': prophet_df['ds'],
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
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download options
                st.markdown("### Download Results")
                
                # Prepare forecast data for download
                download_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                download_df.columns = ['Date', 'Predicted_Sales', 'Lower_Bound', 'Upper_Bound']
                
                # Add actual values where available
                merged_df = download_df.merge(
                    df[['Date', 'Sales']], 
                    on='Date', 
                    how='left'
                )
                
                # Format date as string
                merged_df['Date'] = merged_df['Date'].dt.strftime('%Y-%m-%d')
                
                # Create download buttons
                col1, col2 = st.columns(2)
                with col1:
                    csv_data = merged_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Full Forecast (CSV)",
                        data=csv_data,
                        file_name='sales_forecast.csv',
                        mime='text/csv',
                    )
                
                with col2:
                    # Only future predictions
                    future_df = merged_df[merged_df['Sales'].isnull()]
                    future_csv = future_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Future Predictions Only (CSV)",
                        data=future_csv,
                        file_name='future_sales_forecast.csv',
                        mime='text/csv',
                    )
                
        except Exception as e:
            st.error(f"Error generating forecast: {e}")
            st.error("Please check your data format and try again.")
    else:
        st.info("Please upload your sales data in the 'Data Upload' tab to generate a forecast.")

with tab3:
    st.markdown('<p class="sub-header">Advanced Analysis</p>', unsafe_allow_html=True)
    
    if 'forecast' in locals():
        # Time period selection for analysis
        st.markdown("### Analysis Period")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=pd.to_datetime(df['Date']).min().date(),
                min_value=pd.to_datetime(df['Date']).min().date(),
                max_value=pd.to_datetime(forecast['ds']).max().date()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=pd.to_datetime(forecast['ds']).max().date(),
                min_value=pd.to_datetime(df['Date']).min().date(),
                max_value=pd.to_datetime(forecast['ds']).max().date()
            )
        
        # Filter data for selected period
        filtered_forecast = forecast[
            (forecast['ds'].dt.date >= start_date) & 
            (forecast['ds'].dt.date <= end_date)
        ]
        
        # Calculate period metrics
        period_total = filtered_forecast['yhat'].sum()
        period_avg = filtered_forecast['yhat'].mean()
        period_max = filtered_forecast['yhat'].max()
        period_min = filtered_forecast['yhat'].min()
        
        # Display period metrics
        st.markdown("### Period Analysis")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Sales", f"{period_total:,.2f}")
        col2.metric("Average Sales", f"{period_avg:,.2f}")
        col3.metric("Peak Sales", f"{period_max:,.2f}")
        col4.metric("Lowest Sales", f"{period_min:,.2f}")
        
        # Monthly breakdown
        st.markdown("### Monthly Breakdown")
        
        # Group by month and calculate metrics
        filtered_forecast['year_month'] = filtered_forecast['ds'].dt.strftime('%Y-%m')
        monthly_forecast = filtered_forecast.groupby('year_month').agg({
            'yhat': ['sum', 'mean', 'min', 'max'],
            'ds': 'first'  # Keep first date for sorting
        }).reset_index()
        
        # Sort by date
        monthly_forecast = monthly_forecast.sort_values(by=[('ds', 'first')])
        
        # Rename columns for clarity
        monthly_forecast.columns = ['year_month', 'total_sales', 'avg_sales', 'min_sales', 'max_sales', 'first_date']
        
        # Create monthly chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_forecast['year_month'],
            y=monthly_forecast['total_sales'],
            name='Monthly Sales',
            marker_color='rgb(55, 83, 109)'
        ))
        
        fig.update_layout(
            title='Monthly Sales Forecast',
            xaxis_title='Month',
            yaxis_title='Total Sales',
            hovermode='x'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show monthly breakdown in table
        with st.expander("Monthly Details"):
            monthly_display = monthly_forecast[['year_month', 'total_sales', 'avg_sales', 'min_sales', 'max_sales']].copy()
            monthly_display.columns = ['Month', 'Total Sales', 'Average Daily Sales', 'Minimum Daily Sales', 'Maximum Daily Sales']
            st.dataframe(monthly_display.style.format({
                'Total Sales': '{:,.2f}',
                'Average Daily Sales': '{:,.2f}',
                'Minimum Daily Sales': '{:,.2f}',
                'Maximum Daily Sales': '{:,.2f}'
            }))
        
        # Seasonality Analysis
        st.markdown("### Seasonality Analysis")
        
        # Create day of week analysis
        if len(filtered_forecast) >= 7:  # Ensure we have at least a week of data
            filtered_forecast['day_of_week'] = filtered_forecast['ds'].dt.day_name()
            dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            dow_forecast = filtered_forecast.groupby('day_of_week').agg({
                'yhat': ['mean', 'sum']
            }).reset_index()
            
            # Reindex to maintain day order
            dow_forecast['day_of_week'] = pd.Categorical(
                dow_forecast['day_of_week'], 
                categories=dow_order,
                ordered=True
            )
            dow_forecast = dow_forecast.sort_values('day_of_week')
            
            # Create day of week chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=dow_forecast['day_of_week'],
                y=dow_forecast[('yhat', 'mean')],
                name='Average Sales by Day of Week',
                marker_color='rgb(26, 118, 255)'
            ))
            
            fig.update_layout(
                title='Average Sales by Day of Week',
                xaxis_title='Day of Week',
                yaxis_title='Average Sales'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Growth Analysis
        st.markdown("### Growth Analysis")
        
        # Calculate growth rates
        if 'df' in locals() and len(df) > 0:
            historical_end_date = df['Date'].max()
            
            # Get historical and future data
            historical_data = forecast[forecast['ds'] <= historical_end_date]
            future_data = forecast[forecast['ds'] > historical_end_date]
            
            if len(historical_data) > 0 and len(future_data) > 0:
                # Calculate average historical and future daily sales
                hist_avg = historical_data['yhat'].mean()
                future_avg = future_data['yhat'].mean()
                
                # Calculate growth percentage
                growth_rate = ((future_avg - hist_avg) / hist_avg) * 100
                
                # Display growth metrics
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Historical Avg. Daily Sales", 
                    f"{hist_avg:,.2f}"
                )
                col2.metric(
                    "Forecasted Avg. Daily Sales", 
                    f"{future_avg:,.2f}", 
                    f"{growth_rate:+.2f}%"
                )
                col3.metric(
                    "Growth Rate", 
                    f"{growth_rate:+.2f}%"
                )
                
                # Create growth visualization
                # Resample to smooth the trend for better visualization
                historical_monthly = historical_data.resample('M', on='ds').mean().reset_index()
                future_monthly = future_data.resample('M', on='ds').mean().reset_index()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=historical_monthly['ds'],
                    y=historical_monthly['yhat'],
                    mode='lines+markers',
                    name='Historical Trend',
                    line=dict(color='blue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=future_monthly['ds'],
                    y=future_monthly['yhat'],
                    mode='lines+markers',
                    name='Future Trend',
                    line=dict(color='red')
                ))
                
                fig.add_vline(
                    x=historical_end_date,
                    line_width=2,
                    line_dash="dash",
                    line_color="green"
                )
                
                fig.add_annotation(
                    x=historical_end_date,
                    y=forecast['yhat'].max() * 0.9,
                    text="Forecast Start",
                    showarrow=True,
                    arrowhead=1
                )
                
                fig.update_layout(
                    title='Historical vs Future Trend',
                    xaxis_title='Date',
                    yaxis_title='Average Sales'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please generate a forecast in the 'Forecast' tab to view advanced analysis.")

with tab4:
    st.markdown('<p class="sub-header">Help & Information</p>', unsafe_allow_html=True)
    
    with st.expander("About This Tool"):
        st.markdown("""
        ### Sales and Revenue Prediction Tool
        
        This tool helps businesses forecast their future sales based on historical data using the Prophet forecasting model developed by Facebook. 
        
        **Key Features:**
        - Upload your historical sales data
        - Generate forecasts for future periods
        - Visualize sales trends and seasonality
        - Analyze growth patterns
        - Download forecast results
        
        **How It Works:**
        Prophet is an additive regression model with components for trend, seasonality, and holidays. It works best with time series that have strong seasonal effects and several seasons of historical data.
        """)
    
    with st.expander("Data Requirements"):
        st.markdown("""
        ### Data Format Requirements
        
        Your CSV file should contain at least two columns:
        
        1. **Date**: In YYYY-MM-DD format (e.g., 2022-01-01)
        2. **Sales**: Numeric values representing your sales figures
        
        **Optional:**
        - You can include additional numeric columns that might influence your sales (e.g., marketing spend, price changes)
        - The tool works best with at least 1 year of historical data to capture seasonality
        
        **Example:**
        
        | Date       | Sales  |
        |------------|--------|
        | 2022-01-01 | 1000   |
        | 2022-01-02 | 1200   |
        | 2022-01-03 | 980    |
        """)
    
    with st.expander("Interpretation Guide"):
        st.markdown("""
        ### How to Interpret Results
        
        **Forecast Components:**
        - **Trend**: The non-periodic changes in the time series
        - **Seasonality**: Periodic changes (weekly, yearly, etc.)
        - **Holidays**: Effects of holidays or special events
        
        **Confidence Intervals:**
        The upper and lower bounds represent the uncertainty in the forecast. A wider interval means less certainty.
        
        **Model Metrics:**
        - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
        - **RMSE (Root Mean Squared Error)**: Square root of the average squared differences
        - **MAPE (Mean Absolute Percentage Error)**: Percentage representation of the error
        
        Lower values for these metrics indicate better model performance.
        """)
    
    with st.expander("Tips for Better Forecasts"):
        st.markdown("""
        ### Tips for Better Forecasts
        
        1. **Provide more historical data**: Longer history leads to better forecasts, especially for capturing seasonal patterns
        
        2. **Clean your data**: Remove outliers or explain them with additional variables
        
        3. **Adjust model parameters**:
           - Increase `changepoint_prior_scale` if the trend changes rapidly
           - Decrease it if the trend is smoother
           - Adjust seasonality settings based on your business patterns
        
        4. **Include relevant variables**: If you have factors that influence sales (promotions, pricing changes), include them as additional columns
        
        5. **Update regularly**: Refresh your forecasts as new data becomes available
        """)