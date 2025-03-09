import streamlit as st

def show_help():
    """Display the help tab"""
    st.markdown('<p class="sub-header">Help & Information</p>', unsafe_allow_html=True)
    
    with st.expander("About This Tool", expanded=True):
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