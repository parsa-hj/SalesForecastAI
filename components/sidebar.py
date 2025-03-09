import streamlit as st

def create_sidebar():
    """Create sidebar with settings and return the selected values"""
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
        
        # Return all settings as a dictionary
        return {
            'forecast_period': forecast_period,
            'confidence_interval': confidence_interval,
            'enable_yearly': enable_yearly,
            'enable_weekly': enable_weekly,
            'enable_daily': enable_daily,
            'changepoint_prior_scale': changepoint_prior_scale,
            'seasonality_prior_scale': seasonality_prior_scale
        }