import streamlit as st
from components.sidebar import create_sidebar
from components.data_upload import show_data_upload
from components.forecast import show_forecast
from components.analysis import show_analysis
from components.help import show_help
from styles.custom import apply_custom_css

# Page configuration
st.set_page_config(
    page_title="Sales Prediction Tool",
    page_icon="📈",
    layout="wide"
)

# Apply custom CSS
apply_custom_css()

# Main content
st.markdown('<p class="main-header">📊 Sales and Revenue Prediction Tool</p>', unsafe_allow_html=True)
st.markdown('<p class="info-text">Upload your historical sales data to get insights and predictions for future sales trends.</p>', unsafe_allow_html=True)

# Sidebar settings
settings = create_sidebar()

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["Data Upload", "Forecast", "Analysis", "Help"])

# Load the content for each tab
with tab1:
    data = show_data_upload()

with tab2:
    show_forecast(data, settings)

with tab3:
    show_analysis(data, settings)

with tab4:
    show_help()