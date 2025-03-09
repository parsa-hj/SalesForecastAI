import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_processing import validate_data, create_example_template

def show_data_upload():
    """Display the data upload tab and return the uploaded data"""
    st.markdown('<p class="sub-header">Data Upload</p>', unsafe_allow_html=True)
    
    # File upload with example template
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    with col2:
        # Create and provide example template for download
        example_csv = create_example_template()
        st.download_button(
            label="Download Template",
            data=example_csv,
            file_name='sales_template.csv',
            mime='text/csv',
        )
    
    st.markdown('<div>Your CSV should have at least two columns: "Date" (in YYYY-MM-DD format) and "Sales" (numeric values)</div>', unsafe_allow_html=True)
    
    # Process uploaded data if available
    df = None
    if uploaded_file is not None:
        try:
            # Load and validate the data
            df = pd.read_csv(uploaded_file)
            validation_result = validate_data(df)
            
            if not validation_result['valid']:
                st.error(validation_result['message'])
                st.stop()
            
            # Successfully loaded data
            df = validation_result['data']
            
            # Data preview with option to see more
            st.write("### Preview of Uploaded Data")
            st.dataframe(df.head(10))
            
            with st.expander("Data Statistics"):
                st.write("#### Summary Statistics")
                st.dataframe(df.describe())
                
                # Date range info
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
    
    # Return the processed dataframe (or None if no file uploaded)
    return df