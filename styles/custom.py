import streamlit as st

def apply_custom_css():
    """Apply custom CSS styles to the app"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem !important;
            color: #1E88E5;
        }
        .sub-header {
            font-size: 1.5rem !important;
            color: white;
        }
        .info-text {
            color: white;
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