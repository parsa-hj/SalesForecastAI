import streamlit as st
import pandas as pd
import numpy as np
import datetime
from utils.forecasting import create_forecast_model, calculate_forecast_metrics
from utils.visualization import create_forecast_plot
from utils.metrics import calculate_error_metrics, create_residuals_plot
from io import BytesIO
from fpdf import FPDF

def show_forecast(df, settings):
    """Display the forecast tab"""
    st.markdown('<p class="sub-header">Sales Forecast</p>', unsafe_allow_html=True)

    if 'forecast_period' not in settings:
        settings['forecast_period'] = 365  # Default forecast period

    if df is not None:
        try:
            # Show progress bar during model training
            with st.spinner("Training forecast model..."):
                # Create and fit the model
                model, prophet_df, forecast = create_forecast_model(df, settings)

                # Calculate forecast metrics
                metrics = calculate_forecast_metrics(df, forecast)

                # Handle missing metrics gracefully
                total_forecast_sales = metrics.get('total_forecast_sales', 0)
                average_daily_sales = metrics.get('average_daily_sales', 0)

                # Display key metrics
                st.markdown("### Forecast Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Forecast Period", 
                    f"{settings['forecast_period']} days", 
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

                # Create forecast plot
                fig = create_forecast_plot(prophet_df, forecast, df['Date'].max())
                st.plotly_chart(fig, use_container_width=True)

                # Forecast components
                with st.expander("Forecast Components"):
                    components_fig = model.plot_components(forecast)
                    st.pyplot(components_fig)

                # Model performance metrics
                with st.expander("Model Performance"):
                    # Calculate error metrics on training data
                    metrics = calculate_error_metrics(prophet_df['y'].values, 
                                                      forecast[:len(prophet_df)]['yhat'].values)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("MAE", f"{metrics['mae']:.2f}")
                    col2.metric("RMSE", f"{metrics['rmse']:.2f}")
                    col3.metric("MAPE", f"{metrics['mape']:.2f}%")
                    col4.metric("MSE", f"{metrics['mse']:.2f}")

                    # Residuals plot
                    fig = create_residuals_plot(prophet_df['ds'], 
                                                prophet_df['y'].values, 
                                                forecast[:len(prophet_df)]['yhat'].values)
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

                # Create CSV download buttons
                st.markdown("#### CSV Downloads")
                csv_col1, csv_col2 = st.columns(2)
                with csv_col1:
                    csv_data = merged_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Full Forecast (CSV)",
                        data=csv_data,
                        file_name='sales_forecast.csv',
                        mime='text/csv',
                    )

                with csv_col2:
                    # Only future predictions
                    future_df = merged_df[merged_df['Sales'].isnull()]
                    future_csv = future_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Future Predictions Only (CSV)",
                        data=future_csv,
                        file_name='future_sales_forecast.csv',
                        mime='text/csv',
                    )
                
                # PDF Downloads section
                st.markdown("#### PDF Downloads")
                pdf_col1, pdf_col2 = st.columns(2)
                
                with pdf_col1:
                    # Generate Full Forecast PDF
                    pdf = FPDF()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.add_page()
                    pdf.set_font("Arial", "B", size=16)
                    pdf.cell(200, 10, txt="Sales Forecast Report - Full Forecast", ln=True, align='C')
                    pdf.ln(10)
                    
                    # Add summary section
                    pdf.set_font("Arial", "B", size=12)
                    pdf.cell(200, 10, txt="Forecast Summary", ln=True)
                    pdf.set_font("Arial", size=10)
                    pdf.cell(200, 10, txt=f"Forecast Period: {settings['forecast_period']} days", ln=True)
                    pdf.cell(200, 10, txt=f"Total Forecasted Sales: {total_forecast_sales:,.2f}", ln=True)
                    pdf.cell(200, 10, txt=f"Average Daily Sales: {average_daily_sales:,.2f}", ln=True)
                    pdf.ln(5)
                    
                    # Add forecast data table header
                    pdf.set_font("Arial", "B", size=10)
                    pdf.cell(40, 10, "Date", 1, 0, 'C')
                    pdf.cell(40, 10, "Predicted Sales", 1, 0, 'C')
                    pdf.cell(40, 10, "Lower Bound", 1, 0, 'C')
                    pdf.cell(40, 10, "Upper Bound", 1, 0, 'C')
                    pdf.cell(40, 10, "Actual Sales", 1, 1, 'C')
                    
                    # Add data rows (limit to first 30 rows to avoid huge PDFs)
                    pdf.set_font("Arial", size=8)
                    for i, row in merged_df.head(30).iterrows():
                        pdf.cell(40, 10, str(row['Date']), 1, 0, 'C')
                        pdf.cell(40, 10, f"{row['Predicted_Sales']:.2f}", 1, 0, 'C')
                        pdf.cell(40, 10, f"{row['Lower_Bound']:.2f}", 1, 0, 'C')
                        pdf.cell(40, 10, f"{row['Upper_Bound']:.2f}", 1, 0, 'C')
                        
                        if pd.isna(row['Sales']):
                            pdf.cell(40, 10, "N/A", 1, 1, 'C')
                        else:
                            pdf.cell(40, 10, f"{row['Sales']:.2f}", 1, 1, 'C')
                    
                    # If there are more rows, add a note
                    if len(merged_df) > 30:
                        pdf.ln(5)
                        pdf.cell(200, 10, f"Note: Showing only 30 out of {len(merged_df)} rows", ln=True)
                    
                    # Create a BytesIO object to store the PDF
                    pdf_output = BytesIO()
                    pdf_output.write(pdf.output(dest='S').encode('latin-1'))
                    pdf_output.seek(0)
                    
                    # Create download button
                    st.download_button(
                        label="Download Full Forecast (PDF)",
                        data=pdf_output,
                        file_name='sales_forecast.pdf',
                        mime='application/pdf',
                    )
                
                with pdf_col2:
                    # Generate Future Forecast PDF (only future predictions)
                    future_df = merged_df[merged_df['Sales'].isnull()]
                    
                    pdf = FPDF()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.add_page()
                    pdf.set_font("Arial", "B", size=16)
                    pdf.cell(200, 10, txt="Sales Forecast Report - Future Predictions Only", ln=True, align='C')
                    pdf.ln(10)
                    
                    # Add summary section
                    pdf.set_font("Arial", "B", size=12)
                    pdf.cell(200, 10, txt="Forecast Summary", ln=True)
                    pdf.set_font("Arial", size=10)
                    pdf.cell(200, 10, txt=f"Forecast Period: {settings['forecast_period']} days", ln=True)
                    pdf.cell(200, 10, txt=f"Total Forecasted Sales: {total_forecast_sales:,.2f}", ln=True)
                    pdf.cell(200, 10, txt=f"Average Daily Sales: {average_daily_sales:,.2f}", ln=True)
                    pdf.ln(5)
                    
                    # Add forecast data table header
                    pdf.set_font("Arial", "B", size=10)
                    pdf.cell(50, 10, "Date", 1, 0, 'C')
                    pdf.cell(50, 10, "Predicted Sales", 1, 0, 'C')
                    pdf.cell(50, 10, "Lower Bound", 1, 0, 'C')
                    pdf.cell(50, 10, "Upper Bound", 1, 1, 'C')
                    
                    # Add data rows (limit to first 30 rows to avoid huge PDFs)
                    pdf.set_font("Arial", size=8)
                    for i, row in future_df.head(30).iterrows():
                        pdf.cell(50, 10, str(row['Date']), 1, 0, 'C')
                        pdf.cell(50, 10, f"{row['Predicted_Sales']:.2f}", 1, 0, 'C')
                        pdf.cell(50, 10, f"{row['Lower_Bound']:.2f}", 1, 0, 'C')
                        pdf.cell(50, 10, f"{row['Upper_Bound']:.2f}", 1, 1, 'C')
                    
                    # If there are more rows, add a note
                    if len(future_df) > 30:
                        pdf.ln(5)
                        pdf.cell(200, 10, f"Note: Showing only 30 out of {len(future_df)} rows", ln=True)
                    
                    # Create a BytesIO object to store the PDF
                    future_pdf_output = BytesIO()
                    future_pdf_output.write(pdf.output(dest='S').encode('latin-1'))
                    future_pdf_output.seek(0)
                    
                    # Create download button
                    st.download_button(
                        label="Download Future Forecast (PDF)",
                        data=future_pdf_output,
                        file_name='future_sales_forecast.pdf',
                        mime='application/pdf',
                    )

                # Store forecast in session state for other tabs
                st.session_state['forecast'] = forecast
                st.session_state['model'] = model
                st.session_state['prophet_df'] = prophet_df

        except Exception as e:
            st.error(f"Error generating forecast: {e}")
            st.error("Please check your data format and try again.")
    else:
        st.info("Please upload your sales data in the 'Data Upload' tab to generate a forecast.")