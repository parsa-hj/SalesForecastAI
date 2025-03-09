import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_analysis(df, settings):
    """Display the analysis tab"""
    st.markdown('<p class="sub-header">Advanced Analysis</p>', unsafe_allow_html=True)
    
    # Check if we have forecast data in session state
    if 'forecast' in st.session_state and df is not None:
        forecast = st.session_state['forecast']
        
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
        if len(df) > 0:
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
                
                # Add seasonal decomposition
                if settings.get('show_seasonal_decomposition', False):
                    show_seasonal_decomposition(forecast, historical_end_date)
    else:
        st.info("Please generate a forecast in the 'Forecast' tab to view advanced analysis.")

def show_seasonal_decomposition(forecast, cutoff_date=None):
    """Show seasonal decomposition of the forecast"""
    st.markdown("### Seasonal Decomposition")
    
    # Create the subplot figure
    fig = make_subplots(rows=3, cols=1, 
                      shared_xaxes=True,
                      subplot_titles=("Weekly Pattern", "Monthly Pattern", "Yearly Pattern"),
                      vertical_spacing=0.1)
    
    # Extract components if they exist
    if 'weekly' in forecast.columns:
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['weekly'], name='Weekly'),
            row=1, col=1
        )
    
    if 'monthly' in forecast.columns:
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['monthly'], name='Monthly'),
            row=2, col=1
        )
        
    if 'yearly' in forecast.columns:
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['yearly'], name='Yearly'),
            row=3, col=1
        )
    
    # Add vertical line for forecast cutoff if provided
    if cutoff_date is not None:
        for i in range(1, 4):
            fig.add_vline(
                x=cutoff_date,
                line_width=1,
                line_dash="dash",
                line_color="red",
                row=i, col=1
            )
    
    fig.update_layout(
        height=600,
        title_text="Seasonal Components of the Forecast",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_anomaly_detection(df, forecast):
    """Detect and display anomalies in historical data"""
    st.markdown("### Anomaly Detection")
    
    # Merge actual data with predictions
    merged_data = pd.merge(
        df,
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        left_on='Date',
        right_on='ds',
        how='inner'
    )
    
    # Identify anomalies
    merged_data['anomaly'] = (
        (merged_data['Sales'] < merged_data['yhat_lower']) | 
        (merged_data['Sales'] > merged_data['yhat_upper'])
    )
    
    # Calculate deviation
    merged_data['deviation'] = merged_data['Sales'] - merged_data['yhat']
    merged_data['deviation_pct'] = (merged_data['deviation'] / merged_data['yhat']) * 100
    
    # Create visualization
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=merged_data['Date'],
        y=merged_data['Sales'],
        mode='markers',
        name='Actual Sales',
        marker=dict(
            color=merged_data['anomaly'].map({True: 'red', False: 'blue'}),
            size=merged_data['anomaly'].map({True: 10, False: 6})
        )
    ))
    
    # Add prediction interval
    fig.add_trace(go.Scatter(
        x=merged_data['Date'],
        y=merged_data['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=merged_data['Date'],
        y=merged_data['yhat_lower'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(68, 68, 68, 0.1)',
        name='Prediction Interval'
    ))
    
    # Add prediction line
    fig.add_trace(go.Scatter(
        x=merged_data['Date'],
        y=merged_data['yhat'],
        mode='lines',
        name='Expected Value',
        line=dict(color='rgba(31, 119, 180, 1)', width=2)
    ))
    
    fig.update_layout(
        title='Anomaly Detection in Historical Sales Data',
        xaxis_title='Date',
        yaxis_title='Sales',
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display anomalies in table
    anomalies = merged_data[merged_data['anomaly']]
    if len(anomalies) > 0:
        st.markdown(f"Found {len(anomalies)} anomalies in the historical data:")
        anomalies_display = anomalies[['Date', 'Sales', 'yhat', 'deviation', 'deviation_pct']].copy()
        anomalies_display.columns = ['Date', 'Actual Sales', 'Expected Sales', 'Deviation', 'Deviation (%)']
        st.dataframe(anomalies_display.style.format({
            'Actual Sales': '{:,.2f}',
            'Expected Sales': '{:,.2f}',
            'Deviation': '{:,.2f}',
            'Deviation (%)': '{:+,.2f}%'
        }))
    else:
        st.success("No anomalies detected in the historical data.")

def show_what_if_analysis(df, forecast, settings):
    """Show what-if scenario analysis"""
    st.markdown("### What-If Analysis")
    
    # Settings for scenarios
    col1, col2 = st.columns(2)
    with col1:
        change_pct = st.slider(
            "Sales Change Percentage", 
            min_value=-50, 
            max_value=50, 
            value=10, 
            step=5,
            help="Adjust sales by this percentage for the scenario analysis"
        )
    with col2:
        scenario_label = st.text_input(
            "Scenario Name", 
            value="Sales Boost Campaign",
            help="A label for this scenario"
        )
    
    # Create scenario forecast
    scenario_forecast = forecast.copy()
    future_mask = scenario_forecast['ds'] > df['Date'].max()
    
    # Apply percentage change to future values
    scenario_forecast.loc[future_mask, 'yhat'] *= (1 + change_pct/100)
    scenario_forecast.loc[future_mask, 'yhat_lower'] *= (1 + change_pct/100)
    scenario_forecast.loc[future_mask, 'yhat_upper'] *= (1 + change_pct/100)
    
    # Visualize the scenario
    fig = go.Figure()
    
    # Add base forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Base Forecast',
        line=dict(color='rgba(31, 119, 180, 1)', width=2)
    ))
    
    # Add scenario forecast
    fig.add_trace(go.Scatter(
        x=scenario_forecast['ds'],
        y=scenario_forecast['yhat'],
        mode='lines',
        name=f'Scenario: {scenario_label}',
        line=dict(color='rgba(255, 127, 14, 1)', width=2)
    ))
    
    # Add vertical line for forecast start
    historical_end_date = df['Date'].max()
    fig.add_vline(
        x=historical_end_date,
        line_width=2,
        line_dash="dash",
        line_color="green"
    )
    
    fig.add_annotation(
        x=historical_end_date,
        y=forecast['yhat'].max() * 0.9,
        text="Scenario Start",
        showarrow=True,
        arrowhead=1
    )
    
    fig.update_layout(
        title=f'What-If Scenario: {change_pct:+}% Sales Change',
        xaxis_title='Date',
        yaxis_title='Sales',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show impact metrics
    base_total = forecast[future_mask]['yhat'].sum()
    scenario_total = scenario_forecast[future_mask]['yhat'].sum()
    difference = scenario_total - base_total
    difference_pct = (difference / base_total) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Base Case Total Sales", 
        f"{base_total:,.2f}"
    )
    col2.metric(
        f"{scenario_label} Total Sales", 
        f"{scenario_total:,.2f}",
        f"{difference_pct:+.2f}%"
    )
    col3.metric(
        "Difference", 
        f"{difference:+,.2f}"
    )