# Sales Forecast AI

A web-based application for sales forecasting and analysis using machine learning. Built with Streamlit and Facebook's Prophet forecasting model, this tool helps businesses predict future sales trends based on historical data.

## Features

### Core Functionality

- **Data Upload**: Upload CSV files with historical sales data
- **Sales Forecasting**: Generate predictions using Facebook's Prophet model
- **Advanced Analysis**: Deep dive into sales patterns and seasonality
- **Interactive Visualizations**: Charts and graphs using Plotly
- **Export Results**: Download forecasts and analysis as CSV or PDF

### Analysis Capabilities

- **Trend Analysis**: Identify long-term growth patterns
- **Seasonality Detection**: Uncover weekly, monthly, and yearly patterns
- **Anomaly Detection**: Find unusual sales patterns
- **What-if Scenarios**: Test different forecasting parameters
- **Performance Metrics**: MAE, RMSE, MAPE, and MSE calculations

### Visualization Features

- **Forecast Plots**: Interactive charts with confidence intervals
- **Component Analysis**: Break down trend, seasonality, and holiday effects
- **Residual Analysis**: Model performance visualization
- **Monthly Breakdowns**: Detailed period analysis
- **Seasonal Decomposition**: Advanced time series analysis

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone or download the project**

   ```bash
   cd SalesForecastAI
   ```

2. **Install required dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   ```bash
   streamlit run app.py
   ```

4. **Access the app**
   - The application will automatically open in your default web browser
   - Default URL: `http://localhost:8501`

## Data Requirements

### Required Format

Your CSV file must contain at least two columns:

| Column    | Format     | Example    |
| --------- | ---------- | ---------- |
| **Date**  | YYYY-MM-DD | 2023-01-01 |
| **Sales** | Numeric    | 1000.50    |

### Optional Columns

You can include additional numeric columns that might influence sales:

- Marketing spend
- Price changes
- Promotional events
- External factors

### Data Quality Tips

- **Minimum Data**: At least 1 year of historical data for better seasonality detection
- **Consistent Format**: Ensure dates are in YYYY-MM-DD format
- **Clean Data**: Remove or explain outliers
- **Regular Intervals**: Daily data works best, but weekly/monthly is also supported

## Configuration

### Model Parameters

The application uses Facebook's Prophet model with configurable parameters:

| Parameter                 | Description            | Default | Impact                        |
| ------------------------- | ---------------------- | ------- | ----------------------------- |
| `changepoint_prior_scale` | Trend flexibility      | 0.05    | Higher = more flexible trend  |
| `seasonality_prior_scale` | Seasonality strength   | 10.0    | Higher = stronger seasonality |
| `confidence_interval`     | Prediction uncertainty | 0.95    | 95% confidence interval       |
| `forecast_period`         | Days to predict        | 365     | One year forecast             |

### Seasonality Settings

- **Yearly Seasonality**: Enable for annual patterns
- **Weekly Seasonality**: Enable for weekly patterns
- **Daily Seasonality**: Enable for daily patterns

## Understanding Results

### Forecast Components

- **Trend**: Long-term growth or decline patterns
- **Seasonality**: Recurring patterns (weekly, monthly, yearly)
- **Holidays**: Special event effects
- **Confidence Intervals**: Uncertainty ranges around predictions

### Performance Metrics

- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **RMSE (Root Mean Squared Error)**: Square root of average squared errors
- **MAPE (Mean Absolute Percentage Error)**: Percentage error representation
- **MSE (Mean Squared Error)**: Average squared prediction errors

### Interpretation Tips

- **Lower metric values** indicate better model performance
- **Wider confidence intervals** suggest higher uncertainty
- **Strong seasonality** shows clear recurring patterns
- **Anomalies** may indicate special events or data issues

## Technical Architecture

### Frontend

- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Custom CSS**: Styled user interface

### Backend

- **Prophet**: Facebook's forecasting library
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Additional machine learning utilities

### Project Structure

```
SalesForecastAI/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── components/           # UI components
│   ├── data_upload.py   # Data upload functionality
│   ├── forecast.py      # Forecasting interface
│   ├── analysis.py      # Analysis features
│   ├── help.py          # Help documentation
│   └── sidebar.py       # Settings sidebar
├── utils/               # Utility functions
│   ├── data_processing.py  # Data validation and processing
│   ├── forecasting.py      # Prophet model wrapper
│   ├── metrics.py          # Performance calculations
│   └── visualization.py    # Chart creation
└── styles/              # Custom styling
    └── custom.py        # CSS styles
```

## 🔧 Troubleshooting

### Common Issues

**Installation Problems**

```bash
# If you encounter dependency issues:
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**Data Upload Errors**

- Ensure CSV format is correct
- Check date format (YYYY-MM-DD)
- Verify sales column contains numeric values
- Remove any empty rows or invalid data

**Forecast Generation Issues**

- Ensure sufficient historical data (at least 1 year recommended)
- Try adjusting model parameters in sidebar
- Check for data quality issues or outliers

**Performance Issues**

- Reduce forecast period for faster processing
- Simplify model parameters
- Use smaller datasets for testing

## Best Practices

### For Better Forecasts

1. **Provide More Data**: Longer history improves accuracy
2. **Clean Your Data**: Remove outliers and errors
3. **Include Relevant Variables**: Add factors that influence sales
4. **Regular Updates**: Refresh forecasts with new data
5. **Validate Results**: Compare predictions with actual outcomes

### For Data Preparation

1. **Consistent Formatting**: Use standardized date formats
2. **Handle Missing Data**: Fill gaps or remove incomplete records
3. **Document Changes**: Note any business events or anomalies
4. **Quality Checks**: Verify data accuracy before uploading
