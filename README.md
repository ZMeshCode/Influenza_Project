# Influenza Prediction System

An advanced machine learning system that leverages LSTM neural networks to predict and analyze influenza trends. The system processes CDC FluView data to forecast Influenza-Like Illness (ILI) activity with up to 4 weeks of advance notice, achieving an R-squared score of 0.89.

## Dataset

The project utilizes CDC FluView data (2019-2025) with focus on typical flu seasons (October to May), including:
- Weekly ILI percentages (range: 1.44% - 7.0%)
- Total specimen counts (range: 1,820 - 5,000)
- Influenza type A and B positive cases
- Percent positivity rates (range: 6.21% - 35.0%)

## Key Features

- **Predictive Modeling**: LSTM-based neural network with:
  - 2 LSTM layers (64 and 32 units)
  - Dropout rate: 0.2
  - Mean Absolute Error: 0.42% ILI
  - R-squared Score: 0.89
- **Time Series Analysis**: Advanced visualization of ILI trends and positivity rates
- **Seasonal Pattern Detection**: Decomposition into trend, seasonal, and residual components
- **Automated Reporting**: Generation of comprehensive analysis reports and visualizations

## Technical Requirements

- Python 3.x
- Dependencies:
  - TensorFlow (Neural network implementation)
  - Pandas (Data processing)
  - Scikit-learn (Model evaluation)
  - Matplotlib/Seaborn (Visualization)
  - Statsmodels (Time series analysis)
  - NumPy (Numerical computations)

## Installation & Usage

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the prediction model:
   ```bash
   python Disease_Prediction_Model.py
   ```

## Model Performance

- **Prediction Accuracy**:
  - 4-week advance predictions with 95% confidence intervals
  - Mean Absolute Error: 0.42% ILI
  - R-squared Score: 0.89
- **Yearly Trends**:
  - Consistent improvement in prediction accuracy (2019-2025)
  - Average ILI range: 2.91% (2019) to 5.93% (2025)
  - Stable positivity rates averaging 29.73%

## Output Files

The system generates multiple visualization and analysis files:
- `time_series_analysis.png`: ILI trends and positivity rates
- `seasonal_decomposition.png`: Trend, seasonal, and residual components
- `monthly_patterns.png`: Monthly ILI patterns
- `summary_statistics.png`: Statistical analysis visualization
- `predictions.png`: Model predictions vs actual values
- `yearly_comparison.csv`: Year-over-year analysis
- `analysis_report.md`: Comprehensive findings

## Project Structure
├── Disease_Prediction_Model.py   # Main LSTM model implementation
├── analysis_report.md           # Detailed analysis findings
├── flu_analysis_output/         # Generated visualizations directory
│   ├── time_series_analysis.png
│   ├── seasonal_decomposition.png
│   ├── monthly_patterns.png
│   ├── summary_statistics.png
│   ├── predictions.png
│   └── yearly_comparison.png
├── yearly_comparison.csv        # Year-over-year statistical data
└── summary_statistics.csv       # Overall model performance metrics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
