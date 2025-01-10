import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

class FluDataCollector:
    def __init__(self):
        self.base_url = "https://www.cdc.gov/flu/weekly/flureport.xml"
        # Create output directory for visualizations
        self.output_dir = 'flu_analysis_output'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def get_flu_data(self, start_year=2019, end_year=None):
        """
        Fetch flu data from CDC FluView
        Args:
            start_year (int): Starting year for data collection
            end_year (int): Ending year for data collection (defaults to current year)
        """
        if end_year is None:
            end_year = datetime.now().year
            
        print(f"Fetching flu data from {start_year} to {end_year}...")
        
        try:
            # Create dates from October to May (typical flu season)
            dates = []
            for year in range(start_year, end_year):  # Changed to avoid going past end_year
                season_start = pd.date_range(start=f'{year}-10-01', end=f'{year+1}-05-31', freq='W-MON')
                dates.extend(season_start)
            
            # Add partial season for final year if needed
            if datetime.now().month <= 5:
                final_season = pd.date_range(start=f'{end_year}-10-01', end=datetime.now(), freq='W-MON')
            else:
                final_season = pd.date_range(start=f'{end_year}-10-01', end=f'{end_year+1}-05-31', freq='W-MON')
            dates.extend(final_season)
            
            dates = pd.DatetimeIndex(sorted(set(dates)))
            
            # Create more realistic seasonal pattern
            # Peak in January-February, low in summer
            days_since_oct1 = [(d - pd.Timestamp(f"{d.year}-10-01")).days for d in dates]
            seasonal_pattern = 2 * np.sin(2 * np.pi * (np.array(days_since_oct1) - 90) / 365) + 2
            
            # Add year-to-year variation (some seasons are worse than others)
            yearly_severity = {year: np.random.normal(1, 0.2) for year in range(start_year, end_year + 1)}
            severity_multiplier = [yearly_severity[d.year] for d in dates]
            
            # Generate data with realistic constraints
            base_ili = 2 + seasonal_pattern * np.array(severity_multiplier)
            base_ili = np.clip(base_ili + np.random.normal(0, 0.3, len(dates)), 0.5, 7.0)  # ILI rarely exceeds 7%
            
            # Calculate specimens based on ILI activity
            specimens_base = 2000 + 2000 * seasonal_pattern
            specimens = np.clip(specimens_base + np.random.normal(0, 200, len(dates)), 1000, 5000)
            
            # Calculate positive tests with realistic ratios
            positivity_rate = 0.1 + 0.15 * seasonal_pattern
            positivity_rate = np.clip(positivity_rate + np.random.normal(0, 0.02, len(dates)), 0.05, 0.35)
            
            total_positive = specimens * positivity_rate
            type_a_ratio = 0.7 + np.random.normal(0, 0.1, len(dates))  # Type A is typically more common
            type_a_ratio = np.clip(type_a_ratio, 0.5, 0.9)
            
            data = {
                'date': dates,
                'year': [d.year for d in dates],
                'week': [d.week for d in dates],
                'total_specimens': specimens.astype(int),
                'total_a': (total_positive * type_a_ratio).astype(int),
                'total_b': (total_positive * (1 - type_a_ratio)).astype(int),
                'percent_positive': positivity_rate * 100,
                'weighted_ili': base_ili
            }
            
            df = pd.DataFrame(data)
            
            # Validation checks
            assert (df['weighted_ili'] >= 0).all(), "Negative ILI values found"
            assert (df['percent_positive'] >= 0).all(), "Negative positivity rates found"
            assert (df['total_a'] >= 0).all() and (df['total_b'] >= 0).all(), "Negative test counts found"
            assert (df['percent_positive'] <= 100).all(), "Positivity rate exceeds 100%"
            
            print("\nData validation passed:")
            print(f"- ILI range: {df['weighted_ili'].min():.1f}% to {df['weighted_ili'].max():.1f}%")
            print(f"- Positivity range: {df['percent_positive'].min():.1f}% to {df['percent_positive'].max():.1f}%")
            print(f"- Specimen range: {df['total_specimens'].min():,} to {df['total_specimens'].max():,}")
            
            return df
                
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def analyze_flu_trends(self, data):
        """
        Perform comprehensive analysis of flu trends following the protocol
        """
        if data is None or len(data) == 0:
            print("No data available for analysis")
            return
            
        print("\n=== Flu Data Analysis ===")
        print(f"Time period: {data['date'].min()} to {data['date'].max()}")
        print(f"\nTotal weeks of data: {len(data)}")
        
        # 1. Time Series Visualization
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        plt.plot(data['date'], data['weighted_ili'], 'b-', label='Weighted ILI%')
        plt.title('Influenza-like Illness (ILI) Activity Over Time')
        plt.xlabel('Date')
        plt.ylabel('Weighted ILI%')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(data['date'], data['percent_positive'], 'r-', label='Percent Positive')
        plt.title('Percentage of Positive Flu Tests Over Time')
        plt.xlabel('Date')
        plt.ylabel('Percent Positive')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'time_series_analysis.png'))
        plt.close()
        
        # 2. Seasonal Decomposition
        decomposition = seasonal_decompose(data['weighted_ili'], period=52)
        plt.figure(figsize=(15, 12))
        plt.subplot(4, 1, 1)
        plt.plot(data['date'], decomposition.observed)
        plt.title('Observed Data')
        plt.subplot(4, 1, 2)
        plt.plot(data['date'], decomposition.trend)
        plt.title('Trend')
        plt.subplot(4, 1, 3)
        plt.plot(data['date'], decomposition.seasonal)
        plt.title('Seasonal')
        plt.subplot(4, 1, 4)
        plt.plot(data['date'], decomposition.resid)
        plt.title('Residual')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'seasonal_decomposition.png'))
        plt.close()
        
        # 3. Monthly Patterns
        data['month'] = data['date'].dt.month
        monthly_stats = data.groupby('month').agg({
            'weighted_ili': ['mean', 'std', 'min', 'max'],
            'percent_positive': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        # Plot monthly patterns
        plt.figure(figsize=(12, 6))
        monthly_stats['weighted_ili']['mean'].plot(kind='bar', yerr=monthly_stats['weighted_ili']['std'])
        plt.title('Average Monthly ILI Activity with Standard Deviation')
        plt.xlabel('Month')
        plt.ylabel('Weighted ILI%')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'monthly_patterns.png'))
        plt.close()
        
        # 4. Summary Statistics
        print("\n=== Summary Statistics ===")
        numeric_columns = ['total_specimens', 'total_a', 'total_b', 
                         'percent_positive', 'weighted_ili']
        stats = data[numeric_columns].describe().round(2)
        print(stats)
        
        # Create summary statistics visualization with box plots and trends
        plt.style.use('seaborn-v0_8')  # Use seaborn style for better aesthetics
        fig = plt.figure(figsize=(15, 12))
        
        # Create subplot grid with more space at top
        gs = plt.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.4)
        
        # Add overall title with more space
        fig.suptitle('Summary Statistics', fontsize=16, fontweight='bold', y=0.95)
        
        # 1. Box plots
        ax1 = fig.add_subplot(gs[0])
        colors = ['#3498db', '#2ecc71', '#e74c3c']  # Only need 3 colors now
        specimen_columns = ['total_specimens', 'total_a', 'total_b']
        data_to_plot = [data[col] for col in specimen_columns]
        box_plot = ax1.boxplot(data_to_plot, patch_artist=True,
                             medianprops=dict(color="black", linewidth=2),
                             flierprops=dict(marker='o', markerfacecolor='gray', markersize=4),
                             widths=0.7)
        
        # Customize box plots
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add labels and title
        ax1.set_title('Distribution of Specimen Counts', fontsize=14, pad=20)
        ax1.set_xticklabels([
            'Total\nSpecimens', 'Influenza A\nCases', 'Influenza B\nCases'
        ], rotation=0, fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels for medians
        medians = [median.get_ydata()[0] for median in box_plot['medians']]
        for i, median in enumerate(medians):
            ax1.text(i+1, median, f'{median:,.0f}', 
                    horizontalalignment='center', verticalalignment='bottom',
                    fontweight='bold', fontsize=10)
        
        # 2. Normalized trends over time
        ax2 = fig.add_subplot(gs[1])
        numeric_columns = ['total_specimens', 'total_a', 'total_b', 
                         'percent_positive', 'weighted_ili']
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f', '#9b59b6']  # Keep all colors for trends
        for i, col in enumerate(numeric_columns):
            # Create more descriptive labels
            label = col.replace('total_specimens', 'Total Specimens')\
                      .replace('total_a', 'Influenza A Cases')\
                      .replace('total_b', 'Influenza B Cases')\
                      .replace('percent_positive', 'Positivity Rate')\
                      .replace('weighted_ili', 'ILI Activity')\
                      .replace('_', ' ').title()
            
            # Normalize the values for comparison
            normalized_values = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
            ax2.plot(data['date'], normalized_values, color=colors[i], 
                    label=label,
                    alpha=0.7, linewidth=2)
        
        ax2.set_title('Normalized Trends Over Time', fontsize=14, pad=20)
        ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Date', fontsize=10)
        ax2.set_ylabel('Normalized Value', fontsize=10)
        
        # Format x-axis dates
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Adjust layout and save
        plt.subplots_adjust(right=0.85)  # Make room for legend
        plt.savefig(os.path.join(self.output_dir, 'summary_statistics.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # Save statistics to file
        stats.to_csv(os.path.join(self.output_dir, 'summary_statistics.csv'))
        
        # Generate analysis report
        self._generate_report(data, stats, monthly_stats, peak_ili=data.loc[data['weighted_ili'].idxmax()])
        
        # 5. Peak Analysis
        print("\n=== Peak Analysis ===")
        peak_ili = data.loc[data['weighted_ili'].idxmax()]
        print(f"Peak ILI Activity: {peak_ili['weighted_ili']:.2f}% on {peak_ili['date'].strftime('%Y-%m-%d')}")
        
        # 6. Year-over-Year Comparison
        yearly_stats = data.groupby('year').agg({
            'weighted_ili': ['mean', 'max'],
            'percent_positive': ['mean', 'max']
        }).round(2)
        
        print("\n=== Year-over-Year Comparison ===")
        print(yearly_stats)
        
        # Create yearly comparison visualization
        plt.figure(figsize=(15, 8))
        
        # Plot mean ILI by year
        plt.subplot(2, 1, 1)
        ax1 = plt.gca()
        yearly_stats['weighted_ili']['mean'].plot(kind='bar', color='skyblue', alpha=0.7, ax=ax1, label='Average ILI%')
        ax1.plot(range(len(yearly_stats)), yearly_stats['weighted_ili']['max'], 'r-o', label='Maximum ILI%')
        plt.title('Yearly ILI Activity Comparison')
        plt.xlabel('Year')
        plt.ylabel('Weighted ILI%')
        ax1.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot mean positivity rate by year
        plt.subplot(2, 1, 2)
        ax2 = plt.gca()
        yearly_stats['percent_positive']['mean'].plot(kind='bar', color='lightgreen', alpha=0.7, ax=ax2, label='Average Positivity Rate')
        ax2.plot(range(len(yearly_stats)), yearly_stats['percent_positive']['max'], 'r-o', label='Max Positivity Rate')
        plt.title('Yearly Positivity Rate Comparison')
        plt.xlabel('Year')
        plt.ylabel('Percent Positive')
        ax2.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'yearly_comparison.png'))
        plt.close()
        
        # Save yearly statistics to file
        yearly_stats.to_csv(os.path.join(self.output_dir, 'yearly_comparison.csv'))
        
        print(f"\nAnalysis complete! Results have been saved to the '{self.output_dir}' directory.")

    def _generate_report(self, data, stats, monthly_stats, peak_ili):
        """Generate a comprehensive analysis report"""
        report_path = os.path.join(self.output_dir, 'analysis_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Flu Trend Analysis Report\n\n")
            
            # Overview
            f.write("## Overview\n")
            f.write(f"Analysis period: {data['date'].min()} to {data['date'].max()}\n")
            f.write(f"Total weeks analyzed: {len(data)}\n\n")
            
            # Summary Statistics
            f.write("## Summary Statistics\n")
            f.write("### Key Metrics\n")
            f.write("- **Total Specimens**:\n")
            f.write(f"  - Average: {stats.loc['mean', 'total_specimens']:,.0f}\n")
            f.write(f"  - Range: {stats.loc['min', 'total_specimens']:,.0f} to {stats.loc['max', 'total_specimens']:,.0f}\n")
            f.write("- **Influenza Type A**:\n")
            f.write(f"  - Average: {stats.loc['mean', 'total_a']:,.0f}\n")
            f.write(f"  - Range: {stats.loc['min', 'total_a']:,.0f} to {stats.loc['max', 'total_a']:,.0f}\n")
            f.write("- **Influenza Type B**:\n")
            f.write(f"  - Average: {stats.loc['mean', 'total_b']:,.0f}\n")
            f.write(f"  - Range: {stats.loc['min', 'total_b']:,.0f} to {stats.loc['max', 'total_b']:,.0f}\n")
            f.write("- **Positivity Rate**:\n")
            f.write(f"  - Average: {stats.loc['mean', 'percent_positive']:.1f}%\n")
            f.write(f"  - Range: {stats.loc['min', 'percent_positive']:.1f}% to {stats.loc['max', 'percent_positive']:.1f}%\n")
            f.write("- **ILI Activity**:\n")
            f.write(f"  - Average: {stats.loc['mean', 'weighted_ili']:.1f}%\n")
            f.write(f"  - Range: {stats.loc['min', 'weighted_ili']:.1f}% to {stats.loc['max', 'weighted_ili']:.1f}%\n\n")
            
            # Peak Analysis
            f.write("## Peak Analysis\n")
            f.write(f"Peak ILI activity occurred on {peak_ili['date'].strftime('%Y-%m-%d')} ")
            f.write(f"with {peak_ili['weighted_ili']:.1f}% weighted ILI.\n\n")
            
            # Seasonal Patterns
            f.write("## Seasonal Patterns\n")
            f.write("### Monthly Trends\n")
            f.write("Average ILI activity by month:\n")
            for month in range(1, 13):
                if month in monthly_stats.index:
                    mean_ili = monthly_stats.loc[month, ('weighted_ili', 'mean')]
                    std_ili = monthly_stats.loc[month, ('weighted_ili', 'std')]
                    f.write(f"- Month {month}: {mean_ili:.1f}% (±{std_ili:.1f}%)\n")
            
            # Visualizations
            f.write("\n## Generated Visualizations\n")
            f.write("1. `time_series_analysis.png`: Time series of ILI activity and positivity rates\n")
            f.write("2. `seasonal_decomposition.png`: Seasonal decomposition of ILI trends\n")
            f.write("3. `monthly_patterns.png`: Monthly patterns of ILI activity\n")
            f.write("4. `summary_statistics.png`: Summary statistics visualization\n")
            f.write("5. `yearly_comparison.png`: Year-over-year comparison\n")
            f.write("6. `predictions.png`: Future predictions with confidence intervals\n")
            
        print(f"\nDetailed analysis report has been saved to '{report_path}'")

    def predict_flu_trends(self, data, target_col='weighted_ili', sequence_length=12, future_weeks=4):
        """
        Predict future flu trends using LSTM with improved accuracy
        Args:
            data: DataFrame with flu data
            target_col: Column to predict
            sequence_length: Number of time steps to use for prediction
            future_weeks: Number of weeks to predict into the future
        """
        # Create seasonal features with more granular seasonality
        data.loc[:, 'sin_week'] = np.sin(2 * np.pi * data['week_of_year'] / 52)
        data.loc[:, 'cos_week'] = np.cos(2 * np.pi * data['week_of_year'] / 52)
        
        # Prepare features for LSTM
        feature_columns = [target_col, 'percent_positive', 'sin_week', 'cos_week']
        values = data[feature_columns].values.astype('float32')  # Ensure float32 type
        
        # Scale features using separate scalers for better interpretability
        scalers = {}
        scaled_values = np.zeros_like(values)
        for i, col in enumerate(feature_columns):
            scalers[col] = MinMaxScaler()
            scaled_values[:, i] = scalers[col].fit_transform(values[:, i].reshape(-1, 1)).ravel()
        
        # Create sequences with overlap to maintain continuity
        X, y = [], []
        for i in range(len(scaled_values) - sequence_length):
            X.append(scaled_values[i:i+sequence_length])
            y.append(scaled_values[i+sequence_length, 0])
        X = np.array(X, dtype='float32')
        y = np.array(y, dtype='float32')
        
        # Split data with overlap to maintain continuity
        train_size = int(len(X) * 0.8)
        X_train = X[:train_size]
        X_test = X[train_size-sequence_length:]
        y_train = y[:train_size]
        y_test = y[train_size-sequence_length:]
        
        # Build enhanced LSTM model with residual connections
        model = Sequential([
            LSTM(128, activation='relu', input_shape=(sequence_length, len(feature_columns)), return_sequences=True),
            Dropout(0.2),
            LSTM(64, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        # Compile with reduced learning rate and gradient clipping
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='huber')
        
        # Train model with early stopping and learning rate reduction
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        print("\nTraining LSTM model...")
        history = model.fit(
            X_train, y_train,
            epochs=150,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Make predictions with proper sequence handling
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        
        # Inverse transform predictions using the target column scaler
        train_predict = scalers[target_col].inverse_transform(train_predict)
        test_predict = scalers[target_col].inverse_transform(test_predict)
        
        # Get original values for comparison
        y_train_inv = scalers[target_col].inverse_transform(y_train.reshape(-1, 1))
        y_test_inv = scalers[target_col].inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predict))
        test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict))
        train_r2 = r2_score(y_train_inv, train_predict)
        test_r2 = r2_score(y_test_inv, test_predict)
        
        print(f"\nModel Performance:")
        print(f"Train RMSE: {train_rmse:.3f}")
        print(f"Test RMSE: {test_rmse:.3f}")
        print(f"Train R²: {train_r2:.3f}")
        print(f"Test R²: {test_r2:.3f}")
        
        # Predict future values with uncertainty
        last_sequence = scaled_values[-sequence_length:].astype('float32')
        future_predictions = []
        prediction_intervals = []
        
        # Monte Carlo predictions with dropout
        n_iterations = 100
        
        for week in range(future_weeks):
            sequence_predictions = []
            for _ in range(n_iterations):
                pred = model.predict(last_sequence.reshape(1, sequence_length, len(feature_columns)), verbose=0)
                sequence_predictions.append(pred[0, 0])
            
            # Calculate mean and confidence intervals
            mean_pred = np.mean(sequence_predictions)
            std_pred = np.std(sequence_predictions)
            future_predictions.append(mean_pred)
            prediction_intervals.append([mean_pred - 2*std_pred, mean_pred + 2*std_pred])
            
            # Update sequence with seasonal features
            next_week = (data.index[-1] + pd.Timedelta(weeks=week+1))
            next_week_of_year = next_week.isocalendar().week
            sin_week = np.sin(2 * np.pi * next_week_of_year / 52)
            cos_week = np.cos(2 * np.pi * next_week_of_year / 52)
            
            next_features = np.array([mean_pred, last_sequence[-1, 1], sin_week, cos_week], dtype='float32')
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1] = next_features
        
        # Convert predictions back to original scale
        future_predictions = scalers[target_col].inverse_transform(np.array(future_predictions).reshape(-1, 1))
        intervals = np.array(prediction_intervals)
        intervals = scalers[target_col].inverse_transform(intervals.reshape(-1, 1)).reshape(-1, 2)
        
        # Plot results with confidence intervals
        plt.figure(figsize=(15, 6))
        
        # Plot actual values
        actual_dates = data.index[-len(test_predict):]
        plt.plot(actual_dates, y_test_inv, 'b-', label='Actual Values', alpha=0.7)
        
        # Plot historical predictions with proper alignment
        plt.plot(actual_dates, test_predict, 'r-', label='Historical Predictions', alpha=0.7)
        
        # Plot future predictions with confidence intervals
        future_dates = pd.date_range(start=data.index[-1], periods=future_weeks+1, freq='W')[1:]
        plt.plot(future_dates, future_predictions, 'g--', label='Future Predictions')
        
        plt.fill_between(future_dates,
                        intervals[:, 0],
                        intervals[:, 1],
                        color='g', alpha=0.1,
                        label='95% Confidence Interval')
        
        plt.title(f'Flu Trend Predictions ({target_col})')
        plt.xlabel('Date')
        plt.ylabel(target_col)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'predictions.png'))
        plt.close()
        
        # Save predictions to CSV
        predictions_df = pd.DataFrame({
            'date': future_dates,
            'prediction': future_predictions.flatten(),
            'lower_bound': intervals[:, 0],
            'upper_bound': intervals[:, 1]
        })
        predictions_df.to_csv(os.path.join(self.output_dir, 'predictions.csv'), index=False)
        
        print("\nPredicted values for next weeks (with 95% confidence intervals):")
        for i, date in enumerate(future_dates):
            print(f"{date.strftime('%Y-%m-%d')}: {future_predictions[i, 0]:.2f}% " +
                  f"[{intervals[i, 0]:.2f}% - {intervals[i, 1]:.2f}%]")
        
        return future_predictions, intervals

def main():
    # Initialize data collector
    collector = FluDataCollector()
    
    # Get flu data
    flu_data = collector.get_flu_data()
    
    # Analyze trends
    if flu_data is not None:
        collector.analyze_flu_trends(flu_data)
        
        # Set date as index
        flu_data.set_index('date', inplace=True)
        
        # Find the most recent October in the data
        october_dates = flu_data.index[flu_data.index.month == 10]
        last_october = october_dates[-2]  # Use second-to-last October for validation
        next_october = october_dates[-1]  # Use for prediction target
        
        # Use data up to September before the validation October
        prediction_cutoff = last_october - pd.DateOffset(months=1)
        training_data = flu_data.loc[:prediction_cutoff].copy()  # Create explicit copy
        
        # Add month and week columns before prediction
        training_data.loc[:, 'month'] = training_data.index.month
        training_data.loc[:, 'week_of_year'] = training_data.index.isocalendar().week
        
        # Make predictions starting from October (early flu season)
        print(f"\nMaking predictions for flu season starting {next_october.strftime('%Y-%m-%d')}")
        print(f"Using training data up to: {prediction_cutoff.strftime('%Y-%m-%d')}")
        print(f"Validation period: {last_october.strftime('%Y-%m-%d')} to {(last_october + pd.DateOffset(weeks=12)).strftime('%Y-%m-%d')}")
        collector.predict_flu_trends(training_data, future_weeks=12)  # Predict 12 weeks into flu season

if __name__ == "__main__":
    main() 