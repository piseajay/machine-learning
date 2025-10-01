# Linear Regression with California Housing Dataset

This example demonstrates how to build and evaluate a Linear Regression model using a real-world dataset.

## Dataset Information

**California Housing Dataset**
- **Source**: Sklearn datasets (originally from StatLib repository, based on 1990 California census data)
- **Samples**: 20,640 observations
- **Features**: 8 numerical features
- **Target**: Median house value for California districts (in $100,000s)

### Features

1. **MedInc**: Median income in block group
2. **HouseAge**: Median house age in block group
3. **AveRooms**: Average number of rooms per household
4. **AveBedrms**: Average number of bedrooms per household
5. **Population**: Block group population
6. **AveOccup**: Average number of household members
7. **Latitude**: Block group latitude
8. **Longitude**: Block group longitude

## What This Example Covers

1. **Data Loading**: Loading a real-world dataset from sklearn
2. **Data Exploration**: 
   - Dataset statistics
   - Missing value analysis
   - Feature descriptions
3. **Data Preprocessing**:
   - Train-test split (80-20)
   - Feature scaling using StandardScaler
4. **Model Training**: Training a Linear Regression model
5. **Model Evaluation**:
   - Multiple metrics (MSE, RMSE, MAE, R²)
   - Training vs testing performance
6. **Visualization**:
   - Actual vs Predicted plot
   - Residuals analysis
   - Feature importance (coefficients)
   - Residual distribution
7. **Interpretation**: Understanding model coefficients and predictions

## How to Run

```bash
python linear_regression_kaggle_example.py
```

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
```

## Expected Output

The script will output:
- Dataset information and statistics
- Training and testing metrics
- Model coefficients for each feature
- Visualizations saved as PNG file
- Sample predictions with actual vs predicted comparisons

## Key Metrics Explained

- **R² Score**: Proportion of variance in the target explained by the model (0 to 1, higher is better)
- **RMSE** (Root Mean Squared Error): Average prediction error in the same units as target
- **MAE** (Mean Absolute Error): Average absolute difference between predictions and actual values
- **MSE** (Mean Squared Error): Average squared difference between predictions and actual values

## Understanding the Results

The model's coefficients tell us how each feature impacts house prices:
- **Positive coefficients**: Feature increases house price
- **Negative coefficients**: Feature decreases house price
- **Magnitude**: Indicates the strength of the relationship

For example, if MedInc has a large positive coefficient, it means higher median income strongly predicts higher house prices.

## Alternative: Using an Actual Kaggle Dataset

If you want to use a dataset directly from Kaggle:

1. Install the Kaggle API:
   ```bash
   pip install kaggle
   ```

2. Set up Kaggle credentials (kaggle.json)

3. Download a dataset (e.g., House Prices dataset):
   ```bash
   kaggle competitions download -c house-prices-advanced-regression-techniques
   ```

4. Modify the script to load from CSV:
   ```python
   df = pd.read_csv('path/to/train.csv')
   ```

## Suggested Kaggle Datasets for Linear Regression

1. **House Prices - Advanced Regression Techniques**
   - Competition dataset with 79 features
   - URL: kaggle.com/c/house-prices-advanced-regression-techniques

2. **USA Housing Dataset**
   - Simple dataset for beginners
   - URL: kaggle.com/datasets/vedavyasv/usa-housing

3. **Student Performance Dataset**
   - Predicting student grades
   - URL: kaggle.com/datasets/larsen0966/student-performance-data-set

4. **Medical Cost Personal Dataset**
   - Predicting medical costs
   - URL: kaggle.com/datasets/mirichoi0218/insurance
