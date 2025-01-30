# Credit Risk Modeling

## Overview
A comprehensive credit risk assessment model using machine learning techniques to predict loan defaults. The model achieves an AUROC of 0.85 and a Gini coefficient of 0.71, demonstrating strong predictive performance for credit risk evaluation.

## Key Features
- Advanced credit risk modeling using logistic regression
- Extensive feature engineering with Weight of Evidence (WoE) transformation
- Handling of imbalanced data using oversampling techniques
- Comprehensive model evaluation with multiple metrics
- Built-in data preprocessing pipeline

## Technical Details
### Model Performance
- AUROC: 0.85
- Gini Coefficient: 0.71
- KS Statistic: 0.56
- Precision-Recall AUC: 0.98

### Data Processing
- Feature engineering using Weight of Evidence (WoE)
- Information Value (IV) calculations for feature selection
- Handling of categorical variables through dummy encoding
- Treatment of imbalanced data using Random Oversampling

## Technologies Used
- Python 3.x
- Key Libraries:
  - scikit-learn for model building
  - pandas for data manipulation
  - numpy for numerical operations
  - matplotlib and seaborn for visualization
  - imblearn for handling imbalanced data
  - yellowbrick for model visualization

## Project Structure
```
├── data/
│   └── loan_data_2007_2014.csv
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_building.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── model.py
├── models/
│   └── credit_risk_model.sav
└── README.md
```

## Installation & Usage
1. Clone the repository:
```bash
git clone https://github.com/username/credit-risk-modeling.git
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the notebooks in sequence:
```bash
jupyter notebook notebooks/01_data_preprocessing.ipynb
```

## Model Features
The model uses a variety of features including:
- Loan characteristics (term, interest rate, grade)
- Borrower demographics (home ownership, employment length)
- Credit history (credit inquiries, revolving credit utilization)
- Payment behavior (total payments, recovery amounts)
- Geographic information (state-level data)

## Model Evaluation
The model has been evaluated using multiple metrics:
- ROC curve analysis
- Precision-Recall curves
- KS statistics
- Confusion matrix
- Classification reports

## Data Preprocessing Steps
1. Missing value treatment
2. Feature engineering using WoE
3. Information Value calculation
4. Categorical variable encoding
5. Feature selection based on IV
6. Data balancing using oversampling

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions and feedback, please open an issue in the repository.

## Acknowledgments
- Thanks to all contributors who have helped with the development
- Special thanks to the scikit-learn and imblearn communities
