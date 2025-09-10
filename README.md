# 🚢 Titanic Survival Prediction - Kaggle Competition

![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue?style=for-the-badge&logo=kaggle)
![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)

A comprehensive machine learning solution for the famous Kaggle Titanic competition. This project implements multiple classification algorithms to predict passenger survival with advanced model comparison and hyperparameter optimization techniques.

## 🎯 Competition Overview

The **Titanic: Machine Learning from Disaster** is one of Kaggle's most popular introductory competitions. The challenge is to predict which passengers survived the tragic Titanic shipwreck based on passenger data such as age, gender, ticket class, and family information.

**Goal**: Predict binary survival outcomes (0 = Did not survive, 1 = Survived)

## ✨ Key Features & Approach

- **Multiple ML Algorithms**: Implementation of Logistic Regression, Random Forest, and Support Vector Machine
- **Model Comparison**: Systematic evaluation and comparison of different algorithms
- **Hyperparameter Optimization**: GridSearchCV for finding optimal model parameters
- **Cross-Validation**: Robust model evaluation using k-fold cross-validation
- **Comprehensive Evaluation**: Classification reports with precision, recall, and F1-scores
- **Competition-Ready**: Proper submission format for Kaggle leaderboard

## 🛠️ Technologies Used

- **Python 3.x** - Primary programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine learning algorithms and evaluation metrics
- **Matplotlib** - Data visualization
- **Jupyter Notebook** - Interactive development environment

## 📊 Dataset Information

The Titanic dataset contains information about passengers aboard the RMS Titanic:

### Features:
| Feature | Description | Type |
|---------|-------------|------|
| PassengerId | Unique identifier | Integer |
| Pclass | Passenger class (1st, 2nd, 3rd) | Integer |
| Name | Passenger name | String |
| Sex | Gender | String |
| Age | Age in years | Float |
| SibSp | Number of siblings/spouses aboard | Integer |
| Parch | Number of parents/children aboard | Integer |
| Ticket | Ticket number | String |
| Fare | Passenger fare | Float |
| Cabin | Cabin number | String |
| Embarked | Port of embarkation | String |

### Target Variable:
- **Survived** - Survival status (0 = No, 1 = Yes)

## 🚀 Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/Zyttik-m/Kaggle_Titanic_compet.git
cd Kaggle_Titanic_compet
```

2. **Install required dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib jupyter seaborn
```

3. **Launch Jupyter Notebook**
```bash
jupyter notebook titanic.ipynb
```

## 💻 Usage

1. Open the `titanic.ipynb` notebook
2. Run all cells to execute the complete analysis pipeline
3. The notebook includes:
   - Data loading and exploration
   - Feature engineering and preprocessing
   - Model training and evaluation
   - Hyperparameter tuning
   - Competition submission preparation

## 🤖 Models Implemented

### 1. Logistic Regression
- Linear classification algorithm
- Baseline model for binary classification
- Provides probability estimates

### 2. Random Forest Classifier
- Ensemble method using multiple decision trees
- Handles non-linear relationships
- Feature importance analysis

### 3. Support Vector Machine (SVM)
- Finds optimal decision boundary
- Effective for high-dimensional data
- Kernel trick for non-linear classification

## 📈 Model Pipeline

### 1. Data Exploration
- Dataset overview and statistics
- Missing value analysis
- Survival rate analysis by different features

### 2. Data Preprocessing
- Handling missing values
- Feature encoding for categorical variables
- Feature scaling and normalization

### 3. Feature Engineering
- Creating meaningful features from existing data
- Family size calculation
- Title extraction from names

### 4. Model Training & Evaluation
- Train-validation split
- Cross-validation implementation
- Hyperparameter tuning with GridSearchCV
- Performance comparison across models

### 5. Competition Submission
- Prediction on test dataset
- Submission file preparation
- Kaggle leaderboard submission

## 📁 Project Structure

```
Kaggle_Titanic_compet/
├── titanic.ipynb            # Main analysis notebook
├── submission.csv           # Kaggle submission file
├── data/                    # Dataset directory
│   ├── train.csv           # Training data
│   ├── test.csv            # Test data
│   └── gender_submission.csv # Sample submission
└── README.md               # Project documentation
```

## 🏆 Key Insights & Results

### Survival Factors Analysis:
- **Gender**: Women had significantly higher survival rates
- **Passenger Class**: First-class passengers had better survival chances
- **Age**: Children had higher survival rates
- **Family Size**: Moderate family sizes showed optimal survival rates

### Model Performance:
- Comprehensive evaluation using accuracy, precision, recall, and F1-score
- Cross-validation ensures robust performance estimates
- Model comparison identifies best-performing algorithm

## 🔮 Future Improvements

- [ ] Advanced feature engineering (titles, deck extraction, fare binning)
- [ ] Ensemble methods (Voting Classifier, Stacking)
- [ ] Deep learning approaches with neural networks
- [ ] Advanced preprocessing techniques
- [ ] Feature selection optimization
- [ ] Hyperparameter optimization with Bayesian methods

## 📊 Competition Performance

This solution demonstrates:
- Systematic approach to machine learning competitions
- Professional model development workflow
- Comprehensive evaluation methodology
- Competition-ready submission format

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/enhancement`)
5. Create a Pull Request

## 📚 Learning Outcomes

This project demonstrates proficiency in:
- **Machine Learning**: Multiple algorithm implementation and comparison
- **Data Science**: Complete end-to-end workflow
- **Model Evaluation**: Cross-validation and comprehensive metrics
- **Competition Skills**: Kaggle submission preparation
- **Python Programming**: Professional data science libraries

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Kittithat Chalermvisutkul**
- GitHub: [@Zyttik-m](https://github.com/Zyttik-m)
