# Loan Repayment Prediction

## Overview
This project demonstrates a supervised classification approach to predict whether a loan will be repaid based on customer data. The notebook uses machine learning algorithms to build and evaluate predictive models to identify repayment patterns and optimize financial decision-making.

---

## Features
- **Data Preprocessing**:
  - Handling missing values.
  - Encoding categorical variables.
  - Standardizing numerical features.
- **Exploratory Data Analysis (EDA)**:
  - Visualizing key trends and patterns in the data.
  - Identifying correlations between features and loan repayment status.
- **Model Training**:
  - Logistic Regression.
  - Random Forest Classifier.
- **Model Evaluation**:
  - Performance metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
  - Confusion matrix and feature importance analysis.
- **Visualization**:
  - Data distribution plots.
  - Performance metric charts for model evaluation.

---

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - pandas: For data manipulation.
  - numpy: For numerical computations.
  - matplotlib & seaborn: For data visualization.
  - scikit-learn: For machine learning models and evaluation.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/supervised-classification-projects.git
   ```
2. Navigate to the project directory:
   ```bash
   cd supervised-classification-projects
   ```
3. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Loan_Repayment_Precidtion.ipynb
   ```
2. Follow the notebook sections to:
   - Preprocess the data.
   - Perform exploratory data analysis.
   - Train and evaluate the models.
3. Experiment with different hyperparameters or models to improve predictions.

---

## Results
- **Logistic Regression**:
  - Accuracy: 63%
- **Random Forest Classifier**:
  - Accuracy: 83%

---

## Future Work
- Implement additional machine learning algorithms (e.g., XGBoost, SVM).
- Tune hyperparameters for better performance.
- Deploy the model using Flask or Streamlit for real-time predictions.

---

## License
This project is licensed under the MIT License.


# Product Analysis and Price Prediction

## Overview
This project focuses on analyzing product data and building a predictive model to estimate product prices. Using supervised learning techniques, the project explores patterns in the data, applies feature engineering, and evaluates various machine learning models to achieve accurate predictions.

## Objectives
- Perform exploratory data analysis (EDA) to uncover insights about the dataset.
- Clean and preprocess data for use in machine learning models.
- Build and evaluate predictive models for product price prediction.
- Visualize trends and patterns to aid decision-making.

## Dataset
The dataset includes product-related features such as:
- **Product name**
- **Category**
- **Price**
- **Ratings**
- **Other metadata**


## Methodology
1. **Exploratory Data Analysis (EDA)**:
   - Identified missing values and outliers.
   - Analyzed relationships between features using visualizations (e.g., heatmaps, pair plots).
   - Uncovered patterns in product pricing based on categories and features.

2. **Data Preprocessing**:
   - Encoded categorical variables using techniques such as `LabelEncoder`.
   - Scaled numerical features for consistency in model training.
   - Split the dataset into training and testing sets using `train_test_split`.

3. **Modeling**:
   - Applied various regression models, including:
     - Linear Regression
     - Ridge Regression
     - Lasso Regression
   - Evaluated models using metrics such as Mean Squared Error (MSE) and R² score.

4. **Visualization**:
   - Used libraries like Matplotlib, Seaborn, and Plotly for creating interactive and static plots.
   - Visualized correlations and distribution of target variables.

## Libraries and Tools
The following libraries were utilized:
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Matplotlib** & **Seaborn**: Data visualization.
- **Plotly**: Interactive visualizations.
- **Scikit-learn**: Machine learning models and preprocessing.

## Results
The project demonstrated:
- Effective preprocessing techniques that improved model performance.
- Linear regression models achieving an R² score of X (replace with actual value).
- Insights into the features most strongly correlated with product pricing.

## Future Work
- Incorporate more advanced models like Gradient Boosting or Neural Networks for improved predictions.
- Perform hyperparameter tuning for better model performance.
- Expand the dataset with more features and observations for generalization.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/supervised-learning-project.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   Open `Product_analysis_and_Price_Prediction.ipynb` in Jupyter Notebook and execute the cells.

## File Structure
- **Product_analysis_and_Price_Prediction.ipynb**: Main notebook containing code and analysis.
- **data/**: Folder containing dataset files (if available).
- **requirements.txt**: List of required Python libraries.

