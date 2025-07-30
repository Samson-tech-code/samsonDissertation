import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm


def load_data(path='data/mental_health.csv'):
    """Load the mental health dataset."""
    df = pd.read_csv(path)
    return df


def descriptive_stats(df):
    """Generate descriptive statistics and plots."""
    desc = df[['GHQ12', 'income']].describe()
    freq_gender = df['gender'].value_counts()
    freq_ethnicity = df['ethnicity'].value_counts()
    # Plot GHQ12 distribution
    plt.figure()
    sns.histplot(df['GHQ12'], kde=True)
    plt.title('GHQ-12 Distribution')
    plt.savefig('analysis/ghq12_distribution.png')
    plt.close()
    return {'describe': desc, 'freq_gender': freq_gender, 'freq_ethnicity': freq_ethnicity}


def bivariate_analysis(df):
    """Run simple t-tests and chi-square tests."""
    from scipy import stats
    # Example t-test: GHQ12 by gender
    males = df[df['gender'] == 'Male']['GHQ12']
    females = df[df['gender'] == 'Female']['GHQ12']
    t_stat, p_val = stats.ttest_ind(males, females, equal_var=False)
    # Chi-square: employment stability vs gender
    contingency = pd.crosstab(df['employment_stable'], df['gender'])
    chi2, chi_p, _, _ = stats.chi2_contingency(contingency)
    return {'t_test_gender': (t_stat, p_val), 'chi2_employment_gender': (chi2, chi_p)}


def regression_models(df):
    """Run linear and logistic regression models."""
    results = {}
    # Linear regression for GHQ12
    X = df[['age', 'income']]
    X = sm.add_constant(X)
    lm = sm.OLS(df['GHQ12'], X).fit()
    results['linear'] = lm.summary().as_text()
    # Logistic regression for employment stability
    X_log = df[['age', 'income']]
    X_log = sm.add_constant(X_log)
    logit = sm.Logit(df['employment_stable'], X_log).fit(disp=False)
    results['logistic_employment'] = logit.summary().as_text()
    return results


def ml_models(df):
    """Train basic ML models and return metrics."""
    features = ['age', 'income']
    X = df[features]
    y = df['employment_stable']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, pred),
        'precision': precision_score(y_test, pred),
        'recall': recall_score(y_test, pred),
        'f1': f1_score(y_test, pred)
    }
    return metrics


def run_all(path='data/mental_health.csv'):
    df = load_data(path)
    stats = descriptive_stats(df)
    biv = bivariate_analysis(df)
    reg = regression_models(df)
    ml = ml_models(df)
    return {'descriptive': stats, 'bivariate': biv, 'regression': reg, 'ml': ml}


if __name__ == '__main__':
    results = run_all()
    for section, res in results.items():
        print(f'--- {section.upper()} ---')
        print(res)
