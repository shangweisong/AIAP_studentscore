import matplotlib.pyplot as plt
import seaborn as sns

def plot_r2_scores(model_results):
    model_names = [result[0] for result in model_results]
    r2_scores = [result[1] for result in model_results]

    plt.figure(figsize=(8, 6))
    plt.bar(model_names, r2_scores, color='skyblue')
    plt.title('R² Scores of Models', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('R² Score', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_residuals(y_test, y_pred, model_name):
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 2})
    plt.title(f'Residual Plot: {model_name}', fontsize=16)
    plt.xlabel('Predicted Values', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.show()

def plot_predicted_vs_actual(y_test, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f'Predicted vs. Actual: {model_name}', fontsize=16)
    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.tight_layout()
    plt.show()