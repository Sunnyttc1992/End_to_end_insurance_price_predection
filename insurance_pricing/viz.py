import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_insurance_distribution(df):
    plt.style.use('seaborn-v0_8-darkgrid')

    # --------------------
    # Continuous variables
    # --------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.histplot(df['age'], ax=axes[0, 0], color='steelblue').set_title('Age Distribution')
    sns.histplot(df['bmi'], ax=axes[0, 1], color='green').set_title('BMI Distribution')
    sns.histplot(df['children'], ax=axes[1, 0], color='purple').set_title('Children Distribution')
    sns.histplot(df['charges'], ax=axes[1, 1], color='skyblue').set_title('Charges Distribution')

    plt.tight_layout()
    plt.show()

    # --------------------
    # Categorical variables
    # --------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Pie chart for sex
    sex_counts = df['sex'].value_counts()
    axes[0].pie(
        sex_counts.values,
        labels=sex_counts.index,
        autopct='%1.1f%%',
        startangle=90
    )
    axes[0].set_title('Distribution of Sex')

    # Bar chart for smoker (aligned with others)
    smoker_counts = df['smoker'].value_counts()
    colors = ['#90EE90', '#FFB6C1']

    axes[1].bar(
        smoker_counts.index,
        smoker_counts.values,
        color=colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=2
    )
    axes[1].set_xlabel('Smoking Status')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Smoker Distribution', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Countplot for region
    sns.countplot(x='region', data=df, ax=axes[2])
    axes[2].set_title('Distribution of Region')

    plt.tight_layout()
    plt.show()
