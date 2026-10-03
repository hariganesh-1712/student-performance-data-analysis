"""
Data Science Fundamentals Assessment
Project: Student Performance Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("student_performance.csv")

# 2. Inspect data
print(df.head())
print("\nData types:\n", df.dtypes)
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# 3. Basic statistics
print("\nDescriptive statistics:\n", df.describe(numeric_only=True))

# 4. Useful calculations
print("\nAverage final score:", round(df["Final_Score"].mean(), 2))
print("Median final score:", round(df["Final_Score"].median(), 2))
print("Standard deviation:", round(df["Final_Score"].std(), 2))
print("Pass percentage:", round((df["Passed"] == "Yes").mean() * 100, 2))

# 5. Correlation analysis
numeric_cols = [
    "Study_Hours_Per_Week",
    "Attendance_Percent",
    "Assignment_Average",
    "Previous_Semester_Score",
    "Final_Score"
]
print("\nCorrelation matrix:\n", df[numeric_cols].corr().round(2))

# 6. Group analysis
print("\nAverage score by pass status:")
print(df.groupby("Passed")["Final_Score"].mean().round(2))

# 7. Visualization: study hours vs final score
plt.figure(figsize=(7, 5))
plt.scatter(df["Study_Hours_Per_Week"], df["Final_Score"])
plt.xlabel("Study Hours Per Week")
plt.ylabel("Final Score")
plt.title("Study Hours vs Final Score")
plt.tight_layout()
plt.savefig("study_hours_vs_score.png", dpi=150)
plt.show()

# 8. Visualization: score distribution
plt.figure(figsize=(7, 5))
plt.hist(df["Final_Score"], bins=10, edgecolor="black")
plt.xlabel("Final Score")
plt.ylabel("Number of Students")
plt.title("Distribution of Final Scores")
plt.tight_layout()
plt.savefig("final_score_distribution.png", dpi=150)
plt.show()
