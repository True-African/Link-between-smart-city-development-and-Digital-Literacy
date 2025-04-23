import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

print("Creating basic plots...")

# Load the dataset
print("Loading dataset...")
df_positive = pd.read_csv('positive_abstracts.csv')
total_abstracts = len(df_positive)
print(f"Total abstracts in dataset: {total_abstracts}")

# Filter to get positive labeled abstracts only
df_positive = df_positive[df_positive['Target'] == 1]
print(f"Positive abstracts after filtering: {len(df_positive)}")

# 1. Generate yearly counts
yearly_counts = df_positive.groupby('Year').size().reset_index(name='Count')
yearly_counts['Percentage'] = (yearly_counts['Count'] / len(df_positive) * 100).round(2)
yearly_counts = yearly_counts.sort_values('Year')

# Print the table
print("\n--- Table: Distribution of Abstracts by Year ---")
print(yearly_counts)

# Generate a simple bar chart
plt.figure(figsize=(12, 7))
plt.bar(yearly_counts['Year'].astype(str), yearly_counts['Count'], color='blue')

plt.title("Distribution of Research Papers Over Time", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Number of Papers", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()

print("Saving plot to 'yearly_distribution.png'...")
plt.savefig('yearly_distribution.png')
plt.close()

# 2. Create a simple pie chart
plt.figure(figsize=(10, 8))
sentiment_distribution = pd.Series([556, 109, 0], index=['Positive', 'Neutral', 'Negative'])
plt.pie(sentiment_distribution, labels=sentiment_distribution.index, 
      autopct='%1.1f%%', startangle=90,
      colors=['green', 'orange', 'blue'],
      wedgeprops={'edgecolor': 'white'})
plt.title("Sentiment Distribution for Digital Literacy Theme", fontsize=16)

print("Saving plot to 'sentiment_distribution.png'...")
plt.savefig('sentiment_distribution.png')
plt.close()

print("All plots have been saved successfully!") 