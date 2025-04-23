import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the dataset
print("Loading dataset...")
df_positive = pd.read_csv('positive_abstracts.csv')
print(f"Total abstracts in dataset: {len(df_positive)}")

# Filter to get positive labeled abstracts only
df_positive = df_positive[df_positive['Target'] == 1]
print(f"Positive abstracts after filtering: {len(df_positive)}")

# Generate yearly counts
yearly_counts = df_positive.groupby('Year').size().reset_index(name='Count')
yearly_counts['Percentage'] = (yearly_counts['Count'] / len(df_positive) * 100).round(2)
yearly_counts = yearly_counts.sort_values('Year')

# Create a clean table for the paper
table_data = yearly_counts[['Year', 'Count', 'Percentage']]

# Save the table with full path
current_dir = os.getcwd()
csv_path = os.path.join(current_dir, 'yearly_table.csv')
print(f"Saving table to: {csv_path}")
table_data.to_csv(csv_path, index=False)

# Print the table directly to console for verification
print("\nTable: Number of Positive Papers Published Each Year (2000-2024)")
print("=" * 60)
print(f"{'Year':<10}{'Count':<10}{'Percentage (%)':<15}")
print("-" * 60)

# Print all rows to make sure we see the complete table
for _, row in table_data.iterrows():
    print(f"{int(row['Year']):<10}{int(row['Count']):<10}{row['Percentage']:<15.2f}")
print("=" * 60)
print(f"Total papers: {table_data['Count'].sum()}")

# Generate the bar chart with the updated title
plt.figure(figsize=(12, 7))
bars = plt.bar(yearly_counts['Year'].astype(str), yearly_counts['Count'], color='#3182bd')

# Add data labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold')

plt.title("Distribution of Research Papers Over Time (for positive abstracts)", 
         fontsize=16, fontweight='bold')
plt.xlabel("Year", fontsize=14)
plt.ylabel("Number of Papers", fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# Add a trend line
years = yearly_counts['Year'].values
counts = yearly_counts['Count'].values
z = np.polyfit(years, counts, 1)
p = np.poly1d(z)
plt.plot(years, p(years), "r--", linewidth=2, 
       label=f"Trend (slope: {z[0]:.2f} papers/year)")

plt.legend()
plt.tight_layout()

# Save the plot with the new title
png_path = os.path.join(current_dir, 'yearly_distribution.png')
print(f"Saving plot to: {png_path}")
plt.savefig(png_path, dpi=300)
plt.close()

print("\nYearly distribution table and plot have been created successfully.") 