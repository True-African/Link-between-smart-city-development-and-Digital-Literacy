import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tabulate import tabulate
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set style for better visualizations
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Use current directory for results
results_dir = "."

print("Generating research paper visualizations and structure...")

# Load the dataset
print("Loading dataset...")
df_positive = pd.read_csv('positive_abstracts.csv')
total_abstracts = len(df_positive)
print(f"Total abstracts in dataset: {total_abstracts}")

# Filter to get positive labeled abstracts only
df_positive = df_positive[df_positive['Target'] == 1]
print(f"Positive abstracts after filtering: {len(df_positive)}")

# 1. Generate table of abstracts/papers per year
yearly_counts = df_positive.groupby('Year').size().reset_index(name='Count')
yearly_counts['Percentage'] = (yearly_counts['Count'] / len(df_positive) * 100).round(2)
yearly_counts = yearly_counts.sort_values('Year')

# Print the table to console in a formatted way
print("\n--- Table 1: Distribution of Abstracts by Year ---")
table_data = [["Year", "Count", "Percentage (%)"]] + yearly_counts[['Year', 'Count', 'Percentage']].values.tolist()
print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

# Save the table as CSV
yearly_counts.to_csv(f"{results_dir}/table1_yearly_distribution.csv", index=False)

# 2. Create bar chart showing abstracts per year
plt.figure(figsize=(12, 7))
ax = sns.barplot(x='Year', y='Count', data=yearly_counts, palette='viridis')
ax.bar_label(ax.containers[0], fontsize=10, fontweight='bold')

plt.title("Distribution of Research Papers Over Time", fontsize=16, fontweight='bold')
plt.xlabel("Year", fontsize=14)
plt.ylabel("Number of Papers", fontsize=14)
plt.xticks(rotation=45, ha='right')
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
plt.savefig(f"{results_dir}/fig1_yearly_distribution.png", dpi=300)
plt.close()

# Define target texts for digital literacy and smart city development
target_texts = [
    "for Smart city development, digital literacy should be a driving factor where Citizens should possess digital skills in order for smart city development to take off quickly",
    "for Smart city development, ICT should be a driving factor where Citizens, organisations, systems, and players should use ICT tools in order for smart city development to take off quickly",
    "for Smart city stagnation, digital divide should be a driving factor where Citizens without digital skills hinder smart city development to take off quickly",
    "for Smart city stagnation, lack of ICTs should be a limiting factor where limited use of ICTs hinder smart city development to take off quickly"
]

# Theme names for better labeling
theme_names = [
    "Digital Literacy as Driver",
    "ICT Tools as Driver",
    "Digital Divide as Barrier",
    "Lack of ICTs as Barrier"
]

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate embeddings
def get_embeddings(text_list):
    return model.encode(text_list, convert_to_tensor=True)

# Function to calculate cosine similarity
def calculate_similarity_with_target(doc_embeddings, target_embedding):
    return cosine_similarity(doc_embeddings, target_embedding.reshape(1, -1))

# Process each theme and get sentiment counts
all_sentiments = []

for idx, target_text in enumerate(target_texts):
    print(f"Processing theme {idx+1}: {theme_names[idx]}")
    
    # Generate target embedding
    target_embedding = get_embeddings([target_text])[0]
    
    # Generate embeddings for positive abstracts
    positive_abstracts_embeddings = get_embeddings(df_positive['Combined(Title+Abstract)'].astype(str).tolist())
    
    # Calculate similarity scores
    similarity_scores = calculate_similarity_with_target(positive_abstracts_embeddings, target_embedding)
    
    # Add similarity scores to DataFrame
    df_positive[f"similarity_{idx}"] = similarity_scores.flatten()
    
    # Apply threshold for classification
    df_positive[f"theme_{idx}_sentiment"] = df_positive[f"similarity_{idx}"].apply(
        lambda x: 1 if x > 0.5 else (0 if 0.2 <= x <= 0.5 else -1)
    )
    
    # Get counts for each sentiment category
    sentiment_counts = df_positive[f"theme_{idx}_sentiment"].value_counts().reindex([1, 0, -1], fill_value=0)
    all_sentiments.append({
        'Theme': theme_names[idx],
        'Positive': sentiment_counts.get(1, 0),
        'Neutral': sentiment_counts.get(0, 0),
        'Negative': sentiment_counts.get(-1, 0)
    })
    
# Create DataFrame from sentiment counts
sentiment_df = pd.DataFrame(all_sentiments)
print("\n--- Table 2: Sentiment Distribution by Theme ---")
table_data = [["Theme", "Positive", "Neutral", "Negative"]] + sentiment_df[['Theme', 'Positive', 'Neutral', 'Negative']].values.tolist()
print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

# Save the table as CSV
sentiment_df.to_csv(f"{results_dir}/table2_sentiment_distribution.csv", index=False)

# 3. Create stacked bar chart showing positive, neutral, negative themes
sentiment_df_melted = pd.melt(sentiment_df, 
                            id_vars=['Theme'],
                            value_vars=['Positive', 'Neutral', 'Negative'],
                            var_name='Sentiment', 
                            value_name='Count')

plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Theme', y='Count', hue='Sentiment', data=sentiment_df_melted, 
               palette={'Positive': '#2ca02c', 'Neutral': '#ff7f0e', 'Negative': '#1f77b4'})

# Add data labels to each segment of the bars
for container in ax.containers:
    ax.bar_label(container, fontsize=10, fontweight='bold')

plt.title("Distribution of Sentiments Across Digital Literacy and Smart City Themes", 
        fontsize=16, fontweight='bold')
plt.xlabel("Theme", fontsize=14)
plt.ylabel("Number of Abstracts", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.savefig(f"{results_dir}/fig2_sentiment_distribution.png", dpi=300)
plt.close()

# 4. Create a pie chart showing the overall distribution of sentiments
plt.figure(figsize=(10, 8))
overall_sentiments = sentiment_df[['Positive', 'Neutral', 'Negative']].sum()
plt.pie(overall_sentiments, labels=['Positive', 'Neutral', 'Negative'], 
      autopct='%1.1f%%', startangle=90,
      colors=['#2ca02c', '#ff7f0e', '#1f77b4'],
      wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
plt.title("Overall Distribution of Sentiments in Research Papers", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{results_dir}/fig3_overall_sentiment_pie.png", dpi=300)
plt.close()

# Generate research paper structure with placeholders for interpretations
research_text = f"""
# Digital Literacy and Smart City Development: A Research Study

## Abstract
This research examines the relationship between digital literacy and smart city development through an analysis of {len(df_positive)} research papers published between {yearly_counts['Year'].min()} and {yearly_counts['Year'].max()}. Using natural language processing and sentiment analysis, we investigate how digital literacy impacts smart city initiatives and development.

## 1. Introduction

Smart cities represent the future of urban development, integrating technology and data to improve the quality of life, efficiency of operations, and sustainability. Digital literacy—the ability to use information and communication technologies—is increasingly recognized as a crucial factor in the successful implementation of smart city initiatives.

This research seeks to understand the relationship between digital literacy and smart city development by analyzing research publications in this domain. We aim to answer the following questions:

1. How has research interest in digital literacy and smart cities evolved over time?
2. What are the prevailing sentiments regarding digital literacy as either a driver or barrier to smart city development?
3. How do digital skills compare with ICT infrastructure in their perceived importance for smart city initiatives?

## 2. Methodology

Our analysis utilized a dataset of {len(df_positive)} research papers focusing on smart cities and digital literacy. We employed natural language processing techniques, specifically:

- Semantic similarity analysis using sentence transformers
- Sentiment classification based on similarity thresholds
- Temporal trend analysis of publication patterns

We categorized papers based on their alignment with four key themes:
1. Digital literacy as a driver for smart city development
2. ICT tools as a driver for smart city development
3. Digital divide as a barrier to smart city development
4. Lack of ICTs as a barrier to smart city development

## 3. Results

### 3.1 Temporal Distribution of Research

[FIGURE 1: Yearly distribution of research papers]

Figure 1 shows the distribution of research papers over time. The data reveals a {"growing" if z[0] > 0 else "declining"} trend in publications related to digital literacy and smart cities, with an average increase of {abs(z[0]):.2f} papers per year. The year {yearly_counts.loc[yearly_counts['Count'].idxmax(), 'Year']} saw the highest number of publications ({yearly_counts['Count'].max()}), representing {yearly_counts['Percentage'].max()}% of the total corpus.

This trend suggests that academic interest in the relationship between digital literacy and smart city development has {"gained significant momentum" if z[0] > 0 else "somewhat declined"} over the years, potentially reflecting the growing importance of digital skills in increasingly connected urban environments.

### 3.2 Sentiment Analysis Across Themes

[FIGURE 2: Distribution of sentiments across themes]

Figure 2 illustrates the distribution of sentiments across our four key themes. The analysis reveals:

- **{sentiment_df.iloc[0]['Theme']}**: This theme attracted {sentiment_df.iloc[0]['Positive']} positive mentions, {sentiment_df.iloc[0]['Neutral']} neutral mentions, and {sentiment_df.iloc[0]['Negative']} negative mentions.
  
- **{sentiment_df.iloc[1]['Theme']}**: This theme attracted {sentiment_df.iloc[1]['Positive']} positive mentions, {sentiment_df.iloc[1]['Neutral']} neutral mentions, and {sentiment_df.iloc[1]['Negative']} negative mentions.
  
- **{sentiment_df.iloc[2]['Theme']}**: This theme attracted {sentiment_df.iloc[2]['Positive']} positive mentions, {sentiment_df.iloc[2]['Neutral']} neutral mentions, and {sentiment_df.iloc[2]['Negative']} negative mentions.
  
- **{sentiment_df.iloc[3]['Theme']}**: This theme attracted {sentiment_df.iloc[3]['Positive']} positive mentions, {sentiment_df.iloc[3]['Neutral']} neutral mentions, and {sentiment_df.iloc[3]['Negative']} negative mentions.

The predominant sentiment across all themes was {overall_sentiments.idxmax()}, representing {(overall_sentiments.max() / overall_sentiments.sum() * 100):.1f}% of all classified sentiments.

### 3.3 Overall Sentiment Distribution

[FIGURE 3: Overall distribution of sentiments]

Figure 3 presents the overall distribution of sentiments across all themes. Positive sentiments account for {(overall_sentiments['Positive'] / overall_sentiments.sum() * 100):.1f}% of all classifications, while neutral and negative sentiments represent {(overall_sentiments['Neutral'] / overall_sentiments.sum() * 100):.1f}% and {(overall_sentiments['Negative'] / overall_sentiments.sum() * 100):.1f}% respectively.

This distribution suggests that research literature predominantly views digital literacy as {"an enabling factor" if overall_sentiments.idxmax() == 'Positive' else "a potential challenge"} in smart city development.

## 4. Discussion

Our analysis reveals several important insights about the relationship between digital literacy and smart city development:

1. **Temporal Trends**: The {"increasing" if z[0] > 0 else "decreasing"} number of publications over time indicates that this research area is {"gaining importance" if z[0] > 0 else "receiving less attention"} in academic discourse. This could reflect the {"growing recognition" if z[0] > 0 else "shifting focus away from"} digital literacy as a critical factor in smart city success.

2. **Theme Comparison**: The data shows that {"digital literacy as a driver" if sentiment_df['Positive'].idxmax() == 0 else "ICT tools as a driver" if sentiment_df['Positive'].idxmax() == 1 else "digital divide as a barrier" if sentiment_df['Positive'].idxmax() == 2 else "lack of ICTs as a barrier"} received the most positive sentiment ({sentiment_df['Positive'].max()} mentions), suggesting this aspect is considered particularly important in the literature.

3. **Sentiment Balance**: The overall predominance of {overall_sentiments.idxmax().lower()} sentiments ({(overall_sentiments.max() / overall_sentiments.sum() * 100):.1f}%) indicates that researchers generally view the relationship between digital literacy and smart city development as {"beneficial and enabling" if overall_sentiments.idxmax() == 'Positive' else "complex and context-dependent" if overall_sentiments.idxmax() == 'Neutral' else "potentially problematic and challenging"}.

These findings highlight the {"critical importance" if overall_sentiments.idxmax() == 'Positive' else "mixed perceptions"} of digital literacy in the smart city context. The research suggests that {"enhancing citizens' digital skills" if sentiment_df['Positive'].idxmax() == 0 else "improving ICT infrastructure" if sentiment_df['Positive'].idxmax() == 1 else "addressing the digital divide" if sentiment_df['Positive'].idxmax() == 2 else "addressing the lack of ICT resources"} should be a priority for policymakers and city planners.

## 5. Conclusion

This research provides evidence-based insights into the relationship between digital literacy and smart city development. Our analysis of {len(df_positive)} research papers reveals that digital literacy is predominantly viewed as {"an enabling factor" if overall_sentiments.idxmax() == 'Positive' else "a complex factor" if overall_sentiments.idxmax() == 'Neutral' else "a challenging factor"} in smart city initiatives.

The findings suggest that successful smart city development requires a balanced approach that addresses both {sentiment_df.iloc[sentiment_df['Positive'].idxmax()]['Theme'].lower()} and {sentiment_df.iloc[sentiment_df.iloc[[0, 1, 2, 3]].drop(sentiment_df['Positive'].idxmax())['Positive'].idxmax()]['Theme'].lower()}.

For policymakers and urban planners, these results emphasize the importance of investing in digital literacy programs alongside technological infrastructure to ensure inclusive and effective smart city development. Future research should further explore the specific mechanisms through which digital literacy influences smart city outcomes and evaluate the effectiveness of different intervention strategies.

"""

# Save the research text structure
with open(f"{results_dir}/research_paper_template.md", "w") as f:
    f.write(research_text)

print(f"\nAll visualizations and research paper structure have been saved to the current directory.")
print("You can now use the generated tables and figures in your research paper.")
print("A template for your research paper has been saved as 'research_paper_template.md'") 