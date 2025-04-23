import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate embeddings
def get_embeddings(text_list):
    return model.encode(text_list, convert_to_tensor=True)

# Function to calculate cosine similarity
def calculate_similarity_with_target(doc_embeddings, target_embedding):
    return cosine_similarity(doc_embeddings, target_embedding.reshape(1, -1))

# Load your positive abstracts DataFrame from current directory
df_positive = pd.read_csv('positive_abstracts.csv')
df_positive = df_positive[df_positive['Target'] == 1] # Filter to get positive labeled abstracts only

# Define target texts focused on digital literacy and smart city development
target_texts = [
    "for Smart city development, digital literacy should be a driving factor where Citizens should possess digital skills in order for smart city development to take off quickly",
    "for Smart city development, ICT should be a driving factor where Citizens, organisations, systems, and players should use ICT tools in order for smart city development to take off quickly",
    "for Smart city stagnation, digital divide should be a driving factor where Citizens without digital skills hinder smart city development to take off quickly",
    "for Smart city stagnation, lack of ICTs should be a limiting factor where limited use of ICTs hinder smart city development to take off quickly"
]

# Store results for each target text
all_results = []

for target_text in target_texts:
    # Generate target embedding
    target_embedding = get_embeddings([target_text])[0]

    # Generate embeddings for positive abstracts
    positive_abstracts_embeddings = get_embeddings(df_positive['Combined(Title+Abstract)'].astype(str).tolist())

    # Calculate similarity scores
    similarity_scores = calculate_similarity_with_target(positive_abstracts_embeddings, target_embedding)

    # Add similarity scores to DataFrame
    df_positive[target_text] = similarity_scores.flatten()

    # Apply threshold for classification (adjust as needed)
    df_positive[target_text] = df_positive[target_text].apply(lambda x: 1 if x > 0.5 else (0 if 0.2 <= x <= 0.5 else -1))

    # Group by year and count classifications
    yearly_counts = df_positive.groupby('Year')[target_text].value_counts().unstack(fill_value=0).reset_index()

    # Store results
    # Explicitly check for the existence of the columns and assign 0 if not present
    results = {
        'Year': yearly_counts['Year'].values,
        'Target Theme': [target_text] * len(yearly_counts),
        'Positive': yearly_counts.get(1, pd.Series(0, index=yearly_counts.index)).values,
        'Neutral': yearly_counts.get(0, pd.Series(0, index=yearly_counts.index)).values,
        'Negative': yearly_counts.get(-1, pd.Series(0, index=yearly_counts.index)).values
    }
    all_results.append(pd.DataFrame(results))

# Combine all results into a single DataFrame
combined_results = pd.concat(all_results, ignore_index=True)

# Plotting the results (grouped bar chart)
barWidth = 0.25
years = combined_results['Year'].unique()
r1 = range(len(years))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Group data by year and target theme, summing classifications
grouped_data = combined_results.groupby(['Year', 'Target Theme'])[['Positive', 'Neutral', 'Negative']].sum().reset_index()

# Plotting the grouped bar chart
fig, ax = plt.subplots(figsize=(16, 8))

# Create bars for each classification and theme
for i, theme in enumerate(grouped_data['Target Theme'].unique()):
    theme_data = grouped_data[grouped_data['Target Theme'] == theme]
    ax.bar(theme_data['Year'] + (i * barWidth), theme_data['Positive'], color='#2ca02c', width=barWidth, edgecolor='white', label=f'Theme {i+1} - Positive' if i == 0 else "")
    ax.bar(theme_data['Year'] + (i * barWidth), theme_data['Neutral'], color='#ff7f0e', width=barWidth, edgecolor='white', label=f'Theme {i+1} - Neutral' if i == 0 else "")
    ax.bar(theme_data['Year'] + (i * barWidth), theme_data['Negative'], color='#1f77b4', width=barWidth, edgecolor='white', label=f'Theme {i+1} - Negative' if i == 0 else "")

    # Individual plots for each target text
    yearly_counts = df_positive.groupby('Year')[target_text].value_counts().unstack(fill_value=0).reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_counts['Year'], yearly_counts.get(1, pd.Series(0, index=yearly_counts.index)), label='Positive', marker='o')
    plt.plot(yearly_counts['Year'], yearly_counts.get(0, pd.Series(0, index=yearly_counts.index)), label='Neutral', marker='o')
    plt.plot(yearly_counts['Year'], yearly_counts.get(-1, pd.Series(0, index=yearly_counts.index)), label='Negative', marker='o')
    plt.title(f"Theme {i+1}: Digital Literacy and Smart City Development")
    plt.xlabel("Year")
    plt.ylabel("Number of Abstracts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"theme_{i+1}_trend.png")
    plt.close()

    # Correlation plot
    yearly_counts_corr = df_positive.groupby('Year')[target_text].mean().reset_index()
    correlation = yearly_counts_corr['Year'].corr(yearly_counts_corr[target_text])
    plt.figure(figsize=(10, 6))
    plt.scatter(yearly_counts_corr['Year'], yearly_counts_corr[target_text])
    plt.title(f"Theme {i+1} Correlation: {correlation:.2f}")
    plt.xlabel("Year")
    plt.ylabel("Average Relevance Score")
    plt.tight_layout()
    plt.savefig(f"theme_{i+1}_correlation.png")
    plt.close()

# Customize the main plot
ax.set_xlabel('Year', fontweight='bold')
ax.set_ylabel('Count of Abstracts', fontweight='bold')
ax.set_xticks([r + barWidth for r in range(len(years))])
ax.set_xticklabels(years, rotation=45, ha='right')
ax.legend(title='Digital Literacy and Smart City Themes')
ax.set_title("Influence of Digital Literacy on Smart City Development", fontsize=14)

plt.tight_layout()
plt.savefig("combined_themes.png")
plt.close()

# Overall trends
overall_yearly_counts = df_positive.groupby('Year')['Target'].value_counts().unstack(fill_value=0).reset_index()
plt.figure(figsize=(10, 6))
plt.plot(overall_yearly_counts['Year'], overall_yearly_counts.get(1, pd.Series(0, index=overall_yearly_counts.index)), label='Positive', marker='o')
plt.title("Overall Digital Literacy Impact on Smart City Development")
plt.xlabel("Year")
plt.ylabel("Number of Publications")
plt.legend()
plt.tight_layout()
plt.savefig("overall_trend.png")
plt.close()

print("Analysis complete! All visualizations have been saved.")