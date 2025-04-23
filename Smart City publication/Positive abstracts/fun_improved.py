import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from datetime import datetime

# Set seaborn style for better visualizations
sns.set_theme(style="whitegrid")

# Create a timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create directory for results if it doesn't exist
results_dir = f"results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# Theme names for better labeling
theme_names = [
    "Digital Literacy as Driver",
    "ICT Tools as Driver",
    "Digital Divide as Barrier",
    "Lack of ICTs as Barrier"
]

try:
    # Load the pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Function to generate embeddings
    def get_embeddings(text_list):
        return model.encode(text_list, convert_to_tensor=True)
    
    # Function to calculate cosine similarity
    def calculate_similarity_with_target(doc_embeddings, target_embedding):
        return cosine_similarity(doc_embeddings, target_embedding.reshape(1, -1))
    
    # Load your positive abstracts DataFrame from current directory
    print("Loading dataset...")
    df_positive = pd.read_csv('positive_abstracts.csv')
    print(f"Total abstracts loaded: {len(df_positive)}")
    
    # Filter to get positive labeled abstracts only
    df_positive = df_positive[df_positive['Target'] == 1]
    print(f"Positive abstracts filtered: {len(df_positive)}")
    
    # Save years range for reference
    years_range = f"{df_positive['Year'].min()}-{df_positive['Year'].max()}"
    print(f"Analyzing data across years: {years_range}")
    
    # Define target texts focused on digital literacy and smart city development
    target_texts = [
        "for Smart city development, digital literacy should be a driving factor where Citizens should possess digital skills in order for smart city development to take off quickly",
        "for Smart city development, ICT should be a driving factor where Citizens, organisations, systems, and players should use ICT tools in order for smart city development to take off quickly",
        "for Smart city stagnation, digital divide should be a driving factor where Citizens without digital skills hinder smart city development to take off quickly",
        "for Smart city stagnation, lack of ICTs should be a limiting factor where limited use of ICTs hinder smart city development to take off quickly"
    ]
    
    # Store results for each target text
    all_results = []
    
    # Create a summary dataframe to store correlation values
    correlation_summary = pd.DataFrame(columns=['Theme', 'Correlation', 'p_value'])
    
    for idx, target_text in enumerate(target_texts):
        theme_name = theme_names[idx]
        print(f"Processing theme {idx+1}: {theme_name}")
        
        # Generate target embedding
        target_embedding = get_embeddings([target_text])[0]
        
        # Generate embeddings for positive abstracts
        positive_abstracts_embeddings = get_embeddings(df_positive['Combined(Title+Abstract)'].astype(str).tolist())
        
        # Calculate similarity scores
        similarity_scores = calculate_similarity_with_target(positive_abstracts_embeddings, target_embedding)
        
        # Add similarity scores to DataFrame
        df_positive[f"similarity_{idx}"] = similarity_scores.flatten()
        
        # Apply threshold for classification (adjust as needed)
        df_positive[target_text] = df_positive[f"similarity_{idx}"].apply(
            lambda x: 1 if x > 0.5 else (0 if 0.2 <= x <= 0.5 else -1)
        )
        
        # Group by year and count classifications
        yearly_counts = df_positive.groupby('Year')[target_text].value_counts().unstack(fill_value=0).reset_index()
        
        # Store results
        # Explicitly check for the existence of the columns and assign 0 if not present
        results = {
            'Year': yearly_counts['Year'].values,
            'Target Theme': [theme_name] * len(yearly_counts),
            'Positive': yearly_counts.get(1, pd.Series(0, index=yearly_counts.index)).values,
            'Neutral': yearly_counts.get(0, pd.Series(0, index=yearly_counts.index)).values,
            'Negative': yearly_counts.get(-1, pd.Series(0, index=yearly_counts.index)).values
        }
        all_results.append(pd.DataFrame(results))
        
        # Individual plots for each target text with enhanced styling
        yearly_counts = df_positive.groupby('Year')[target_text].value_counts().unstack(fill_value=0).reset_index()
        
        plt.figure(figsize=(12, 7))
        plt.plot(yearly_counts['Year'], yearly_counts.get(1, pd.Series(0, index=yearly_counts.index)), 
                label='Positive', marker='o', linewidth=3, color='#2ca02c')
        plt.plot(yearly_counts['Year'], yearly_counts.get(0, pd.Series(0, index=yearly_counts.index)), 
                label='Neutral', marker='s', linewidth=3, color='#ff7f0e')
        plt.plot(yearly_counts['Year'], yearly_counts.get(-1, pd.Series(0, index=yearly_counts.index)), 
                label='Negative', marker='^', linewidth=3, color='#1f77b4')
        
        plt.title(f"Theme {idx+1}: {theme_name}", fontsize=16, fontweight='bold')
        plt.xlabel("Year", fontsize=14)
        plt.ylabel("Number of Abstracts", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # Add data labels for better readability
        for i, year in enumerate(yearly_counts['Year']):
            if year in yearly_counts['Year'].values:
                row = yearly_counts[yearly_counts['Year'] == year].iloc[0]
                for col, color, offset in zip(
                    [1, 0, -1], 
                    ['#2ca02c', '#ff7f0e', '#1f77b4'],
                    [0.3, 0, -0.3]
                ):
                    if col in row and row[col] > 0:
                        plt.annotate(f"{int(row[col])}", 
                                    xy=(year, row[col]), 
                                    xytext=(0, 10),
                                    textcoords='offset points',
                                    ha='center', va='bottom',
                                    fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/theme_{idx+1}_{theme_name.replace(' ', '_')}_trend.png", dpi=300)
        plt.close()
        
        # Correlation plot with enhanced styling
        yearly_counts_corr = df_positive.groupby('Year')[f"similarity_{idx}"].mean().reset_index()
        from scipy import stats
        correlation, p_value = stats.pearsonr(yearly_counts_corr['Year'], yearly_counts_corr[f"similarity_{idx}"])
        
        # Store correlation results
        correlation_summary = pd.concat([
            correlation_summary, 
            pd.DataFrame({'Theme': [theme_name], 'Correlation': [correlation], 'p_value': [p_value]})
        ])
        
        # Create the correlation plot
        plt.figure(figsize=(12, 7))
        sns.regplot(x='Year', y=f"similarity_{idx}", data=yearly_counts_corr, 
                  scatter_kws={'s': 100}, line_kws={'color': 'red', 'linewidth': 2})
        
        plt.title(f"{theme_name} Correlation with Time\nPearson r: {correlation:.3f} (p-value: {p_value:.3f})", 
                fontsize=16, fontweight='bold')
        plt.xlabel("Year", fontsize=14)
        plt.ylabel("Average Relevance Score", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # Add a horizontal line at y=0.5 to show the positive threshold
        plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Positive Threshold')
        # Add a horizontal line at y=0.2 to show the neutral threshold
        plt.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='Neutral Threshold')
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/theme_{idx+1}_{theme_name.replace(' ', '_')}_correlation.png", dpi=300)
        plt.close()
    
    # Combine all results into a single DataFrame
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Save the combined results to CSV for future reference
    combined_results.to_csv(f"{results_dir}/combined_results.csv", index=False)
    
    # Save correlation summary
    correlation_summary.to_csv(f"{results_dir}/correlation_summary.csv", index=False)
    
    # Plotting the results (grouped bar chart) with enhanced styling
    barWidth = 0.2
    years = sorted(combined_results['Year'].unique())
    
    # Group data by year and target theme, summing classifications
    grouped_data = combined_results.groupby(['Year', 'Target Theme'])[['Positive', 'Neutral', 'Negative']].sum().reset_index()
    
    # Create a larger figure for better visualization
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Set background color for better aesthetics
    ax.set_facecolor('#f8f9fa')
    
    # Dictionary to track bar positions for annotation
    bar_positions = {}
    
    # Create bars for each classification and theme with enhanced styling
    for i, theme in enumerate(theme_names):
        theme_data = grouped_data[grouped_data['Target Theme'] == theme]
        
        # Sort the data by year to ensure proper order
        theme_data = theme_data.sort_values('Year')
        
        # Calculate positions
        positions = np.array(range(len(years))) + (i * barWidth)
        bar_positions[theme] = positions
        
        # Map theme data years to position indices
        pos_map = {year: idx for idx, year in enumerate(years)}
        theme_positions = [pos_map[year] + (i * barWidth) for year in theme_data['Year']]
        
        # Plot bars with distinct colors
        positive_bars = ax.bar(theme_positions, theme_data['Positive'], 
                             color=f'C{i}', width=barWidth, 
                             edgecolor='white', linewidth=1,
                             label=f'{theme} - Positive' if i == 0 else "")
        
        # Stack neutral and negative on top
        cumulative = theme_data['Positive'].values
        neutral_bars = ax.bar(theme_positions, theme_data['Neutral'], 
                            bottom=cumulative,
                            color=f'C{i}', width=barWidth, alpha=0.6,
                            edgecolor='white', linewidth=1, 
                            label=f'{theme} - Neutral' if i == 0 else "")
        
        cumulative = cumulative + theme_data['Neutral'].values
        negative_bars = ax.bar(theme_positions, theme_data['Negative'], 
                             bottom=cumulative,
                             color=f'C{i}', width=barWidth, alpha=0.3,
                             edgecolor='white', linewidth=1,
                             label=f'{theme} - Negative' if i == 0 else "")
        
        # Add value labels to the bars
        for j, (p_bar, n_bar, neg_bar) in enumerate(zip(positive_bars, neutral_bars, negative_bars)):
            if p_bar.get_height() > 0:
                ax.text(p_bar.get_x() + p_bar.get_width()/2, p_bar.get_height()/2,
                      int(p_bar.get_height()), ha='center', va='center',
                      fontsize=9, fontweight='bold', color='white')
            
            if n_bar.get_height() > 0:
                ax.text(n_bar.get_x() + n_bar.get_width()/2, p_bar.get_height() + n_bar.get_height()/2,
                      int(n_bar.get_height()), ha='center', va='center',
                      fontsize=9, fontweight='bold', color='white')
            
            if neg_bar.get_height() > 0:
                ax.text(neg_bar.get_x() + neg_bar.get_width()/2, p_bar.get_height() + n_bar.get_height() + neg_bar.get_height()/2,
                      int(neg_bar.get_height()), ha='center', va='center',
                      fontsize=9, fontweight='bold', color='white')
    
    # Add theme labels
    for i, theme in enumerate(theme_names):
        # Place theme labels at the bottom of each group
        ax.text(np.mean(bar_positions[theme]), -1, theme, 
              ha='center', va='top', fontsize=12, fontweight='bold',
              color=f'C{i}', rotation=0)
    
    # Customize the main plot with enhanced styling
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count of Abstracts', fontsize=14, fontweight='bold')
    
    # Set x-ticks at the middle of each year's group
    middle_positions = [np.mean([pos + 1.5*barWidth for pos in range(len(years))]) for year in years]
    ax.set_xticks(middle_positions)
    ax.set_xticklabels(years, fontsize=12, rotation=0)
    
    ax.legend(title='Digital Literacy and Smart City Classifications', fontsize=12, title_fontsize=13)
    ax.set_title(f"Influence of Digital Literacy on Smart City Development\n{years_range}", 
                fontsize=16, fontweight='bold')
    
    # Add a grid for better readability
    ax.grid(axis='y', alpha=0.3)
    
    # Add a text box with summary info
    textstr = '\n'.join((
        'Summary:',
        f'Total abstracts analyzed: {len(df_positive)}',
        f'Year range: {years_range}',
        f'Most positive theme: {correlation_summary.iloc[correlation_summary["Correlation"].argmax()]["Theme"]}',
        f'Strongest correlation: {correlation_summary["Correlation"].max():.3f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
          verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/combined_themes_enhanced.png", dpi=300)
    plt.close()
    
    # Create a separate plot for each theme showing yearly trends
    plt.figure(figsize=(15, 10))
    
    theme_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
    
    for i, theme in enumerate(theme_names):
        theme_data = combined_results[combined_results['Target Theme'] == theme]
        theme_data = theme_data.sort_values('Year')
        
        plt.subplot(2, 2, i+1)
        plt.plot(theme_data['Year'], theme_data['Positive'], marker='o', 
               linewidth=3, color=theme_colors[i], label='Positive')
        plt.plot(theme_data['Year'], theme_data['Neutral'], marker='s', 
               linewidth=2, color=theme_colors[i], linestyle='--', alpha=0.7, label='Neutral')
        plt.plot(theme_data['Year'], theme_data['Negative'], marker='^', 
               linewidth=2, color=theme_colors[i], linestyle=':', alpha=0.5, label='Negative')
        
        plt.title(theme, fontsize=14, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        corr_val = correlation_summary[correlation_summary['Theme'] == theme]['Correlation'].values[0]
        p_val = correlation_summary[correlation_summary['Theme'] == theme]['p_value'].values[0]
        plt.annotate(f"r = {corr_val:.2f}, p = {p_val:.3f}", 
                   xy=(0.05, 0.92), xycoords='axes fraction', 
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.suptitle("Digital Literacy and Smart City Development by Theme", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(f"{results_dir}/themes_comparison.png", dpi=300)
    plt.close()
    
    # Overall trends with enhanced styling
    yearly_trend = df_positive.groupby('Year').size().reset_index(name='count')
    
    plt.figure(figsize=(12, 7))
    sns.lineplot(x='Year', y='count', data=yearly_trend, marker='o', color='#3182bd', linewidth=3)
    
    # Add a trend line
    z = np.polyfit(yearly_trend['Year'], yearly_trend['count'], 1)
    p = np.poly1d(z)
    plt.plot(yearly_trend['Year'], p(yearly_trend['Year']), "r--", linewidth=2, 
           label=f"Trend line (slope: {z[0]:.2f})")
    
    plt.title("Overall Digital Literacy Impact on Smart City Development", 
            fontsize=16, fontweight='bold')
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Number of Publications", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add data labels
    for i, row in yearly_trend.iterrows():
        plt.annotate(f"{int(row['count'])}", 
                   xy=(row['Year'], row['count']),
                   xytext=(0, 10),  # 10 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/overall_trend_enhanced.png", dpi=300)
    plt.close()
    
    # Create a heatmap showing the correlation between themes
    similarity_columns = [f"similarity_{i}" for i in range(len(target_texts))]
    correlation_matrix = df_positive[similarity_columns].corr()
    
    # Rename columns and index for better readability
    correlation_matrix.columns = theme_names
    correlation_matrix.index = theme_names
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
              linewidths=0.5, fmt='.2f')
    plt.title('Correlation Between Digital Literacy Themes', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/theme_correlations_heatmap.png", dpi=300)
    plt.close()
    
    print(f"Analysis complete! All enhanced visualizations have been saved to {results_dir}/")
    
except Exception as e:
    print(f"Error during analysis: {e}")
    import traceback
    traceback.print_exc() 