# Smart City Research Analysis

## Project Overview

This project analyzes research abstracts related to smart city development, focusing on the role of digital literacy and ICT. It utilizes natural language processing techniques, specifically Sentence-BERT embeddings and cosine similarity, to classify abstracts based on their relevance to the target theme.

## Step-by-Step Guidelines

1. **Data Preparation**
    - Load the research abstracts dataset from the CSV file (`Datasheet.csv`).
    - Clean the data by removing unnecessary columns, handling missing values, and combining the title and abstract into a single text field.
    - Save the cleaned data to a new CSV file (`df_cleaned1.csv`).

2. **Theme Matching using Sentence-BERT**
    - Load the pre-trained Sentence-BERT model ('all-MiniLM-L6-v2').
    - Generate sentence embeddings for the combined title and abstract of each research paper.
    - Define the target theme related to digital literacy and ICT in smart cities.
    - Generate sentence embedding for the target theme.
    - Calculate the cosine similarity between each research paper's embedding and the target theme embedding.
    - Classify research papers based on similarity scores using predefined thresholds (Positive, Neutral, Negative).

3. **Visualization and Analysis**
    - Visualize the distribution of relevance classifications using a bar chart.
    - Analyze the trend of relevance over time by plotting the number of positive abstracts per year.
    - Calculate the correlation between relevance and year of publication.
    - Visualize the correlation using a scatter plot.
    - Further Analysis : Analyze the impact of specific themes within the positive abstracts and trend and correlations for this target themes.

4. **Conclusion**
    - Summarize the findings and insights derived from the analysis.
    - Discuss the overall trend and implications for smart city development.

## Code Implementation

The project is implemented using Python and the following libraries:

- pandas
- sentence_transformers
- scikit-learn
- matplotlib
- plotly

The code is available in the accompanying Jupyter Notebook (`your_notebook_name.ipynb`).

## Further Exploration

- Explore the content of the classified abstracts to gain deeper insights into specific themes and discussions.
- Adjust the similarity thresholds and target theme to refine the classification and analysis.
- Investigate other NLP techniques and models for theme matching and analysis.
- Extend the analysis to include other aspects of smart city development, such as sustainability and citizen engagement.

## Contributions

This project was developed by [Your Name].

## License

Contact: iyunva@gmail.com
