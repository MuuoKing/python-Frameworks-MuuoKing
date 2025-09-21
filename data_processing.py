import pandas as pd
import numpy as np
from datetime import datetime
import re
from collections import Counter

def load_and_explore_data(file_path='cord19_sample_metadata.csv'):
    """
    Load the CORD-19 dataset and perform initial exploration
    """
    print("Loading CORD-19 dataset...")
    
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        
        # Basic information
        print("\n=== DATASET OVERVIEW ===")
        print(f"Number of papers: {len(df):,}")
        print(f"Number of columns: {len(df.columns)}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Column information
        print("\n=== COLUMN INFORMATION ===")
        print(df.info())
        
        # Missing values
        print("\n=== MISSING VALUES ===")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Percentage': missing_percent
        }).sort_values('Missing Count', ascending=False)
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # Sample data
        print("\n=== SAMPLE DATA ===")
        print(df.head())
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        print("Please run generate_sample_data.py first to create sample data.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def clean_data(df):
    """
    Clean and preprocess the dataset
    """
    print("\n=== DATA CLEANING ===")
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Convert publish_time to datetime
    print("Converting publish_time to datetime...")
    df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
    
    # Extract year and month for analysis
    df_clean['year'] = df_clean['publish_time'].dt.year
    df_clean['month'] = df_clean['publish_time'].dt.month
    df_clean['year_month'] = df_clean['publish_time'].dt.to_period('M')
    
    # Clean journal names (remove extra whitespace, standardize)
    df_clean['journal'] = df_clean['journal'].str.strip()
    
    # Clean titles and abstracts
    df_clean['title'] = df_clean['title'].str.strip()
    df_clean['abstract'] = df_clean['abstract'].str.strip()
    
    # Create text length features
    df_clean['title_length'] = df_clean['title'].str.len()
    df_clean['abstract_length'] = df_clean['abstract'].str.len()
    
    # Count number of authors
    df_clean['author_count'] = df_clean['authors'].str.count(';') + 1
    df_clean.loc[df_clean['authors'].isnull(), 'author_count'] = 0
    
    # Filter out papers with invalid dates
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['publish_time'])
    print(f"Removed {initial_count - len(df_clean)} papers with invalid dates")
    
    # Filter reasonable date range (2019-2024)
    df_clean = df_clean[
        (df_clean['year'] >= 2019) & 
        (df_clean['year'] <= 2024)
    ]
    
    print(f"Final dataset shape after cleaning: {df_clean.shape}")
    print(f"Date range: {df_clean['publish_time'].min()} to {df_clean['publish_time'].max()}")
    
    return df_clean

def extract_keywords(text_series, top_n=50):
    """
    Extract most common keywords from text series
    """
    if text_series.isnull().all():
        return Counter()
    
    # Combine all text
    all_text = ' '.join(text_series.dropna().astype(str))
    
    # Convert to lowercase and extract words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
    
    # Common stop words to exclude
    stop_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
        'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
        'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 
        'did', 'she', 'use', 'way', 'will', 'with', 'this', 'that', 'have',
        'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time',
        'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many',
        'over', 'such', 'take', 'than', 'them', 'well', 'were', 'what', 'your',
        'study', 'studies', 'analysis', 'research', 'results', 'data', 'method',
        'methods', 'conclusion', 'background', 'objective', 'purpose', 'using'
    }
    
    # Filter out stop words and count
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    word_counts = Counter(filtered_words)
    
    return word_counts.most_common(top_n)

if __name__ == "__main__":
    # Load and explore data
    df = load_and_explore_data()
    
    if df is not None:
        # Clean data
        df_clean = clean_data(df)
        
        # Extract keywords from titles
        print("\n=== TOP KEYWORDS IN TITLES ===")
        title_keywords = extract_keywords(df_clean['title'], top_n=20)
        for word, count in title_keywords:
            print(f"{word}: {count}")
        
        # Save cleaned data
        df_clean.to_csv('cord19_cleaned_data.csv', index=False)
        print(f"\nCleaned data saved to 'cord19_cleaned_data.csv'")
