import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

def analyze_publications_over_time(df, save_plot=True):
    """
    Analyze publication trends over time
    """
    print("=== PUBLICATIONS OVER TIME ANALYSIS ===")
    
    # Group by year-month
    monthly_counts = df.groupby('year_month').size().reset_index(name='count')
    monthly_counts['date'] = monthly_counts['year_month'].dt.to_timestamp()
    
    # Group by year
    yearly_counts = df.groupby('year').size().reset_index(name='count')
    
    print(f"Total publications: {len(df):,}")
    print(f"Date range: {df['publish_time'].min().strftime('%Y-%m-%d')} to {df['publish_time'].max().strftime('%Y-%m-%d')}")
    print(f"Peak month: {monthly_counts.loc[monthly_counts['count'].idxmax(), 'year_month']} ({monthly_counts['count'].max()} papers)")
    print(f"Peak year: {yearly_counts.loc[yearly_counts['count'].idxmax(), 'year']} ({yearly_counts['count'].max()} papers)")
    
    if save_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Monthly trend
        ax1.plot(monthly_counts['date'], monthly_counts['count'], marker='o', linewidth=2, markersize=4)
        ax1.set_title('COVID-19 Research Publications Over Time (Monthly)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Number of Publications')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Yearly trend
        ax2.bar(yearly_counts['year'], yearly_counts['count'], color='skyblue', edgecolor='navy', alpha=0.7)
        ax2.set_title('COVID-19 Research Publications by Year', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Number of Publications')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('publications_over_time.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return monthly_counts, yearly_counts

def analyze_top_journals(df, top_n=15, save_plot=True):
    """
    Analyze top journals by publication count
    """
    print(f"\n=== TOP {top_n} JOURNALS ANALYSIS ===")
    
    # Count publications by journal
    journal_counts = df['journal'].value_counts().head(top_n)
    
    print(f"Total unique journals: {df['journal'].nunique()}")
    print(f"Top {top_n} journals account for {journal_counts.sum():,} papers ({journal_counts.sum()/len(df)*100:.1f}% of total)")
    
    print(f"\nTop {top_n} journals:")
    for i, (journal, count) in enumerate(journal_counts.items(), 1):
        print(f"{i:2d}. {journal}: {count:,} papers")
    
    if save_plot:
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(journal_counts)), journal_counts.values, color='lightcoral', edgecolor='darkred', alpha=0.8)
        plt.yticks(range(len(journal_counts)), journal_counts.index)
        plt.xlabel('Number of Publications')
        plt.title(f'Top {top_n} Journals by Publication Count', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.01 * max(journal_counts.values), bar.get_y() + bar.get_height()/2, 
                    f'{int(width):,}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('top_journals.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return journal_counts

def analyze_word_frequency(df, text_column='title', top_n=30, save_plot=True):
    """
    Analyze word frequency in titles or abstracts
    """
    print(f"\n=== WORD FREQUENCY ANALYSIS ({text_column.upper()}) ===")
    
    # Extract keywords
    def extract_keywords(text_series, top_n=50):
        if text_series.isnull().all():
            return Counter()
        
        # Combine all text
        all_text = ' '.join(text_series.dropna().astype(str))
        
        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
        
        # Enhanced stop words for academic papers
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 
            'did', 'she', 'use', 'way', 'will', 'with', 'this', 'that', 'have',
            'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time',
            'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many',
            'over', 'such', 'take', 'than', 'them', 'well', 'were', 'what', 'your',
            'study', 'studies', 'analysis', 'research', 'results', 'data', 'method',
            'methods', 'conclusion', 'background', 'objective', 'purpose', 'using',
            'based', 'among', 'between', 'during', 'after', 'before', 'within',
            'associated', 'related', 'significant', 'clinical', 'patients', 'patient'
        }
        
        # Filter out stop words and count
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        word_counts = Counter(filtered_words)
        
        return word_counts.most_common(top_n)
    
    # Get word frequencies
    word_freq = extract_keywords(df[text_column], top_n)
    
    if not word_freq:
        print(f"No words found in {text_column} column")
        return None
    
    print(f"Top {len(word_freq)} words in {text_column}:")
    for i, (word, count) in enumerate(word_freq, 1):
        print(f"{i:2d}. {word}: {count:,}")
    
    if save_plot and word_freq:
        words, counts = zip(*word_freq)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(words)), counts, color='lightgreen', edgecolor='darkgreen', alpha=0.8)
        plt.yticks(range(len(words)), words)
        plt.xlabel('Frequency')
        plt.title(f'Top {len(word_freq)} Words in {text_column.title()}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.01 * max(counts), bar.get_y() + bar.get_height()/2, 
                    f'{int(width):,}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'word_frequency_{text_column}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return word_freq

def analyze_author_statistics(df, save_plot=True):
    """
    Analyze author-related statistics
    """
    print("\n=== AUTHOR STATISTICS ===")
    
    # Author count distribution
    author_dist = df['author_count'].value_counts().sort_index()
    
    print(f"Papers with author information: {(df['author_count'] > 0).sum():,}")
    print(f"Average authors per paper: {df['author_count'].mean():.2f}")
    print(f"Median authors per paper: {df['author_count'].median():.1f}")
    print(f"Max authors on a single paper: {df['author_count'].max()}")
    
    print("\nAuthor count distribution:")
    for count, papers in author_dist.head(10).items():
        print(f"{count} authors: {papers:,} papers")
    
    if save_plot:
        plt.figure(figsize=(10, 6))
        author_dist_plot = author_dist[author_dist.index <= 10]  # Limit to 10 authors for readability
        
        plt.bar(author_dist_plot.index, author_dist_plot.values, color='gold', edgecolor='orange', alpha=0.8)
        plt.xlabel('Number of Authors')
        plt.ylabel('Number of Papers')
        plt.title('Distribution of Author Count per Paper', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for x, y in zip(author_dist_plot.index, author_dist_plot.values):
            plt.text(x, y + 0.01 * max(author_dist_plot.values), f'{y:,}', 
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('author_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return author_dist

def generate_summary_statistics(df):
    """
    Generate comprehensive summary statistics
    """
    print("\n=== COMPREHENSIVE SUMMARY STATISTICS ===")
    
    # Basic statistics
    print("Dataset Overview:")
    print(f"  Total papers: {len(df):,}")
    print(f"  Date range: {df['publish_time'].min().strftime('%Y-%m-%d')} to {df['publish_time'].max().strftime('%Y-%m-%d')}")
    print(f"  Unique journals: {df['journal'].nunique():,}")
    print(f"  Papers with abstracts: {df['abstract'].notna().sum():,} ({df['abstract'].notna().sum()/len(df)*100:.1f}%)")
    
    # Text statistics
    print("\nText Statistics:")
    print(f"  Average title length: {df['title_length'].mean():.1f} characters")
    print(f"  Average abstract length: {df['abstract_length'].mean():.1f} characters")
    print(f"  Average authors per paper: {df['author_count'].mean():.2f}")
    
    # Temporal statistics
    yearly_stats = df.groupby('year').size()
    print(f"\nTemporal Statistics:")
    print(f"  Most productive year: {yearly_stats.idxmax()} ({yearly_stats.max():,} papers)")
    print(f"  Least productive year: {yearly_stats.idxmin()} ({yearly_stats.min():,} papers)")
    
    # Journal statistics
    journal_stats = df['journal'].value_counts()
    print(f"\nJournal Statistics:")
    print(f"  Most prolific journal: {journal_stats.index[0]} ({journal_stats.iloc[0]:,} papers)")
    print(f"  Journals with only 1 paper: {(journal_stats == 1).sum():,}")
    
    return {
        'total_papers': len(df),
        'date_range': (df['publish_time'].min(), df['publish_time'].max()),
        'unique_journals': df['journal'].nunique(),
        'avg_title_length': df['title_length'].mean(),
        'avg_abstract_length': df['abstract_length'].mean(),
        'avg_authors': df['author_count'].mean(),
        'most_productive_year': yearly_stats.idxmax(),
        'most_prolific_journal': journal_stats.index[0]
    }

def run_complete_analysis(data_file='cord19_cleaned_data.csv'):
    """
    Run the complete analysis pipeline
    """
    print("Starting comprehensive CORD-19 dataset analysis...")
    print("=" * 60)
    
    try:
        # Load cleaned data
        df = pd.read_csv(data_file)
        df['publish_time'] = pd.to_datetime(df['publish_time'])
        df['year_month'] = df['publish_time'].dt.to_period('M')
        
        print(f"Loaded {len(df):,} papers for analysis")
        
        # Run all analyses
        monthly_counts, yearly_counts = analyze_publications_over_time(df)
        journal_counts = analyze_top_journals(df)
        title_words = analyze_word_frequency(df, 'title')
        abstract_words = analyze_word_frequency(df, 'abstract')
        author_dist = analyze_author_statistics(df)
        summary_stats = generate_summary_statistics(df)
        
        print("\n" + "=" * 60)
        print("Analysis complete! Generated visualizations:")
        print("  - publications_over_time.png")
        print("  - top_journals.png") 
        print("  - word_frequency_title.png")
        print("  - word_frequency_abstract.png")
        print("  - author_statistics.png")
        
        return {
            'monthly_counts': monthly_counts,
            'yearly_counts': yearly_counts,
            'journal_counts': journal_counts,
            'title_words': title_words,
            'abstract_words': abstract_words,
            'author_dist': author_dist,
            'summary_stats': summary_stats
        }
        
    except FileNotFoundError:
        print(f"Error: Could not find {data_file}")
        print("Please run data_processing.py first to generate cleaned data.")
        return None
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return None

if __name__ == "__main__":
    # Run complete analysis
    results = run_complete_analysis()
