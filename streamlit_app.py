import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add scripts directory to path for imports
sys.path.append('scripts')

# Import our analysis functions
try:
    from analysis_functions import (
        analyze_publications_over_time,
        analyze_top_journals, 
        analyze_word_frequency,
        analyze_author_statistics,
        generate_summary_statistics
    )
    from data_processing import load_and_explore_data, clean_data
except ImportError:
    st.error("Could not import analysis functions. Please ensure scripts are in the correct directory.")

# Page configuration
st.set_page_config(
    page_title="CORD-19 Dataset Analysis Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar-content {
        background-color: #ffffff;
    }
    .stSelectbox > div > div {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        # Try to load cleaned data first
        if os.path.exists('cord19_cleaned_data.csv'):
            df = pd.read_csv('cord19_cleaned_data.csv')
            df['publish_time'] = pd.to_datetime(df['publish_time'])
            df['year_month'] = df['publish_time'].dt.to_period('M')
            return df
        # If cleaned data doesn't exist, try sample data
        elif os.path.exists('cord19_sample_metadata.csv'):
            df = pd.read_csv('cord19_sample_metadata.csv')
            df = clean_data(df)
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🦠 CORD-19 Dataset Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading dataset..."):
        df = load_data()
    
    if df is None:
        st.error("❌ Could not load dataset. Please ensure the data files are available.")
        st.info("💡 Run the following scripts first:")
        st.code("""
        # Generate sample data
        python scripts/generate_sample_data.py
        
        # Process and clean data  
        python scripts/data_processing.py
        """)
        return
    
    # Sidebar controls
    st.sidebar.title("📊 Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Dataset overview in sidebar
    st.sidebar.subheader("📈 Dataset Overview")
    st.sidebar.metric("Total Papers", f"{len(df):,}")
    st.sidebar.metric("Date Range", f"{df['publish_time'].min().strftime('%Y')} - {df['publish_time'].max().strftime('%Y')}")
    st.sidebar.metric("Unique Journals", f"{df['journal'].nunique():,}")
    
    # Filters
    st.sidebar.subheader("🔍 Filters")
    
    # Date range filter
    min_date = df['publish_time'].min().date()
    max_date = df['publish_time'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Journal filter
    all_journals = ['All Journals'] + sorted(df['journal'].dropna().unique().tolist())
    selected_journal = st.sidebar.selectbox("Select Journal", all_journals)
    
    # Author count filter
    max_authors = int(df['author_count'].max())
    author_range = st.sidebar.slider(
        "Author Count Range", 
        min_value=0, 
        max_value=max_authors, 
        value=(0, max_authors)
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    # Date filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['publish_time'].dt.date >= start_date) & 
            (filtered_df['publish_time'].dt.date <= end_date)
        ]
    
    # Journal filter
    if selected_journal != 'All Journals':
        filtered_df = filtered_df[filtered_df['journal'] == selected_journal]
    
    # Author count filter
    filtered_df = filtered_df[
        (filtered_df['author_count'] >= author_range[0]) & 
        (filtered_df['author_count'] <= author_range[1])
    ]
    
    # Display filtered dataset info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Filtered Results")
    st.sidebar.metric("Filtered Papers", f"{len(filtered_df):,}")
    st.sidebar.metric("Percentage of Total", f"{len(filtered_df)/len(df)*100:.1f}%")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📈 Time Analysis", 
        "📰 Journal Analysis", 
        "🔤 Word Analysis", 
        "👥 Author Analysis"
    ])
    
    with tab1:
        st.header("📊 Dataset Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Publications", 
                f"{len(filtered_df):,}",
                delta=f"{len(filtered_df) - len(df):,}" if len(filtered_df) != len(df) else None
            )
        
        with col2:
            avg_authors = filtered_df['author_count'].mean()
            st.metric("Avg Authors/Paper", f"{avg_authors:.1f}")
        
        with col3:
            if 'title_length' in filtered_df.columns:
                avg_title_len = filtered_df['title_length'].mean()
                st.metric("Avg Title Length", f"{avg_title_len:.0f} chars")
        
        with col4:
            unique_journals = filtered_df['journal'].nunique()
            st.metric("Unique Journals", f"{unique_journals:,}")
        
        # Summary statistics
        st.subheader("📋 Summary Statistics")
        
        if len(filtered_df) > 0:
            summary_stats = generate_summary_statistics(filtered_df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Temporal Information:**")
                st.write(f"• Date Range: {filtered_df['publish_time'].min().strftime('%Y-%m-%d')} to {filtered_df['publish_time'].max().strftime('%Y-%m-%d')}")
                yearly_counts = filtered_df.groupby('year').size()
                if len(yearly_counts) > 0:
                    st.write(f"• Most Productive Year: {yearly_counts.idxmax()} ({yearly_counts.max():,} papers)")
            
            with col2:
                st.write("**Content Information:**")
                st.write(f"• Papers with Abstracts: {filtered_df['abstract'].notna().sum():,} ({filtered_df['abstract'].notna().sum()/len(filtered_df)*100:.1f}%)")
                if 'abstract_length' in filtered_df.columns:
                    st.write(f"• Avg Abstract Length: {filtered_df['abstract_length'].mean():.0f} characters")
        
        # Raw data preview
        st.subheader("🔍 Data Preview")
        st.dataframe(
            filtered_df[['title', 'journal', 'publish_time', 'author_count']].head(10),
            use_container_width=True
        )
    
    with tab2:
        st.header("📈 Time Series Analysis")
        
        if len(filtered_df) > 0:
            # Monthly publications trend
            monthly_counts = filtered_df.groupby('year_month').size().reset_index(name='count')
            monthly_counts['date'] = monthly_counts['year_month'].dt.to_timestamp()
            
            # Yearly publications trend  
            yearly_counts = filtered_df.groupby('year').size().reset_index(name='count')
            
            # Interactive monthly trend with Plotly
            fig_monthly = px.line(
                monthly_counts, 
                x='date', 
                y='count',
                title='Publications Over Time (Monthly)',
                labels={'count': 'Number of Publications', 'date': 'Date'},
                markers=True
            )
            fig_monthly.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Publications",
                hovermode='x unified'
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
            
            # Interactive yearly bar chart
            fig_yearly = px.bar(
                yearly_counts,
                x='year',
                y='count', 
                title='Publications by Year',
                labels={'count': 'Number of Publications', 'year': 'Year'},
                color='count',
                color_continuous_scale='Blues'
            )
            fig_yearly.update_layout(
                xaxis_title="Year",
                yaxis_title="Number of Publications",
                showlegend=False
            )
            st.plotly_chart(fig_yearly, use_container_width=True)
            
            # Time series insights
            col1, col2 = st.columns(2)
            with col1:
                peak_month = monthly_counts.loc[monthly_counts['count'].idxmax()]
                st.metric("Peak Month", f"{peak_month['year_month']}", f"{peak_month['count']} papers")
            
            with col2:
                peak_year = yearly_counts.loc[yearly_counts['count'].idxmax()]
                st.metric("Peak Year", f"{peak_year['year']}", f"{peak_year['count']} papers")
        else:
            st.warning("No data available for the selected filters.")
    
    with tab3:
        st.header("📰 Journal Analysis")
        
        if len(filtered_df) > 0:
            # Top journals analysis
            top_n = st.slider("Number of top journals to display", min_value=5, max_value=25, value=15)
            journal_counts = filtered_df['journal'].value_counts().head(top_n)
            
            # Interactive horizontal bar chart
            fig_journals = px.bar(
                x=journal_counts.values,
                y=journal_counts.index,
                orientation='h',
                title=f'Top {top_n} Journals by Publication Count',
                labels={'x': 'Number of Publications', 'y': 'Journal'},
                color=journal_counts.values,
                color_continuous_scale='Reds'
            )
            fig_journals.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False,
                height=max(400, top_n * 25)
            )
            st.plotly_chart(fig_journals, use_container_width=True)
            
            # Journal statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Unique Journals", f"{filtered_df['journal'].nunique():,}")
            with col2:
                top_journal_papers = journal_counts.iloc[0] if len(journal_counts) > 0 else 0
                st.metric("Top Journal Papers", f"{top_journal_papers:,}")
            with col3:
                single_paper_journals = (filtered_df['journal'].value_counts() == 1).sum()
                st.metric("Single-Paper Journals", f"{single_paper_journals:,}")
            
            # Journal distribution pie chart
            st.subheader("Journal Distribution")
            
            # Group smaller journals into "Others"
            top_journals = journal_counts.head(8)
            others_count = journal_counts.iloc[8:].sum() if len(journal_counts) > 8 else 0
            
            if others_count > 0:
                pie_data = pd.concat([top_journals, pd.Series({'Others': others_count})])
            else:
                pie_data = top_journals
            
            fig_pie = px.pie(
                values=pie_data.values,
                names=pie_data.index,
                title="Distribution of Publications by Journal"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")
    
    with tab4:
        st.header("🔤 Word Frequency Analysis")
        
        if len(filtered_df) > 0:
            # Text column selection
            text_options = []
            if 'title' in filtered_df.columns and filtered_df['title'].notna().any():
                text_options.append('title')
            if 'abstract' in filtered_df.columns and filtered_df['abstract'].notna().any():
                text_options.append('abstract')
            
            if text_options:
                selected_text = st.selectbox("Select text field to analyze", text_options)
                top_words_n = st.slider("Number of top words to display", min_value=10, max_value=50, value=20)
                
                # Extract word frequencies
                from collections import Counter
                import re
                
                def extract_keywords(text_series, top_n=50):
                    if text_series.isnull().all():
                        return []
                    
                    all_text = ' '.join(text_series.dropna().astype(str))
                    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
                    
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
                        'based', 'among', 'between', 'during', 'after', 'before', 'within'
                    }
                    
                    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
                    word_counts = Counter(filtered_words)
                    return word_counts.most_common(top_n)
                
                word_freq = extract_keywords(filtered_df[selected_text], top_words_n)
                
                if word_freq:
                    words, counts = zip(*word_freq)
                    
                    # Interactive word frequency bar chart
                    fig_words = px.bar(
                        x=list(counts),
                        y=list(words),
                        orientation='h',
                        title=f'Top {len(word_freq)} Words in {selected_text.title()}',
                        labels={'x': 'Frequency', 'y': 'Words'},
                        color=list(counts),
                        color_continuous_scale='Greens'
                    )
                    fig_words.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        showlegend=False,
                        height=max(400, len(word_freq) * 20)
                    )
                    st.plotly_chart(fig_words, use_container_width=True)
                    
                    # Word cloud alternative - top words table
                    st.subheader(f"Top Words in {selected_text.title()}")
                    word_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
                    word_df['Rank'] = range(1, len(word_df) + 1)
                    st.dataframe(
                        word_df[['Rank', 'Word', 'Frequency']].set_index('Rank'),
                        use_container_width=True
                    )
                else:
                    st.warning(f"No words found in {selected_text} field.")
            else:
                st.warning("No text fields available for analysis.")
        else:
            st.warning("No data available for the selected filters.")
    
    with tab5:
        st.header("👥 Author Analysis")
        
        if len(filtered_df) > 0:
            # Author count distribution
            author_dist = filtered_df['author_count'].value_counts().sort_index()
            
            # Limit to reasonable range for visualization
            max_authors_display = st.slider("Maximum authors to display", min_value=5, max_value=20, value=10)
            author_dist_display = author_dist[author_dist.index <= max_authors_display]
            
            # Interactive author distribution bar chart
            fig_authors = px.bar(
                x=author_dist_display.index,
                y=author_dist_display.values,
                title='Distribution of Author Count per Paper',
                labels={'x': 'Number of Authors', 'y': 'Number of Papers'},
                color=author_dist_display.values,
                color_continuous_scale='Oranges'
            )
            fig_authors.update_layout(showlegend=False)
            st.plotly_chart(fig_authors, use_container_width=True)
            
            # Author statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                papers_with_authors = (filtered_df['author_count'] > 0).sum()
                st.metric("Papers with Authors", f"{papers_with_authors:,}")
            
            with col2:
                avg_authors = filtered_df['author_count'].mean()
                st.metric("Avg Authors/Paper", f"{avg_authors:.2f}")
            
            with col3:
                median_authors = filtered_df['author_count'].median()
                st.metric("Median Authors/Paper", f"{median_authors:.0f}")
            
            with col4:
                max_authors = filtered_df['author_count'].max()
                st.metric("Max Authors", f"{max_authors}")
            
            # Author collaboration trends over time
            st.subheader("Author Collaboration Trends")
            
            # Group by year and calculate average author count
            yearly_author_trends = filtered_df.groupby('year')['author_count'].agg(['mean', 'median', 'count']).reset_index()
            yearly_author_trends = yearly_author_trends[yearly_author_trends['count'] >= 10]  # Filter years with few papers
            
            if len(yearly_author_trends) > 1:
                fig_trends = go.Figure()
                
                fig_trends.add_trace(go.Scatter(
                    x=yearly_author_trends['year'],
                    y=yearly_author_trends['mean'],
                    mode='lines+markers',
                    name='Average Authors',
                    line=dict(color='blue', width=3)
                ))
                
                fig_trends.add_trace(go.Scatter(
                    x=yearly_author_trends['year'],
                    y=yearly_author_trends['median'],
                    mode='lines+markers',
                    name='Median Authors',
                    line=dict(color='red', width=3, dash='dash')
                ))
                
                fig_trends.update_layout(
                    title='Author Collaboration Trends Over Time',
                    xaxis_title='Year',
                    yaxis_title='Number of Authors',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_trends, use_container_width=True)
            else:
                st.info("Not enough data points to show collaboration trends over time.")
            
            # Detailed author count breakdown
            st.subheader("Author Count Breakdown")
            author_breakdown = pd.DataFrame({
                'Author Count': author_dist.index,
                'Number of Papers': author_dist.values,
                'Percentage': (author_dist.values / author_dist.sum() * 100).round(2)
            })
            st.dataframe(author_breakdown, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")

# Store filtered_df in session state for other tabs
st.session_state.filtered_df = filtered_df

if __name__ == "__main__":
    main()
