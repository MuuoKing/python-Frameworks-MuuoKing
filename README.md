# CORD-19 Dataset Analysis Dashboard

A comprehensive data analysis and visualization dashboard for COVID-19 research papers using the CORD-19 dataset.

## Features

### 📊 Interactive Dashboard
- **Streamlit Web Application**: User-friendly interface with multiple analysis tabs
- **Real-time Filtering**: Filter by date range, journal, and author count
- **Responsive Design**: Works on desktop and mobile devices

### 📈 Analysis Capabilities
- **Time Series Analysis**: Publication trends over time (monthly and yearly)
- **Journal Analysis**: Top journals, publication distribution, and statistics
- **Word Frequency Analysis**: Most common words in titles and abstracts
- **Author Analysis**: Collaboration patterns and author count distributions
- **Statistical Summaries**: Comprehensive dataset statistics

### 🎨 Interactive Visualizations
- **Plotly Charts**: Interactive line charts, bar charts, and pie charts
- **Dynamic Filtering**: All visualizations update based on selected filters
- **Hover Information**: Detailed information on hover
- **Customizable Views**: Adjustable parameters for different perspectives

## Installation & Setup

### 1. Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 2. Generate Sample Data
\`\`\`bash
python scripts/generate_sample_data.py
\`\`\`

### 3. Process and Clean Data
\`\`\`bash
python scripts/data_processing.py
\`\`\`

### 4. Run Analysis (Optional)
\`\`\`bash
python scripts/analysis_functions.py
\`\`\`

### 5. Launch Dashboard
\`\`\`bash
streamlit run streamlit_app.py
\`\`\`

## Project Structure

\`\`\`
├── streamlit_app.py              # Main Streamlit dashboard
├── requirements.txt              # Python dependencies
├── README.md                    # Project documentation
├── scripts/
│   ├── generate_sample_data.py  # Generate sample CORD-19 data
│   ├── data_processing.py       # Data loading and cleaning
│   └── analysis_functions.py    # Analysis and visualization functions
└── Generated Files/
    ├── cord19_sample_metadata.csv    # Sample dataset
    ├── cord19_cleaned_data.csv       # Cleaned dataset
    └── *.png                         # Static visualization outputs
\`\`\`

## Usage Guide

### Dashboard Navigation
1. **Overview Tab**: Dataset summary and key metrics
2. **Time Analysis Tab**: Publication trends over time
3. **Journal Analysis Tab**: Journal-specific insights
4. **Word Analysis Tab**: Text analysis and keyword extraction
5. **Author Analysis Tab**: Collaboration patterns and statistics

### Interactive Features
- **Sidebar Filters**: Adjust date range, journal selection, and author count
- **Dynamic Charts**: All visualizations update based on filters
- **Customizable Parameters**: Adjust number of items displayed in charts
- **Data Export**: View and analyze filtered datasets

### Key Insights Available
- Publication volume trends during COVID-19 pandemic
- Most prolific journals in COVID-19 research
- Common research themes and keywords
- Collaboration patterns among researchers
- Temporal analysis of research focus areas

## Technical Implementation

### Data Processing Pipeline
1. **Data Generation**: Creates realistic sample data mimicking CORD-19 structure
2. **Data Cleaning**: Handles missing values, date formatting, and feature engineering
3. **Analysis Functions**: Modular functions for different types of analysis
4. **Interactive Dashboard**: Streamlit-based web application

### Key Technologies
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Matplotlib/Seaborn**: Static visualizations
- **NumPy**: Numerical computations

### Performance Optimizations
- **Data Caching**: Streamlit caching for faster load times
- **Efficient Filtering**: Optimized data filtering operations
- **Modular Design**: Separate analysis functions for maintainability

## Customization

### Adding New Analysis
1. Create analysis function in `scripts/analysis_functions.py`
2. Add new tab in `streamlit_app.py`
3. Implement visualization using Plotly or Matplotlib

### Modifying Visualizations
- Edit chart parameters in respective tab sections
- Customize colors, layouts, and interactive features
- Add new chart types using Plotly Express

### Data Source Integration
- Replace sample data generation with real CORD-19 data loading
- Modify data processing functions for different data formats
- Update column mappings as needed

## Assignment Requirements Fulfilled

✅ **Data Loading & Exploration**: Comprehensive dataset loading with structure analysis  
✅ **Data Cleaning**: Missing value handling, date formatting, feature engineering  
✅ **Statistical Analysis**: Descriptive statistics, grouping, and trend analysis  
✅ **Multiple Visualizations**: 4+ different chart types with customization  
✅ **Streamlit Dashboard**: Interactive web application with widgets  
✅ **Documentation**: Detailed code comments and user guide  
✅ **Error Handling**: Robust error handling throughout the application  

## Future Enhancements

- **Real CORD-19 Integration**: Connect to actual CORD-19 dataset
- **Advanced NLP**: Implement topic modeling and sentiment analysis
- **Machine Learning**: Add predictive models for research trends
- **Export Features**: PDF reports and data export functionality
- **User Authentication**: Multi-user support with saved preferences
