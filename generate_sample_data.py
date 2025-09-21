import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_cord19_sample_data(n_papers=5000):
    """
    Generate sample data that mimics the CORD-19 dataset structure
    """
    print("Generating sample CORD-19 dataset...")
    
    # Sample journal names (mix of real and fictional)
    journals = [
        'Nature', 'Science', 'Cell', 'The Lancet', 'New England Journal of Medicine',
        'PLOS ONE', 'BMJ', 'Journal of Virology', 'Proceedings of the National Academy of Sciences',
        'Nature Medicine', 'Cell Host & Microbe', 'Journal of Infectious Diseases',
        'Virology', 'Antiviral Research', 'Vaccine', 'Clinical Infectious Diseases',
        'Emerging Infectious Diseases', 'Journal of Medical Virology', 'Epidemiology',
        'Public Health Reports', 'International Journal of Infectious Diseases'
    ]
    
    # Sample authors
    author_names = [
        'Smith, J.', 'Johnson, M.', 'Williams, R.', 'Brown, S.', 'Jones, D.',
        'Garcia, L.', 'Miller, K.', 'Davis, A.', 'Rodriguez, C.', 'Wilson, P.',
        'Martinez, E.', 'Anderson, T.', 'Taylor, N.', 'Thomas, B.', 'Hernandez, F.',
        'Moore, G.', 'Martin, H.', 'Jackson, I.', 'Thompson, O.', 'White, Q.'
    ]
    
    # COVID-related keywords for titles and abstracts
    covid_keywords = [
        'COVID-19', 'SARS-CoV-2', 'coronavirus', 'pandemic', 'vaccine', 'treatment',
        'symptoms', 'transmission', 'prevention', 'diagnosis', 'therapy', 'antiviral',
        'immunity', 'antibody', 'respiratory', 'pneumonia', 'outbreak', 'epidemiology',
        'public health', 'social distancing', 'mask', 'lockdown', 'quarantine'
    ]
    
    # Generate data
    data = []
    start_date = datetime(2019, 12, 1)
    end_date = datetime(2023, 12, 31)
    
    for i in range(n_papers):
        # Generate random publication date
        random_days = random.randint(0, (end_date - start_date).days)
        pub_date = start_date + timedelta(days=random_days)
        
        # Generate title with COVID keywords
        title_keywords = random.sample(covid_keywords, random.randint(2, 4))
        title = f"Study on {' and '.join(title_keywords[:2])}: {' '.join(title_keywords[2:])}"
        
        # Generate abstract with more keywords
        abstract_keywords = random.sample(covid_keywords, random.randint(5, 10))
        abstract = f"This study investigates {abstract_keywords[0]} in relation to {abstract_keywords[1]}. " \
                  f"We analyzed {abstract_keywords[2]} and {abstract_keywords[3]} to understand " \
                  f"the impact of {abstract_keywords[4]} on {abstract_keywords[5]}. " \
                  f"Our findings suggest that {abstract_keywords[6]} plays a crucial role in " \
                  f"{abstract_keywords[7]} prevention and treatment."
        
        # Generate authors (1-5 authors per paper)
        num_authors = random.randint(1, 5)
        authors = '; '.join(random.sample(author_names, num_authors))
        
        # Select journal
        journal = random.choice(journals)
        
        # Generate DOI
        doi = f"10.1000/sample.{i+1:06d}"
        
        # Some papers might have missing data
        if random.random() < 0.05:  # 5% missing titles
            title = None
        if random.random() < 0.1:   # 10% missing abstracts
            abstract = None
        if random.random() < 0.02:  # 2% missing journals
            journal = None
        if random.random() < 0.03:  # 3% missing authors
            authors = None
            
        data.append({
            'cord_uid': f'cord-{i+1:06d}',
            'title': title,
            'abstract': abstract,
            'authors': authors,
            'journal': journal,
            'publish_time': pub_date.strftime('%Y-%m-%d'),
            'doi': doi,
            'url': f'https://example.com/paper/{i+1}',
            'source_x': random.choice(['PMC', 'Elsevier', 'arXiv', 'bioRxiv'])
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('cord19_sample_metadata.csv', index=False)
    print(f"Generated {len(df)} sample papers and saved to 'cord19_sample_metadata.csv'")
    print(f"Date range: {df['publish_time'].min()} to {df['publish_time'].max()}")
    print(f"Unique journals: {df['journal'].nunique()}")
    print(f"Missing values:")
    print(df.isnull().sum())
    
    return df

if __name__ == "__main__":
    df = generate_cord19_sample_data()
    print("\nSample data preview:")
    print(df.head())
