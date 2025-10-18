"""
Data preprocessing utilities for the NovaBert Recommendation System.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any

def load_and_preprocess_data(data_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and preprocess the dataset for the recommendation system.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        Tuple of (processed_dataframe, metadata_dict)
    """
    # Load data
    df = pd.read_csv(data_path)
    df['review_date'] = pd.to_datetime(df['review_date'])
    df_sorted = df.sort_values('review_date')
    df_sorted.rename(columns={
        'rented for': 'rented_for',
        'body type': 'body_type', 
        'bust size': 'bust_size'
    }, inplace=True)
    
    # Create mappings
    unique_item_id = set(df_sorted['item_id'])
    side_feature = ['rented_for', 'Topic']
    
    # Item mappings
    item_to_index = {item: idx + 1 for idx, item in enumerate(unique_item_id)}
    index_to_item = {idx + 1: item for idx, item in enumerate(unique_item_id)}
    
    # Side feature mappings
    mappings = {}
    for feature in side_feature:
        unique_values = set(df_sorted[feature])
        feature_to_index = {val: idx + 1 for idx, val in enumerate(unique_values)}
        index_to_feature = {idx + 1: val for idx, val in enumerate(unique_values)}
        
        mappings[f'{feature}_to_index'] = feature_to_index
        mappings[f'index_to_{feature}'] = index_to_feature
        mappings[f'unique_{feature}'] = unique_values
    
    # Add item mappings
    mappings['item_to_index'] = item_to_index
    mappings['index_to_item'] = index_to_item
    mappings['unique_item_id'] = unique_item_id
    
    return df_sorted, mappings

def create_user_sequences(df: pd.DataFrame, min_sequence_length: int = 2, 
                         max_sequence_length: int = 21) -> Dict[int, Dict[str, List]]:
    """
    Create user-item sequences from the dataset.
    
    Args:
        df: Processed dataframe
        min_sequence_length: Minimum sequence length
        max_sequence_length: Maximum sequence length
        
    Returns:
        Dictionary of user sequences
    """
    side_feature = ['rented_for', 'Topic']
    
    # Group by user and aggregate sequences
    user_item_sequence = (
        df.groupby('user_id')[['item_id'] + side_feature]
        .agg(list)
        .to_dict(orient='index')
    )
    
    # Filter by sequence length
    user_item_sequence = {
        user: val
        for user, val in user_item_sequence.items()
        if min_sequence_length <= len(val['item_id']) <= max_sequence_length
    }
    
    return user_item_sequence

def prepare_sequence_for_model(user_sequence: Dict[str, List], 
                             mappings: Dict[str, Any]) -> Dict[str, List]:
    """
    Prepare a user sequence for model input.
    
    Args:
        user_sequence: User sequence with item_ids, rented_for, Topic
        mappings: Dictionary containing all mappings
        
    Returns:
        Prepared sequence for model input
    """
    side_feature = ['rented_for', 'Topic']
    
    prepared_sequence = {
        'item_id': [mappings['item_to_index'].get(item, 0) for item in user_sequence['item_id']]
    }
    
    for feature in side_feature:
        mapping_key = f'{feature}_to_index'
        prepared_sequence[feature] = [
            mappings[mapping_key].get(val, 0) for val in user_sequence[feature]
        ]
    
    return prepared_sequence

def get_item_details(item_id: int, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get detailed information about an item.
    
    Args:
        item_id: Item ID
        df: Dataframe containing item information
        
    Returns:
        Dictionary with item details
    """
    item_data = df[df['item_id'] == item_id]
    
    if len(item_data) == 0:
        return {"error": "Item not found"}
    
    # Get most common values for categorical features
    details = {
        'item_id': item_id,
        'category': item_data['category'].mode().iloc[0] if len(item_data['category'].mode()) > 0 else 'Unknown',
        'average_rating': item_data['rating'].mean(),
        'review_count': len(item_data),
        'most_common_occasion': item_data['rented_for'].mode().iloc[0] if len(item_data['rented_for'].mode()) > 0 else 'Unknown',
        'recent_reviews': item_data.nlargest(3, 'review_date')[['review_summary', 'rating', 'review_date']].to_dict('records')
    }
    
    return details

def save_mappings(mappings: Dict[str, Any], filepath: str):
    """
    Save mappings to a pickle file.
    
    Args:
        mappings: Dictionary of mappings
        filepath: Path to save the file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(mappings, f)

def load_mappings(filepath: str) -> Dict[str, Any]:
    """
    Load mappings from a pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Dictionary of mappings
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def create_sample_sequences(df: pd.DataFrame, n_samples: int = 5) -> List[Dict[str, Any]]:
    """
    Create sample user sequences for demonstration.
    
    Args:
        df: Processed dataframe
        n_samples: Number of sample sequences to create
        
    Returns:
        List of sample sequences with details
    """
    user_sequences = create_user_sequences(df)
    sample_users = list(user_sequences.keys())[:n_samples]
    
    samples = []
    for user_id in sample_users:
        sequence = user_sequences[user_id]
        sample = {
            'user_id': user_id,
            'sequence_length': len(sequence['item_id']),
            'items': sequence['item_id'][:5],  # First 5 items
            'occasions': sequence['rented_for'][:5],
            'topics': sequence['Topic'][:5]
        }
        samples.append(sample)
    
    return samples

def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the dataset and return statistics.
    
    Args:
        df: Processed dataframe
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_reviews': len(df),
        'unique_users': df['user_id'].nunique(),
        'unique_items': df['item_id'].nunique(),
        'unique_categories': df['category'].nunique(),
        'date_range': {
            'start': df['review_date'].min(),
            'end': df['review_date'].max()
        },
        'rating_distribution': df['rating'].value_counts().to_dict(),
        'category_distribution': df['category'].value_counts().head(10).to_dict(),
        'occasion_distribution': df['rented_for'].value_counts().to_dict(),
        'topic_distribution': df['Topic'].value_counts().to_dict()
    }
    
    return stats

if __name__ == "__main__":
    # Example usage
    data_path = "/Users/baonguyen/IU/thesis/data/clean_data/data_with_bertopic_column.csv"
    
    print("Loading and preprocessing data...")
    df, mappings = load_and_preprocess_data(data_path)
    
    print("Creating user sequences...")
    user_sequences = create_user_sequences(df)
    
    print("Analyzing dataset...")
    stats = analyze_dataset(df)
    
    print(f"Dataset statistics:")
    print(f"- Total reviews: {stats['total_reviews']:,}")
    print(f"- Unique users: {stats['unique_users']:,}")
    print(f"- Unique items: {stats['unique_items']:,}")
    print(f"- User sequences: {len(user_sequences):,}")
    
    print("\nSample sequences:")
    samples = create_sample_sequences(df, 3)
    for i, sample in enumerate(samples):
        print(f"User {sample['user_id']}: {sample['sequence_length']} items")

