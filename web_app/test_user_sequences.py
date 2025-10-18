#!/usr/bin/env python3
"""
Test script to verify user-based sequences work correctly
"""

import sys
import os
sys.path.append('/Users/baonguyen/IU/thesis/web_app')

import pandas as pd
from streamlit_app import load_data, load_model

def test_user_sequences():
    print("🧪 Testing user-based sequences...")
    
    # Load data
    print("📊 Loading data...")
    df_sorted, topic_info = load_data()
    if df_sorted is None:
        print("❌ Failed to load data")
        return False
    
    # Create user sequences (same as in notebook)
    print("👥 Creating user sequences...")
    user_item_sequence = (
        df_sorted.groupby('user_id')[['item_id', 'rented for', 'Topic']]
        .agg(list)
        .to_dict(orient='index')
    )
    
    # Filter users with valid sequences
    user_item_sequence = {
        user: val
        for user, val in user_item_sequence.items()
        if len(val['item_id']) >= 2 and len(val['item_id']) <= 21
    }
    
    print(f"✅ Found {len(user_item_sequence)} users with valid sequences")
    
    # Show some example users
    print("\n📋 Example users:")
    example_users = list(user_item_sequence.keys())[:5]
    for user_id in example_users:
        user_seq = user_item_sequence[user_id]
        print(f"• User {user_id}: {len(user_seq['item_id'])} items")
        print(f"  Occasions: {', '.join(list(set(user_seq['rented for']))[:3])}")
        print(f"  Topics: {', '.join(map(str, list(set(user_seq['Topic']))[:3]))}")
        print(f"  Items: {user_seq['item_id'][:5]}...")
    
    # Load model
    print("\n🤖 Loading model...")
    model_data = load_model()
    if model_data is None:
        print("❌ Failed to load model")
        return False
    
    # Test recommendation for a user
    print("\n🎯 Testing recommendation for a user...")
    test_user = example_users[0]
    user_sequence = user_item_sequence[test_user]
    
    print(f"Testing with User {test_user}:")
    print(f"  Items: {user_sequence['item_id'][:5]}...")
    print(f"  Occasions: {user_sequence['rented for'][:5]}...")
    print(f"  Topics: {user_sequence['Topic'][:5]}...")
    
    # Convert to model format
    user_sequence_model = {
        'item_ids': [model_data['item_to_index'].get(item_id, 0) for item_id in user_sequence['item_id']],
        'rented_for': [model_data['rented_for_to_index'].get(rented_for, 0) for rented_for in user_sequence['rented for']],
        'Topic': [model_data['topic_to_index'].get(topic, 0) for topic in user_sequence['Topic']]
    }
    
    # Pad sequences
    max_len = 21
    user_sequence_model['item_ids'] = user_sequence_model['item_ids'] + [0] * (max_len - len(user_sequence_model['item_ids']))
    user_sequence_model['rented_for'] = user_sequence_model['rented_for'] + [0] * (max_len - len(user_sequence_model['rented_for']))
    user_sequence_model['Topic'] = user_sequence_model['Topic'] + [0] * (max_len - len(user_sequence_model['Topic']))
    
    print(f"  Converted to model format:")
    print(f"    Item indices: {user_sequence_model['item_ids'][:5]}...")
    print(f"    Occasion indices: {user_sequence_model['rented_for'][:5]}...")
    print(f"    Topic indices: {user_sequence_model['Topic'][:5]}...")
    
    # Test model forward pass
    try:
        import torch
        model = model_data['model']
        device = model_data['device']
        
        item_tensor = torch.tensor([user_sequence_model['item_ids']], dtype=torch.long).to(device)
        side_input_dict = {
            'rented_for': torch.tensor([user_sequence_model['rented_for']], dtype=torch.long).to(device),
            'Topic': torch.tensor([user_sequence_model['Topic']], dtype=torch.long).to(device)
        }
        key_padding_mask = (item_tensor == 0)
        
        with torch.no_grad():
            output = model(item_tensor, side_input_dict, key_padding_mask=key_padding_mask)
            print(f"✅ Model forward pass successful! Output shape: {output.shape}")
            
            # Get top predictions
            probabilities = torch.softmax(output[:, -1, :], dim=-1)
            top_items = torch.topk(probabilities, k=5).indices[0].tolist()
            print(f"   Top 5 predicted items: {top_items}")
            
            # Convert back to original item IDs
            predicted_items = [model_data['index_to_item'].get(idx, 0) for idx in top_items if idx in model_data['index_to_item']]
            print(f"   Predicted item IDs: {predicted_items}")
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False
    
    print("\n🎉 User-based sequence test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_user_sequences()
    if success:
        print("\n✅ All tests passed! User-based interface is working correctly.")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
