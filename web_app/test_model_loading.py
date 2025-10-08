#!/usr/bin/env python3
"""
Test script to verify model loading works correctly
"""

import sys
import os
sys.path.append('/Users/baonguyen/IU/thesis/web_app')

import torch
import pandas as pd
from streamlit_app import load_model, load_data

def test_model_loading():
    print("🧪 Testing model loading...")
    
    # Load data
    print("📊 Loading data...")
    df_sorted, topic_info = load_data()
    if df_sorted is None:
        print("❌ Failed to load data")
        return False
    
    print(f"✅ Data loaded: {len(df_sorted)} records")
    print(f"   - Unique items: {df_sorted['item_id'].nunique()}")
    print(f"   - Unique rented_for: {df_sorted['rented for'].nunique()}")
    print(f"   - Unique topics: {df_sorted['Topic'].nunique()}")
    
    # Load model
    print("\n🤖 Loading model...")
    model_data = load_model()
    if model_data is None:
        print("❌ Failed to load model")
        return False
    
    print("✅ Model loaded successfully!")
    print(f"   - Model device: {model_data['device']}")
    print(f"   - Item mappings: {len(model_data['item_to_index'])} items")
    print(f"   - Rented_for mappings: {len(model_data['rented_for_to_index'])} occasions")
    print(f"   - Topic mappings: {len(model_data['topic_to_index'])} topics")
    
    # Test a simple prediction
    print("\n🎯 Testing prediction...")
    try:
        # Create a simple test sequence
        test_sequence = {
            'item_ids': [1, 2, 3],  # Some item indices
            'rented_for': [1, 1, 1],  # Some occasion indices
            'Topic': [1, 1, 1]  # Some topic indices
        }
        
        # Pad the sequence
        max_len = 21
        test_sequence['item_ids'] = test_sequence['item_ids'] + [0] * (max_len - len(test_sequence['item_ids']))
        test_sequence['rented_for'] = test_sequence['rented_for'] + [0] * (max_len - len(test_sequence['rented_for']))
        test_sequence['Topic'] = test_sequence['Topic'] + [0] * (max_len - len(test_sequence['Topic']))
        
        # Test model forward pass
        model = model_data['model']
        device = model_data['device']
        
        item_tensor = torch.tensor([test_sequence['item_ids']], dtype=torch.long).to(device)
        side_input_dict = {
            'rented_for': torch.tensor([test_sequence['rented_for']], dtype=torch.long).to(device),
            'Topic': torch.tensor([test_sequence['Topic']], dtype=torch.long).to(device)
        }
        key_padding_mask = (item_tensor == 0)
        
        with torch.no_grad():
            output = model(item_tensor, side_input_dict, key_padding_mask=key_padding_mask)
            print(f"✅ Model forward pass successful! Output shape: {output.shape}")
            
            # Get top predictions
            probabilities = torch.softmax(output[:, -1, :], dim=-1)
            top_items = torch.topk(probabilities, k=5).indices[0].tolist()
            print(f"   Top 5 predicted items: {top_items}")
            
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Model is working correctly.")
    return True

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n✅ Model loading test completed successfully!")
    else:
        print("\n❌ Model loading test failed!")
        sys.exit(1)

