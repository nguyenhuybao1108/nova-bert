"""
FastAPI-based REST API for the NovaBert Recommendation System.
Provides programmatic access to recommendation functionality.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from data_utils import load_and_preprocess_data, prepare_sequence_for_model, get_item_details

# Import model classes (same as in streamlit_app.py)
class GatingFusor(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(h, 1))

    def forward(self, features):   
        gates = torch.sigmoid(features @ self.weight)         
        fused = torch.sum(gates * features, dim=2)            
        return fused

class NovabertEmbedding(nn.Module):
    def __init__(self,num_item,num_side_feature_ids:dict,embedding_dim,max_len=64):
        super(NovabertEmbedding,self).__init__()
        self.item_embedding = nn.Embedding(num_item,embedding_dim)
        self.side_embedding_  = nn.ModuleDict()
        for feat_name,num_feat in num_side_feature_ids.items():
            self.side_embedding_[feat_name]=nn.Embedding(num_feat,embedding_dim)
        self.position_encoding = nn.Embedding(max_len, embedding_dim)
    def forward(self, item_ids, side_feature_ids: dict):
        position_ids = torch.arange(item_ids.size(1), dtype=torch.long, device=item_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_ids)
        item_embed = self.item_embedding(item_ids)
        pos_embed = self.position_encoding(position_ids)
        item_embed = item_embed + pos_embed
        side_emb_list = []

        for feat_name in self.side_embedding_:
            feat_ids = side_feature_ids[feat_name]
            side_embed = self.side_embedding_[feat_name](feat_ids) + pos_embed
            side_emb_list.append(side_embed)
            
        return item_embed, side_emb_list

class NovabertCrossAttention(nn.Module):
    def __init__(self,embedding_dim,num_heads=8):
        super(NovabertCrossAttention,self).__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim //num_heads
        self.value_proj = nn.Linear(embedding_dim,embedding_dim)
        self.query_proj = nn.Linear(embedding_dim,embedding_dim)
        self.key_proj = nn.Linear(embedding_dim,embedding_dim)
        self.fusor = GatingFusor(h=embedding_dim)
        self.output_proj = nn.Sequential()
        
    def forward(self,item_embed,side_feature_embed:list,attn_mask=None,key_padding_mask=None):
        batch_size , sequence_len , embedding_dim = item_embed.size()
        def reshape(x:torch.tensor):
            return x.view(batch_size,sequence_len,self.num_heads,self.head_dim).transpose(1,2)
        
        features = torch.stack([item_embed] + side_feature_embed, dim=2)
        fused_features = self.fusor(features)
        V = self.value_proj(item_embed)
        Q = self.query_proj(fused_features)
        K = self.key_proj(fused_features)
        Q = reshape(Q)
        K = reshape(K)
        V = reshape(V)
        scores = torch.matmul(Q,K.transpose(-2,-1)) / np.sqrt(self.head_dim)
        
        if attn_mask is not None:
            scores += attn_mask.unsqueeze(0)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding_mask,float('-inf'))

        attn_weights = torch.softmax(scores,dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_output = torch.matmul(attn_weights,V) 

        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size,sequence_len,embedding_dim)
        return self.output_proj(attn_output)

class NovabertLayer(nn.Module):
    def __init__(self,embedding_dim,num_heads):
        super(NovabertLayer,self).__init__()
        self.cross_attn = NovabertCrossAttention(embedding_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, id_embed,side_feature_embed:list , attention_mask=None,key_padding_mask = None):
        guided = self.cross_attn(id_embed,side_feature_embed,attention_mask,key_padding_mask)
        x = self.norm1(id_embed + self.dropout(guided))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class NovabertModel(nn.Module):
    def __init__(self, num_items,num_side_feature_ids:dict, embedding_dim, max_len=64, num_layers=4, num_heads=8):
        super(NovabertModel, self).__init__()
        self.embedding = NovabertEmbedding(num_item=num_items,
                                           num_side_feature_ids=num_side_feature_ids,
                                           embedding_dim=embedding_dim,
                                           max_len=max_len)
        self.nova_layer = nn.ModuleList([
            NovabertLayer(embedding_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, num_items)
        )

    def forward(self, item_ids, side_feature_ids: dict, attention_mask=None, key_padding_mask=None):
        x, y = self.embedding(item_ids, side_feature_ids)
        for layer in self.nova_layer:
            x = layer(x, y, attention_mask, key_padding_mask)
        return self.output_layer(x)

# Pydantic models for API
class RecommendationRequest(BaseModel):
    item_ids: List[int]
    rented_for: List[str]
    topics: List[int]
    k: Optional[int] = 10

class RecommendationResponse(BaseModel):
    recommendations: List[int]
    item_details: List[Dict[str, Any]]
    success: bool
    message: Optional[str] = None

class ItemDetailsRequest(BaseModel):
    item_id: int

class ItemDetailsResponse(BaseModel):
    item_id: int
    category: str
    average_rating: float
    review_count: int
    most_common_occasion: str
    recent_reviews: List[Dict[str, Any]]

# Global variables
model = None
mappings = None
df = None

def load_model_and_data():
    """Load model and data on startup"""
    global model, mappings, df
    
    try:
        # Load data
        df, mappings = load_and_preprocess_data('/Users/baonguyen/IU/thesis/data/clean_data/data_with_bertopic_column.csv')
        
        # Load model
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        num_side_feature_ids = {
            'rented_for': len(mappings['unique_rented_for']), 
            'Topic': len(mappings['unique_Topic'])
        }
        
        model = NovabertModel(
            num_items=len(mappings['unique_item_id'])+1,
            num_side_feature_ids=num_side_feature_ids,
            embedding_dim=256,
            max_len=21,
            num_layers=2,
            num_heads=4
        ).to(device)
        
        # Load model weights
        model_path = '/Users/baonguyen/IU/thesis/src/models/models_item_with_novabert_gatingfusor_/fold_1/best_model.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load model and data: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="NovaBert Recommendation API",
    description="REST API for the NovaBert sequential recommendation system",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Load model and data on startup"""
    load_model_and_data()

def padding(mask_seq, labels, max_len=21, pad_item=0, pad_label=-100):
    """Pad sequences to max_len"""
    def pad(seq, max_len, pad_value):
        if len(seq) < max_len:
            return seq + [pad_value] * (max_len - len(seq))
        else:
            return seq[len(seq)-max_len:len(seq)]
    
    padded_mask_seq = {}
    padded_labels = {}
    side_feature = ['rented_for', 'Topic']

    for user in mask_seq:
        padded_mask_seq[user] = {
            **{'item_id': pad(mask_seq[user]['item_id'], max_len, pad_item)},
            **{i: pad(mask_seq[user][i], max_len, pad_item) for i in side_feature}
        }
        padded_labels[user] = pad(labels[user], max_len, pad_label)
    
    return padded_mask_seq, padded_labels

def get_recommendations_api(user_sequence: Dict[str, List], k: int = 10) -> List[int]:
    """Get recommendations for a user sequence"""
    try:
        device = next(model.parameters()).device
        
        # Convert user sequence to model format
        side_feature = ['rented_for', 'Topic']
        mask_seq = {'user': {'item_id': user_sequence['item_ids']}}
        for feat in side_feature:
            mask_seq['user'][feat] = user_sequence[feat]
        
        padded_seq, _ = padding(mask_seq, {'user': []}, max_len=21)
        
        # Prepare tensors
        item_tensor = torch.tensor([padded_seq['user']['item_id']], dtype=torch.long).to(device)
        side_input_dict = {
            feat: torch.tensor([padded_seq['user'][feat]], dtype=torch.long).to(device)
            for feat in side_feature
        }
        key_padding_mask = (item_tensor == 0)
        
        # Get predictions
        with torch.no_grad():
            logits = model(item_tensor, side_input_dict, key_padding_mask=key_padding_mask)
            last_pos = min(len(user_sequence['item_ids'])-1, 20)
            probabilities = torch.softmax(logits[:, last_pos, :], dim=-1)
            topk = torch.topk(probabilities, k=k).indices[0].tolist()
        
        # Convert back to original item IDs
        recommendations = [mappings['index_to_item'][i] for i in topk if i in mappings['index_to_item']]
        
        return recommendations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "NovaBert Recommendation API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """Get recommendations for a user"""
    try:
        # Prepare user sequence
        user_sequence = {
            'item_ids': [mappings['item_to_index'].get(item_id, 0) for item_id in request.item_ids],
            'rented_for': [mappings['rented_for_to_index'].get(rented_for, 0) for rented_for in request.rented_for],
            'Topic': [mappings['topic_to_index'].get(topic, 0) for topic in request.topics]
        }
        
        # Pad sequences to same length
        max_len = max(len(user_sequence['item_ids']), len(user_sequence['rented_for']), len(user_sequence['Topic']))
        user_sequence['item_ids'] = user_sequence['item_ids'] + [0] * (max_len - len(user_sequence['item_ids']))
        user_sequence['rented_for'] = user_sequence['rented_for'] + [0] * (max_len - len(user_sequence['rented_for']))
        user_sequence['Topic'] = user_sequence['Topic'] + [0] * (max_len - len(user_sequence['Topic']))
        
        # Get recommendations
        recommendations = get_recommendations_api(user_sequence, request.k)
        
        # Get item details
        item_details = []
        for item_id in recommendations:
            details = get_item_details(item_id, df)
            if 'error' not in details:
                item_details.append(details)
        
        return RecommendationResponse(
            recommendations=recommendations,
            item_details=item_details,
            success=True,
            message=f"Found {len(recommendations)} recommendations"
        )
        
    except Exception as e:
        return RecommendationResponse(
            recommendations=[],
            item_details=[],
            success=False,
            message=f"Error: {str(e)}"
        )

@app.post("/item/{item_id}", response_model=ItemDetailsResponse)
async def get_item_details_endpoint(item_id: int):
    """Get detailed information about an item"""
    try:
        details = get_item_details(item_id, df)
        if 'error' in details:
            raise HTTPException(status_code=404, detail=details['error'])
        
        return ItemDetailsResponse(**details)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/items")
async def get_available_items(limit: int = 100):
    """Get list of available items"""
    try:
        items = df[['item_id', 'category', 'review_summary']].drop_duplicates().head(limit)
        return {"items": items.to_dict('records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/topics")
async def get_available_topics():
    """Get list of available topics"""
    try:
        topic_info = pd.read_csv('/Users/baonguyen/IU/thesis/data/topic_info.csv')
        return {"topics": topic_info[['Topic', 'Name', 'Representation']].to_dict('records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/occasions")
async def get_available_occasions():
    """Get list of available occasions"""
    try:
        occasions = list(mappings['unique_rented_for'])
        return {"occasions": occasions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

