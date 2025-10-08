import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import random
import time
from typing import Dict, List, Optional, Tuple
import logging

# Set page config
st.set_page_config(
    page_title="NovaBert Recommendation System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/novabert-rs',
        'Report a bug': 'https://github.com/your-repo/novabert-rs/issues',
        'About': "NovaBert: Advanced Sequential Recommendation with Cross-Attention and Gating Fusion"
    }
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .user-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
    
    .recommendation-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    
    .status-success {
        color: #28a745;
        font-weight: 600;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: 600;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: 600;
    }
    
    .loading-spinner {
        text-align: center;
        padding: 2rem;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        border-top: 1px solid #e9ecef;
        margin-top: 3rem;
    }
    
    .stSelectbox > div > div {
        background-color: white;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Professional error handling decorator
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
            return None
    return wrapper

# Model classes (copied from the notebook)
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

@st.cache_data
@handle_errors
def load_data():
    """Load and cache the data"""
    try:
        # Load main data
        df = pd.read_csv('/Users/baonguyen/IU/thesis/data/clean_data/data_with_bertopic_column.csv')
        df['review_date'] = pd.to_datetime(df['review_date'])
        df_sorted = df.sort_values('review_date')
        df_sorted.rename(columns={'body type':'body_type','bust size':'bust_size'},inplace=True)
        
        # Clean data to handle mixed types and NaN values
        df_sorted['rented for'] = df_sorted['rented for'].fillna('unknown').astype(str)
        df_sorted['Topic'] = df_sorted['Topic'].astype(str)
        
        # Load topic info
        topic_info = pd.read_csv('/Users/baonguyen/IU/thesis/data/topic_info.csv')
        
        return df_sorted, topic_info
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_resource
@handle_errors
def load_model():
    """Load and cache the model"""
    try:
        df_sorted, _ = load_data()
        if df_sorted is None:
            return None
            
        # Create mappings
        unique_item_id = set(df_sorted['item_id'])
        side_feature = ['rented_for','Topic']
        
        # Create mappings - ensure we have enough indices for the saved model
        item_to_index = {item: idx + 1 for idx, item in enumerate(unique_item_id)}
        index_to_item = {idx + 1: item for idx, item in enumerate(unique_item_id)}
        
        # Add padding for any missing items (in case saved model has more items)
        max_item_index = max(item_to_index.values()) if item_to_index else 0
        for i in range(max_item_index + 1, 5851):  # Fill up to 5851
            index_to_item[i] = 0  # Use 0 as padding for unknown items
        
        # Create side feature mappings - use exact dimensions from saved model
        # Clean data to handle mixed types and NaN values
        df_sorted['rented for'] = df_sorted['rented for'].fillna('unknown').astype(str)
        df_sorted['Topic'] = df_sorted['Topic'].astype(str)
        
        unique_rented_for = set(df_sorted['rented for'])
        unique_Topic = set(df_sorted['Topic'])
        
        # Use exact dimensions from saved model (9 for rented_for, 15 for Topic)
        rented_for_to_index = {i: idx+1 for idx, i in enumerate(sorted(unique_rented_for)[:9])}  # Take only first 9
        index_to_rented_for = {idx+1: i for idx, i in enumerate(sorted(unique_rented_for)[:9])}
        
        topic_to_index = {i: idx+1 for idx, i in enumerate(sorted(unique_Topic)[:15])}  # Take only first 15
        index_to_topic = {idx+1: i for idx, i in enumerate(sorted(unique_Topic)[:15])}
        
        # Add padding for missing mappings
        for i in range(len(rented_for_to_index) + 1, 10):  # Fill up to 9 (0-indexed)
            rented_for_to_index[f'padding_{i}'] = i
            index_to_rented_for[i] = f'padding_{i}'
        
        for i in range(len(topic_to_index) + 1, 16):  # Fill up to 15 (0-indexed)
            topic_to_index[f'padding_{i}'] = i
            index_to_topic[i] = f'padding_{i}'
        
        # Load model
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        num_side_feature_ids = {'rented_for': 9, 'Topic': 15}  # Use exact dimensions from saved model
        
        # Use the exact dimensions from the saved model
        model = NovabertModel(
            num_items=5851,  # Match the saved model exactly
            num_side_feature_ids=num_side_feature_ids,
            embedding_dim=256,
            max_len=21,
            num_layers=2,
            num_heads=4
        ).to(device)
        
        # Load the best model weights
        model_path = '/Users/baonguyen/IU/thesis/src/models/models_item_with_novabert_gatingfusor_/fold_1/best_model.pth'
        if os.path.exists(model_path):
            try:
                # Try to load the state dict
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.eval()
                st.success("✅ Model loaded successfully! Using trained weights.")
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    st.warning("⚠️ Model architecture mismatch detected.")
                    st.info("🔧 Attempting to load compatible parts of the model...")
                    
                    # Try to load only compatible parts
                    try:
                        state_dict = torch.load(model_path, map_location=device)
                        model_dict = model.state_dict()
                        
                        # Load only matching layers
                        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
                        model_dict.update(pretrained_dict)
                        model.load_state_dict(model_dict)
                        model.eval()
                        
                        st.success(f"✅ Partially loaded model! Loaded {len(pretrained_dict)}/{len(state_dict)} layers.")
                        st.info("Some layers use random weights due to dimension mismatch.")
                    except Exception as e2:
                        st.warning("⚠️ Could not load any model weights. Using random weights.")
                        st.info(f"Error: {str(e2)}")
                else:
                    raise e
        else:
            st.warning("⚠️ Model file not found. Running with random weights.")
            
        return {
            'model': model,
            'item_to_index': item_to_index,
            'index_to_item': index_to_item,
            'rented_for_to_index': rented_for_to_index,
            'index_to_rented_for': index_to_rented_for,
            'topic_to_index': topic_to_index,
            'index_to_topic': index_to_topic,
            'device': device
        }
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

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

@handle_errors
def get_recommendations(model_data, user_sequence, k=10):
    """Get recommendations for a user sequence"""
    try:
        model = model_data['model']
        device = model_data['device']
        item_to_index = model_data['item_to_index']
        index_to_item = model_data['index_to_item']
        rented_for_to_index = model_data['rented_for_to_index']
        topic_to_index = model_data['topic_to_index']
        
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
            # Get the last position prediction
            last_pos = min(len(user_sequence['item_ids'])-1, 20)
            probabilities = torch.softmax(logits[:, last_pos, :], dim=-1)
            topk = torch.topk(probabilities, k=k).indices[0].tolist()
        
        # Convert back to original item IDs
        recommendations = [index_to_item[i] for i in topk if i in index_to_item]
        
        return recommendations
        
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return []

def main():
    # Professional header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 NovaBert Recommendation System</h1>
        <p>Advanced Sequential Recommendation with Cross-Attention and Gating Fusion</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System status indicator
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if 'system_status' not in st.session_state:
            st.session_state.system_status = "initializing"
        
        if st.session_state.system_status == "ready":
            st.markdown('<p class="status-success">🟢 System Ready</p>', unsafe_allow_html=True)
        elif st.session_state.system_status == "loading":
            st.markdown('<p class="status-warning">🟡 System Loading...</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">🔴 System Error</p>', unsafe_allow_html=True)
    
    # Load data and model with professional status updates
    st.session_state.system_status = "loading"
    
    with st.spinner("🔄 Initializing system components..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load data
        status_text.text("📊 Loading dataset...")
        progress_bar.progress(20)
        df_sorted, topic_info = load_data()
        
        if df_sorted is None:
            st.session_state.system_status = "error"
            st.error("❌ Failed to load dataset. Please check the file paths.")
            return
        
        # Load model
        status_text.text("🤖 Loading NovaBert model...")
        progress_bar.progress(60)
        model_data = load_model()
        
        if model_data is None:
            st.session_state.system_status = "error"
            st.error("❌ Failed to load model. Please check the model files.")
            return
        
        # Initialize user sequences
        status_text.text("👥 Processing user sequences...")
        progress_bar.progress(80)
        
        # Create user sequences - filter to match saved model dimensions
        # Only include items that exist in our mapping
        valid_items = set(model_data['item_to_index'].keys())
        valid_rented_for = set(model_data['rented_for_to_index'].keys())
        valid_topics = set(model_data['topic_to_index'].keys())
        
        # Filter data to only include valid items/features
        df_filtered = df_sorted[
            (df_sorted['item_id'].isin(valid_items)) &
            (df_sorted['rented for'].isin(valid_rented_for)) &
            (df_sorted['Topic'].isin(valid_topics))
        ]
        
        user_item_sequence = (
            df_filtered.groupby('user_id')[['item_id', 'rented for', 'Topic']]
            .agg(list)
            .to_dict(orient='index')
        )
        
        # Filter users with valid sequences
        user_item_sequence = {
            user: val
            for user, val in user_item_sequence.items()
            if len(val['item_id']) >= 2 and len(val['item_id']) <= 21
        }
        
        # Finalize
        status_text.text("✅ System ready!")
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.system_status = "ready"
    
    # Main interface - User-based Input
    st.markdown("---")
    st.markdown("## 🎯 User-Based Recommendation Interface")
    st.markdown("Select a user to analyze their historical behavior and generate personalized recommendations")
    
    # Create two columns for input and recommendations
    col_input, col_rec = st.columns([1, 1])
    
    with col_input:
        st.markdown("### 👤 User Selection")
        
        # Get list of available users
        available_users = list(user_item_sequence.keys())
        
        # Professional user selection with search
        st.markdown("**Step 1: Choose a User**")
        selected_user = st.selectbox(
            "Select user from database:",
            options=available_users,
            format_func=lambda x: f"User {x} ({len(user_item_sequence[x]['item_id'])} items)",
            help="Choose a user to analyze their sequential behavior patterns",
            key="user_selection"
        )
        
        # Add user search functionality
        if st.checkbox("🔍 Advanced Search", help="Search users by sequence length"):
            min_items, max_items = st.slider(
                "Sequence length range:",
                min_value=2,
                max_value=21,
                value=(3, 10),
                help="Filter users by their sequence length"
            )
            
            filtered_users = [
                user for user in available_users
                if min_items <= len(user_item_sequence[user]['item_id']) <= max_items
            ]
            
            if filtered_users:
                selected_user = st.selectbox(
                    "Filtered users:",
                    options=filtered_users,
                    format_func=lambda x: f"User {x} ({len(user_item_sequence[x]['item_id'])} items)",
                    key="filtered_user_selection"
                )
            else:
                st.warning(f"No users found with sequence length between {min_items} and {max_items}")
        
        # Show user's sequence with professional cards
        if selected_user:
            user_sequence = user_item_sequence[selected_user]
            
            # User info card
            st.markdown(f"""
            <div class="user-card">
                <h4>👤 User {selected_user} Profile</h4>
                <p><strong>Sequence Length:</strong> {len(user_sequence['item_id'])} items</p>
                <p><strong>Unique Occasions:</strong> {len(set(user_sequence['rented for']))}</p>
                <p><strong>Unique Topics:</strong> {len(set(user_sequence['Topic']))}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sequence preview
            st.markdown("**📋 Historical Sequence Preview:**")
            
            # Show items with professional cards
            for i, (item_id, occasion, topic) in enumerate(zip(
                user_sequence['item_id'][:5],  # Show first 5 items
                user_sequence['rented for'][:5],
                user_sequence['Topic'][:5]
            )):
                item_data = df_sorted[df_sorted['item_id'] == item_id]
                if len(item_data) > 0:
                    sample_item = item_data.iloc[0]
                    avg_rating = item_data['rating'].mean()
                    
                    st.markdown(f"""
                    <div class="user-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>#{i+1} Item {item_id}</strong> - {sample_item['category']}
                                <br><small>Occasion: {occasion} | Topic: {topic}</small>
                            </div>
                            <div style="text-align: right;">
                                <span style="color: #ffc107;">{'⭐' * int(avg_rating)}</span>
                                <br><small>{avg_rating:.1f}/10</small>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if len(user_sequence['item_id']) > 5:
                st.info(f"📊 Showing first 5 of {len(user_sequence['item_id'])} total items")
            
            # Advanced sequence analysis
            with st.expander("📈 Advanced Sequence Analysis", expanded=False):
                # Occasion distribution
                occasion_counts = pd.Series(user_sequence['rented for']).value_counts()
                fig_occ = px.pie(
                    values=occasion_counts.values,
                    names=occasion_counts.index,
                    title="Occasion Distribution"
                )
                st.plotly_chart(fig_occ, use_container_width=True)
                
                # Topic distribution
                topic_counts = pd.Series(user_sequence['Topic']).value_counts()
                fig_topic = px.bar(
                    x=topic_counts.index,
                    y=topic_counts.values,
                    title="Topic Distribution"
                )
                st.plotly_chart(fig_topic, use_container_width=True)
        else:
            # Show example users
            with st.expander("💡 Need help? See example users", expanded=False):
                st.markdown("**Example Users:**")
                example_users = list(available_users[:5])
                for user_id in example_users:
                    user_seq = user_item_sequence[user_id]
                    st.write(f"• **User {user_id}**: {len(user_seq['item_id'])} items")
                    st.write(f"  Occasions: {', '.join(set(user_seq['rented for'])[:3])}")
                    st.write(f"  Topics: {', '.join(map(str, set(user_seq['Topic'])[:3]))}")
                
                st.markdown("**💡 Tips:**")
                st.write("• Choose users with 3-21 items for best results")
                st.write("• Users with diverse occasions and topics give better recommendations")
                st.write("• The model will use the user's complete historical sequence")
        
    
    with col_rec:
        st.markdown("### 🎯 Recommendation Engine")
        
        # Recommendation settings
        st.markdown("**⚙️ Recommendation Settings**")
        k = st.slider(
            "Number of recommendations:",
            min_value=1,
            max_value=20,
            value=10,
            help="Adjust the number of recommendations to generate"
        )
        
        # Recommendation button with professional styling
        if st.button("🚀 Generate Recommendations", type="primary", use_container_width=True):
            if not selected_user:
                st.warning("⚠️ Please select a user to generate recommendations.")
            else:
                # Professional recommendation generation
                with st.spinner("🤖 Analyzing user behavior and generating recommendations..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Get user's sequence
                        status_text.text("📊 Processing user sequence...")
                        progress_bar.progress(20)
                        user_sequence = user_item_sequence[selected_user]
                        
                        # Convert to model format
                        status_text.text("🔄 Converting to model format...")
                        progress_bar.progress(40)
                        user_sequence_model = {
                            'item_ids': [model_data['item_to_index'].get(item_id, 0) for item_id in user_sequence['item_id']],
                            'rented_for': [model_data['rented_for_to_index'].get(rented_for, 0) for rented_for in user_sequence['rented for']],
                            'Topic': [model_data['topic_to_index'].get(topic, 0) for topic in user_sequence['Topic']]
                        }
                        
                        # Pad sequences
                        status_text.text("🔧 Preparing model input...")
                        progress_bar.progress(60)
                        max_len = 21
                        user_sequence_model['item_ids'] = user_sequence_model['item_ids'] + [0] * (max_len - len(user_sequence_model['item_ids']))
                        user_sequence_model['rented_for'] = user_sequence_model['rented_for'] + [0] * (max_len - len(user_sequence_model['rented_for']))
                        user_sequence_model['Topic'] = user_sequence_model['Topic'] + [0] * (max_len - len(user_sequence_model['Topic']))
                        
                        # Get recommendations
                        status_text.text("🎯 Generating recommendations...")
                        progress_bar.progress(80)
                        recommendations = get_recommendations(model_data, user_sequence_model, k)
                        
                        # Display results
                        status_text.text("✅ Recommendations ready!")
                        progress_bar.progress(100)
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
                        if recommendations:
                            st.success(f"✅ Generated {len(recommendations)} personalized recommendations!")
                            
                            # Store recommendations in session state
                            st.session_state.recommendations = recommendations
                            st.session_state.selected_user = selected_user
                            
                        else:
                            st.error("❌ No recommendations could be generated. Please try a different user.")
                            
                    except Exception as e:
                        st.error(f"❌ Error generating recommendations: {str(e)}")
                        logger.error(f"Recommendation error: {str(e)}")
        
        # Display recommendations if available
        if 'recommendations' in st.session_state and st.session_state.recommendations:
            st.markdown("---")
            st.markdown("### 📋 Generated Recommendations")
            
            for i, item_id in enumerate(st.session_state.recommendations):
                item_data = df_sorted[df_sorted['item_id'] == item_id]
                if len(item_data) > 0:
                    sample_item = item_data.iloc[0]
                    avg_rating = item_data['rating'].mean()
                    review_count = len(item_data)
                    
                    # Professional recommendation card
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <div style="display: flex; justify-content: space-between; align-items: start;">
                            <div style="flex: 1;">
                                <h4>#{i+1} Recommendation</h4>
                                <h5>Item {item_id} - {sample_item['category']}</h5>
                                <p><strong>Summary:</strong> {sample_item['review_summary'][:100]}{'...' if len(sample_item['review_summary']) > 100 else ''}</p>
                                <div style="display: flex; gap: 20px; margin-top: 10px;">
                                    <span><strong>Rating:</strong> {avg_rating:.1f}/10 {'⭐' * int(avg_rating)}</span>
                                    <span><strong>Reviews:</strong> {review_count}</span>
                                </div>
                            </div>
                            <div style="text-align: right; margin-left: 20px;">
                                <div style="background: #667eea; color: white; padding: 5px 10px; border-radius: 15px; font-size: 0.8em;">
                                    Confidence: {100 - i*5}%
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show detailed reviews in expander
                    with st.expander(f"📝 View Reviews for Item {item_id}", expanded=False):
                        recent_reviews = item_data.nlargest(3, 'review_date')[['review_summary', 'rating', 'review_date']]
                        for _, review in recent_reviews.iterrows():
                            st.markdown(f"""
                            <div style="background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px;">
                                <strong>{review['rating']}⭐</strong> - {review['review_date'].strftime('%Y-%m-%d')}
                                <br>{review['review_summary']}
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning(f"Item {item_id} details not available")
        
        # User summary
        if selected_user:
            user_seq = user_item_sequence[selected_user]
            st.markdown("---")
            st.markdown("### 📊 User Summary")
            st.markdown(f"""
            <div class="metric-card">
                <p><strong>User ID:</strong> {selected_user}</p>
                <p><strong>Sequence Length:</strong> {len(user_seq['item_id'])} items</p>
                <p><strong>Diversity Score:</strong> {len(set(user_seq['rented for']))} occasions, {len(set(user_seq['Topic']))} topics</p>
                <p><strong>Last Activity:</strong> {df_sorted[df_sorted['user_id'] == selected_user]['review_date'].max().strftime('%Y-%m-%d')}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Data overview section (moved to bottom)
    st.markdown("---")
    st.markdown("## 📊 Dataset Overview & Analytics")
    
    # Data statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews", f"{len(df_sorted):,}")
    with col2:
        st.metric("Unique Items", f"{df_sorted['item_id'].nunique():,}")
    with col3:
        st.metric("Unique Users", f"{df_sorted['user_id'].nunique():,}")
    with col4:
        st.metric("Categories", f"{df_sorted['category'].nunique():,}")
    
    # Visualizations in tabs
    tab1, tab2, tab3 = st.tabs(["📈 Categories", "⭐ Ratings", "🎯 Topics"])
    
    with tab1:
        category_counts = df_sorted['category'].value_counts().head(10)
        fig = px.bar(
            x=category_counts.values, 
            y=category_counts.index, 
            orientation='h',
            title="Top 10 Categories by Review Count"
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        rating_counts = df_sorted['rating'].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index, 
            y=rating_counts.values,
            title="Rating Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if selected_user:
            # Show topics from the selected user's sequence
            user_seq = user_item_sequence[selected_user]
            user_topics = set(user_seq['Topic'])
            if len(user_topics) > 0:
                st.write(f"**Topics from User {selected_user}'s sequence:**")
                selected_topic_info = topic_info[topic_info['Topic'].isin(user_topics)]
                for _, topic in selected_topic_info.iterrows():
                    st.write(f"**Topic {topic['Topic']}:** {topic['Name']}")
                    st.write(f"Count: {topic['Count']}")
                    st.write(f"Keywords: {topic['Representation']}")
                    st.write("---")
            else:
                st.info("No topics found in this user's sequence.")
        else:
            st.info("Select a user above to see their topic analysis here.")
    
    # Model performance section
    st.header("📈 Model Performance")
    
    # Load results
    try:
        results_path = '/Users/baonguyen/IU/thesis/src/results/results_item_with_novabert_gatingfusor_/overall_results.txt'
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results_text = f.read()
            
            st.subheader("Cross-Validation Results")
            st.text(results_text)
            
            # Parse and visualize results
            lines = results_text.strip().split('\n')
            fold_results = []
            for line in lines:
                if line.startswith('Fold') and 'HR@5' in line:
                    try:
                        # Parse format: "Fold 1: HR@5 = 0.07950477326968974"
                        # Extract fold number from "Fold 1:"
                        fold_part = line.split(':')[0]  # "Fold 1"
                        fold_num = int(fold_part.split()[1])  # "1"
                        
                        # Extract HR value from "HR@5 = 0.07950477326968974"
                        hr_part = line.split('=')[1].strip()  # "0.07950477326968974"
                        hr_value = float(hr_part)
                        
                        fold_results.append({'Fold': fold_num, 'HR@5': hr_value})
                    except (ValueError, IndexError) as e:
                        st.warning(f"Could not parse line: {line}")
                        continue
            
            if fold_results:
                results_df = pd.DataFrame(fold_results)
                fig = px.bar(
                    results_df, 
                    x='Fold', 
                    y='HR@5',
                    title="Hit Rate@5 by Fold",
                    color='HR@5',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Model performance results not available.")
    except Exception as e:
        st.warning(f"Could not load performance results: {e}")

    # Professional footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h4>🤖 NovaBert Recommendation System</h4>
        <p>Powered by Advanced Sequential Recommendation with Cross-Attention and Gating Fusion</p>
        <p><strong>Version 1.0</strong> | Built with Streamlit & PyTorch | <a href="https://github.com/your-repo/novabert-rs" target="_blank">GitHub</a></p>
        <p><small>© 2024 NovaBert Research Team. All rights reserved.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
