# 🤖 NovaBert Recommendation System - Professional Web Application

A **professional-grade** web application for the NovaBert sequential recommendation model with cross-attention and gating fusion mechanisms. Features a modern Streamlit interface with advanced analytics and a comprehensive FastAPI REST API.

## ✨ Professional Features

### 🎨 **Modern UI/UX Design**

- **Gradient Headers**: Beautiful gradient backgrounds with professional styling
- **Card-Based Layout**: Clean, organized information display with shadow effects
- **Interactive Components**: Smooth animations and hover effects
- **Responsive Design**: Optimized for different screen sizes
- **Professional Color Scheme**: Consistent branding throughout the application

### 🚀 **Enhanced User Experience**

- **Real-time Status Indicators**: System status with color-coded indicators (🟢 Ready, 🟡 Loading, 🔴 Error)
- **Progress Tracking**: Step-by-step progress bars for all operations
- **Advanced Search**: Filter users by sequence length and other criteria
- **Interactive Analytics**: Expandable sections with detailed visualizations
- **Professional Error Handling**: Graceful error messages and recovery mechanisms

### 📊 **Advanced Analytics**

- **User Behavior Analysis**: Detailed sequence analysis with interactive visualizations
- **Recommendation Confidence**: Confidence scores for each recommendation
- **Performance Metrics**: Model performance visualization with interactive charts
- **Data Insights**: Comprehensive dataset overview and statistics
- **Interactive Charts**: Plotly-powered interactive visualizations

### 🔧 **Technical Excellence**

- **Robust Error Handling**: Professional error management with comprehensive logging
- **Caching System**: Optimized performance with intelligent caching
- **Session Management**: Persistent state across user interactions
- **Type Hints**: Full type annotation for better code quality and maintainability
- **Logging System**: Comprehensive logging for debugging and monitoring

## Features

- 🤖 **Interactive Recommendations**: Get personalized item recommendations based on user history
- 📊 **Data Visualization**: Explore dataset statistics and model performance
- 🎯 **Multi-feature Input**: Support for item sequences, occasions, and topic preferences
- 📈 **Model Performance**: View cross-validation results and hit rates
- 🔍 **Topic Analysis**: Explore BERTopic-generated topics and their characteristics
- 🔌 **REST API**: Programmatic access via FastAPI endpoints
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Model Architecture

The NovaBert model includes:

- **NovabertEmbedding**: Item and side feature embeddings with positional encoding
- **NovabertCrossAttention**: Cross-attention mechanism with gating fusion
- **NovabertLayer**: Transformer layer with cross-attention and feed-forward network
- **GatingFusor**: Learnable gating mechanism for feature fusion

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
cd /Users/baonguyen/IU/thesis/web_app
python setup.py
```

### Option 2: Manual Setup

```bash
cd /Users/baonguyen/IU/thesis/web_app
pip install -r requirements.txt
```

## Usage

### Streamlit Web Interface

1. **Start the application:**

   ```bash
   # Option 1: Use the launcher script
   ./run_streamlit.sh

   # Option 2: Use the Python launcher
   python run_app.py

   # Option 3: Direct Streamlit command
   streamlit run streamlit_app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Use the application:**
   - Select items from your history in the sidebar
   - Choose occasions and topics
   - Click "Get Recommendations" to see personalized suggestions
   - Explore data visualizations and model performance metrics

### REST API

1. **Start the API server:**

   ```bash
   # Option 1: Use the launcher script
   ./run_api.sh

   # Option 2: Direct uvicorn command
   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the API:**

   - API Documentation: `http://localhost:8000/docs`
   - Interactive API: `http://localhost:8000/redoc`
   - Health Check: `http://localhost:8000/health`

3. **Example API usage:**

   ```python
   import requests

   # Get recommendations
   response = requests.post("http://localhost:8000/recommend", json={
       "item_ids": [123456, 789012],
       "rented_for": ["wedding", "party"],
       "topics": [0, 1],
       "k": 10
   })

   recommendations = response.json()
   print(recommendations)
   ```

## Application Structure

```
web_app/
├── streamlit_app.py          # Main Streamlit application
├── api.py                    # FastAPI REST API
├── data_utils.py             # Data preprocessing utilities
├── run_app.py               # Streamlit launcher with checks
├── run_streamlit.sh         # Shell script launcher
├── run_api.sh               # API launcher script
├── setup.py                 # Automated setup script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## API Endpoints

### Core Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /recommend` - Get recommendations
- `GET /item/{item_id}` - Get item details
- `GET /items` - List available items
- `GET /topics` - List available topics
- `GET /occasions` - List available occasions

### Example API Calls

```bash
# Get recommendations
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "item_ids": [123456, 789012],
    "rented_for": ["wedding", "party"],
    "topics": [0, 1],
    "k": 5
  }'

# Get item details
curl "http://localhost:8000/item/123456"

# List available items
curl "http://localhost:8000/items?limit=50"
```

## Model Performance

The model achieves the following performance across 5-fold cross-validation:

- Mean HR@5: ~0.082 (8.2% hit rate)
- Individual fold performance varies from 0.078 to 0.090

## Data Requirements

The application expects the following data structure:

- **Main dataset**: CSV with columns including `item_id`, `user_id`, `category`, `rating`, `rented_for`, `Topic`, etc.
- **Topic information**: CSV with topic metadata including names and representations
- **Trained model**: PyTorch model state dictionary

Required file paths:

- Model weights: `/Users/baonguyen/IU/thesis/src/models/models_item_with_novabert_gatingfusor_/fold_1/best_model.pth`
- Data files: `/Users/baonguyen/IU/thesis/data/clean_data/data_with_bertopic_column.csv`
- Topic info: `/Users/baonguyen/IU/thesis/data/topic_info.csv`

## Configuration

### Environment Variables

- `MODEL_PATH`: Path to model weights (default: auto-detected)
- `DATA_PATH`: Path to dataset (default: auto-detected)
- `TOPIC_PATH`: Path to topic info (default: auto-detected)
- `DEVICE`: Device to use ('auto', 'cpu', 'cuda', 'mps')

### Model Parameters

- `embedding_dim`: 256
- `max_len`: 21
- `num_layers`: 2
- `num_heads`: 4

## Troubleshooting

### Common Issues

1. **Model not loading**:

   - Check that model file exists at the expected path
   - Verify PyTorch installation and device compatibility
   - Check available memory (4GB+ recommended)

2. **Data loading errors**:

   - Verify CSV files are in the correct locations
   - Check file permissions and format
   - Ensure sufficient disk space

3. **Memory issues**:

   - Close other applications to free memory
   - Consider using CPU instead of GPU if VRAM is limited
   - Reduce batch size or sequence length

4. **Dependencies**:
   - Run `pip install -r requirements.txt`
   - Check Python version (3.8+ required)
   - Update pip: `pip install --upgrade pip`

### Debug Mode

Enable debug mode for detailed error information:

```bash
# Streamlit
streamlit run streamlit_app.py --logger.level debug

# API
uvicorn api:app --log-level debug
```

## Customization

### Modifying the Model

1. Edit model parameters in `load_model()` function
2. Update architecture in the model class definitions
3. Adjust preprocessing in `data_utils.py`

### Adding Features

1. **Streamlit**: Add new components in `streamlit_app.py`
2. **API**: Add new endpoints in `api.py`
3. **Data**: Extend `data_utils.py` for new data processing

### Styling

- Modify Streamlit theme in `streamlit_app.py`
- Update API documentation in `api.py`
- Customize visualizations using Plotly

## Performance Optimization

### Caching

- Streamlit uses `@st.cache_data` and `@st.cache_resource`
- API uses in-memory caching for model and data
- Consider Redis for production deployments

### Scaling

- Use multiple API workers: `uvicorn api:app --workers 4`
- Implement load balancing for high traffic
- Consider model quantization for faster inference

## Production Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000 8501
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Deployment

- **Streamlit Cloud**: Deploy Streamlit app directly
- **Heroku**: Use Procfile for both services
- **AWS/GCP/Azure**: Use container services or serverless functions

## Technical Details

- **Web Framework**: Streamlit + FastAPI
- **ML Framework**: PyTorch
- **Visualization**: Plotly
- **Data Processing**: Pandas + NumPy
- **API Documentation**: OpenAPI/Swagger
- **Caching**: Streamlit built-in + in-memory
- **Async Support**: FastAPI async/await

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of a research thesis. Please cite appropriately if used in academic work.

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Check logs for detailed error messages
4. Ensure all dependencies are correctly installed
