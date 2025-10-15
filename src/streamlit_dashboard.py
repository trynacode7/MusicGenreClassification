"""
Streamlit Music Genre Classification Dashboard

A browser-based dashboard for visualizing music genre classification results from CSV files.
Supports both batch predictions (full songs) and near-real-time predictions (chunked).

Features:
- Read predictions from CSV files
- Bar chart of genre probabilities for current chunk
- Rolling history of last N chunk predictions
- Table showing batch prediction results
- Dynamic updates for near-real-time simulation
- Professional layout with Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Music Genre Classification Dashboard",
    page_icon="🎵",
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-prediction {
        background-color: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
    }
    .error-prediction {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Genre colors for consistent visualization
GENRE_COLORS = {
    'blues': '#1f77b4',
    'classical': '#ff7f0e', 
    'country': '#2ca02c',
    'disco': '#d62728',
    'hiphop': '#9467bd',
    'jazz': '#8c564b',
    'metal': '#e377c2',
    'pop': '#7f7f7f',
    'reggae': '#bcbd22',
    'rock': '#17becf',
    'Unknown': '#ff9999'
}

def load_csv_data(file_path):
    """Load data from CSV file."""
    try:
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        return None
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

def create_genre_probability_chart(data, chunk_index):
    """Create genre probability bar chart for current chunk."""
    if data is None or len(data) == 0:
        return None
        
    # Get current chunk data
    current_chunk = data.iloc[chunk_index % len(data)]
    
    # Create genre probability distribution
    genres = list(GENRE_COLORS.keys())
    probabilities = np.zeros(len(genres))
    
    # Set probability for predicted genre
    predicted_genre = current_chunk['predicted_genre']
    confidence = current_chunk['confidence']
    
    if predicted_genre in genres:
        genre_idx = genres.index(predicted_genre)
        probabilities[genre_idx] = confidence
    else:
        # Unknown genre
        probabilities[-1] = confidence
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=genres,
            y=probabilities,
            marker_color=[GENRE_COLORS.get(g, '#cccccc') for g in genres],
            text=[f'{p:.3f}' if p > 0 else '' for p in probabilities],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.3f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=f"Genre Probabilities - Chunk {chunk_index}",
        xaxis_title="Genre",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        showlegend=False,
        height=400
    )
    
    # Highlight the predicted genre
    if predicted_genre in genres:
        genre_idx = genres.index(predicted_genre)
        fig.data[0].marker.color[genre_idx] = '#ff0000'  # Red for predicted
    
    return fig

def create_rolling_history_chart(data, chunk_index, history_length):
    """Create rolling history chart."""
    if data is None or len(data) == 0:
        return None
        
    start_idx = max(0, chunk_index - history_length + 1)
    end_idx = min(len(data), chunk_index + 1)
    
    history_data = data.iloc[start_idx:end_idx]
    
    if len(history_data) == 0:
        return None
    
    # Create timeline plot
    fig = go.Figure()
    
    for i, (_, row) in enumerate(history_data.iterrows()):
        genre = row['predicted_genre']
        confidence = row['confidence']
        time_val = row['chunk_start']
        
        color = GENRE_COLORS.get(genre, '#cccccc')
        
        fig.add_trace(go.Scatter(
            x=[time_val],
            y=[i],
            mode='markers+text',
            marker=dict(size=15, color=color, line=dict(width=2, color='black')),
            text=[f'{genre}<br>{confidence:.2f}'],
            textposition='middle center',
            name=genre,
            showlegend=False,
            hovertemplate=f'<b>{genre}</b><br>Time: {time_val:.1f}s<br>Confidence: {confidence:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"Prediction History (Last {history_length} Chunks)",
        xaxis_title="Time (seconds)",
        yaxis_title="Chunk Index",
        height=400,
        showlegend=False
    )
    
    # Add current position indicator
    if chunk_index < len(data):
        current_time = data.iloc[chunk_index]['chunk_start']
        fig.add_vline(x=current_time, line_dash="dash", line_color="red", line_width=2)
    
    return fig

def create_confidence_timeline(data, chunk_index):
    """Create confidence timeline chart."""
    if data is None or len(data) == 0:
        return None
    
    # Create line plot with color-coded points
    fig = go.Figure()
    
    # Add line
    fig.add_trace(go.Scatter(
        x=data['chunk_index'],
        y=data['confidence'],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Confidence',
        hovertemplate='Chunk: %{x}<br>Confidence: %{y:.3f}<extra></extra>'
    ))
    
    # Add colored scatter points for each genre
    for i, row in data.iterrows():
        genre = row['predicted_genre']
        confidence = row['confidence']
        chunk_idx = row['chunk_index']
        
        color = GENRE_COLORS.get(genre, '#cccccc')
        
        fig.add_trace(go.Scatter(
            x=[chunk_idx],
            y=[confidence],
            mode='markers',
            marker=dict(size=8, color=color, line=dict(width=1, color='black')),
            name=genre,
            showlegend=False,
            hovertemplate=f'<b>{genre}</b><br>Chunk: {chunk_idx}<br>Confidence: {confidence:.3f}<extra></extra>'
        ))
    
    # Highlight current position
    if chunk_index < len(data):
        current_conf = data.iloc[chunk_index]['confidence']
        current_idx = data.iloc[chunk_index]['chunk_index']
        
        fig.add_trace(go.Scatter(
            x=[current_idx],
            y=[current_conf],
            mode='markers',
            marker=dict(size=15, color='red', line=dict(width=3, color='black')),
            name='Current',
            showlegend=False,
            hovertemplate=f'<b>Current Position</b><br>Chunk: {current_idx}<br>Confidence: {current_conf:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Confidence Over Time",
        xaxis_title="Chunk Index",
        yaxis_title="Confidence",
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False
    )
    
    return fig

def create_batch_results_table(data):
    """Create batch results table."""
    if data is None or len(data) == 0:
        return None
    
    # Create styled dataframe
    styled_data = data.copy()
    styled_data['Correct'] = styled_data['true_genre'] == styled_data['predicted_genre']
    styled_data['Status'] = styled_data['Correct'].apply(lambda x: '✓' if x else '✗')
    
    return styled_data

def create_sample_data():
    """Create sample CSV files for testing."""
    # Create sample batch data
    batch_data = {
        'file_name': ['song1.wav', 'song2.wav', 'song3.wav', 'song4.wav', 'song5.wav', 'song6.wav'],
        'true_genre': ['blues', 'classical', 'rock', 'jazz', 'pop', 'metal'],
        'predicted_genre': ['blues', 'classical', 'rock', 'jazz', 'metal', 'metal'],
        'confidence': [0.85, 0.92, 0.78, 0.88, 0.65, 0.91],
        'latency': [0.5, 0.6, 0.4, 0.7, 0.5, 0.3],
        'model': ['svm', 'svm', 'svm', 'svm', 'svm', 'svm']
    }
    
    batch_df = pd.DataFrame(batch_data)
    batch_df.to_csv('sample_batch_results.csv', index=False)
    
    # Create sample streaming data
    streaming_data = []
    genres = ['blues', 'classical', 'rock', 'jazz', 'pop', 'metal', 'country', 'disco']
    
    for i in range(30):
        genre = np.random.choice(genres)
        confidence = np.random.uniform(0.3, 0.95)
        
        streaming_data.append({
            'chunk_index': i,
            'chunk_start': i * 3.0,
            'chunk_end': (i + 1) * 3.0,
            'predicted_genre': genre,
            'confidence': confidence,
            'latency': np.random.uniform(0.3, 0.8)
        })
    
    streaming_df = pd.DataFrame(streaming_data)
    streaming_df.to_csv('sample_streaming_results.csv', index=False)

def main():
    """Main Streamlit app."""
    # Header
    st.markdown('<h1 class="main-header">🎵 Music Genre Classification Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📁 Data Sources")
    
    # File upload
    batch_file = st.sidebar.file_uploader("Upload Batch Results CSV", type=['csv'])
    streaming_file = st.sidebar.file_uploader("Upload Streaming Results CSV", type=['csv'])
    
    # Create sample data button
    if st.sidebar.button("Create Sample Data"):
        create_sample_data()
        st.sidebar.success("Sample data created!")
        st.experimental_rerun()
    
    # Load data
    batch_data = None
    streaming_data = None
    
    if batch_file is not None:
        batch_data = pd.read_csv(batch_file)
    elif os.path.exists('sample_batch_results.csv'):
        batch_data = pd.read_csv('sample_batch_results.csv')
    
    if streaming_file is not None:
        streaming_data = pd.read_csv(streaming_file)
    elif os.path.exists('sample_streaming_results.csv'):
        streaming_data = pd.read_csv('sample_streaming_results.csv')
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    if streaming_data is not None and len(streaming_data) > 0:
        # Animation controls
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("▶️ Start Animation"):
                st.session_state.animating = True
                st.session_state.chunk_index = 0
        
        with col2:
            if st.button("⏹️ Stop Animation"):
                st.session_state.animating = False
        
        # Animation speed
        speed = st.sidebar.slider("Animation Speed (seconds)", 0.1, 3.0, 1.0, 0.1)
        
        # History length
        history_length = st.sidebar.slider("History Length", 5, 50, 10)
        
        # Current chunk index
        if 'chunk_index' not in st.session_state:
            st.session_state.chunk_index = 0
        
        chunk_index = st.sidebar.slider("Current Chunk", 0, len(streaming_data)-1, st.session_state.chunk_index)
        st.session_state.chunk_index = chunk_index
        
        # Auto-animation
        if st.session_state.get('animating', False):
            time.sleep(speed)
            st.session_state.chunk_index = (st.session_state.chunk_index + 1) % len(streaming_data)
            st.experimental_rerun()
    
    # Main content
    if batch_data is not None or streaming_data is not None:
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if batch_data is not None:
                total_files = len(batch_data)
                st.metric("Total Files", total_files)
        
        with col2:
            if batch_data is not None:
                correct_predictions = (batch_data['true_genre'] == batch_data['predicted_genre']).sum()
                accuracy = correct_predictions / len(batch_data) * 100
                st.metric("Accuracy", f"{accuracy:.1f}%")
        
        with col3:
            if streaming_data is not None:
                st.metric("Total Chunks", len(streaming_data))
        
        with col4:
            if streaming_data is not None:
                avg_confidence = streaming_data['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        
        # Charts row
        if streaming_data is not None and len(streaming_data) > 0:
            st.header("📊 Streaming Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Genre probabilities
                prob_chart = create_genre_probability_chart(streaming_data, chunk_index)
                if prob_chart:
                    st.plotly_chart(prob_chart, use_container_width=True)
            
            with col2:
                # Rolling history
                history_chart = create_rolling_history_chart(streaming_data, chunk_index, history_length)
                if history_chart:
                    st.plotly_chart(history_chart, use_container_width=True)
            
            # Confidence timeline
            conf_chart = create_confidence_timeline(streaming_data, chunk_index)
            if conf_chart:
                st.plotly_chart(conf_chart, use_container_width=True)
        
        # Batch results
        if batch_data is not None and len(batch_data) > 0:
            st.header("📋 Batch Results")
            
            # Create styled table
            styled_data = create_batch_results_table(batch_data)
            
            # Display table with styling
            st.dataframe(
                styled_data[['file_name', 'true_genre', 'predicted_genre', 'confidence', 'Status']],
                use_container_width=True,
                height=400
            )
            
            # Summary statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Genre Distribution")
                genre_counts = batch_data['predicted_genre'].value_counts()
                fig = px.pie(values=genre_counts.values, names=genre_counts.index, title="Predicted Genres")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Confidence Distribution")
                fig = px.histogram(batch_data, x='confidence', nbins=20, title="Confidence Scores")
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👆 Please upload CSV files or create sample data to get started!")
        
        st.markdown("""
        ### How to use this dashboard:
        
        1. **Upload your CSV files** using the file uploaders in the sidebar
        2. **Or create sample data** by clicking the "Create Sample Data" button
        3. **Use the controls** to navigate through streaming data
        4. **Start animation** to see real-time simulation
        5. **View batch results** in the table below
        
        ### Expected CSV format:
        
        **Batch Results CSV:**
        - `file_name`: Name of the audio file
        - `true_genre`: True genre label
        - `predicted_genre`: Predicted genre
        - `confidence`: Prediction confidence (0-1)
        - `latency`: Processing time in seconds
        
        **Streaming Results CSV:**
        - `chunk_index`: Index of the chunk
        - `chunk_start`: Start time of chunk (seconds)
        - `chunk_end`: End time of chunk (seconds)
        - `predicted_genre`: Predicted genre for this chunk
        - `confidence`: Prediction confidence (0-1)
        - `latency`: Processing time in seconds
        """)

if __name__ == "__main__":
    main()
