"""
Embedding visualization utilities for ChromaDB collections
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

class EmbeddingVisualizer:
    """Visualize embeddings from ChromaDB collections"""
    
    def __init__(self, chroma_client: chromadb.Client, collection_name: str = "data_lineage_queries_txt", 
                 embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.client = chroma_client
        self.collection_name = collection_name
        self.collection = None
        self.embeddings = None
        self.metadata = None
        self.documents = None
        
        # Load embedding model with offline support
        self.embedding_model = self._load_embedding_model_offline(embedding_model_name)
        
    def _find_cached_model_path(self, model_name: str) -> Optional[str]:
        """Find the cached model path for offline loading"""
        # Common cache locations
        cache_locations = [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "torch" / "sentence_transformers",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub",
        ]
        
        # Transform model name to hub format
        hub_model_name = f"models--sentence-transformers--{model_name}"
        
        for cache_dir in cache_locations:
            if not cache_dir.exists():
                continue
                
            model_dir = cache_dir / hub_model_name
            if model_dir.exists():
                # Find the latest snapshot
                snapshots_dir = model_dir / "snapshots"
                if snapshots_dir.exists():
                    snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                    if snapshots:
                        # Get the most recent snapshot
                        latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
                        print(f"📸 Found cached model at: {latest_snapshot}")
                        return str(latest_snapshot)
        
        return None
    
    def _load_embedding_model_offline(self, model_name: str) -> Optional[SentenceTransformer]:
        """Load embedding model with offline fallback"""
        print(f"🔄 Loading embedding model: {model_name}")
        
        # First, try to find cached model for offline use
        cached_path = self._find_cached_model_path(model_name)
        
        if cached_path:
            try:
                # Set offline mode
                os.environ.update({
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1"
                })
                
                model = SentenceTransformer(cached_path, local_files_only=True)
                print(f"✅ Loaded model offline from cache: {model_name}")
                return model
                
            except Exception as e:
                print(f"⚠️  Offline loading failed: {e}")
                print("🔄 Trying online mode...")
        
        # Fallback to online mode
        try:
            # Clear offline environment variables
            for key in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]:
                os.environ.pop(key, None)
            
            model = SentenceTransformer(model_name)
            print(f"✅ Loaded model online: {model_name}")
            return model
            
        except ImportError as e:
            print(f"❌ sentence-transformers not installed: {e}")
            print("💡 Install with: pip install sentence-transformers")
            return None
            
        except Exception as e:
            print(f"❌ Error loading model {model_name}: {e}")
            
            # Try fallback models
            fallback_models = ["paraphrase-MiniLM-L6-v2", "all-mpnet-base-v2"]
            for fallback in fallback_models:
                try:
                    print(f"🔄 Trying fallback model: {fallback}")
                    model = SentenceTransformer(fallback)
                    print(f"✅ Loaded fallback model: {fallback}")
                    return model
                except Exception as fe:
                    print(f"❌ Fallback {fallback} failed: {fe}")
            
            print("❌ All models failed to load!")
            return None
    
    def load_collection_data(self) -> Dict[str, Any]:
        """Load embeddings and metadata from ChromaDB collection"""
        try:
            self.collection = self.client.get_collection(self.collection_name)
            
            # Get all data from collection
            results = self.collection.get(include=['embeddings', 'metadatas', 'documents'])
            
            self.embeddings = np.array(results['embeddings'])
            self.metadata = results['metadatas']
            self.documents = results['documents']
            
            print(f"✅ Loaded {len(self.embeddings)} embeddings from {self.collection_name}")
            print(f"📊 Embedding dimension: {self.embeddings.shape[1]}")
            
            return {
                'embeddings': self.embeddings,
                'metadata': self.metadata,
                'documents': self.documents,
                'count': len(self.embeddings)
            }
            
        except Exception as e:
            print(f"❌ Error loading collection data: {e}")
            return {}
    
    def reduce_dimensions_tsne(self, perplexity: int = 30, n_components: int = 2) -> np.ndarray:
        """Reduce embeddings to 2D/3D using t-SNE"""
        if self.embeddings is None:
            self.load_collection_data()
            
        tsne = TSNE(
            n_components=n_components,
            perplexity=min(perplexity, len(self.embeddings) - 1),
            random_state=42,
            init='pca',
            learning_rate='auto'
        )
        
        reduced_embeddings = tsne.fit_transform(self.embeddings)
        print(f"✅ Reduced embeddings to {n_components}D using t-SNE")
        
        return reduced_embeddings
    
    def reduce_dimensions_pca(self, n_components: int = 2) -> np.ndarray:
        """Reduce embeddings to 2D/3D using PCA"""
        if self.embeddings is None:
            self.load_collection_data()
            
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(self.embeddings)
        
        print(f"✅ Reduced embeddings to {n_components}D using PCA")
        print(f"📊 Explained variance ratio: {pca.explained_variance_ratio_}")
        
        return reduced_embeddings
    
    def create_2d_scatter_plot(self, method: str = 'tsne', color_by: str = 'source') -> go.Figure:
        """Create interactive 2D scatter plot of embeddings"""
        if method == 'tsne':
            coords = self.reduce_dimensions_tsne(n_components=2)
        else:
            coords = self.reduce_dimensions_pca(n_components=2)
        
        # Prepare color mapping
        color_values = []
        hover_texts = []
        unique_sources = set()
        
        # First pass: collect unique sources
        for i, (doc, meta) in enumerate(zip(self.documents, self.metadata)):
            if color_by in meta:
                unique_sources.add(meta[color_by])
            else:
                unique_sources.add('Unknown')
        
        # Create color mapping (convert strings to numbers for consistent coloring)
        unique_sources = sorted(list(unique_sources))
        color_map = {source: i for i, source in enumerate(unique_sources)}
        
        # Second pass: assign colors and create hover texts
        for i, (doc, meta) in enumerate(zip(self.documents, self.metadata)):
            # Extract color value based on metadata
            if color_by in meta:
                color_values.append(color_map[meta[color_by]])
            else:
                color_values.append(color_map['Unknown'])
            
            # Create hover text
            hover_text = f"<b>Document {i}</b><br>"
            hover_text += f"Text: {doc[:100]}...<br>"
            for key, value in meta.items():
                hover_text += f"{key}: {value}<br>"
            hover_texts.append(hover_text)
        
        fig = px.scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            color=color_values,
            hover_name=[f"Doc {i}" for i in range(len(coords))],
            title=f'Embedding Visualization - {method.upper()} (2D)',
            labels={'x': f'{method.upper()} Component 1', 'y': f'{method.upper()} Component 2'},
            color_continuous_scale='Viridis'
        )
        
        # Update hover template
        fig.update_traces(
            hovertemplate='<b>%{hovertext}</b><br>' +
                         'X: %{x:.3f}<br>' +
                         'Y: %{y:.3f}<br>' +
                         '<extra></extra>',
            hovertext=hover_texts
        )
        
        # Add colorbar with proper labels
        fig.update_layout(
            width=800,
            height=600,
            title_font_size=16,
            coloraxis_colorbar=dict(
                title=color_by.title(),
                tickvals=list(range(len(unique_sources))),
                ticktext=unique_sources
            )
        )
        
        return fig
    
    def create_3d_scatter_plot(self, method: str = 'tsne', color_by: str = 'source') -> go.Figure:
        """Create interactive 3D scatter plot of embeddings"""
        if method == 'tsne':
            coords = self.reduce_dimensions_tsne(n_components=3)
        else:
            coords = self.reduce_dimensions_pca(n_components=3)
        
        # Prepare color mapping and hover texts
        color_values = []
        hover_texts = []
        unique_sources = set()
        
        # First pass: collect unique sources
        for i, (doc, meta) in enumerate(zip(self.documents, self.metadata)):
            if color_by in meta:
                unique_sources.add(meta[color_by])
            else:
                unique_sources.add('Unknown')
        
        # Create color mapping (convert strings to numbers)
        unique_sources = sorted(list(unique_sources))
        color_map = {source: i for i, source in enumerate(unique_sources)}
        
        # Second pass: assign colors and create hover texts
        for i, (doc, meta) in enumerate(zip(self.documents, self.metadata)):
            if color_by in meta:
                color_values.append(color_map[meta[color_by]])
            else:
                color_values.append(color_map['Unknown'])
            
            hover_text = f"Doc {i}: {doc[:50]}..."
            hover_texts.append(hover_text)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=color_values,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(
                    title=color_by.title(),
                    tickvals=list(range(len(unique_sources))),
                    ticktext=unique_sources
                )
            ),
            text=hover_texts,
            hovertemplate='<b>%{text}</b><br>' +
                         'X: %{x:.3f}<br>' +
                         'Y: %{y:.3f}<br>' +
                         'Z: %{z:.3f}<br>' +
                         '<extra></extra>'
        )])
        
        fig.update_layout(
            title=f'Embedding Visualization - {method.upper()} (3D)',
            scene=dict(
                xaxis_title=f'{method.upper()} Component 1',
                yaxis_title=f'{method.upper()} Component 2',
                zaxis_title=f'{method.upper()} Component 3'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_cluster_heatmap(self) -> plt.Figure:
        """Create a heatmap showing embedding similarities"""
        if self.embeddings is None:
            self.load_collection_data()
        
        # Calculate cosine similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(self.embeddings)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            similarity_matrix,
            cmap='viridis',
            center=0,
            square=True,
            cbar_kws={'label': 'Cosine Similarity'}
        )
        plt.title('Embedding Similarity Heatmap')
        plt.xlabel('Document Index')
        plt.ylabel('Document Index')
        
        return plt.gcf()
    
    def save_visualization(self, fig, filename: str, format: str = 'html'):
        """Save visualization to file"""
        output_dir = Path('visualizations')
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / f"{filename}.{format}"
        
        if hasattr(fig, 'write_html'):  # Plotly figure
            if format == 'html':
                fig.write_html(str(filepath))
            elif format == 'png':
                fig.write_image(str(filepath))
        else:  # Matplotlib figure
            fig.savefig(str(filepath), dpi=300, bbox_inches='tight')
        
        print(f"💾 Saved visualization to {filepath}")
    
    def add_question_embeddings(self, questions: List[str]) -> None:
        """Add question embeddings to the visualization data"""
        if self.embedding_model is None:
            print("❌ No embedding model available. Cannot generate question embeddings.")
            return
        
        try:
            # Generate embeddings for questions
            print(f"🔄 Generating embeddings for {len(questions)} questions...")
            question_embeddings = self.embedding_model.encode(questions)
            print(f"✅ Generated question embeddings with shape: {question_embeddings.shape}")
            
            # Add to existing data
            if self.embeddings is not None:
                # Combine document and question embeddings
                self.embeddings = np.vstack([self.embeddings, question_embeddings])
                
                # Add metadata for questions
                for i, question in enumerate(questions):
                    question_meta = {
                        'source': 'question',
                        'type': 'user_query',
                        'content': question[:100] + '...' if len(question) > 100 else question
                    }
                    self.metadata.append(question_meta)
                    self.documents.append(question)
                    
                print(f"✅ Combined embeddings now have shape: {self.embeddings.shape}")
            else:
                print("❌ No document embeddings loaded. Load collection data first.")
                
        except Exception as e:
            print(f"❌ Error generating question embeddings: {e}")
    
    def create_combined_visualization(self, questions: List[str], method: str = 'tsne', 
                                    n_components: int = 2) -> go.Figure:
        """Create visualization with both documents and questions"""
        if self.embedding_model is None:
            print("❌ No embedding model available for questions")
            return None
            
        # Load document embeddings first
        if self.embeddings is None:
            self.load_collection_data()
        
        # Store original counts
        original_doc_count = len(self.embeddings) if self.embeddings is not None else 0
        
        # Add question embeddings
        self.add_question_embeddings(questions)
        
        if self.embeddings is None:
            print("❌ No embeddings available for visualization")
            return None
        
        # Reduce dimensions
        if method == 'tsne':
            coords = self.reduce_dimensions_tsne(n_components=n_components)
        else:
            coords = self.reduce_dimensions_pca(n_components=n_components)
        
        # Prepare data for plotting
        point_types = []
        colors = []
        hover_texts = []
        sizes = []
        
        for i, (doc, meta) in enumerate(zip(self.documents, self.metadata)):
            # Determine point type and color
            if i >= original_doc_count:  # This is a question
                point_types.append('Question')
                colors.append(0)  # Questions get color 0
                sizes.append(10)  # Larger size for questions
                hover_text = f"<b>Question {i - original_doc_count + 1}</b><br>"
                hover_text += f"Text: {doc}<br>"
            else:  # This is a document
                point_types.append('Document')
                colors.append(1)  # Documents get color 1
                sizes.append(6)  # Smaller size for documents
                hover_text = f"<b>Document {i + 1}</b><br>"
                hover_text += f"Text: {doc[:100]}...<br>"
                for key, value in meta.items():
                    hover_text += f"{key}: {value}<br>"
            
            hover_texts.append(hover_text)
        
        # Create the plot
        if n_components == 2:
            fig = go.Figure()
            
            # Add documents
            doc_indices = [i for i, pt in enumerate(point_types) if pt == 'Document']
            if doc_indices:
                fig.add_trace(go.Scatter(
                    x=[coords[i, 0] for i in doc_indices],
                    y=[coords[i, 1] for i in doc_indices],
                    mode='markers',
                    marker=dict(
                        size=[sizes[i] for i in doc_indices],
                        color='blue',
                        opacity=0.6,
                        symbol='circle'
                    ),
                    name='Documents',
                    hovertemplate='%{text}<extra></extra>',
                    text=[hover_texts[i] for i in doc_indices]
                ))
            
            # Add questions
            q_indices = [i for i, pt in enumerate(point_types) if pt == 'Question']
            if q_indices:
                fig.add_trace(go.Scatter(
                    x=[coords[i, 0] for i in q_indices],
                    y=[coords[i, 1] for i in q_indices],
                    mode='markers',
                    marker=dict(
                        size=[sizes[i] for i in q_indices],
                        color='red',
                        opacity=0.8,
                        symbol='diamond'
                    ),
                    name='Questions',
                    hovertemplate='%{text}<extra></extra>',
                    text=[hover_texts[i] for i in q_indices]
                ))
            
            fig.update_layout(
                title=f'Combined Visualization - Documents & Questions ({method.upper()} 2D)',
                xaxis_title=f'{method.upper()} Component 1',
                yaxis_title=f'{method.upper()} Component 2',
                width=900,
                height=700,
                showlegend=True
            )
            
        else:  # 3D plot
            fig = go.Figure()
            
            # Add documents
            doc_indices = [i for i, pt in enumerate(point_types) if pt == 'Document']
            if doc_indices:
                fig.add_trace(go.Scatter3d(
                    x=[coords[i, 0] for i in doc_indices],
                    y=[coords[i, 1] for i in doc_indices],
                    z=[coords[i, 2] for i in doc_indices],
                    mode='markers',
                    marker=dict(
                        size=[sizes[i] for i in doc_indices],
                        color='blue',
                        opacity=0.6,
                        symbol='circle'
                    ),
                    name='Documents',
                    hovertemplate='%{text}<extra></extra>',
                    text=[hover_texts[i] for i in doc_indices]
                ))
            
            # Add questions
            q_indices = [i for i, pt in enumerate(point_types) if pt == 'Question']
            if q_indices:
                fig.add_trace(go.Scatter3d(
                    x=[coords[i, 0] for i in q_indices],
                    y=[coords[i, 1] for i in q_indices],
                    z=[coords[i, 2] for i in q_indices],
                    mode='markers',
                    marker=dict(
                        size=[sizes[i] for i in q_indices],
                        color='red',
                        opacity=0.8,
                        symbol='diamond'
                    ),
                    name='Questions',
                    hovertemplate='%{text}<extra></extra>',
                    text=[hover_texts[i] for i in q_indices]
                ))
            
            fig.update_layout(
                title=f'Combined Visualization - Documents & Questions ({method.upper()} 3D)',
                scene=dict(
                    xaxis_title=f'{method.upper()} Component 1',
                    yaxis_title=f'{method.upper()} Component 2',
                    zaxis_title=f'{method.upper()} Component 3'
                ),
                width=900,
                height=700,
                showlegend=True
            )
        
        return fig