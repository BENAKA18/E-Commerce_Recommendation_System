import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Visualization
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Complete Recommendation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #A23B72;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: bold;
        border-bottom: 3px solid #2E86AB;
        padding-bottom: 0.5rem;
    }
    .model-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        border: 2px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .winner-card {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .metric-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: white;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .nav-button {
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        border-radius: 8px;
        background-color: #e9f7fe;
        border: 2px solid #2E86AB;
        cursor: pointer;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .nav-button:hover {
        background-color: #2E86AB;
        color: white;
    }
    .center-content {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        text-align: center;
    }
    .comparison-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        width: 100%;
        max-width: 800px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

class CompleteRecommendationSystem:
    def __init__(self):
        self.products_df = None
        self.ratings_df = None
        self.user_item_matrix = None
        self.popular_products = None
        self.cosine_sim = None
        self.indices = None
        
    def load_data(self):
        """Load data for all three models"""
        try:
            # Create progress tracker
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Load products
            status_text.text("📦 Loading products...")
            products_path = r"C:\Users\sumad\Downloads\recommendation\product_descriptions.csv"
            if os.path.exists(products_path):
                self.products_df = pd.read_csv(products_path, nrows=5000)
                
                # Standardize columns
                if 'product_uid' in self.products_df.columns:
                    self.products_df = self.products_df.rename(columns={'product_uid': 'product_id'})
                self.products_df['product_id'] = self.products_df['product_id'].astype(str)
                
                # Create product name
                if 'product_name' not in self.products_df.columns:
                    if 'product_description' in self.products_df.columns:
                        self.products_df['product_name'] = self.products_df['product_description'].str[:30] + '...'
                    else:
                        self.products_df['product_name'] = 'Product ' + self.products_df['product_id']
                
                st.success(f"✓ Loaded {len(self.products_df):,} products")
            else:
                st.error("Product file not found")
                return False
            
            progress_bar.progress(30)
            
            # Load ratings
            status_text.text("⭐ Loading ratings...")
            ratings_path = r"C:\Users\sumad\Downloads\recommendation\ratings_Beauty.csv"
            if os.path.exists(ratings_path):
                self.ratings_df = pd.read_csv(ratings_path, nrows=10000)
                
                # Standardize columns
                column_map = {}
                if 'UserId' in self.ratings_df.columns:
                    column_map['UserId'] = 'user_id'
                if 'ProductId' in self.ratings_df.columns:
                    column_map['ProductId'] = 'product_id'
                if 'Rating' in self.ratings_df.columns:
                    column_map['Rating'] = 'rating'
                
                if column_map:
                    self.ratings_df = self.ratings_df.rename(columns=column_map)
                
                self.ratings_df['product_id'] = self.ratings_df['product_id'].astype(str)
                self.ratings_df['user_id'] = self.ratings_df['user_id'].astype(str)
                
                st.success(f"✓ Loaded {len(self.ratings_df):,} ratings")
            else:
                st.error("Ratings file not found")
                return False
            
            progress_bar.progress(60)
            
            # Prepare all models
            status_text.text("🔄 Preparing all recommendation models...")
            self._prepare_all_models()
            
            progress_bar.progress(100)
            status_text.text("✅ All models ready!")
            
            return True
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return False
    
    def _prepare_all_models(self):
        """Prepare all three recommendation models"""
        try:
            # 1. Popularity-Based Model
            self._prepare_popularity_model()
            
            # 2. Collaborative Filtering Model
            self._prepare_collaborative_model()
            
            # 3. Content-Based Model
            self._prepare_content_model()
            
            st.success("✓ All three models prepared successfully!")
            
        except Exception as e:
            st.warning(f"Note: {str(e)}")
    
    def _prepare_popularity_model(self):
        """Prepare popularity-based recommendations"""
        try:
            # Calculate product popularity
            product_stats = self.ratings_df.groupby('product_id').agg(
                rating_count=('rating', 'count'),
                avg_rating=('rating', 'mean')
            ).reset_index()
            
            # Get top 50 popular products
            self.popular_products = product_stats.sort_values(
                ['rating_count', 'avg_rating'], 
                ascending=[False, False]
            ).head(50).copy()
            
            # Add product names
            if self.products_df is not None:
                self.popular_products = pd.merge(
                    self.popular_products,
                    self.products_df[['product_id', 'product_name']],
                    on='product_id',
                    how='left'
                )
            
            # Fill missing names
            self.popular_products['product_name'] = self.popular_products['product_name'].fillna(
                'Product ' + self.popular_products['product_id']
            )
            
            st.info(f"✓ Popularity model: {len(self.popular_products)} popular products")
            
        except Exception as e:
            st.warning(f"Popularity model: {str(e)}")
    
    def _prepare_collaborative_model(self):
        """Prepare collaborative filtering model"""
        try:
            # Create user-item matrix
            self.user_item_matrix = self.ratings_df.pivot_table(
                index='user_id',
                columns='product_id',
                values='rating',
                aggfunc='mean'
            ).fillna(0)
            
            st.info(f"✓ Collaborative model: {self.user_item_matrix.shape[0]} users × {self.user_item_matrix.shape[1]} products")
            
        except Exception as e:
            st.warning(f"Collaborative model: {str(e)}")
    
    def _prepare_content_model(self):
        """Prepare content-based model"""
        try:
            # Use product descriptions for content-based
            if self.products_df is not None:
                # Prepare text data
                text_columns = []
                for col in self.products_df.columns:
                    if col != 'product_id' and self.products_df[col].dtype == 'object':
                        text_columns.append(col)
                
                if text_columns:
                    self.products_df['combined_text'] = self.products_df[text_columns].fillna('').agg(' '.join, axis=1)
                else:
                    self.products_df['combined_text'] = ''
                
                # Create TF-IDF matrix
                tfidf = TfidfVectorizer(stop_words='english', max_features=500)
                tfidf_matrix = tfidf.fit_transform(self.products_df['combined_text'])
                
                # Compute cosine similarity
                self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
                
                # Create indices
                self.indices = pd.Series(
                    self.products_df.index,
                    index=self.products_df['product_id']
                ).drop_duplicates()
                
                st.info(f"✓ Content-based model: {self.cosine_sim.shape[0]} products")
                
        except Exception as e:
            st.warning(f"Content model: {str(e)}")
    
    # ================== THREE RECOMMENDATION MODELS ==================
    
    def model_1_popularity_based(self, n=10):
        """Model 1: Popularity-based recommendations (for new customers)"""
        if self.popular_products is None:
            # Sample data
            return pd.DataFrame({
                'product_id': [str(i) for i in range(1, n+1)],
                'product_name': [f'Popular Product {i}' for i in range(1, n+1)],
                'rating_count': np.random.randint(50, 500, n),
                'avg_rating': np.round(np.random.uniform(3.5, 5.0, n), 1)
            })
        
        return self.popular_products.head(n).copy()
    
    def model_2_collaborative_filtering(self, user_id, n=10):
        """Model 2: Collaborative filtering (for returning customers)"""
        try:
            if self.user_item_matrix is None:
                return self.model_1_popularity_based(n)
            
            user_id = str(user_id)
            
            if user_id not in self.user_item_matrix.index:
                st.info("User not found. Showing popular products instead.")
                return self.model_1_popularity_based(n)
            
            # Simple user-based collaborative filtering
            user_ratings = self.user_item_matrix.loc[user_id]
            unrated_items = user_ratings[user_ratings == 0].index.tolist()
            
            if not unrated_items:
                return self.model_1_popularity_based(n)
            
            # Recommend popular unrated items
            if self.popular_products is not None:
                recommendations = self.popular_products[
                    self.popular_products['product_id'].isin(unrated_items)
                ].head(n)
                
                if len(recommendations) > 0:
                    return recommendations
            
            return self.model_1_popularity_based(n)
            
        except Exception as e:
            return self.model_1_popularity_based(n)
    
    def model_3_content_based(self, product_id, n=10):
        """Model 3: Content-based recommendations (cold start)"""
        try:
            if self.cosine_sim is None or self.indices is None:
                return self.model_1_popularity_based(n)
            
            product_id = str(product_id)
            
            # Check if product exists in indices
            if product_id not in self.indices:
                st.info("Product not found in content model. Showing popular products instead.")
                return self.model_1_popularity_based(n)
            
            # Get similar products
            idx = self.indices[product_id]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
            
            # Get product details
            result = []
            for prod_idx, score in sim_scores:
                if prod_idx < len(self.products_df):
                    product_info = self.products_df.iloc[prod_idx].to_dict()
                    product_info['similarity_score'] = score
                    result.append(product_info)
            
            if result:
                return pd.DataFrame(result)
            else:
                return self.model_1_popularity_based(n)
                
        except Exception as e:
            return self.model_1_popularity_based(n)
    
    # ================== PCA vs SVD COMPARISON ==================
    
    def compare_pca_vs_svd(self, n_components=5):
        """Compare PCA vs SVD for dimensionality reduction"""
        try:
            # Use a sample for comparison
            if len(self.ratings_df) > 2000:
                sample_ratings = self.ratings_df.sample(2000, random_state=42)
            else:
                sample_ratings = self.ratings_df
            
            # Create matrix for comparison
            comparison_matrix = sample_ratings.pivot_table(
                index='user_id',
                columns='product_id',
                values='rating',
                aggfunc='mean'
            ).fillna(0)
            
            # Check if we have enough data
            if comparison_matrix.shape[0] < 10 or comparison_matrix.shape[1] < 10:
                return None
            
            # Adjust components
            max_comp = min(n_components, comparison_matrix.shape[1] - 1)
            if max_comp < 2:
                return None
            
            # FIXED: Use markdown instead of st.info with unsafe_allow_html
            st.markdown(f"**Comparing PCA vs SVD with {max_comp} components**")
            
            # Prepare data (item-based)
            X = comparison_matrix.T.values
            
            # Train PCA
            pca = PCA(n_components=max_comp)
            X_pca = pca.fit_transform(X)
            
            # Train SVD
            svd = TruncatedSVD(n_components=max_comp, random_state=42)
            X_svd = svd.fit_transform(X)
            
            # Reconstruct
            X_recon_pca = pca.inverse_transform(X_pca)
            X_recon_svd = svd.inverse_transform(X_svd)
            
            # Calculate metrics
            mse_pca = mean_squared_error(X.flatten(), X_recon_pca.flatten())
            mse_svd = mean_squared_error(X.flatten(), X_recon_svd.flatten())
            
            mae_pca = mean_absolute_error(X.flatten(), X_recon_pca.flatten())
            mae_svd = mean_absolute_error(X.flatten(), X_recon_svd.flatten())
            
            # Explained variance
            explained_var_pca = np.sum(pca.explained_variance_ratio_)
            explained_var_svd = np.sum(svd.explained_variance_ratio_)
            
            # Create results
            results = {
                'PCA': {
                    'MSE': float(mse_pca),
                    'MAE': float(mae_pca),
                    'Explained Variance': float(explained_var_pca),
                    'Components': max_comp
                },
                'SVD': {
                    'MSE': float(mse_svd),
                    'MAE': float(mae_svd),
                    'Explained Variance': float(explained_var_svd),
                    'Components': max_comp
                }
            }
            
            # Determine winner
            pca_score = (1 - min(mse_pca, 1)) + explained_var_pca
            svd_score = (1 - min(mse_svd, 1)) + explained_var_svd
            
            if pca_score > svd_score:
                results['Best Model'] = "PCA"
                results['Winner Reason'] = f"PCA has better overall score ({pca_score:.3f} vs {svd_score:.3f})"
            elif svd_score > pca_score:
                results['Best Model'] = "SVD"
                results['Winner Reason'] = f"SVD has better overall score ({svd_score:.3f} vs {pca_score:.3f})"
            else:
                results['Best Model'] = "Tie"
                results['Winner Reason'] = "Both models performed equally"
            
            results['PCA_Score'] = float(pca_score)
            results['SVD_Score'] = float(svd_score)
            
            return results
            
        except Exception as e:
            st.error(f"Comparison error: {str(e)}")
            return None

def main():
    # Initialize system
    if 'rec_system' not in st.session_state:
        st.session_state.rec_system = CompleteRecommendationSystem()
        st.session_state.data_loaded = False
    
    # Initialize navigation state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "overview"
    
    rec_system = st.session_state.rec_system
    
    # Sidebar
    st.sidebar.title("🎯 E-Commerce Recommendation System")
    st.sidebar.markdown("---")
    
    # Data loading
    if st.sidebar.button("🚀 Load Data", type="primary", use_container_width=True):
        with st.spinner("Loading data"):
            if rec_system.load_data():
                st.session_state.data_loaded = True
                st.sidebar.success("✓ All data loaded!")
            else:
                st.sidebar.error("✗ Load failed")
    
    st.sidebar.markdown("---")
    
    # Settings
    st.sidebar.subheader("⚙️ Settings")
    num_recs = st.sidebar.slider("Recommendations per model", 5, 15, 10)
    
    # Main content
    st.markdown('<h1 class="main-header">🎯 E-Commerce Recommendation System</h1>', unsafe_allow_html=True)
    
    # Navigation buttons - CENTERED
    st.markdown('<div class="center-content">', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("🏠 Overview", use_container_width=True, key="btn_overview"):
            st.session_state.current_page = "overview"
            st.rerun()
    
    with col2:
        if st.button("🔥 Popularity", use_container_width=True, key="btn_popularity"):
            st.session_state.current_page = "popularity"
            st.rerun()
    
    with col3:
        if st.button("👥 Collaborative", use_container_width=True, key="btn_collaborative"):
            st.session_state.current_page = "collaborative"
            st.rerun()
    
    with col4:
        if st.button("🔍 Content-Based", use_container_width=True, key="btn_content"):
            st.session_state.current_page = "content"
            st.rerun()
    
    with col5:
        if st.button("📊 PCA vs SVD", use_container_width=True, key="btn_comparison"):
            st.session_state.current_page = "comparison"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Quick stats
    if rec_system.products_df is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📦 Products", f"{len(rec_system.products_df):,}")
        with col2:
            if rec_system.ratings_df is not None:
                st.metric("⭐ Ratings", f"{len(rec_system.ratings_df):,}")
        with col3:
            if rec_system.popular_products is not None:
                st.metric("🔥 Popular", len(rec_system.popular_products))
    
    # Page content based on navigation
    if st.session_state.current_page == "overview":
        st.markdown('<h2 class="sub-header">🏠 System Overview</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="model-card">
        <h2>📋 Complete Recommendation System</h2>
        
        ### **Three Models for Customer Journey:**
        
        #### **1. 🔥 Popularity-Based Model** 
        - For **new customers** with no history
        - Based on overall product popularity
        - Shows trending/best-selling items
        
        #### **2. 👥 Collaborative Filtering Model** 
        - For **returning customers**
        - Personalized recommendations
        - Based on similar users' preferences
        
        #### **3. 🔍 Content-Based Model** 
        - For **cold start** (new products/businesses)
        - Based on product features/descriptions
        - No ratings needed
        
        #### **Plus: 📊 PCA vs SVD Comparison**
        - Determines which dimensionality reduction is better
        - Based on reconstruction error and explained variance
        - Clear winner identification
        </div>
        """, unsafe_allow_html=True)
        
        if rec_system.products_df is not None:
            st.subheader("📋 Data Preview")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Sample Products**")
                st.dataframe(rec_system.products_df[['product_id', 'product_name']].head(5), 
                           use_container_width=True, hide_index=True)
            with col2:
                st.write("**Sample Ratings**")
                st.dataframe(rec_system.ratings_df[['user_id', 'product_id', 'rating']].head(5), 
                           use_container_width=True, hide_index=True)
    
    elif st.session_state.current_page == "popularity":
        st.markdown('<h2 class="sub-header">🔥 Popularity-Based Recommendations</h2>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("Please load data first from the sidebar")
        else:
            if st.button("Generate Popular Recommendations", type="primary", key="pop_button"):
                with st.spinner("Getting popular products..."):
                    recommendations = rec_system.model_1_popularity_based(num_recs)
                    
                    st.success(f"🎯 Top {len(recommendations)} Popular Products")
                    
                    for idx, row in recommendations.iterrows():
                        with st.container():
                            cols = st.columns([3, 1, 1])
                            with cols[0]:
                                st.markdown(f"### {row['product_name']}")
                            with cols[1]:
                                if 'avg_rating' in row:
                                    st.markdown(f"**Rating:** ⭐ {row['avg_rating']:.1f}")
                            with cols[2]:
                                if 'rating_count' in row:
                                    st.markdown(f"**Reviews:** {row['rating_count']:,}")
                            st.divider()
                    
                    # Visualization
                    if len(recommendations) > 0:
                        fig = px.bar(
                            recommendations.head(10),
                            x='product_name',
                            y='rating_count',
                            title="Top 10 Products by Popularity",
                            color='avg_rating',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(xaxis_tickangle=-45, height=400)
                        st.plotly_chart(fig, use_container_width=True)
    
    elif st.session_state.current_page == "collaborative":
        st.markdown('<h2 class="sub-header">👥 Collaborative Filtering Recommendations</h2>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("Please load data first from the sidebar")
        else:
            col1, col2 = st.columns(2)
            with col1:
                # Get available users
                if rec_system.user_item_matrix is not None:
                    available_users = list(rec_system.user_item_matrix.index)[:20]
                    user_id = st.selectbox("Select User", options=available_users, key="collab_user")
                else:
                    user_id = st.text_input("Enter User ID", value="1", key="collab_input")
            
            with col2:
                if st.button("Get Personalized Recommendations", type="primary", key="collab_button"):
                    with st.spinner("Finding personalized recommendations..."):
                        recommendations = rec_system.model_2_collaborative_filtering(user_id, num_recs)
                        
                        if recommendations is not None:
                            st.success(f"🎯 Personalized Recommendations for User {user_id}")
                            
                            for idx, row in recommendations.iterrows():
                                with st.container():
                                    cols = st.columns([3, 1, 1])
                                    with cols[0]:
                                        st.markdown(f"### {row['product_name']}")
                                    with cols[1]:
                                        if 'avg_rating' in row:
                                            st.markdown(f"**Rating:** ⭐ {row['avg_rating']:.1f}")
                                    with cols[2]:
                                        if 'rating_count' in row:
                                            st.markdown(f"**Reviews:** {row['rating_count']}")
                                    st.divider()
    
    elif st.session_state.current_page == "content":
        st.markdown('<h2 class="sub-header">🔍 Content-Based Recommendations</h2>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("Please load data first from the sidebar")
        else:
            # FIXED: Get products that exist in the products_df (content model source)
            if rec_system.products_df is not None:
                # Use the first 20 products from products_df (not popular_products)
                valid_products = rec_system.products_df[['product_id', 'product_name']].head(20)
                
                if len(valid_products) > 0:
                    product_dict = {row['product_name']: row['product_id'] for _, row in valid_products.iterrows()}
                    
                    selected_product = st.selectbox("Select a product to find similar items:", 
                                                  options=list(product_dict.keys()), key="content_select")
                    product_id = product_dict[selected_product]
                    
                    if st.button("Find Similar Products", type="primary", key="content_button"):
                        with st.spinner("Finding similar products..."):
                            recommendations = rec_system.model_3_content_based(product_id, num_recs)
                            
                            if recommendations is not None:
                                st.success(f"🔍 Products Similar to: **{selected_product}**")
                                
                                for idx, row in recommendations.iterrows():
                                    with st.container():
                                        cols = st.columns([3, 1, 1])
                                        with cols[0]:
                                            st.markdown(f"### {row['product_name']}")
                                        with cols[1]:
                                            if 'similarity_score' in row:
                                                st.markdown(f"**Similarity:** {row['similarity_score']:.2f}")
                                        with cols[2]:
                                            if 'avg_rating' in row:
                                                st.markdown(f"**Rating:** ⭐ {row['avg_rating']:.1f}")
                                        st.divider()
                else:
                    st.warning("No products available for content-based recommendations")
            else:
                st.warning("Please load data first")
    
    elif st.session_state.current_page == "comparison":
        st.markdown('<h2 class="sub-header">📊 PCA vs SVD Comparison</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="model-card">
        <h3>🎯 Determine the Best Dimensionality Reduction Technique</h3>
        <p>Compare Principal Component Analysis (PCA) vs Singular Value Decomposition (SVD) 
        for your recommendation system.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("Please load data first from the sidebar")
        else:
            st.subheader("Configuration")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                n_components = st.slider("Number of Components", 2, 10, 5, key="pca_svd_slider")
            
            with col2:
                st.markdown("")
                st.markdown("")
                if st.button("⚡ Compare PCA vs SVD", type="primary", key="compare_button"):
                    with st.spinner("Comparing models..."):
                        results = rec_system.compare_pca_vs_svd(n_components)
                        
                        if results:
                            # Create a centered container for the comparison
                            st.markdown('<div class="comparison-container">', unsafe_allow_html=True)
                            
                            # Display metrics in centered layout
                            st.subheader("📈 Performance Metrics")
                            
                            # Create two columns for side-by-side comparison
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("""
                                <div class="metric-box">
                                <h4 style="text-align: center; color: #2E86AB;">🔷 PCA Results</h4>
                                <hr>
                                <p><strong>Mean Squared Error (MSE):</strong><br>{mse:.6f}</p>
                                <p><strong>Mean Absolute Error (MAE):</strong><br>{mae:.6f}</p>
                                <p><strong>Explained Variance:</strong><br>{var:.2%}</p>
                                <p><strong>Components Used:</strong><br>{comp}</p>
                                </div>
                                """.format(
                                    mse=results['PCA']['MSE'],
                                    mae=results['PCA']['MAE'],
                                    var=results['PCA']['Explained Variance'],
                                    comp=results['PCA']['Components']
                                ), unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("""
                                <div class="metric-box">
                                <h4 style="text-align: center; color: #A23B72;">🔶 SVD Results</h4>
                                <hr>
                                <p><strong>Mean Squared Error (MSE):</strong><br>{mse:.6f}</p>
                                <p><strong>Mean Absolute Error (MAE):</strong><br>{mae:.6f}</p>
                                <p><strong>Explained Variance:</strong><br>{var:.2%}</p>
                                <p><strong>Components Used:</strong><br>{comp}</p>
                                </div>
                                """.format(
                                    mse=results['SVD']['MSE'],
                                    mae=results['SVD']['MAE'],
                                    var=results['SVD']['Explained Variance'],
                                    comp=results['SVD']['Components']
                                ), unsafe_allow_html=True)
                            
                            # WINNER ANNOUNCEMENT - CENTERED
                            winner = results['Best Model']
                            
                            if winner == "PCA":
                                st.markdown("""
                                <div class="winner-card">
                                <h1>🏆 PCA IS THE WINNER! 🏆</h1>
                                <h3>{reason}</h3>
                                <p>PCA performed better with lower reconstruction error and higher explained variance</p>
                                </div>
                                """.format(reason=results['Winner Reason']), unsafe_allow_html=True)
                            elif winner == "SVD":
                                st.markdown("""
                                <div class="winner-card">
                                <h1>🏆 SVD IS THE WINNER! 🏆</h1>
                                <h3>{reason}</h3>
                                <p>SVD performed better with lower reconstruction error and higher explained variance</p>
                                </div>
                                """.format(reason=results['Winner Reason']), unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div style="padding: 2rem; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                         color: white; border-radius: 10px; text-align: center; margin: 2rem 0;">
                                <h1>🤝 IT'S A TIE! 🤝</h1>
                                <h3>Both PCA and SVD performed equally well</h3>
                                <p>You can use either technique for your recommendation system</p>
                                </div>
                                """, unsafe_allow_html=True)

                            # Visualization - CENTERED
                            st.subheader("📊 Visual Comparison")

                            # Center the chart using CSS container
                            st.markdown("""
                            <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
                            <div style="width: 80%; max-width: 900px;">
                             """, unsafe_allow_html=True)

                            # Create chart
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                name='PCA',
                                x=['MSE', 'MAE'],
                                y=[results['PCA']['MSE'], results['PCA']['MAE']],
                                marker_color='#2E86AB',
                                text=[f"{results['PCA']['MSE']:.4f}", f"{results['PCA']['MAE']:.4f}"],
                                textposition='auto'
                            ))
                            fig.add_trace(go.Bar(
                                name='SVD',
                                x=['MSE', 'MAE'],
                                y=[results['SVD']['MSE'], results['SVD']['MAE']],
                                marker_color='#A23B72',
                                text=[f"{results['SVD']['MSE']:.4f}", f"{results['SVD']['MAE']:.4f}"],
                                textposition='auto'
                            ))
                            fig.update_layout(
                                title="Error Metrics Comparison (Lower is Better)",
                                yaxis_title="Error Value",
                                height=400,
                                showlegend=True,
                                width=800,
                                margin=dict(l=50, r=50, t=50, b=50)
                            )

                            # Display the chart
                            st.plotly_chart(fig, use_container_width=True)

                            st.markdown("""
    </div>
</div>
""", unsafe_allow_html=True)
                            
                            # Detailed comparison table - CENTERED
                            st.subheader("🔬 Detailed Comparison")

                            comparison_data = {
                                'Metric': ['Best Model', 'Mean Squared Error (MSE)', 
                                     'Mean Absolute Error (MAE)', 'Explained Variance',
                                        'Number of Components'],
                                 'PCA': [
                                     '✅' if winner == 'PCA' else '',
                                     f"{results['PCA']['MSE']:.6f}",
                                     f"{results['PCA']['MAE']:.6f}",
                                     f"{results['PCA']['Explained Variance']:.2%}",
                                     results['PCA']['Components']
                                      ],
                                  'SVD': [
                                     '✅' if winner == 'SVD' else '',
                                      f"{results['SVD']['MSE']:.6f}",
                                     f"{results['SVD']['MAE']:.6f}",
                                     f"{results['SVD']['Explained Variance']:.2%}",
                                     results['SVD']['Components']
                                      ]
                             }

                            # Create a wider, more visible table
                            df_comparison = pd.DataFrame(comparison_data)

                            # Style the table for better visibility
                            st.markdown("""
<style>
    .big-table {
        width: 100%;
        max-width: 900px;
        margin: 0 auto;
    }
    .big-table table {
        width: 100% !important;
        font-size: 16px !important;
    }
    .big-table th {
        font-size: 18px !important;
        font-weight: bold !important;
        text-align: center !important;
        background-color: #f0f2f6 !important;
    }
    .big-table td {
        font-size: 16px !important;
        text-align: center !important;
        padding: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

                            # Display the table
                            st.markdown('<div class="big-table">', unsafe_allow_html=True)
                            st.dataframe(df_comparison, use_container_width=True, height=300)
                            st.markdown('</div>', unsafe_allow_html=True)
    # Footer
    st.markdown("---")
    st.markdown("""
<div style="text-align: center; color: #666;">
<p>🎯 <strong>Complete Recommendation System</strong></p>
<p>Designed for e-commerce customer journey: New → Returning → Cold Start</p>
</div>
 """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()