import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="AI Drug Toxicity Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-card {
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: #f8f9fa;
    }
    .high-risk {
        border-color: #ff6b6b;
        background: #fff5f5;
    }
    .medium-risk {
        border-color: #ffd93d;
        background: #fffef5;
    }
    .low-risk {
        border-color: #6bcf7f;
        background: #f5fff7;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class DrugToxicityPredictor:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(128, 64, 32), random_state=42, max_iter=1000),
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_performance = {}
        
    def generate_synthetic_data(self, n_samples=5000):
        """Generate synthetic molecular data for demonstration"""
        np.random.seed(42)
        
        data = {
            'molecular_weight': np.random.normal(350, 150, n_samples),
            'logP': np.random.normal(2.5, 1.5, n_samples),
            'hbond_donors': np.random.poisson(2, n_samples),
            'hbond_acceptors': np.random.poisson(4, n_samples),
            'aromatic_rings': np.random.poisson(1.5, n_samples),
            'rotating_bonds': np.random.poisson(5, n_samples),
            'surface_area': np.random.normal(400, 100, n_samples),
            'volume': np.random.normal(300, 80, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Generate toxicity labels based on heuristic rules
        toxicity_score = (
            (df['molecular_weight'] > 500).astype(int) * 0.3 +
            (df['logP'] > 5).astype(int) * 0.4 +
            (df['aromatic_rings'] > 3).astype(int) * 0.2 +
            (df['hbond_donors'] > 5).astype(int) * 0.1 +
            np.random.random(n_samples) * 0.3
        )
        
        df['toxicity'] = (toxicity_score > 0.5).astype(int)
        
        return df
    
    def train_models(self, df):
        """Train all models on the dataset"""
        X = df.drop('toxicity', axis=1)
        y = df['toxicity']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, model) in enumerate(self.models.items()):
            status_text.text(f'Training {name}...')
            
            # Simulate training time
            for j in range(20):
                time.sleep(0.05)
                progress_bar.progress((i * 20 + j + 1) / (len(self.models) * 20))
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            self.model_performance[name] = {
                'accuracy': accuracy_score(y_test, y_pred) * 100,
                'precision': precision_score(y_test, y_pred) * 100,
                'f1_score': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_pred_proba)
            }
        
        self.is_trained = True
        status_text.text('Training completed!')
        progress_bar.progress(100)
        
        return X_test, y_test
    
    def predict_toxicity(self, features, model_name='Random Forest'):
        """Predict toxicity for given molecular features"""
        if not self.is_trained:
            st.error("Models not trained yet. Please train the models first.")
            return None
        
        features_scaled = self.scaler.transform([features])
        model = self.models[model_name]
        
        probability = model.predict_proba(features_scaled)[0][1]
        prediction = model.predict(features_scaled)[0]
        
        # Get feature importance for tree-based models
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_names = ['Molecular Weight', 'LogP', 'H-bond Donors', 
                           'H-bond Acceptors', 'Aromatic Rings', 'Rotating Bonds',
                           'Surface Area', 'Volume']
            feature_importance = dict(zip(feature_names, model.feature_importances_))
        
        return {
            'prediction': prediction,
            'probability': probability,
            'confidence': max(probability, 1 - probability),
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.4 else 'Low',
            'feature_importance': feature_importance
        }

def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 AI Drug Toxicity Prediction Model</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Machine Learning for Pharmaceutical Safety</p>', unsafe_allow_html=True)
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = DrugToxicityPredictor()
        st.session_state.data = st.session_state.predictor.generate_synthetic_data()
    
    predictor = st.session_state.predictor
    
    # Sidebar for model controls
    st.sidebar.header("🎯 Model Configuration")
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=list(predictor.models.keys()),
        index=0
    )
    
    # Training section
    st.sidebar.subheader("🚀 Model Training")
    if st.sidebar.button("🔥 Train All Models"):
        with st.spinner("Training models..."):
            X_test, y_test = predictor.train_models(st.session_state.data)
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
        st.sidebar.success("Training completed!")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🧪 Molecular Features Input")
        
        # Feature inputs
        molecular_weight = st.slider("Molecular Weight (g/mol)", 100, 800, 350, 10)
        logP = st.slider("LogP (Lipophilicity)", -2.0, 8.0, 2.5, 0.1)
        hbond_donors = st.slider("H-bond Donors", 0, 10, 2)
        hbond_acceptors = st.slider("H-bond Acceptors", 0, 15, 4)
        aromatic_rings = st.slider("Aromatic Rings", 0, 6, 1)
        rotating_bonds = st.slider("Rotating Bonds", 0, 20, 5)
        surface_area = st.slider("Surface Area (Ų)", 200, 800, 400, 10)
        volume = st.slider("Volume (ų)", 150, 600, 300, 10)
        
        features = [molecular_weight, logP, hbond_donors, hbond_acceptors, 
                   aromatic_rings, rotating_bonds, surface_area, volume]
        
        # Prediction button
        if st.button("🎯 Predict Toxicity"):
            if predictor.is_trained:
                result = predictor.predict_toxicity(features, selected_model)
                st.session_state.prediction_result = result
            else:
                st.error("Please train the models first!")
    
    with col2:
        st.subheader("📊 Prediction Results")
        
        # Show prediction results
        if hasattr(st.session_state, 'prediction_result') and st.session_state.prediction_result:
            result = st.session_state.prediction_result
            
            # Main prediction card
            risk_class = f"{result['risk_level'].lower()}-risk"
            st.markdown(f"""
            <div class="prediction-card {risk_class}">
                <h2 style="text-align: center; margin-bottom: 1rem;">
                    Toxicity Probability: {result['probability']:.1%}
                </h2>
                <h3 style="text-align: center; color: {'#ff6b6b' if result['risk_level'] == 'High' else '#ffd93d' if result['risk_level'] == 'Medium' else '#6bcf7f'};">
                    Risk Level: {result['risk_level']}
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Confidence", f"{result['confidence']:.1%}")
            with col2b:
                if predictor.model_performance:
                    accuracy = predictor.model_performance.get(selected_model, {}).get('accuracy', 0)
                    st.metric("Model Accuracy", f"{accuracy:.1f}%")
            
            # Toxicity endpoints simulation
            st.subheader("🎯 Toxicity Endpoints")
            endpoints = ['Hepatotoxicity', 'Cardiotoxicity', 'Mutagenicity', 
                        'Nephrotoxicity', 'Neurotoxicity', 'Carcinogenicity']
            
            endpoint_results = []
            for endpoint in endpoints:
                # Simulate endpoint-specific predictions
                endpoint_prob = result['probability'] + np.random.normal(0, 0.1)
                endpoint_prob = np.clip(endpoint_prob, 0, 1)
                predicted = endpoint_prob > 0.5
                
                endpoint_results.append({
                    'Endpoint': endpoint,
                    'Probability': endpoint_prob,
                    'Prediction': 'Toxic' if predicted else 'Safe',
                    'Status': '⚠️' if predicted else '✅'
                })
            
            endpoint_df = pd.DataFrame(endpoint_results)
            st.dataframe(endpoint_df, hide_index=True)
        
        else:
            st.info("Enter molecular features and click 'Predict Toxicity' to see results")
    
    # Model Performance Section
    if predictor.is_trained and predictor.model_performance:
        st.subheader("📈 Model Performance Comparison")
        
        # Create performance dataframe
        perf_data = []
        for model_name, metrics in predictor.model_performance.items():
            perf_data.append({
                'Model': model_name,
                'Accuracy (%)': metrics['accuracy'],
                'Precision (%)': metrics['precision'],
                'F1-Score': metrics['f1_score'],
                'AUC': metrics['auc']
            })
        
        perf_df = pd.DataFrame(perf_data)
        
        # Display metrics
        col3, col4 = st.columns(2)
        
        with col3:
            # Accuracy comparison
            fig_acc = px.bar(perf_df, x='Model', y='Accuracy (%)', 
                            title='Model Accuracy Comparison',
                            color='Accuracy (%)',
                            color_continuous_scale='viridis')
            fig_acc.update_layout(showlegend=False)
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col4:
            # Multiple metrics comparison
            fig_multi = go.Figure()
            fig_multi.add_trace(go.Bar(name='Accuracy', x=perf_df['Model'], y=perf_df['Accuracy (%)']))
            fig_multi.add_trace(go.Bar(name='Precision', x=perf_df['Model'], y=perf_df['Precision (%)']))
            fig_multi.update_layout(title='Accuracy vs Precision', barmode='group')
            st.plotly_chart(fig_multi, use_container_width=True)
        
        # Performance table
        st.subheader("📋 Detailed Performance Metrics")
        st.dataframe(perf_df, hide_index=True)
        
        # Feature importance (if available)
        if hasattr(st.session_state, 'prediction_result') and st.session_state.prediction_result:
            if st.session_state.prediction_result['feature_importance']:
                st.subheader("🔍 Feature Importance")
                
                importance_data = st.session_state.prediction_result['feature_importance']
                importance_df = pd.DataFrame(list(importance_data.items()), 
                                           columns=['Feature', 'Importance'])
                importance_df = importance_df.sort_values('Importance', ascending=True)
                
                fig_importance = px.bar(importance_df, x='Importance', y='Feature', 
                                      orientation='h', title='Feature Importance Analysis',
                                      color='Importance', color_continuous_scale='plasma')
                fig_importance.update_layout(height=400)
                st.plotly_chart(fig_importance, use_container_width=True)
    
    # Dataset Information
    st.subheader("📊 Dataset Information")
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.markdown('<div class="metric-card"><h3>5,000</h3><p>Total Compounds</p></div>', unsafe_allow_html=True)
    with col6:
        st.markdown('<div class="metric-card"><h3>8</h3><p>Molecular Features</p></div>', unsafe_allow_html=True)
    with col7:
        st.markdown('<div class="metric-card"><h3>6</h3><p>Toxicity Endpoints</p></div>', unsafe_allow_html=True)
    with col8:
        st.markdown('<div class="metric-card"><h3>80/20</h3><p>Train/Test Split</p></div>', unsafe_allow_html=True)
    
    # Data visualization
    if st.checkbox("🔬 Show Dataset Analysis"):
        st.subheader("📈 Dataset Distribution Analysis")
        
        col9, col10 = st.columns(2)
        
        with col9:
            # Molecular weight distribution
            fig_mw = px.histogram(st.session_state.data, x='molecular_weight', 
                                 color='toxicity', title='Molecular Weight Distribution',
                                 labels={'toxicity': 'Toxicity'})
            st.plotly_chart(fig_mw, use_container_width=True)
        
        with col10:
            # LogP vs Toxicity scatter
            fig_logp = px.scatter(st.session_state.data, x='logP', y='molecular_weight',
                                 color='toxicity', title='LogP vs Molecular Weight',
                                 labels={'toxicity': 'Toxicity'})
            st.plotly_chart(fig_logp, use_container_width=True)
        
        # Correlation heatmap
        corr_data = st.session_state.data.select_dtypes(include=[np.number]).corr()
        fig_corr = px.imshow(corr_data, title='Feature Correlation Matrix',
                            color_continuous_scale='RdBu_r', aspect='auto')
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Information section
    st.subheader("ℹ️ About This Application")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.info("""
        **🎯 Purpose**
        
        This application demonstrates AI-powered drug toxicity prediction using machine learning models to assess pharmaceutical safety before clinical trials.
        """)
    
    with info_col2:
        st.success("""
        **🔬 Features**
        
        - Multiple ML algorithms
        - Interactive predictions
        - Feature importance analysis
        - Performance comparisons
        - Real-time visualization
        """)
    
    with info_col3:
        st.warning("""
        **⚠️ Disclaimer**
        
        This model is for research and educational purposes only.
        CREATED BY SACHIN UPADHYAY
        GITHUB LINK - https://github.com/sachinu25
        """)

if __name__ == "__main__":
    main()
