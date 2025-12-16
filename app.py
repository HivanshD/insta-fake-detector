"""
Instagram Fake Profile Detector
Professional ML-Powered Web Application

"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px


# PAGE CONFIG


st.set_page_config(
    page_title="Instagram Fake Profile Detector",
    layout="wide"
)


# FEATURE ENGINEERING FUNCTION


def engineer_features(df):
    """
    Create derived features from raw account metadata.
    This MUST match the feature engineering used during training.
    """
    df = df.copy()
    
    df['follower_ratio'] = df['#followers'] / (df['#follows'] + 1)
    df['following_ratio'] = df['#follows'] / (df['#followers'] + 1)
    df['engagement_rate'] = df['#posts'] / (df['#followers'] + 1)
    df['posts_per_following'] = df['#posts'] / (df['#follows'] + 1)
    df['profile_completeness'] = (
        df['profile pic'] + 
        df['external URL'] + 
        (df['description length'] > 0).astype(int)
    ) / 3
    df['suspicious_username'] = (
        (df['name==username'] == 1) & 
        (df['nums/length username'] > 0.3)
    ).astype(int)
    df['very_sparse_profile'] = (
        (df['profile pic'] == 0) & 
        (df['description length'] == 0) &
        (df['external URL'] == 0)
    ).astype(int)
    df['log_followers'] = np.log1p(df['#followers'])
    df['log_follows'] = np.log1p(df['#follows'])
    df['log_posts'] = np.log1p(df['#posts'])
    df['has_posts'] = (df['#posts'] > 0).astype(int)
    df['has_bio'] = (df['description length'] > 0).astype(int)
    
    return df


# MODEL LOADING


@st.cache_resource
def load_model_artifacts():
    """Load trained model and threshold"""
    try:
        artifacts_dir = Path("artifacts")
        model = joblib.load(artifacts_dir / "final_pipe.joblib")
        with open(artifacts_dir / "threshold.txt", "r") as f:
            threshold = float(f.read().strip())
        return model, threshold
    except FileNotFoundError as e:
        st.error(f"Model artifacts not found: {str(e)}")
        st.info("Please ensure the 'artifacts/' folder contains: final_pipe.joblib, threshold.txt")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model, THRESHOLD = load_model_artifacts()


# VISUALIZATION FUNCTIONS


def create_gauge_chart(probability, threshold):
    """Create a gauge chart for fake probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Fake Probability", 'font': {'size': 20}},
        delta={'reference': threshold * 100, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "darkred" if probability >= threshold else "darkgreen"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#D1FAE5'},
                {'range': [40, 70], 'color': '#FEF3C7'},
                {'range': [70, 100], 'color': '#FEE2E2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_feature_importance_chart():
    """Create feature importance bar chart"""
    features = ['log_followers', '#followers', '#posts', 'log_posts', 'nums/length username',
                'profile_completeness', 'following_ratio', 'posts_per_following', 
                'follower_ratio', 'very_sparse_profile']
    importance = [0.1586, 0.1567, 0.1106, 0.1077, 0.0771, 
                  0.0707, 0.0503, 0.0465, 0.0443, 0.0337]
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker=dict(
            color=importance,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=[f'{i:.1%}' for i in importance],
        textposition='outside'
    ))
    fig.update_layout(
        title='Top 10 Feature Importances',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=400,
        margin=dict(l=150, r=50, t=50, b=50),
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig

def create_confusion_matrix():
    """Create confusion matrix heatmap"""
    cm = np.array([[51, 9], [1, 59]])
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Real', 'Predicted Fake'],
        y=['Actual Real', 'Actual Fake'],
        colorscale='RdYlGn_r',
        text=[[f'TN<br>{cm[0,0]}', f'FP<br>{cm[0,1]}'], 
              [f'FN<br>{cm[1,0]}', f'TP<br>{cm[1,1]}']],
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True,
        colorbar=dict(title="Count")
    ))
    fig.update_layout(
        title='Test Set Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=400,
        margin=dict(l=80, r=80, t=80, b=80)
    )
    return fig

def create_metrics_comparison():
    """Create metrics comparison chart"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    values = [91.67, 86.76, 98.33, 92.19, 99.00]
    
    fig = go.Figure(go.Bar(
        x=metrics,
        y=values,
        marker=dict(
            color=values,
            colorscale='Blues',
            showscale=False
        ),
        text=[f'{v:.2f}%' for v in values],
        textposition='outside'
    ))
    fig.update_layout(
        title='Test Set Performance Metrics',
        yaxis_title='Score (%)',
        height=400,
        yaxis=dict(range=[0, 105]),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

def create_radar_chart(input_features):
    """Create radar chart for input profile analysis"""
    categories = ['Followers (scaled)', 'Following (scaled)', 'Posts (scaled)', 
                  'Profile Completeness', 'Engagement Rate']
    
    max_followers = 10000
    max_follows = 5000
    max_posts = 500
    
    values = [
        min(input_features['#followers'] / max_followers * 100, 100),
        min(input_features['#follows'] / max_follows * 100, 100),
        min(input_features['#posts'] / max_posts * 100, 100),
        input_features['profile_completeness'] * 100,
        min(input_features['engagement_rate'] * 1000, 100)
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        marker=dict(color='#667eea'),
        line=dict(color='#764ba2')
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=False,
        title='Profile Behavioral Pattern',
        height=400,
        margin=dict(l=80, r=80, t=80, b=80)
    )
    return fig


# CUSTOM CSS


st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1F2937;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
        padding: 1rem 1.5rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #FEE2E2;
        border-left: 4px solid #EF4444;
        padding: 1rem 1.5rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #F3F4F6;
        border-left: 4px solid #3B82F6;
        padding: 1rem 1.5rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem 1.5rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)


# SIDEBAR


with st.sidebar:
    st.markdown("### Model Information")
    
    st.markdown("**Algorithm**")
    st.markdown("Random Forest Classifier")
    
    st.markdown("**Features**")
    st.markdown("23 total (11 raw + 12 engineered)")
    
    st.markdown("**Training**")
    st.markdown("574 samples, 5-fold CV")
    
    st.markdown("---")
    
    st.markdown("### Test Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "91.67%", delta="High")
        st.metric("Precision", "86.76%", delta="Good")
    with col2:
        st.metric("Recall", "98.33%", delta="Excellent")
        st.metric("F1", "92.19%", delta="High")
    
    st.metric("ROC-AUC", "99.00%", delta="Excellent")
    
    st.markdown("---")
    
    st.markdown("### Model Details")
    with st.expander("Hyperparameters"):
        st.code("""
n_estimators: 153
max_depth: None
max_features: log2
min_samples_split: 7
min_samples_leaf: 6
class_weight: None
        """, language="python")
    
    st.markdown("**Threshold**")
    st.code(f"{THRESHOLD}", language="python")
    st.caption("Optimized for 98% recall")
    
    st.markdown("---")
    st.markdown("### Quick Stats")
    st.markdown("**Confusion Matrix:**")
    st.markdown("• TP: 59 | TN: 51")
    st.markdown("• FP: 9 | FN: 1")


# MAIN CONTENT


st.markdown('<h1 class="main-header">Instagram Fake Profile Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ML-Powered Detection System | 98.33% Recall | 99% ROC-AUC</p>', unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Profile Analysis", "Model Performance", "About"])


# TAB 1: PROFILE ANALYSIS


with tab1:
    st.markdown("""
        <div class="info-box">
        <strong>How It Works:</strong> Enter Instagram account metadata below. The ML model analyzes 
        23 behavioral features to detect fake accounts with 98.33% recall.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### Account Metadata Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Profile Settings**")
        profile_pic = st.selectbox("Profile Picture", options=[1, 0], format_func=lambda x: "Present" if x == 1 else "Missing")
        private = st.selectbox("Account Privacy", options=[0, 1], format_func=lambda x: "Private" if x == 1 else "Public")
        external_url = st.selectbox("External URL", options=[0, 1], format_func=lambda x: "Present" if x == 1 else "Missing")
        name_eq_username = st.selectbox("Name = Username", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        st.markdown("**Engagement Metrics**")
        followers = st.number_input("Followers", min_value=0, value=500, step=10)
        follows = st.number_input("Following", min_value=0, value=300, step=10)
        posts = st.number_input("Posts", min_value=0, value=50, step=1)
        description_length = st.number_input("Bio Length", min_value=0, value=80, step=1)

    with col3:
        st.markdown("**Username Analysis**")
        nums_len_username = st.slider("Numeric Density (Username)", 0.0, 1.0, 0.15, 0.01)
        fullname_words = st.number_input("Name Word Count", min_value=0, value=2, step=1)
        nums_len_fullname = st.slider("Numeric Density (Name)", 0.0, 1.0, 0.0, 0.01)

    st.markdown("---")
    st.markdown("### Computed Behavioral Metrics")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    follower_ratio = follows / (followers + 1) if followers > 0 else follows
    engagement_rate_display = posts / (followers + 1) if followers > 0 else 0
    profile_completeness_val = (profile_pic + external_url + (1 if description_length > 0 else 0)) / 3
    following_ratio_val = followers / (follows + 1) if follows > 0 else 0

    with metric_col1:
        st.metric("Following/Follower", f"{follower_ratio:.3f}")
    with metric_col2:
        st.metric("Engagement Rate", f"{engagement_rate_display:.4f}")
    with metric_col3:
        st.metric("Profile Complete", f"{profile_completeness_val:.1%}")
    with metric_col4:
        st.metric("Follower/Following", f"{following_ratio_val:.3f}")

    input_data = pd.DataFrame([{
        "profile pic": profile_pic,
        "nums/length username": nums_len_username,
        "fullname words": fullname_words,
        "nums/length fullname": nums_len_fullname,
        "name==username": name_eq_username,
        "description length": description_length,
        "external URL": external_url,
        "private": private,
        "#posts": posts,
        "#followers": followers,
        "#follows": follows
    }])

    st.markdown("---")
    if st.button("Analyze Account", type="primary"):
        with st.spinner("Running ML analysis..."):
            try:
                input_engineered = engineer_features(input_data)
                prediction_proba = model.predict_proba(input_engineered)[:, 1][0]
                prediction = int(prediction_proba >= THRESHOLD)
                
                st.markdown("---")
                st.markdown("### Analysis Results")
                
                result_col1, result_col2 = st.columns([1, 1])
                
                with result_col1:
                    if prediction == 1:
                        st.markdown(f"""
                            <div class="danger-box">
                            <h3 style="margin-top: 0; color: #DC2626;">SUSPICIOUS ACCOUNT DETECTED</h3>
                            <p style="margin-bottom: 0;">
                            <strong>Classification:</strong> Fake (Class 1)<br>
                            <strong>Confidence:</strong> {prediction_proba*100:.2f}%<br>
                            <strong>Risk Level:</strong> {'HIGH' if prediction_proba >= 0.8 else 'MODERATE' if prediction_proba >= 0.6 else 'LOW-MODERATE'}
                            </p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="success-box">
                            <h3 style="margin-top: 0; color: #059669;">LEGITIMATE ACCOUNT</h3>
                            <p style="margin-bottom: 0;">
                            <strong>Classification:</strong> Real (Class 0)<br>
                            <strong>Confidence:</strong> {(1-prediction_proba)*100:.2f}%<br>
                            <strong>Assessment:</strong> Normal behavior
                            </p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    col_r1, col_r2, col_r3 = st.columns(3)
                    with col_r1:
                        st.metric("Fake Prob", f"{prediction_proba*100:.1f}%")
                    with col_r2:
                        st.metric("Real Prob", f"{(1-prediction_proba)*100:.1f}%")
                    with col_r3:
                        st.metric("Verdict", "Fake" if prediction else "Real")
                
                with result_col2:
                    st.plotly_chart(create_gauge_chart(prediction_proba, THRESHOLD), use_container_width=True)
                
                st.markdown("---")
                st.markdown("### Profile Behavior Analysis")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    input_features = {
                        '#followers': followers,
                        '#follows': follows,
                        '#posts': posts,
                        'profile_completeness': profile_completeness_val,
                        'engagement_rate': engagement_rate_display
                    }
                    st.plotly_chart(create_radar_chart(input_features), use_container_width=True)
                
                with viz_col2:
                    feature_data = pd.DataFrame({
                        'Feature': ['Follower Ratio', 'Engagement Rate', 'Profile Completeness', 
                                    'Following Ratio', 'Has Bio', 'Has Posts'],
                        'Value': [
                            follower_ratio,
                            engagement_rate_display,
                            profile_completeness_val,
                            following_ratio_val,
                            1 if description_length > 0 else 0,
                            1 if posts > 0 else 0
                        ]
                    })
                    
                    fig = go.Figure(go.Bar(
                        x=feature_data['Value'],
                        y=feature_data['Feature'],
                        orientation='h',
                        marker=dict(color='#667eea')
                    ))
                    fig.update_layout(
                        title='Key Behavioral Features',
                        xaxis_title='Value',
                        height=400,
                        margin=dict(l=150, r=20, t=50, b=50)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("View Technical Details"):
                    st.markdown("**Model Configuration**")
                    st.code(f"""
Decision Threshold: {THRESHOLD:.4f}
Raw Score: {prediction_proba:.6f}
Model: Random Forest (153 trees)
Features Used: 23
Test Recall: 98.33%
Test Precision: 86.76%
                    """)
                    
                    st.markdown("**Top Engineered Features (Sample)**")
                    sample_features = input_engineered[[
                        'follower_ratio', 'engagement_rate', 'profile_completeness',
                        'log_followers', 'suspicious_username', 'very_sparse_profile'
                    ]].T
                    sample_features.columns = ['Value']
                    st.dataframe(sample_features, use_container_width=True)
                
                st.markdown("""
                    <div class="warning-box">
                    <strong>Disclaimer:</strong> This is an ML prediction based on metadata only. 
                    Human review recommended for critical decisions. The model may miss sophisticated 
                    fake accounts (1.67% false negative rate on test set).
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


# TAB 2: MODEL PERFORMANCE


with tab2:
    st.markdown("### Model Performance Dashboard")
    
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        st.plotly_chart(create_metrics_comparison(), use_container_width=True)
    
    with perf_col2:
        st.plotly_chart(create_confusion_matrix(), use_container_width=True)
    
    st.markdown("---")
    st.plotly_chart(create_feature_importance_chart(), use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Key Performance Insights")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        st.markdown("""
        **Excellent Detection**
        - 98.33% Recall
        - Only 1 fake missed
        - 59 out of 60 caught
        """)
    
    with insight_col2:
        st.markdown("""
        **Balanced Precision**
        - 86.76% Precision
        - 9 false positives
        - Manageable review load
        """)
    
    with insight_col3:
        st.markdown("""
        **Near-Perfect AUC**
        - 99.00% ROC-AUC
        - Excellent discrimination
        - Strong generalization
        """)


# TAB 3: ABOUT


with tab3:
    st.markdown("### About This Project")
    
    st.markdown("""
    This Instagram Fake Profile Detector uses machine learning to identify fake accounts based on account metadata. 
    """)
    
    st.markdown("---")
    st.markdown("### Methodology")
    
    col_about1, col_about2 = st.columns(2)
    
    with col_about1:
        st.markdown("""
        **Data & Features**
        - Dataset: 694 Instagram profiles
        - Split: 574 training, 120 test
        - Features: 11 raw + 12 engineered
        - Target: Binary (Real=0, Fake=1)
        """)
        
        st.markdown("""
        **Model Pipeline**
        1. Feature engineering (ratios, logs, indicators)
        2. Standard scaling (mean=0, std=1)
        3. Random Forest classification (153 trees)
        4. Threshold optimization (recall-focused)
        """)
    
    with col_about2:
        st.markdown("""
        **Key Features**
        - log_followers (15.86%)
        - #followers (15.67%)
        - #posts (11.06%)
        - profile_completeness (7.07%)
        - Engineered features: 57.3% importance
        """)
        
        st.markdown("""
        **Performance Highlights**
        - 5-fold CV: 92.34% ± 1.93% accuracy
        - Minimal overfitting (3.44% gap)
        - Threshold 0.40 optimized for recall
        - Production-ready reliability
        """)
    
    st.markdown("---")
    st.markdown("---")
    st.markdown("### Technical Details")
    
    with st.expander("Feature Engineering"):
        st.markdown("""
        **Engineered Features (12 total):**
        1. **Ratios:** follower_ratio, following_ratio
        2. **Engagement:** engagement_rate, posts_per_following
        3. **Completeness:** profile_completeness (0-1 scale)
        4. **Patterns:** suspicious_username, very_sparse_profile
        5. **Log Transforms:** log_followers, log_follows, log_posts
        6. **Binary:** has_posts, has_bio
        """)
    
    with st.expander("Model Architecture"):
        st.markdown("""
        **Random Forest Hyperparameters:**
        - n_estimators: 153
        - max_depth: None (unlimited)
        - max_features: log2
        - min_samples_split: 7
        - min_samples_leaf: 6
        - class_weight: None
        - random_state: 42
        """)
    
    with st.expander("Evaluation Metrics"):
        st.markdown("""
        **Test Set Results (120 samples):**
        - Accuracy: 91.67%
        - Precision: 86.76%
        - Recall: 98.33% 
        - F1 Score: 92.19%
        - ROC-AUC: 99.00% 
        
        **Confusion Matrix:**
        - True Positives: 59
        - True Negatives: 51
        - False Positives: 9
        - False Negatives: 1
        """)

st.markdown("---")







