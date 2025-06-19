import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import time
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Fraud Detection Platform",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4ECDC4;
    }
    .fraud-alert {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
    }
    .safe-alert {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
    }
    .warning-alert {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
    }
    .stTab [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTab [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .feature-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    """Load the trained fraud detection model"""
    try:
        model = joblib.load('fraud_model.pkl')
        return model, None
    except FileNotFoundError:
        return None, "Model file 'fraud_model.pkl' not found. Please ensure the model is in the same directory."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def validate_data(df, model=None):
    """Validate the uploaded data"""
    errors = []
    warnings = []
    
    if df.empty:
        errors.append("The uploaded file is empty.")
        return errors, warnings
    
    if model is not None and hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
        target_columns = ['Class', 'class', 'target', 'Target', 'label', 'Label', 'fraud', 'Fraud', 'y']
        available_cols = [col for col in df.columns if col not in target_columns]
        
        missing_cols = set(expected_features) - set(available_cols)
        if missing_cols:
            warnings.append(f"Some expected features are missing: {', '.join(missing_cols)}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < len(df.columns):
        non_numeric = set(df.columns) - set(numeric_cols)
        warnings.append(f"Non-numeric columns detected: {', '.join(non_numeric)}. These will be excluded from prediction.")
    
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        warnings.append(f"Dataset contains {missing_count} missing values. These will be handled automatically.")
    
    return errors, warnings

def preprocess_data(df, model=None):
    """Preprocess the data for prediction"""
    target_columns = ['Class', 'class', 'target', 'Target', 'label', 'Label', 'fraud', 'Fraud', 'y']
    
    df_clean = df.copy()
    removed_cols = []
    for col in target_columns:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])
            removed_cols.append(col)
    
    numeric_df = df_clean.select_dtypes(include=[np.number])
    
    if model is not None and hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
        available_features = numeric_df.columns.tolist()
        
        missing_features = set(expected_features) - set(available_features)
        extra_features = set(available_features) - set(expected_features)
        
        if missing_features:
            st.warning(f"Missing expected features: {', '.join(missing_features)}")
        
        if extra_features:
            st.info(f"Extra features will be ignored: {', '.join(extra_features)}")
        
        common_features = [f for f in expected_features if f in available_features]
        if common_features:
            numeric_df = numeric_df[common_features]
        else:
            raise ValueError("No matching features found between the uploaded data and the trained model.")
    
    numeric_df = numeric_df.fillna(numeric_df.median())
    
    if removed_cols:
        st.info(f"Removed target columns: {', '.join(removed_cols)}")
    
    return numeric_df

def create_advanced_visualizations(df_with_predictions):
    """Create comprehensive visualizations"""
    
    # 1. Fraud Distribution with Enhanced Pie Chart
    fraud_counts = df_with_predictions['Prediction'].value_counts()
    fig_pie = go.Figure(data=[go.Pie(
        labels=['Genuine', 'Fraudulent'],
        values=[fraud_counts.get(0, 0), fraud_counts.get(1, 0)],
        hole=.3,
        marker_colors=['#4CAF50', '#F44336'],
        textinfo='label+percent+value'
    )])
    fig_pie.update_layout(title="Transaction Distribution", height=400)
    
    # 2. Fraud Probability Distribution with Risk Zones
    fig_hist = px.histogram(
        df_with_predictions, x='Fraud_Probability', nbins=50,
        title="Fraud Probability Distribution with Risk Zones"
    )
    fig_hist.add_vline(x=0.3, line_dash="dash", line_color="orange", 
                       annotation_text="Medium Risk")
    fig_hist.add_vline(x=0.7, line_dash="dash", line_color="red", 
                       annotation_text="High Risk")
    fig_hist.update_layout(height=400)
    
    # 3. Amount vs Fraud Probability Scatter (if Amount column exists)
    if 'Amount' in df_with_predictions.columns:
        fig_scatter = px.scatter(
            df_with_predictions.sample(min(5000, len(df_with_predictions))),
            x='Amount', y='Fraud_Probability',
            color='Prediction',
            color_discrete_map={0: '#4CAF50', 1: '#F44336'},
            title="Transaction Amount vs Fraud Probability",
            labels={'Prediction': 'Fraud Status'}
        )
        fig_scatter.update_layout(height=400)
    else:
        fig_scatter = None
    
    # 4. Time-based Analysis (if Time column exists)
    if 'Time' in df_with_predictions.columns:
        # Convert time to hours
        df_with_predictions['Hour'] = (df_with_predictions['Time'] / 3600) % 24
        hourly_fraud = df_with_predictions.groupby('Hour')['Prediction'].agg(['sum', 'count']).reset_index()
        hourly_fraud['fraud_rate'] = hourly_fraud['sum'] / hourly_fraud['count'] * 100
        
        fig_time = px.line(
            hourly_fraud, x='Hour', y='fraud_rate',
            title="Fraud Rate by Hour of Day",
            labels={'fraud_rate': 'Fraud Rate (%)'}
        )
        fig_time.update_layout(height=400)
    else:
        fig_time = None
    
    return fig_pie, fig_hist, fig_scatter, fig_time

def generate_fraud_report(df_with_predictions):
    """Generate comprehensive fraud analysis report"""
    
    total_transactions = len(df_with_predictions)
    fraud_count = sum(df_with_predictions['Prediction'])
    fraud_rate = fraud_count / total_transactions * 100
    
    # Risk categories
    high_risk = sum(df_with_predictions['Fraud_Probability'] > 0.7)
    medium_risk = sum((df_with_predictions['Fraud_Probability'] > 0.3) & 
                     (df_with_predictions['Fraud_Probability'] <= 0.7))
    low_risk = total_transactions - high_risk - medium_risk
    
    # Amount analysis (if available)
    if 'Amount' in df_with_predictions.columns:
        avg_fraud_amount = df_with_predictions[df_with_predictions['Prediction'] == 1]['Amount'].mean()
        avg_genuine_amount = df_with_predictions[df_with_predictions['Prediction'] == 0]['Amount'].mean()
        total_fraud_amount = df_with_predictions[df_with_predictions['Prediction'] == 1]['Amount'].sum()
    else:
        avg_fraud_amount = avg_genuine_amount = total_fraud_amount = None
    
    return {
        'total_transactions': total_transactions,
        'fraud_count': fraud_count,
        'fraud_rate': fraud_rate,
        'high_risk': high_risk,
        'medium_risk': medium_risk,
        'low_risk': low_risk,
        'avg_fraud_amount': avg_fraud_amount,
        'avg_genuine_amount': avg_genuine_amount,
        'total_fraud_amount': total_fraud_amount
    }

def create_model_performance_tab(df_with_predictions):
    """Create model performance analysis tab"""
    
    st.subheader("🎯 Model Performance Metrics")
    
    # If we have actual labels, show performance metrics
    if 'Actual_Class' in df_with_predictions.columns:
        y_true = df_with_predictions['Actual_Class']
        y_pred = df_with_predictions['Prediction']
        y_prob = df_with_predictions['Fraud_Probability']
        
        # Classification Report
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Classification Report:**")
            report = classification_report(y_true, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3))
        
        with col2:
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                              title="Confusion Matrix",
                              labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig_cm, use_container_width=True)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC Curve (AUC = {roc_auc:.3f})'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier'))
        fig_roc.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    
    else:
        st.info("Actual fraud labels not available. Upload data with 'Class' column to see performance metrics.")

def create_feature_analysis_tab(df_with_predictions, original_data):
    """Create feature importance and analysis tab"""
    
    st.subheader("🔍 Feature Analysis")
    
    # Feature statistics for fraud vs genuine
    numeric_cols = original_data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        feature_stats = []
        
        for col in numeric_cols[:10]:  # Show top 10 features
            if col not in ['Class', 'Prediction', 'Fraud_Probability']:
                fraud_mean = original_data[df_with_predictions['Prediction'] == 1][col].mean()
                genuine_mean = original_data[df_with_predictions['Prediction'] == 0][col].mean()
                fraud_std = original_data[df_with_predictions['Prediction'] == 1][col].std()
                genuine_std = original_data[df_with_predictions['Prediction'] == 0][col].std()
                
                feature_stats.append({
                    'Feature': col,
                    'Fraud_Mean': fraud_mean,
                    'Genuine_Mean': genuine_mean,
                    'Fraud_Std': fraud_std,
                    'Genuine_Std': genuine_std,
                    'Difference': abs(fraud_mean - genuine_mean)
                })
        
        feature_df = pd.DataFrame(feature_stats).sort_values('Difference', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Feature Statistics Comparison:**")
            st.dataframe(feature_df.round(4))
        
        with col2:
            # Feature importance visualization
            if len(feature_df) > 0:
                fig_importance = px.bar(
                    feature_df.head(10), 
                    x='Difference', 
                    y='Feature',
                    title="Feature Importance (Mean Difference)",
                    orientation='h'
                )
                st.plotly_chart(fig_importance, use_container_width=True)

def create_real_time_monitoring_tab():
    """Create real-time monitoring simulation"""
    
    st.subheader("📡 Real-Time Monitoring Simulation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Start Monitoring Simulation"):
            st.session_state.monitoring = True
    
    with col2:
        if st.button("⏸️ Pause Monitoring"):
            st.session_state.monitoring = False
    
    with col3:
        if st.button("🛑 Stop Monitoring"):
            st.session_state.monitoring = False
    
    if st.session_state.get('monitoring', False):
        # Simulate real-time transactions
        placeholder = st.empty()
        
        for i in range(10):
            # Simulate transaction data
            fake_transaction = {
                'Transaction_ID': f"TXN_{i+1:04d}",
                'Amount': np.random.exponential(50),
                'Fraud_Probability': np.random.beta(1, 10),
                'Status': 'Fraud' if np.random.beta(1, 10) > 0.5 else 'Genuine',
                'Timestamp': datetime.now() - timedelta(seconds=i*2)
            }
            
            with placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Transaction ID", fake_transaction['Transaction_ID'])
                with col2:
                    st.metric("Amount", f"${fake_transaction['Amount']:.2f}")
                with col3:
                    st.metric("Fraud Probability", f"{fake_transaction['Fraud_Probability']:.3f}")
                with col4:
                    color = "🔴" if fake_transaction['Status'] == 'Fraud' else "🟢"
                    st.metric("Status", f"{color} {fake_transaction['Status']}")
                
                if fake_transaction['Status'] == 'Fraud':
                    st.error(f"🚨 FRAUD ALERT: Transaction {fake_transaction['Transaction_ID']} flagged!")
            
            time.sleep(1)

def create_alert_system(df_with_predictions):
    """Create intelligent alert system"""
    
    st.subheader("🚨 Intelligent Alert System")
    
    # Configure alert thresholds
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_risk_threshold = st.slider("High Risk Threshold", 0.5, 1.0, 0.8, 0.05)
    with col2:
        amount_threshold = st.number_input("Large Amount Alert ($)", value=1000, step=100)
    with col3:
        velocity_threshold = st.number_input("Velocity Alert (transactions/hour)", value=10, step=1)
    
    # Generate alerts
    alerts = []
    
    # High-risk transaction alerts
    high_risk_transactions = df_with_predictions[
        df_with_predictions['Fraud_Probability'] > high_risk_threshold
    ]
    
    if len(high_risk_transactions) > 0:
        alerts.append({
            'Type': '🔴 High Risk',
            'Count': len(high_risk_transactions),
            'Message': f"{len(high_risk_transactions)} transactions with >{high_risk_threshold*100:.0f}% fraud probability",
            'Severity': 'Critical'
        })
    
    # Large amount alerts (if Amount column exists)
    if 'Amount' in df_with_predictions.columns:
        large_amount_fraud = df_with_predictions[
            (df_with_predictions['Amount'] > amount_threshold) & 
            (df_with_predictions['Prediction'] == 1)
        ]
        
        if len(large_amount_fraud) > 0:
            alerts.append({
                'Type': '💰 Large Amount',
                'Count': len(large_amount_fraud),
                'Message': f"{len(large_amount_fraud)} high-value fraudulent transactions",
                'Severity': 'High'
            })
    
    # Display alerts
    if alerts:
        for alert in alerts:
            if alert['Severity'] == 'Critical':
                st.error(f"{alert['Type']}: {alert['Message']}")
            elif alert['Severity'] == 'High':
                st.warning(f"{alert['Type']}: {alert['Message']}")
            else:
                st.info(f"{alert['Type']}: {alert['Message']}")
    else:
        st.success("✅ No critical alerts at this time")

def create_demo_data():
    """Create demo data for testing purposes"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic transaction data
    demo_data = pd.DataFrame({
        'V1': np.random.normal(0, 1, n_samples),
        'V2': np.random.normal(0, 1, n_samples),
        'V3': np.random.normal(0, 1, n_samples),
        'V4': np.random.normal(0, 1, n_samples),
        'V5': np.random.normal(0, 1, n_samples),
        'Amount': np.random.exponential(50, n_samples),
        'Time': np.random.randint(0, 86400, n_samples),
        'Class': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
    })
    
    return demo_data

def main():
    # Initialize session state
    if 'monitoring' not in st.session_state:
        st.session_state.monitoring = False
    
    # Header
    st.markdown('<h1 class="main-header">🔒 Advanced Credit Card Fraud Detection Platform</h1>', unsafe_allow_html=True)
    
    # Load model
    model, error = load_model()
    
    if error:
        st.error(error)
        st.info("To use this app, please ensure you have a trained model saved as 'fraud_model.pkl' in the same directory.")
        
        # Offer demo mode
        st.markdown("---")
        st.subheader("🧪 Demo Mode")
        st.info("Try the demo with synthetic data to explore the platform features.")
        
        if st.button("🚀 Load Demo Data"):
            demo_data = create_demo_data()
            st.session_state.demo_data = demo_data
            st.success("Demo data loaded! The platform will now work with synthetic data for demonstration purposes.")
            st.rerun()
        
        if 'demo_data' not in st.session_state:
            return
        else:
            # Create a simple demo model for testing
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            
            demo_data = st.session_state.demo_data
            X = demo_data.drop('Class', axis=1)
            y = demo_data['Class']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            st.info("Demo model created and trained on synthetic data.")
    
    # Enhanced Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        st.subheader("Model Settings")
        confidence_threshold = st.slider(
            "Fraud Probability Threshold", 0.0, 1.0, 0.5, 0.05,
            help="Transactions above this threshold are flagged as fraud"
        )
        
        # Display settings
        st.subheader("Display Settings")
        show_probabilities = st.checkbox("Show detailed probabilities", value=True)
        max_display_rows = st.selectbox("Max rows to display", [10, 25, 50, 100, "All"], index=1)
        
        # Analysis settings
        st.subheader("Analysis Settings")
        enable_alerts = st.checkbox("Enable Alert System", value=True)
        auto_refresh = st.checkbox("Auto-refresh monitoring", value=False)
        
        # Export settings
        st.subheader("Export Settings")
        include_raw_data = st.checkbox("Include raw data in export", value=True)
        export_format = st.selectbox("Export format", ["CSV", "JSON", "Excel"])
    
    # Main content with tabs
    st.markdown("""
    ### 📊 Advanced Fraud Detection Analysis
    Upload your transaction data to get comprehensive fraud detection insights with advanced analytics.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload CSV with transaction data. Supports files up to 200MB."
    )
    
    # Check if we have demo data and no uploaded file
    if uploaded_file is None and 'demo_data' in st.session_state:
        st.info("Using demo data. Upload your own CSV file to analyze real transaction data.")
        data = st.session_state.demo_data
        uploaded_file = "demo"  # Flag to indicate we're using demo data
    
    if uploaded_file is not None:
        try:
            # Load and validate data
            with st.spinner("Loading and validating data..."):
                if uploaded_file == "demo":
                    data = st.session_state.demo_data
                else:
                    data = pd.read_csv(uploaded_file)
                errors, warnings = validate_data(data, model)
            
            if errors:
                for error in errors:
                    st.error(error)
                return
            
            if warnings:
                for warning in warnings:
                    st.warning(warning)
            
            # Basic metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Transactions", f"{len(data):,}")
            with col2:
                st.metric("Features", len(data.columns))
            with col3:
                if uploaded_file != "demo":
                    st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
                else:
                    st.metric("Data Type", "Demo")
            with col4:
                st.metric("Data Quality", f"{(1 - data.isnull().sum().sum() / (len(data) * len(data.columns)))*100:.1f}%")
            
            # Process data and make predictions
            with st.spinner("Processing data and making predictions..."):
                processed_data = preprocess_data(data, model)
                
                if processed_data.empty:
                    st.error("No numeric columns found for prediction.")
                    return
                
                predictions = model.predict(processed_data)
                prediction_proba = model.predict_proba(processed_data)[:, 1]
                
                # Create enhanced results dataframe
                results_df = data.copy()
                results_df['Prediction'] = predictions
                results_df['Fraud_Probability'] = prediction_proba
                results_df['Risk_Level'] = pd.cut(
                    prediction_proba,
                    bins=[0, 0.3, 0.7, 1.0],
                    labels=['Low', 'Medium', 'High']
                )
                
                # Add actual class if it exists (for performance evaluation)
                if 'Class' in data.columns:
                    results_df['Actual_Class'] = data['Class']
            
            # Create tabs for different analysis views
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Overview", "📈 Advanced Analytics", "🎯 Model Performance", 
                "🔍 Feature Analysis", "🚨 Alert System", "📡 Real-time Monitoring"
            ])
            
            with tab1:
                # Overview tab (original functionality enhanced)
                st.subheader("🎯 Detection Results Overview")
                
                # Enhanced summary metrics
                report = generate_fraud_report(results_df)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Fraudulent", f"{report['fraud_count']:,}", 
                             delta=f"{report['fraud_rate']:.2f}%")
                with col2:
                    st.metric("Genuine", f"{report['total_transactions'] - report['fraud_count']:,}")
                with col3:
                    st.metric("High Risk", f"{report['high_risk']:,}")
                with col4:
                    st.metric("Medium Risk", f"{report['medium_risk']:,}")
                with col5:
                    st.metric("Low Risk", f"{report['low_risk']:,}")
                
                # Enhanced visualizations
                fig_pie, fig_hist, fig_scatter, fig_time = create_advanced_visualizations(results_df)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_pie, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                if fig_scatter:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    if fig_time:
                        with col2:
                            st.plotly_chart(fig_time, use_container_width=True)
                
                # Top risky transactions
                st.subheader("⚠️ Highest Risk Transactions")
                risky_transactions = results_df.nlargest(10, 'Fraud_Probability')
                
                display_cols = ['Fraud_Probability', 'Risk_Level']
                if 'Amount' in results_df.columns:
                    display_cols.append('Amount')
                if 'Time' in results_df.columns:
                    display_cols.append('Time')
                
                # Combine display and original columns safely
                combined_cols = display_cols + list(data.columns[:5])
                unique_cols = list(dict.fromkeys(combined_cols))
                # Only keep columns that exist in risky_transactions
                existing_cols = [col for col in unique_cols if col in risky_transactions.columns]
                st.dataframe(risky_transactions[existing_cols].round(4))
            
            with tab2:
                # Advanced Analytics tab
                st.subheader("📈 Advanced Analytics Dashboard")
                
                # Risk distribution analysis
                risk_dist = results_df['Risk_Level'].value_counts()
                fig_risk = px.bar(x=risk_dist.index, y=risk_dist.values, 
                                 title="Risk Level Distribution",
                                 color=risk_dist.index,
                                 color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
                st.plotly_chart(fig_risk, use_container_width=True)
                
                # Statistical analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Fraud Probability Statistics:**")
                    prob_stats = results_df['Fraud_Probability'].describe()
                    st.dataframe(prob_stats.to_frame().T.round(4))
                
                with col2:
                    if 'Amount' in results_df.columns:
                        st.write("**Amount Analysis:**")
                        amount_stats = {
                            'Avg Fraud Amount': results_df[results_df['Prediction']==1]['Amount'].mean(),
                            'Avg Genuine Amount': results_df[results_df['Prediction']==0]['Amount'].mean(),
                            'Max Fraud Amount': results_df[results_df['Prediction']==1]['Amount'].max(),
                            'Total Fraud Value': results_df[results_df['Prediction']==1]['Amount'].sum()
                        }
                        st.json(amount_stats)
            
            with tab3:
                create_model_performance_tab(results_df)
            
            with tab4:
                create_feature_analysis_tab(results_df, data)
            
            with tab5:
                if enable_alerts:
                    create_alert_system(results_df)
                else:
                    st.info("Alert system is currently disabled. Enable it in the sidebar settings.")
            
            with tab6:
                create_real_time_monitoring_tab()
            
            # Data export functionality
            st.markdown("---")
            st.subheader("📤 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if include_raw_data:
                    export_df = results_df
                else:
                    export_cols = ['Prediction', 'Fraud_Probability', 'Risk_Level']
                    if 'Actual_Class' in results_df.columns:
                        export_cols.append('Actual_Class')
                    export_df = results_df[export_cols]
                
                if export_format == "CSV":
                    csv = export_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Results as CSV",
                        data=csv,
                        file_name="fraud_detection_results.csv",
                        mime="text/csv"
                    )
                elif export_format == "JSON":
                    json = export_df.to_json(orient='records')
                    st.download_button(
                        "Download Results as JSON",
                        data=json,
                        file_name="fraud_detection_results.json",
                        mime="application/json"
                    )
                else:  # Excel
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        export_df.to_excel(writer, index=False, sheet_name='Fraud_Results')
                    st.download_button(
                        "Download Results as Excel",
                        data=output.getvalue(),
                        file_name="fraud_detection_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with col2:
                st.write("**Export Options:**")
                st.info(f"Format: {export_format}")
                st.info(f"Include raw data: {'Yes' if include_raw_data else 'No'}")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
    
    else:
        # Show demo information when no file is uploaded
        st.markdown("""
        ### Welcome to the Fraud Detection Platform
        
        This advanced platform helps you detect fraudulent transactions using machine learning.
        
        **To get started:**
        1. Upload a CSV file containing transaction data
        2. Or try our **demo mode** with synthetic data
        
        The platform provides:
        - Real-time fraud detection
        - Comprehensive analytics
        - Detailed visualizations
        - Alert system for suspicious transactions
        """)
        
        if st.button("🚀 Try Demo Mode"):
            demo_data = create_demo_data()
            st.session_state.demo_data = demo_data
            st.success("Demo data loaded! The platform will now work with synthetic data for demonstration purposes.")
            st.rerun()

if __name__ == "__main__":
    main()
