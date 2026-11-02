import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path

current_dir = Path(__file__).resolve().parent

# Set page configuration
st.set_page_config(
    page_title="Credit Borrower Segmentation & Risk Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Blue Header & Mild Grey / Off-White Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Font Family */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1E293B;
    }

    /* Main App Background (Mild Grey / Off-White) */
    .stApp {
        background-color: #F4F6F9;
    }

    /* Sidebar Styling (Coordinated Dark Navy/Grey) */
    [data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #CBD5E1;
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .css-10trblm {
        color: #F8FAFC !important;
    }

    /* Premium Header Banner Container */
    .header-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15);
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #93C5FD;
        font-weight: 400;
        margin: 0;
    }

    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
    }
    div[data-testid="stMetric"] label {
        color: #64748B !important;
        font-weight: 500;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 700;
    }

    /* Insight Boxes */
    .insight-box {
        background-color: #FFFFFF;
        border-left: 4px solid #2563EB;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        color: #1E293B;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }
    .insight-box h4 {
        color: #1E3A8A;
        margin-top: 0;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: transparent;
        padding-top: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        background-color: #E2E8F0;
        border-radius: 8px 8px 0px 0px;
        color: #475569;
        font-weight: 500;
        border: 1px solid #CBD5E1;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: #FFFFFF !important;
        font-weight: 600;
        border-color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# Data loading with caching
@st.cache_data(ttl=3600)
def load_credit_data():
    possible_paths = [
        r"D:\Sasi\GUVI HCL AI-ML\Credit_Segmentation_Project\dataset\cleaned\cleaned_loans_granting_model_dataset.csv", # Original local absolute path
        "dataset/cleaned/cleaned_loans_granting_model_dataset.csv", # Relative path structure
        "cleaned_loans_granting_model_dataset.csv"                  # Root fallback path
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
            
    if not file_path:
        st.error("Dataset file not found. Please ensure 'cleaned_loans_granting_model_dataset.csv' is available locally or uploaded to your repository.")
        st.stop()
    
    df = pd.read_csv(file_path)
    
    df['issue_d'] = pd.to_datetime(df['issue_d'])
    df['issue_year'] = df['issue_d'].dt.year
    
    fico_bins = [600, 650, 700, 750, 800, 850]
    fico_labels = ['600-650', '650-700', '700-750', '750-800', '800+']
    df['fico_bin'] = pd.cut(df['fico_n'], bins=fico_bins, labels=fico_labels)
    
    dti_bins = [0, 10, 20, 30, 40, 50, 100]
    dti_labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50%+']
    df['dti_bin'] = pd.cut(df['dti_n'], bins=dti_bins, labels=dti_labels)
    
    loan_bins = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]
    loan_labels = ['$0-5K', '$5K-10K', '$10K-15K', '$15K-20K', '$20K-25K', '$25K-30K', '$30K-35K', '$35K-40K']
    df['loan_bin'] = pd.cut(df['loan_amnt'], bins=loan_bins, labels=loan_labels)
    
    conditions = [
        (df['fico_n'] >= 720) & (df['dti_n'] <= 18) & (df['revenue'] >= 75000),
        (df['fico_n'] >= 680) & (df['dti_n'] <= 25),
        (df['purpose'] == 'small_business') | (df['dti_n'] > 25) | (df['fico_n'] < 660),
        (df['revenue'] >= 100000) & (df['loan_amnt'] >= 25000)
    ]
    choices = ['Prime / Low Risk', 'Near-Prime / Core', 'High Risk / Speculative', 'High Value / Large Capital']
    df['borrower_segment'] = np.select(conditions, choices, default='Standard / Medium Risk')
    
    return df

try:
    df = load_credit_data()
except Exception as e:
    st.error(f"Error loading dataset. Please verify path and structure: {e}")
    st.stop()

# Header Banner Section (Dark Blue Background)
st.markdown("""
<div class="header-container">
    <div class="main-header">Credit Risk & Borrower Segmentation</div>
    <div class="sub-header">Executive Portfolio Intelligence & Behavioral Clustering Dashboard</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Filter Panel
st.sidebar.markdown('<h2 style="color: white; margin-bottom: 10px;">🔍 Global Filters</h2>', unsafe_allow_html=True)

selected_purposes = st.sidebar.multiselect(
    "**Loan Purpose**",
    options=sorted(df['purpose'].dropna().unique().tolist()),
    default=[]
)

selected_home = st.sidebar.multiselect(
    "**Home Ownership**",
    options=sorted(df['home_ownership_n'].dropna().unique().tolist()),
    default=[]
)

selected_segments = st.sidebar.multiselect(
    "**Borrower Segment**",
    options=sorted(df['borrower_segment'].unique().tolist()),
    default=[]
)

fico_range = st.sidebar.slider(
    "**FICO Score Range**",
    min_value=int(df['fico_n'].min()),
    max_value=int(df['fico_n'].max()),
    value=(int(df['fico_n'].min()), int(df['fico_n'].max()))
)

# Apply Filters
filtered_df = df.copy()

if selected_purposes:
    filtered_df = filtered_df[filtered_df['purpose'].isin(selected_purposes)]
if selected_home:
    filtered_df = filtered_df[filtered_df['home_ownership_n'].isin(selected_home)]
if selected_segments:
    filtered_df = filtered_df[filtered_df['borrower_segment'].isin(selected_segments)]

filtered_df = filtered_df[
    (filtered_df['fico_n'] >= fico_range[0]) & 
    (filtered_df['fico_n'] <= fico_range[1])
]

# Top KPI Summary Cards
total_loans = len(filtered_df)
total_volume = filtered_df['loan_amnt'].sum()
overall_default_rate = filtered_df['Default'].mean() * 100 if total_loans > 0 else 0.0
avg_fico = filtered_df['fico_n'].mean() if total_loans > 0 else 0.0
avg_dti = filtered_df['dti_n'].mean() if total_loans > 0 else 0.0

col1, col2, col3, col4, col5 = st.columns([1.25,1.6,1.0,1.0,1.0])

with col1:
    st.metric("Total Loans", f"{total_loans:,}")
with col2:
    st.metric("Portfolio Volume", f"${total_volume/1e6:,.2f}M")
with col3:
    st.metric("Default Rate", f"{overall_default_rate:.2f}%")
with col4:
    st.metric("Avg FICO", f"{avg_fico:.1f}")
with col5:
    st.metric("Avg DTI", f"{avg_dti:.2f}%")

st.markdown("---")

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Summary", 
    "🎯 Borrower Segments", 
    "⚠️ Default Analytics", 
    "💡 Strategic Insights"
])

# Helper to apply clean light-mode layout to plotly charts matching off-white background
def apply_light_theme(fig, height=400):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family="Inter", color="#1E293B")
    )
    return fig

# TAB 1: EXECUTIVE SUMMARY
with tab1:
    st.subheader("Portfolio Distribution Overview")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        purpose_df = filtered_df.groupby('purpose')['Default'].agg(['count', 'mean']).reset_index()
        purpose_df['mean'] = purpose_df['mean'] * 100
        purpose_df = purpose_df.sort_values(by='mean', ascending=True)
        
        fig_purpose = px.bar(
            purpose_df, x='mean', y='purpose', orientation='h',
            title='Default Rate (%) by Loan Purpose',
            labels={'mean': 'Default Rate (%)', 'purpose': 'Loan Purpose'},
            color='mean', color_continuous_scale='Reds'
        )
        fig_purpose.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_light_theme(fig_purpose, 450), use_container_width=True)
        
    with col_b:
        home_df = filtered_df.groupby('home_ownership_n')['Default'].agg(['count', 'mean']).reset_index()
        home_df['mean'] = home_df['mean'] * 100
        
        fig_home = px.bar(
            home_df, x='home_ownership_n', y='mean',
            title='Default Rate (%) by Home Ownership',
            labels={'mean': 'Default Rate (%)', 'home_ownership_n': 'Home Ownership'},
            color='mean', color_continuous_scale='Oranges'
        )
        fig_home.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_light_theme(fig_home, 450), use_container_width=True)
        
    st.markdown("---")
    
    col_c, col_d = st.columns(2)
    
    with col_c:
        corr_cols = ['revenue', 'dti_n', 'loan_amnt', 'fico_n', 'emp_length_n', 'Default']
        corr_matrix = filtered_df[corr_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix, text_auto='.2f', aspect="auto",
            color_continuous_scale='Blues',
            title='Feature Correlation Matrix'
        )
        st.plotly_chart(apply_light_theme(fig_corr, 400), use_container_width=True)
        
    with col_d:
        summary_segment = filtered_df.groupby('borrower_segment').agg(
            total_loans=('id', 'count'),
            avg_loan=('loan_amnt', 'mean'),
            avg_fico=('fico_n', 'mean'),
            default_rate=('Default', lambda x: x.mean() * 100)
        ).reset_index()
        
        fig_bubble = px.scatter(
            summary_segment, x='avg_fico', y='default_rate', size='total_loans',
            color='borrower_segment', hover_name='borrower_segment',
            title='Segment Risk Profile: FICO vs. Default Rate',
            labels={'avg_fico': 'Average FICO Score', 'default_rate': 'Default Rate (%)', 'total_loans': 'Volume'},
            size_max=50
        )
        st.plotly_chart(apply_light_theme(fig_bubble, 400), use_container_width=True)

# TAB 2: BORROWER SEGMENTS & CLUSTERS
with tab2:
    st.subheader("Unsupervised Cluster Segmentation Profiles")
    
    cluster_summary = filtered_df.groupby('borrower_segment').agg(
        Total_Borrowers=('id', 'count'),
        Avg_Revenue=('revenue', 'mean'),
        Avg_Loan_Amount=('loan_amnt', 'mean'),
        Avg_FICO=('fico_n', 'mean'),
        Avg_DTI=('dti_n', 'mean'),
        Default_Rate=('Default', lambda x: x.mean() * 100)
    ).reset_index()
    
    st.dataframe(
        cluster_summary.style.format({
            'Total_Borrowers': '{:,}',
            'Avg_Revenue': '${:,.2f}',
            'Avg_Loan_Amount': '${:,.2f}',
            'Avg_FICO': '{:.1f}',
            'Avg_DTI': '{:.2f}%',
            'Default_Rate': '{:.2f}%'
        }),
        use_container_width=True
    )
    
    col_e, col_f = st.columns(2)
    
    with col_e:
        fig_scatter = px.scatter(
            filtered_df.sample(min(5000, len(filtered_df))),
            x='fico_n', y='dti_n', color='borrower_segment', symbol='Default', opacity=0.7,
            title='Borrower Clusters: FICO Score vs. DTI Ratio',
            labels={'fico_n': 'FICO Score', 'dti_n': 'DTI Ratio (%)'}
        )
        st.plotly_chart(apply_light_theme(fig_scatter, 450), use_container_width=True)
        
    with col_f:
        fig_rev_loan = px.scatter(
            filtered_df.sample(min(5000, len(filtered_df))),
            x='revenue', y='loan_amnt', color='borrower_segment', log_x=True,
            title='Borrower Clusters: Annual Revenue vs. Loan Amount',
            labels={'revenue': 'Annual Revenue ($)', 'loan_amnt': 'Loan Amount ($)'}
        )
        st.plotly_chart(apply_light_theme(fig_rev_loan, 450), use_container_width=True)

# TAB 3: DEFAULT PATTERN ANALYTICS
with tab3:
    st.subheader("Multidimensional Default Drivers")
    
    col_g, col_h = st.columns(2)
    
    with col_g:
        fico_df = filtered_df.groupby('fico_bin', observed=False)['Default'].agg(['count', 'mean']).reset_index()
        fico_df['mean'] = fico_df['mean'] * 100
        
        fig_fico = px.bar(
            fico_df, x='fico_bin', y='mean', text_auto='.1f',
            title='Default Rate (%) by FICO Band',
            labels={'fico_bin': 'FICO Range', 'mean': 'Default Rate (%)'},
            color='mean', color_continuous_scale='Blues'
        )
        fig_fico.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_light_theme(fig_fico, 400), use_container_width=True)
        
    with col_h:
        dti_df = filtered_df.groupby('dti_bin', observed=False)['Default'].agg(['count', 'mean']).reset_index()
        dti_df['mean'] = dti_df['mean'] * 100
        
        fig_dti = px.bar(
            dti_df, x='dti_bin', y='mean', text_auto='.1f',
            title='Default Rate (%) by DTI Band',
            labels={'dti_bin': 'DTI Range', 'mean': 'Default Rate (%)'},
            color='mean', color_continuous_scale='Reds'
        )
        fig_dti.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_light_theme(fig_dti, 400), use_container_width=True)
        
    col_i, col_j = st.columns(2)
    
    with col_i:
        loan_df = filtered_df.groupby('loan_bin', observed=False)['Default'].agg(['count', 'mean']).reset_index()
        loan_df['mean'] = loan_df['mean'] * 100
        
        fig_loan_bin = px.bar(
            loan_df, x='loan_bin', y='mean', text_auto='.1f',
            title='Default Rate (%) by Loan Bracket',
            labels={'loan_bin': 'Loan Amount', 'mean': 'Default Rate (%)'},
            color='mean', color_continuous_scale='Purples'
        )
        fig_loan_bin.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_light_theme(fig_loan_bin, 400), use_container_width=True)
        
    with col_j:
        emp_df = filtered_df.groupby('emp_length_n')['Default'].agg(['count', 'mean']).reset_index()
        emp_df['mean'] = emp_df['mean'] * 100
        emp_df['emp_label'] = emp_df['emp_length_n'].apply(lambda x: "Unknown" if x == -1 else f"{int(x)} yrs")
        
        fig_emp = px.bar(
            emp_df, x='emp_label', y='mean', text_auto='.1f',
            title='Default Rate (%) by Employment Length',
            labels={'emp_label': 'Employment Duration', 'mean': 'Default Rate (%)'},
            color='mean', color_continuous_scale='Teal'
        )
        fig_emp.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_light_theme(fig_emp, 400), use_container_width=True)

# TAB 4: ACTIONABLE BUSINESS INSIGHTS
with tab4:
    st.subheader("Strategic Portfolio Recommendations")
    
    st.markdown("""
    <div class="insight-box">
        <h4>1. Risk-Based Pricing & Tightened Small Business Underwriting</h4>
        <p><b>Observation:</b> Small business loans exhibit the highest default rate (~29.85%) across all categories.<br>
        <b>Recommendation:</b> Enforce strict liquidity reserves, minimum FICO thresholds (680+), and mandatory collateral for commercial loan applications.</p>
    </div>
    <div class="insight-box">
        <h4>2. Debt-To-Income (DTI) Threshold Controls</h4>
        <p><b>Observation:</b> Defaults surge rapidly when DTI exceeds 20%, peaking above 32% for DTI ratios over 40%.<br>
        <b>Recommendation:</b> Implement a strict portfolio underwriting ceiling cap at 30% DTI for uncollateralized personal loans.</p>
    </div>
    <div class="insight-box">
        <h4>3. Automated Tiered Approvals</h4>
        <p><b>Observation:</b> Borrowers in the 600-650 FICO tier default at 32.0%, compared to under 9.3% for prime tiers (&gt;750 FICO).<br>
        <b>Recommendation:</b> Automate loan limit reductions for lower FICO bands while expediting approvals for prime applicants.</p>
    </div>
    <div class="insight-box">
        <h4>4. Housing Tenure Factor Integration</h4>
        <p><b>Observation:</b> Renters show significantly higher default risk (23.23%) than homeowners with active mortgages (17.22%).<br>
        <b>Recommendation:</b> Incorporate verified rental payment history into alternative scoring features for non-mortgage holders.</p>
    </div>
    """, unsafe_allow_html=True)