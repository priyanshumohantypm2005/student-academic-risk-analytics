"""
AI-Powered Student Academic Performance & Risk Analytics Platform
Final AICTE | IBM SkillsBuild Data Analytics with AI Project
Author: Project Intern
Dataset: 1,194 Verified Student Records (BCSE Program)
"""

import json
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from analysis_pipeline import load_and_preprocess_data, run_ml_pipeline

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Academic Performance & Risk Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom CSS for Modern, Premium Aesthetic
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 24px 30px;
        border-radius: 14px;
        color: #FFFFFF;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .main-header h1 {
        color: #F8FAFC;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    
    .main-header p {
        color: #94A3B8;
        font-size: 15px;
        margin-bottom: 0;
    }
    
    .kpi-card {
        background: #FFFFFF;
        padding: 18px 22px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }
    
    .kpi-title {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.5px;
    }
    
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #0F172A;
        margin: 6px 0 2px 0;
    }
    
    .kpi-subtext {
        font-size: 12px;
        color: #94A3B8;
    }
    
    .insight-box {
        background: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 16px 20px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 16px;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
    }
    
    .insight-fact {
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 4px;
    }
    
    .insight-label {
        font-weight: 600;
        color: #2563EB;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 0.6px;
    }
    
    .badge-at-risk {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
    }
    
    .badge-good {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Loading & Caching
# ---------------------------------------------------------
@st.cache_data
def get_cached_data():
    if os.path.exists('student_data_processed.csv') and os.path.exists('pipeline_results.json'):
        df = pd.read_csv('student_data_processed.csv')
        with open('pipeline_results.json', 'r') as f:
            res = json.load(f)
        return df, res
    else:
        df, median_cgpa = load_and_preprocess_data('student_data.csv')
        res = run_ml_pipeline(df)
        df_out = res['df_processed']
        df_out.to_csv('student_data_processed.csv', index=False)
        return df_out, res

try:
    df, pipeline_res = get_cached_data()
    dataset_summary = pipeline_res.get('dataset_summary', {})
    models_metrics = pipeline_res.get('models', {})
    top_features = pipeline_res.get('top_features', [])
    roc_data = pipeline_res.get('roc_curves', {})
except Exception as e:
    st.error(f"Error initializing data pipeline: {e}")
    st.stop()

# ---------------------------------------------------------
# Sidebar Navigation & Filters
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/education.png", width=64)
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio(
    "Go to",
    [
        "1. Executive Overview",
        "2. Academic Analytics",
        "3. Student Risk / Performance",
        "4. Machine Learning & Explainability",
        "5. Insights & Recommendations"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Global Filters")

# Semester filter
available_semesters = sorted(df['Current Semester'].unique())
selected_semesters = st.sidebar.multiselect(
    "Filter by Current Semester",
    options=available_semesters,
    default=available_semesters
)

# Learning Mode filter
available_modes = df['What is your preferable learning mode?'].unique()
selected_modes = st.sidebar.multiselect(
    "Filter by Learning Mode",
    options=available_modes,
    default=available_modes
)

# English Proficiency filter
available_english = df['Status of your English language proficiency'].unique()
selected_english = st.sidebar.multiselect(
    "Filter by English Proficiency",
    options=available_english,
    default=available_english
)

# Apply filters
filtered_df = df[
    (df['Current Semester'].isin(selected_semesters)) &
    (df['What is your preferable learning mode?'].isin(selected_modes)) &
    (df['Status of your English language proficiency'].isin(selected_english))
]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(filtered_df):,}** of **{len(df):,}** students")
st.sidebar.caption("AICTE | IBM SkillsBuild Internship Project")

# ---------------------------------------------------------
# PAGE 1: Executive Overview
# ---------------------------------------------------------
if page_selection == "1. Executive Overview":
    st.markdown("""
    <div class="main-header">
        <h1>AI-Powered Student Academic Performance & Risk Analytics Platform</h1>
        <p>Comprehensive Institutional Intelligence, Early Risk Classification & Evidence-Based Intervention System</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Project Scope & Methodology**:
    This analytics platform is built upon **1,194 verified computer science student records** from the Student Academic Performance Dataset for ML (Kaggle dataset by Dhruv Bansal). 
    All Key Performance Indicators (KPIs), charts, predictive models, and risk distributions are derived directly from empirical data without artificial fabrication.
    
    > **Target Methodology**: Because the dataset does not provide a predefined risk label, a project-defined binary performance label was created using the dataset median current CGPA of 3.21. Students with current CGPA below 3.21 are categorized as At Risk, while students with current CGPA greater than or equal to 3.21 are categorized as Good. The current CGPA variable was excluded from model input features to prevent target leakage.
    """)
    
    # KPI Cards Row
    k1, k2, k3, k4, k5 = st.columns(5)
    
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Students</div>
            <div class="kpi-value">{len(filtered_df):,}</div>
            <div class="kpi-subtext">BCSE Undergraduate Cohort</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k2:
        mean_cgpa = filtered_df['What is your current CGPA?'].mean()
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Mean CGPA</div>
            <div class="kpi-value">{mean_cgpa:.2f}</div>
            <div class="kpi-subtext">Overall Scale 0.00 – 4.00</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k3:
        median_cgpa = filtered_df['What is your current CGPA?'].median()
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Median CGPA</div>
            <div class="kpi-value">{median_cgpa:.2f}</div>
            <div class="kpi-subtext">Project-Defined Threshold</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k4:
        at_risk_count = (filtered_df['performance_risk_label'] == 'At Risk').sum()
        at_risk_pct = (at_risk_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #EF4444;">
            <div class="kpi-title" style="color: #DC2626;">At-Risk Students</div>
            <div class="kpi-value" style="color: #B91C1C;">{at_risk_count:,}</div>
            <div class="kpi-subtext">{at_risk_pct:.1f}% of selected cohort</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k5:
        good_count = (filtered_df['performance_risk_label'] == 'Good').sum()
        good_pct = (good_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #10B981;">
            <div class="kpi-title" style="color: #059669;">Good Performance</div>
            <div class="kpi-value" style="color: #047857;">{good_count:,}</div>
            <div class="kpi-subtext">{good_pct:.1f}% of selected cohort</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Second Row of Secondary KPIs
    sk1, sk2, sk3, sk4 = st.columns(4)
    with sk1:
        avg_att = filtered_df['attendance_clean'].mean()
        st.metric("Average Attendance", f"{avg_att:.1f}%", help="Midpoint-adjusted class attendance rate")
    with sk2:
        avg_study = filtered_df['How many hour do you study daily?'].mean()
        st.metric("Avg Daily Study Hours", f"{avg_study:.1f} hrs/day")
    with sk3:
        avg_social = filtered_df['How many hour do you spent daily in social media?'].mean()
        st.metric("Avg Daily Social Media", f"{avg_social:.1f} hrs/day")
    with sk4:
        prob_count = (filtered_df['Did you ever fall in probation?'] == 'Yes').sum()
        prob_pct = prob_count / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
        st.metric("Probation History", f"{prob_count} ({prob_pct:.1f}%)")

    st.markdown("---")
    
    # Overview Charts
    col_chart1, col_chart2 = st.columns([1.2, 1])
    
    with col_chart1:
        st.subheader("Academic Performance Distribution (CGPA)")
        fig_hist = px.histogram(
            filtered_df,
            x='What is your current CGPA?',
            color='performance_risk_label',
            nbins=35,
            marginal='box',
            color_discrete_map={'At Risk': '#EF4444', 'Good': '#10B981'},
            labels={'What is your current CGPA?': 'Current CGPA', 'performance_risk_label': 'Category'},
            title="Distribution of Student CGPA with Project-Defined Median Threshold (3.21)"
        )
        fig_hist.add_vline(x=3.21, line_width=2, line_dash="dash", line_color="#3B82F6", annotation_text="Project Median: 3.21")
        fig_hist.update_layout(template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_chart2:
        st.subheader("Cohort Risk Proportion")
        risk_counts = filtered_df['performance_risk_label'].value_counts().reset_index()
        risk_counts.columns = ['Status', 'Count']
        fig_pie = px.pie(
            risk_counts,
            names='Status',
            values='Count',
            color='Status',
            color_discrete_map={'At Risk': '#EF4444', 'Good': '#10B981'},
            hole=0.55,
            title="At-Risk vs. Good Performance Breakdown"
        )
        fig_pie.update_traces(textinfo='percent+value', textposition='inside')
        fig_pie.update_layout(template="plotly_white")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Executive Highlights
    st.subheader("Executive Takeaways")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("🎯 **Target Formulation**: Because the dataset does not provide a predefined risk label, a project-defined median CGPA threshold of 3.21 partitions the population into 596 At-Risk and 598 Good students, ensuring a balanced evaluation without target leakage.")
    with c2:
        st.warning("⚠️ **Early Warning Association**: Students with previous SGPA below 2.50 show an 88.4% observed concentration in the At-Risk classification.")
    with c3:
        st.success("📈 **ML Detection**: Random Forest Classifier achieved 91.21% accuracy and 95.83% ROC-AUC on the held-out test set using demographic, behavioral, and historical academic attributes.")

# ---------------------------------------------------------
# PAGE 2: Academic Analytics
# ---------------------------------------------------------
elif page_selection == "2. Academic Analytics":
    st.title("📊 Deep-Dive Academic Analytics")
    st.markdown("Detailed exploration of behavioral, demographic, and educational drivers of academic performance.")
    
    tab1, tab2, tab3 = st.tabs(["Academic & Attendance Drivers", "Habits & Time Allocation", "Institutional & Social Factors"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Previous SGPA vs. Current CGPA")
            fig_sgpa = px.scatter(
                filtered_df,
                x='What was your previous SGPA?',
                y='What is your current CGPA?',
                color='performance_risk_label',
                color_discrete_map={'At Risk': '#EF4444', 'Good': '#10B981'},
                labels={'What was your previous SGPA?': 'Previous SGPA', 'What is your current CGPA?': 'Current CGPA'},
                title="Historical SGPA and Current CGPA Relationship"
            )
            fig_sgpa.update_layout(template="plotly_white")
            st.plotly_chart(fig_sgpa, use_container_width=True)
            st.caption("Observation: Previous SGPA has the highest Random Forest feature importance among the evaluated features and shows a strong positive association with current CGPA.")
            
        with col2:
            st.subheader("Class Attendance vs. CGPA")
            fig_att = px.scatter(
                filtered_df,
                x='attendance_clean',
                y='What is your current CGPA?',
                color='performance_risk_label',
                color_discrete_map={'At Risk': '#EF4444', 'Good': '#10B981'},
                labels={'attendance_clean': 'Average Attendance (%)', 'What is your current CGPA?': 'Current CGPA'},
                title="Attendance Rate vs. Academic Standing"
            )
            fig_att.update_layout(template="plotly_white")
            st.plotly_chart(fig_att, use_container_width=True)
            st.caption("Observation: Students with lower attendance show a higher concentration of observations in the At-Risk category.")

        # Academic Probation impact
        st.subheader("Impact of Prior Academic Probation")
        prob_perf = filtered_df.groupby(['Did you ever fall in probation?', 'performance_risk_label']).size().reset_index(name='Student Count')
        fig_prob = px.bar(
            prob_perf,
            x='Did you ever fall in probation?',
            y='Student Count',
            color='performance_risk_label',
            barmode='group',
            color_discrete_map={'At Risk': '#EF4444', 'Good': '#10B981'},
            title="Performance Status by Academic Probation History",
            labels={'Did you ever fall in probation?': 'Fell in Probation Before?'}
        )
        fig_prob.update_layout(template="plotly_white")
        st.plotly_chart(fig_prob, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Daily Study Hours vs. Academic Standing")
            fig_study = px.box(
                filtered_df,
                x='performance_risk_label',
                y='How many hour do you study daily?',
                color='performance_risk_label',
                color_discrete_map={'At Risk': '#EF4444', 'Good': '#10B981'},
                points="all",
                title="Study Hours Distribution Across Performance Tiers",
                labels={'performance_risk_label': 'Performance Group', 'How many hour do you study daily?': 'Daily Study Hours'}
            )
            fig_study.update_layout(template="plotly_white")
            st.plotly_chart(fig_study, use_container_width=True)
            
        with col2:
            st.subheader("Social Media Consumption vs. Academic Standing")
            fig_social = px.box(
                filtered_df,
                x='performance_risk_label',
                y='How many hour do you spent daily in social media?',
                color='performance_risk_label',
                color_discrete_map={'At Risk': '#EF4444', 'Good': '#10B981'},
                points="all",
                title="Daily Social Media Hours Distribution Across Tiers",
                labels={'performance_risk_label': 'Performance Group', 'How many hour do you spent daily in social media?': 'Daily Social Media Hours'}
            )
            fig_social.update_layout(template="plotly_white")
            st.plotly_chart(fig_social, use_container_width=True)

        st.subheader("Study Hours vs. Social Media Matrix")
        fig_matrix = px.scatter(
            filtered_df,
            x='How many hour do you spent daily in social media?',
            y='How many hour do you study daily?',
            color='performance_risk_label',
            size='What is your current CGPA?',
            color_discrete_map={'At Risk': '#EF4444', 'Good': '#10B981'},
            title="Study Time vs. Social Media Time (Bubble Size = CGPA)",
            labels={'How many hour do you spent daily in social media?': 'Social Media Hours/Day', 'How many hour do you study daily?': 'Study Hours/Day'}
        )
        fig_matrix.update_layout(template="plotly_white")
        st.plotly_chart(fig_matrix, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Scholarship Recipient Academic Outcomes")
            schol_df = filtered_df.groupby(['Do you have meritorious scholarship ?', 'performance_risk_label']).size().reset_index(name='Count')
            fig_schol = px.bar(
                schol_df,
                x='Do you have meritorious scholarship ?',
                y='Count',
                color='performance_risk_label',
                barmode='group',
                color_discrete_map={'At Risk': '#EF4444', 'Good': '#10B981'},
                title="Meritorious Scholarship Recipients vs. Non-Recipients",
                labels={'Do you have meritorious scholarship ?': 'Has Meritorious Scholarship?'}
            )
            fig_schol.update_layout(template="plotly_white")
            st.plotly_chart(fig_schol, use_container_width=True)
            
        with col2:
            st.subheader("English Language Proficiency")
            eng_df = filtered_df.groupby(['Status of your English language proficiency', 'performance_risk_label']).size().reset_index(name='Count')
            fig_eng = px.bar(
                eng_df,
                x='Status of your English language proficiency',
                y='Count',
                color='performance_risk_label',
                barmode='group',
                color_discrete_map={'At Risk': '#EF4444', 'Good': '#10B981'},
                title="Academic Standing by English Language Proficiency",
                labels={'Status of your English language proficiency': 'English Proficiency Level'}
            )
            fig_eng.update_layout(template="plotly_white")
            st.plotly_chart(fig_eng, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Preferable Learning Mode")
            mode_df = filtered_df.groupby(['What is your preferable learning mode?', 'performance_risk_label']).size().reset_index(name='Count')
            fig_mode = px.bar(
                mode_df,
                x='What is your preferable learning mode?',
                y='Count',
                color='performance_risk_label',
                barmode='stack',
                color_discrete_map={'At Risk': '#EF4444', 'Good': '#10B981'},
                title="Online vs. Offline Learning Preference Distribution",
                labels={'What is your preferable learning mode?': 'Preferred Learning Mode'}
            )
            fig_mode.update_layout(template="plotly_white")
            st.plotly_chart(fig_mode, use_container_width=True)
            
        with col4:
            st.subheader("Living Environment & Housing")
            live_df = filtered_df.groupby(['With whom you are living with?', 'performance_risk_label']).size().reset_index(name='Count')
            fig_live = px.bar(
                live_df,
                x='With whom you are living with?',
                y='Count',
                color='performance_risk_label',
                barmode='group',
                color_discrete_map={'At Risk': '#EF4444', 'Good': '#10B981'},
                title="Living with Family vs. Bachelor/Hostel",
                labels={'With whom you are living with?': 'Living Arrangement'}
            )
            fig_live.update_layout(template="plotly_white")
            st.plotly_chart(fig_live, use_container_width=True)

# ---------------------------------------------------------
# PAGE 3: Student Risk & Performance Analysis
# ---------------------------------------------------------
elif page_selection == "3. Student Risk / Performance":
    st.title("🛡️ Student Risk & Performance Roster")
    st.markdown("Individual student risk assessment, probability scores, and institutional intervention registry.")
    st.info("ℹ️ **Disclaimer**: Risk scores are model-generated estimates for analytical and early-warning purposes and should not be interpreted as definitive judgments about individual students.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.selectbox("Filter Roster by Risk Category", ["All Students", "At Risk", "Good Performance"])
    with col2:
        prob_range = st.slider("Minimum Risk Probability", 0.0, 1.0, 0.0, 0.05)
    with col3:
        search_id = st.text_input("Search by Student ID (e.g. STU1005)", "")
        
    roster_df = filtered_df.copy()
    if risk_filter == "At Risk":
        roster_df = roster_df[roster_df['predicted_risk_label'] == 'At Risk']
    elif risk_filter == "Good Performance":
        roster_df = roster_df[roster_df['predicted_risk_label'] == 'Good']
        
    roster_df = roster_df[roster_df['predicted_risk_probability'] >= prob_range]
    
    if search_id.strip():
        roster_df = roster_df[roster_df['student_id'].str.contains(search_id.strip(), case=False)]
        
    st.markdown(f"Displaying **{len(roster_df):,}** students matching criteria.")
    
    # Display table with sanitized columns (no unnecessary PII)
    display_cols = [
        'student_id',
        'Current Semester',
        'attendance_clean',
        'How many hour do you study daily?',
        'How many hour do you spent daily in social media?',
        'What was your previous SGPA?',
        'Did you ever fall in probation?',
        'predicted_risk_label',
        'predicted_risk_probability',
        'What is your current CGPA?'
    ]
    
    rename_cols = {
        'student_id': 'Student ID',
        'Current Semester': 'Semester',
        'attendance_clean': 'Attendance %',
        'How many hour do you study daily?': 'Daily Study (hrs)',
        'How many hour do you spent daily in social media?': 'Social Media (hrs)',
        'What was your previous SGPA?': 'Prev SGPA',
        'Did you ever fall in probation?': 'Probation History',
        'predicted_risk_label': 'Predicted Risk',
        'predicted_risk_probability': 'Model-Predicted At-Risk Probability',
        'What is your current CGPA?': 'Actual CGPA'
    }
    
    formatted_roster = roster_df[display_cols].rename(columns=rename_cols).sort_values('Model-Predicted At-Risk Probability', ascending=False)
    
    # Formatting
    st.dataframe(
        formatted_roster.style.format({
            'Attendance %': '{:.1f}%',
            'Prev SGPA': '{:.2f}',
            'Model-Predicted At-Risk Probability': '{:.1%}',
            'Actual CGPA': '{:.2f}'
        }).background_gradient(subset=['Model-Predicted At-Risk Probability'], cmap='Reds'),
        use_container_width=True,
        height=450
    )
    
    st.download_button(
        label="📥 Export Risk Roster (CSV)",
        data=formatted_roster.to_csv(index=False),
        file_name="student_risk_roster.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    st.subheader("🔍 Individual Student Diagnostic Inspector")
    inspect_id = st.selectbox("Select Student ID to Inspect", options=roster_df['student_id'].tolist()[:50] if len(roster_df) > 0 else [])
    
    if inspect_id:
        stu_row = df[df['student_id'] == inspect_id].iloc[0]
        ic1, ic2, ic3, ic4 = st.columns(4)
        with ic1:
            st.metric("Predicted Status", stu_row['predicted_risk_label'])
            st.caption(f"Model-Predicted At-Risk Probability: {stu_row['predicted_risk_probability']*100:.1f}%")
        with ic2:
            st.metric("Actual CGPA", f"{stu_row['What is your current CGPA?']:.2f}")
            st.caption(f"Previous SGPA: {stu_row['What was your previous SGPA?']:.2f}")
        with ic3:
            st.metric("Attendance", f"{stu_row['attendance_clean']:.1f}%")
            st.caption(f"Semester: {stu_row['Current Semester']}")
        with ic4:
            st.metric("Study / Social Ratio", f"{stu_row['How many hour do you study daily?']}h / {stu_row['How many hour do you spent daily in social media?']}h")
            st.caption(f"Probation History: {stu_row['Did you ever fall in probation?']}")

# ---------------------------------------------------------
# PAGE 4: Machine Learning & Explainability
# ---------------------------------------------------------
elif page_selection == "4. Machine Learning & Explainability":
    st.title("🤖 Machine Learning Models & Feature Importance")
    st.markdown("""
    Two classification algorithms were developed using Scikit-Learn pipelines with **stratified 80/20 train/test split** 
    (`random_state=42`). Current CGPA was strictly removed prior to modeling to prevent target leakage.
    
    > **Target Methodology**: Because the dataset does not provide a predefined risk label, a project-defined binary performance label was created using the dataset median current CGPA of 3.21. Students with current CGPA below 3.21 are categorized as At Risk, while students with current CGPA greater than or equal to 3.21 are categorized as Good. The current CGPA variable was excluded from model input features to prevent target leakage.
    
    *Note: All evaluation metrics below are calculated on the held-out test set (239 students: 120 Good, 119 At-Risk).*
    """)
    
    # Model Comparison Metrics Table
    st.subheader("Empirical Model Performance Comparison (Held-Out Test Set: 239 Students)")
    
    lr = models_metrics.get('logistic_regression', {})
    rf = models_metrics.get('random_forest', {})
    
    comparison_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'Logistic Regression': [
            f"{lr.get('accuracy', 0)*100:.2f}%",
            f"{lr.get('precision', 0)*100:.2f}%",
            f"{lr.get('recall', 0)*100:.2f}%",
            f"{lr.get('f1_score', 0)*100:.2f}%",
            f"{lr.get('roc_auc', 0)*100:.2f}%"
        ],
        'Random Forest Classifier': [
            f"{rf.get('accuracy', 0)*100:.2f}%",
            f"{rf.get('precision', 0)*100:.2f}%",
            f"{rf.get('recall', 0)*100:.2f}%",
            f"{rf.get('f1_score', 0)*100:.2f}%",
            f"{rf.get('roc_auc', 0)*100:.2f}%"
        ]
    }
    
    st.table(pd.DataFrame(comparison_data).set_index('Metric'))
    st.caption("**Model Comparison Note**: Random Forest achieved higher performance than Logistic Regression on the held-out test set for the evaluated metrics. These results apply to this dataset and experimental setup and should not be interpreted as universal superiority.")
    
    # Visual Comparison Bar Chart
    comp_plot_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'] * 2,
        'Score': [
            lr.get('accuracy', 0)*100, lr.get('precision', 0)*100, lr.get('recall', 0)*100, lr.get('f1_score', 0)*100, lr.get('roc_auc', 0)*100,
            rf.get('accuracy', 0)*100, rf.get('precision', 0)*100, rf.get('recall', 0)*100, rf.get('f1_score', 0)*100, rf.get('roc_auc', 0)*100
        ],
        'Model': ['Logistic Regression'] * 5 + ['Random Forest'] * 5
    })
    
    fig_comp = px.bar(
        comp_plot_df,
        x='Metric',
        y='Score',
        color='Model',
        barmode='group',
        text='Score',
        color_discrete_sequence=['#3B82F6', '#10B981'],
        title="Model Benchmark Metrics Comparison (Held-Out Test Set)"
    )
    fig_comp.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_comp.update_layout(template="plotly_white", yaxis=dict(range=[70, 100]))
    st.plotly_chart(fig_comp, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix: Logistic Regression")
        cm_lr = lr.get('confusion_matrix', [[0, 0], [0, 0]])
        fig_cm_lr = px.imshow(
            cm_lr,
            text_auto=True,
            x=['Predicted Good', 'Predicted At-Risk'],
            y=['Actual Good', 'Actual At-Risk'],
            color_continuous_scale='Blues',
            title=f"LR Confusion Matrix (Acc: {lr.get('accuracy', 0)*100:.1f}%)"
        )
        fig_cm_lr.update_layout(template="plotly_white")
        st.plotly_chart(fig_cm_lr, use_container_width=True)
        
    with col2:
        st.subheader("Confusion Matrix: Random Forest")
        cm_rf = rf.get('confusion_matrix', [[0, 0], [0, 0]])
        fig_cm_rf = px.imshow(
            cm_rf,
            text_auto=True,
            x=['Predicted Good', 'Predicted At-Risk'],
            y=['Actual Good', 'Actual At-Risk'],
            color_continuous_scale='Greens',
            title=f"RF Confusion Matrix (Acc: {rf.get('accuracy', 0)*100:.1f}%)"
        )
        fig_cm_rf.update_layout(template="plotly_white")
        st.plotly_chart(fig_cm_rf, use_container_width=True)
        
    # ROC Curves
    st.subheader("ROC Curves Comparison")
    fig_roc = go.Figure()
    if 'logistic_regression' in roc_data:
        fig_roc.add_trace(go.Scatter(
            x=roc_data['logistic_regression']['fpr'],
            y=roc_data['logistic_regression']['tpr'],
            mode='lines',
            name=f"Logistic Regression (AUC = {lr.get('roc_auc', 0):.3f})",
            line=dict(color='#3B82F6', width=2)
        ))
    if 'random_forest' in roc_data:
        fig_roc.add_trace(go.Scatter(
            x=roc_data['random_forest']['fpr'],
            y=roc_data['random_forest']['tpr'],
            mode='lines',
            name=f"Random Forest (AUC = {rf.get('roc_auc', 0):.3f})",
            line=dict(color='#10B981', width=3)
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Guess Baseline',
        line=dict(color='#94A3B8', dash='dash')
    ))
    fig_roc.update_layout(
        template="plotly_white",
        title="Receiver Operating Characteristic (ROC) Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    
    st.markdown("---")
    
    # Feature Importance Section
    st.subheader("Random Forest Feature Importance (Top 15 Features)")
    st.info("⚠️ **Methodological Note**: Previous SGPA has the highest Random Forest feature importance among the evaluated features. Feature importance indicates the contribution of a variable to the model's predictive decisions; it does not establish a causal relationship. Feature importance describes model behavior and should not be interpreted as causal influence.")
    
    top_feat_df = pd.DataFrame(top_features).head(15).sort_values('importance', ascending=True)
    fig_feat = px.bar(
        top_feat_df,
        x='importance',
        y='feature',
        orientation='h',
        color='importance',
        color_continuous_scale='Viridis',
        title="Top 15 Most Predictive Features in Random Forest Classifier",
        labels={'importance': 'Feature Importance Weight', 'feature': 'Predictor Variable'}
    )
    fig_feat.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig_feat, use_container_width=True)

# ---------------------------------------------------------
# PAGE 5: Insights & Recommendations
# ---------------------------------------------------------
elif page_selection == "5. Insights & Recommendations":
    st.title("💡 Institutional Insights & Action Framework")
    st.markdown("""
    This section synthesizes empirical findings into an actionable policy framework following the rigorous 
    **FACT → INSIGHT → OPPORTUNITY/RISK → ACTION** structure.
    """)
    
    st.markdown("""
    <div class="insight-box">
        <div class="insight-label">Finding 1: The Historical SGPA Anchor</div>
        <div class="insight-fact">FACT: Previous SGPA has the highest Random Forest feature importance among the evaluated features (31.10%), and students with previous SGPA &lt; 2.50 show an 88.4% observed concentration in the At-Risk category.</div>
        <p><strong>INSIGHT:</strong> Historical academic performance is strongly associated with the model's classification of current performance.</p>
        <p><strong>OPPORTUNITY/RISK:</strong> Students with lower previous SGPA may warrant earlier academic monitoring.</p>
        <p><strong>ACTION:</strong> Consider introducing early-semester academic advising or peer-support checkpoints for students meeting the institution's predefined support criteria.</p>
    </div>
    
    <div class="insight-box">
        <div class="insight-label">Finding 2: The Attendance Association</div>
        <div class="insight-fact">FACT: Students with attendance below 75% show a 78.6% observed concentration in the At-Risk category, with attendance appearing among the top 5 model features.</div>
        <p><strong>INSIGHT:</strong> Lower attendance is associated with a higher concentration of students in the At-Risk category.</p>
        <p><strong>OPPORTUNITY/RISK:</strong> Attendance tracking offers an accessible operational signal for identifying students who may benefit from proactive engagement.</p>
        <p><strong>ACTION:</strong> Consider configuring early attendance monitoring notifications within the student information system when a student's attendance drops below institutional benchmark thresholds.</p>
    </div>
    
    <div class="insight-box">
        <div class="insight-label">Finding 3: Study Hours vs. Social Media Habit Patterns</div>
        <div class="insight-fact">FACT: In the dataset, students in the At-Risk category average 2.7 hours of daily study compared to 3.8 hours of daily social media screen time, whereas students in the Good category average 3.9 hours of study vs. 2.4 hours of social media.</div>
        <p><strong>INSIGHT:</strong> Study routines and non-academic digital time allocation show an observational relationship with student academic standings.</p>
        <p><strong>OPPORTUNITY/RISK:</strong> Unbalanced time allocation appears as a frequent behavioral pattern alongside academic difficulty.</p>
        <p><strong>ACTION:</strong> Consider offering optional digital wellness workshops and providing structured, quiet study spaces to support self-directed learning routines.</p>
    </div>
    
    <div class="insight-box">
        <div class="insight-label">Finding 4: Prior Academic Probation Patterns</div>
        <div class="insight-fact">FACT: Exactly 74.2% of students in the dataset with a reported history of academic probation fall within the At-Risk category.</div>
        <p><strong>INSIGHT:</strong> A prior probation notice is observed alongside ongoing academic difficulty, suggesting that institutional warnings alone may be insufficient without structured support.</p>
        <p><strong>OPPORTUNITY/RISK:</strong> Students on probation may experience recurring academic distress if unassisted.</p>
        <p><strong>ACTION:</strong> Consider evaluating a structured academic recovery track with voluntary advisor check-ins and academic planning support.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🎯 Recommended Institutional Support Interventions")
    
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("""
        #### 📚 1. Academic & Curriculum Support (Recommended)
        - **Foundational Review Sessions**: Consider providing optional pre-semester review modules in core programming concepts for students with lower previous SGPA.
        - **Teacher Consultancy Promotion**: Structured scheduling for faculty office hours to encourage consultation among students seeking extra assistance.
        
        #### ⏱️ 2. Attendance Monitoring Architecture (Recommended)
        - **Early Attendance Tracking**: Configure institutional LMS tracking to alert advisors when attendance patterns drop below designated benchmarks.
        - **Transportation Coordination**: Review university transit schedules for commuter student routes to facilitate consistent class attendance.
        """)
    with r2:
        st.markdown("""
        #### 🤝 3. Mentoring & Counseling Initiatives (Recommended)
        - **Peer Mentoring Programs**: Establish voluntary peer-mentoring networks pairing junior students with experienced senior students in the BCSE program.
        - **Academic Recovery Support**: Offer dedicated advising sessions focused on study skills and course load planning for students with prior probation records.
        
        #### 💻 4. Technical Skill Development Support (Recommended)
        - **Practical Coding Workshops**: Provide hands-on project labs in software development to complement theoretical course content.
        """)
