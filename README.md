# AI-Powered Student Academic Performance & Risk Analytics Platform

**Final Internship Project for AICTE | IBM SkillsBuild: Data Analytics with AI**

---

## 📌 Project Overview
The **AI-Powered Student Academic Performance & Risk Analytics Platform** is an educational analytics and decision-support system designed to analyze multidimensional student data, identify patterns associated with academic performance, classify students into empirical risk categories, and provide actionable interventions for higher education institutions.

All analyses, models, metrics, and report figures are derived directly and reproducibly from the verified institutional dataset of **1,194 computer science undergraduate students**.

---

## 🎯 Problem Statement
Higher education institutions often experience delays in identifying students who encounter academic distress. Traditional evaluation systems typically assess academic standing only after semester examination results are tabulated, when remedial interventions are often too late to prevent course repeats or academic probation.

By leveraging machine learning and data analytics, institutions can proactively identify at-risk patterns early in the academic term, uncover key academic, behavioral, and environmental indicators, and implement targeted academic support programs.

---

## 🚀 Objectives
1. **Empirical Dataset Audit & Preprocessing**: Clean, audit, and normalize a 31-column student dataset without artificial data injection.
2. **Target Methodology & Leakage Prevention**: Formulate a transparent, project-defined binary performance label using the dataset median current CGPA (3.21) while strictly excluding current CGPA from model features to prevent target leakage.
3. **Exploratory Data Analysis (EDA)**: Statistically evaluate observational relationships between academic performance and attendance, previous SGPA, study hours, social media consumption, living arrangements, scholarship status, and academic probation history.
4. **Supervised Machine Learning**: Train and benchmark two supervised learning models (**Logistic Regression** and **Random Forest Classifier**) on a held-out test set using stratified sampling.
5. **Model Explainability & Feature Importance**: Quantify the relative contribution of behavioral and academic predictors to support evidence-based educational administration, clearly distinguishing model association from causality.
6. **Interactive Multi-Page Streamlit Dashboard**: Provide a responsive, 5-page analytics and decision-support web application for academic leadership and student advisors.

---

## 📊 Dataset Description & Source
- **Dataset Title**: [Student Academic Performance Dataset for ML](https://www.kaggle.com/datasets/dhruvbansal64/student-academic-performance-dataset-for-ml)
- **Author / Source**: Dhruv Bansal (Kaggle)
- **Cohort**: Undergraduate students enrolled in the Bachelor of Computer Science and Engineering (BCSE) program.
- **Total Records (Rows)**: 1,194
- **Total Features (Columns)**: 31
- **Key Features**:
  - *Academic Metrics*: Current CGPA (`what_is_your_current_cgpa?`), Previous SGPA (`what_was_your_previous_sgpa?`), Completed Credits, Current Semester, Class Attendance.
  - *Learning Habits*: Daily study hours, seating frequency, preferred learning mode (Online vs. Offline).
  - *Digital Habits*: Daily social media hours, daily skill development hours, co-curricular participation.
  - *Institutional Engagement*: Class attendance rate, meritorious scholarship status, teacher consultancy attendance, probation history, suspension history.
  - *Demographics & Environment*: Age, gender, relationship status, living arrangement (Family vs. Bachelor/Hostel), monthly family income, health status.

---

## 🏷️ Target Methodology & Leakage Prevention
> **Methodology Statement**: Because the dataset does not provide a predefined risk label, a project-defined binary performance label was created using the dataset median current CGPA of 3.21. Students with current CGPA below 3.21 are categorized as At Risk, while students with current CGPA greater than or equal to 3.21 are categorized as Good. The current CGPA variable was excluded from model input features to prevent target leakage.

- **Project-Defined Threshold**: Median current CGPA = **3.2100**
- **At Risk** (`CGPA < 3.21`): **596 students (49.92%)**
- **Good Performance** (`CGPA >= 3.21`): **598 students (50.08%)**
- **Leakage Prevention**: The continuous column `what_is_your_current_cgpa?` is strictly omitted from the model feature matrix $X$ to prevent target leakage and circular evaluation.

---

## 🛠️ Technologies Used
- **Language**: Python 3.10+
- **Web Application Framework**: Streamlit
- **Data Analytics & Manipulation**: Pandas, NumPy
- **Machine Learning & Preprocessing**: Scikit-Learn
- **Data Visualization**: Plotly Express, Plotly Graph Objects, Seaborn, Matplotlib
- **Document Generation**: Python-docx, Pillow

---

## 📁 Project Structure
```text
IBM INTERNSHIP/
├── student_data.csv             # Raw institutional dataset (1,194 records, 31 attributes)
├── student_data_processed.csv   # Cleaned dataset with risk labels and model predictions
├── pipeline_results.json        # Verified ML evaluation metrics and feature importances
├── analysis_pipeline.py         # Data cleaning, preprocessing & ML pipeline script
├── app.py                       # Main 5-page interactive Streamlit dashboard
├── generate_report.py           # Automated 23-section Word report generation script
├── Project_Report.docx          # Formal 23-section academic internship report with screenshots
├── screenshots/                 # Captured high-resolution dashboard screenshots
│   ├── screenshot_page1_overview.png
│   ├── screenshot_page2_analytics.png
│   ├── screenshot_page3_risk_roster.png
│   ├── screenshot_page4_ml_explainability.png
│   └── screenshot_page5_insights.png
├── requirements.txt             # Pinned package dependencies
└── README.md                    # Project documentation and reproduction guide
```

---

## ⚙️ Installation & Setup Instructions

### 1. Prerequisites
Verify that Python 3.10 or higher is installed:
```bash
python --version
```

### 2. Navigate to Project Directory
```bash
cd "c:\Users\priya\OneDrive\Desktop\PROJECTS\IBM INTERNSHIP"
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Data & Machine Learning Pipeline
To execute data preprocessing, train the machine learning models, and generate verified metrics:
```bash
python analysis_pipeline.py
```

### 5. Launch the Streamlit Interactive Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 6. Generate the Formal Word Report (with Screenshots)
```bash
python generate_report.py
```

---

## 🧠 Machine Learning Methodology & Empirical Results

### Modeling Workflow
1. **Preprocessing Pipeline**:
   - Numerical features (10): Median imputation + Standard Scaling (`StandardScaler`).
   - Categorical features (17): Most-frequent imputation + One-Hot Encoding (`OneHotEncoder(handle_unknown='ignore')`).
2. **Leakage Prevention**: Current CGPA is strictly dropped from predictor inputs.
3. **Validation Strategy**: 80% Train / 20% Test Stratified Split (`random_state=42`) evaluating on 239 held-out test records (120 Good, 119 At-Risk).

### Evaluation Metrics (Evaluated on Held-Out Test Set)
> **Note**: Evaluation metrics are calculated on the held-out test set (n = 239). Random Forest achieved higher performance than Logistic Regression on the held-out test set for the evaluated metrics. These results apply to this dataset and experimental setup and should not be interpreted as universal superiority.

| Evaluation Metric | Logistic Regression | Random Forest Classifier | Difference (Held-Out Test Set) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **86.19%** | **91.21%** | **+5.02%** |
| **Precision** | **84.13%** | **92.24%** | **+8.11%** |
| **Recall (Sensitivity)** | **89.08%** | **89.92%** | **+0.84%** |
| **F1-Score** | **86.53%** | **91.06%** | **+4.53%** |
| **ROC-AUC Score** | **92.68%** | **95.83%** | **+3.15%** |

### Confusion Matrix Breakdown (Held-Out Test Set: 239 Students)
- **Logistic Regression**:
  - True Negatives (Good predicted as Good): **100**
  - False Positives (Good predicted as At-Risk): **20**
  - False Negatives (At-Risk predicted as Good): **13**
  - True Positives (At-Risk predicted as At-Risk): **106**
- **Random Forest Classifier**:
  - True Negatives (Good predicted as Good): **111**
  - False Positives (Good predicted as At-Risk): **9**
  - False Negatives (At-Risk predicted as Good): **12**
  - True Positives (At-Risk predicted as At-Risk): **107**

---

## 🏆 Feature Importance & Explainability
Previous SGPA has the highest Random Forest feature importance among the evaluated features. 
Feature importance indicates the contribution of a variable to the model's predictive decisions; it does not establish a causal relationship. Feature importance describes model behavior and should not be interpreted as causal influence.

1. **Previous SGPA (31.10%)**: Highest feature weight in decision tree partitioning.
2. **Completed Credits (6.96%)**: Degree milestone indicator.
3. **Current Semester (5.26%)**: Curriculum complexity stage.
4. **Monthly Family Income (5.10%)**: Socio-economic context.
5. **Class Attendance Rate (4.76%)**: Classroom presence indicator.
6. **Daily Social Media Hours (3.38%)**: Daily digital time allocation.
7. **Age (2.87%)**: Cohort age distribution.
8. **Daily Study Hours (2.59%)**: Self-reported study time.
9. **Daily Skill Development Hours (2.19%)**: Technical self-investment.
10. **Study Seating Frequency (2.10%)**: Daily study session count.

---

## 🖥️ Dashboard Overview
The Streamlit application features five dedicated pages:
1. **Executive Overview**: High-level KPIs (Total Students 1,194, Mean CGPA 3.17, Median CGPA 3.21, At-Risk 596, Good 598), CGPA distribution histogram with project median threshold, and cohort risk proportion chart.
2. **Academic Analytics**: 3 tabs exploring Previous SGPA vs Current CGPA, Attendance vs CGPA, study hours, social media habits, scholarship status, and probation history.
3. **Student Risk / Performance**: Searchable student roster by Student ID (`STU1000` to `STU2193`), risk category filters, Model-Predicted At-Risk Probability, and an individual diagnostic inspector with an explicit analytical disclaimer.
4. **Machine Learning & Explainability**: Model comparison table on the held-out test set, comparative bar charts, confusion matrix heatmaps, dual ROC curves, and the Top 15 feature importance chart.
5. **Insights & Recommendations**: 4 structured findings using the **FACT → INSIGHT → OPPORTUNITY/RISK → ACTION** framework, followed by recommended institutional support interventions.

---

## ⚠️ Model & Project Limitations
1. **Project-Defined Risk Label**: The risk label is project-defined using the dataset median current CGPA of 3.21 rather than an original target supplied by the dataset.
2. **Observational Data**: The dataset is observational, so statistical associations and feature correlations should not be interpreted as causal relationships.
3. **Estimated Probabilities**: Model-predicted probabilities are statistical estimates based on feature patterns, not guaranteed future outcomes.
4. **Sample & Split Dependency**: Reported evaluation metrics reflect this specific dataset and the 80/20 train/test experimental split.
5. **Non-Causal Feature Importance**: Feature importance reflects Gini impurity reduction within decision trees and does not establish causal mechanisms.
6. **Early-Warning Decision Support Role**: The system should be treated as an analytical and early-warning decision-support tool rather than an automated decision-making system.
7. **Institutional Context Required**: Individual student predictions should always be interpreted with appropriate institutional context and professional advising judgment.

---

## 🔮 Future Scope
1. **Longitudinal Telemetry Integration**: Incorporating continuous, weekly learning management system (LMS) activity logs and assignment submission timelines.
2. **Cross-Discipline Generalization**: Evaluating the pipeline across non-computing academic departments (e.g. business, humanities, life sciences) to test model transferability.
3. **Advisor Feedback Calibration**: Creating feedback loops where academic counselors record student intervention outcomes, enabling continuous retraining and calibration.

---

## 📜 Academic Integrity Statement
This project was developed strictly adhering to academic integrity guidelines. All summary statistics, KPIs, machine learning metrics, confusion matrices, and model comparisons originate from actual execution of `analysis_pipeline.py` on the verified `student_data.csv`. No results, figures, or metrics have been fabricated.
