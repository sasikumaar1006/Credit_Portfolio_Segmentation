# Credit Portfolio Segmentation & Risk Analysis Dashboard

An end-to-end data science and machine learning application designed to conduct deep credit risk profiling, borrower segmentation, and portfolio-level risk assessment on a massive scale (1.34M+ records). 

🌐 **Live Streamlit Application:** [Credit Portfolio Segmentation Dashboard](https://creditportfoliosegmentation-dkxfoxcher3gdl88zsajeg.streamlit.app/)

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Dataset & Architecture](#-dataset--architecture)
4. [Methodology & Machine Learning Pipeline](#-methodology--machine-learning-pipeline)
5. [Key Performance Metrics & Model Insights](#-key-performance-metrics--model-insights)
6. [Business-Driven Recommendations](#-business-driven-recommendations)

---

## 🔍 Project Overview
This project implements an advanced quantitative pipeline that combines dimensionality reduction, unsupervised clustering, and exploratory behavioral analysis. The insights are packaged into a high-performance, interactive executive dashboard deployed on Streamlit Cloud.

---

## 🎯 Problem Statement
Financial institutions face significant exposure to loan defaults, over-leveraged borrowers, and sub-optimal risk-based pricing structures. Traditional linear credit-scoring models often fail to capture non-linear behavioral clusters across multi-million-record portfolios. 

**Core Objectives:**
* **Borrower Profiling:** Group homogeneous borrower profiles using unsupervised clustering to uncover hidden credit tiers.
* **Risk Quantification:** Isolate default drivers using key financial ratios (`dti_n`, `fico_n`, `revenue_log`).
* **Portfolio Optimization:** Provide actionable risk-based pricing, dynamic DTI capping, and FICO tiering strategies for executive decision-making.

---

## 🗄️ Dataset & Architecture
* **Scale:** Processed over 1.34 million borrower records.
* **Storage Strategy:** Handled via **Git Large File Storage (LFS)** to securely manage cleaned datasets (`cleaned_loans_granting_model_dataset.csv`) while maintaining GitHub performance limits.
* **Core Features Engineered:**
  * `revenue_log`: Log-transformed annual income to normalize right-skewed distributions.
  * `dti_n`: Normalized Debt-to-Income ratio for precise indebtedness tracking.
  * `fico_n`: Normalized FICO credit score tiering.

---

## ⚙️ Methodology & Machine Learning Pipeline
1. **Preprocessing & Scaling:** Standardized numerical variables to ensure equal weight during distance-based computations.
2. **Dimensionality Reduction (PCA):** Reduced complex feature spaces down to 10 principal components while capturing ~58% of cumulative explained variance.
3. **Clustering Algorithms:** Applied **K-Means** and **Hierarchical Clustering** with optimal cluster validation achieving robust silhouette scores (0.3–0.4).

---

## 📊 Key Performance Metrics & Model Insights

* **Variance Explained:** The 10-component PCA configuration successfully compresses complex financial features while retaining ~58% of total variance, avoiding overfitting.
* **Cluster Separation:** Silhouette scores ranging between **0.3 and 0.4** indicate meaningful, moderately dense separation between low-, medium-, and high-risk borrower clusters.
* **Default Pattern Identification:** High DTI ratios combined with lower FICO tiers disproportionately concentrate default rates, highlighting the necessity of strict leverage boundaries.

---

## 💡 Business-Driven Recommendations

* **Risk-Based Pricing (RBP):** Automatically adjust interest rates and loan limits proportional to the borrower’s assigned cluster tier rather than a flat portfolio rate.
* **Dynamic DTI Capping:** Implement hard underwriting checkpoints for applications crossing specific `dti_n` thresholds to curb escalating default portfolios.
* **FICO Tiering Strategy:** Establish tailored grace periods and restructuring protocols for borderline tiers before delinquency hardens into default.

---
