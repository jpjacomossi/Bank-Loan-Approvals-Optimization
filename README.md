# Optimization of Bank Loan Approvals

## Table of Contents

1. [Background and Overview](#background-and-overview)  

2. [Data Structure Overview](#data-structure-overview)

3. [Executive Summary](#executive-summary)

4. [Handling Missing Values](#handling-missing-values)  
   4.1 [Overview](#41-overview)  
   4.2 [Missing Data Analysis](#42-missing-data-analysis)  
   4.3 [Diagnosing Missing Data Mechanisms](#43-diagnosing-missing-data-mechanisms)  
   4.4 [Methods for Handling Missing Values](#44-methods-for-handling-missing-values)  
   4.5 [Performance Evaluation](#45-performance-evaluation)  
   4.6 [Final Imputation](#46-final-imputation)  
   4.7 [Conclusion](#47-conclusion)

5. [Credit Score Methodology](#credit-score-methodology)

6. [Insights Deep Dive](#insights-deep-dive)  
   6.1 [Default Rate Overview](#61-default-rate-overview)  
   6.2 [Loan Characteristics](#62-loan-characteristics)  
   6.3 [Behavioral Risk Indicators](#63-behavioral-risk-indicators)  
   6.4 [Comparative Analysis of Loan Defaults](#64-comparative-analysis-of-loan-defaults)  
   6.5 [Credit Score Insights](#65-credit-score-insights)

7. [Predictive Modeling Insights](#predictive-modeling-insights)  
   7.1 [Model Performance and Selection](#71-model-performance-and-selection)  
   7.2 [SHAP Analysis: Enhancing Model Interpretability](#72-shap-analysis-enhancing-model-interpretability)  
   7.3 [Impact of Predictive Model on Loan Approvals and Risk Management](#73-impact-of-predictive-model-on-loan-approvals-and-risk-management)

8. [Recommendations](#recommendations)

9. [Tools and Technologies](#tools-and-technologies)


---

## Background and Overview

### 1. Introduction 

The Consumer Credit Department of a bank aims to enhance its loan approval process by automating decision-making through a data-driven model. This initiative seeks to optimize lending decisions while adhering to the principles of the Equal Credit Opportunity Act (ECOA), ensuring fairness and compliance.

The proposed model will be developed using historical data from recent loan applicants, **enabling the bank to predict the likelihood of default**. By leveraging predictive analytics, the bank intends to **proactively identify high-risk applicants while maintaining transparency in its decision-making process**. This ensures that, in cases of loan denial, clear and understandable explanations can be provided.

The dataset comprises information on 5,960 bank loans, of which 1,189 applicants (approximately 20%) defaulted. The data includes 13 original variables for each applicant, offering a foundation for building a model capable of identifying potential risks with precision and reliability.

This predictive approach will enable the bank to:
- Improve its lending decisions.
- Mitigate financial risks.
- Responsibly extend credit to those in need.

By achieving these objectives, the bank can maintain a balance between profitability and social responsibility, fostering trust and long-term sustainability in its operations.


---

### 2. Business Problem
The bank faces a **high loan default rate of nearly 20%**, significantly exceeding the industry standard of 1–5%. This elevated default rate:
- Negatively impacts the bank’s profitability and reputation.
- Poses a serious challenge to the bank’s long-term success and competitiveness.

---

### 3. Business Goal
The primary goal of this project is to **optimize the loan approval process** to address the high default rate. Through improved risk assessment and transparent decision-making, the bank aims to:
- Reduce loan defaults and mitigate credit risk.
- Enhance profitability and minimize financial losses.
- Strengthen its reputation for responsible lending, fostering sustainable growth and competitive advantage.

---

### 4. Analytics Goals
The analytics efforts focus on achieving the following outcomes:

1. **Exploratory Data Analysis (EDA)**:
   - Gain insights into the dataset.
   - Identify patterns and key factors influencing loan defaults and customer behavior.

2. **Classification Model Development**:
   - Build a predictive model to classify the likelihood of customer default.
   - Support the bank in assessing credit risk and improving decision-making for loan approvals.


---

## Data Structure Overview

The dataset used for this project contains **5,960 observations** and **13 variables**, providing information on loan applicants and their financial behavior. To ensure clarity and consistency during the analysis, the following initial preprocessing steps were performed:

- The variable `BAD` was renamed to `DEFAULT` for better interpretability.
- Missing values were replaced with `NA` to facilitate handling during analysis.
- Categorical and binary variables (`DEFAULT`, `REASON`, and `JOB`) were converted into factors to allow proper handling by statistical models.

### 1. Original Variables

| **Variable Name**  | **Definition**                                                                                   | **Type**          |
|---------------------|-------------------------------------------------------------------------------------------------|-------------------|
| `DEFAULT`          | Indicates whether an applicant defaulted on their loan (1 = Defaulted, 0 = Not Defaulted).       | Categorical/Binary |
| `LOAN`             | The amount of the loan requested by the applicant.                                               | Numeric (Integer) |
| `MORTDUE`          | Amount due on an existing mortgage.                                                              | Numeric           |
| `VALUE`            | Current value of the applicant’s property.                                                       | Numeric           |
| `REASON`           | Reason for the loan (Debt Consolidation = "DebtCon" or Home Improvement = "HomeImp").            | Categorical       |
| `JOB`              | Applicant’s occupational category.                                                               | Categorical       |
| `YOJ`              | Years at present job.                                                                            | Numeric           |
| `DEROG`            | Number of major derogatory credit reports (e.g., bankruptcies, liens).                           | Numeric (Integer) |
| `DELINQ`           | Number of delinquent credit lines.                                                               | Numeric (Integer) |
| `CLAGE`            | Age of the oldest credit line in months.                                                         | Numeric           |
| `NINQ`             | Number of recent credit inquiries.                                                               | Numeric (Integer) |
| `CLNO`             | Total number of credit lines the applicant has.                                                  | Numeric (Integer) |
| `DEBTINC`          | Debt-to-income ratio, representing the applicant’s monthly debt obligations relative to income.   | Numeric           |

### 2. Derived Variables

| **Variable Name**      | **Definition**                                                                                              | **Type** |
|-------------------------|----------------------------------------------------------------------------------------------------------|----------|
| `EQUITY`               | Applicant’s equity in the property (calculated as `VALUE - MORTDUE`).                                      | Numeric  |
| `UNSECURED_LOAN`       | Portion of a loan not covered by the collateral's equity (calculated as `max(0, LOAN - max(EQUITY, 0))`).   | Numeric  |
| `DEFINITIVE_LOSS`      | Actual financial loss incurred by the lender for defaulted applicants (calculated as `max(0, LOAN - max(EQUITY, 0)) * 1_{DEFAULT=1}`). | Numeric  |
| `LTV_RATIO`            | Loan-to-Value ratio (calculated as `LOAN / VALUE`).                                                       | Numeric  |
| `LTE_RATIO`            | Loan-to-Equity ratio (calculated as `LOAN / EQUITY`).                                                     | Numeric  |
| `OWNERSHIP`            | Percentage of property owned by the applicant (calculated as `EQUITY / VALUE`).                           | Numeric  |
| `CREDIT_SCORE`         | Custom credit score reflecting applicant’s financial behavior and risk profile.                            | Numeric  |
| `SCORE_CATEGORY`       | Credit score category (`Poor`, `Fair`, `Good`, `Very Good`, `Excellent`) based on custom thresholds.       | Categorical |

This dataset forms the foundation for exploring loan defaults, assessing credit risk, and building a predictive model to optimize loan approvals while ensuring fairness and transparency.

---

## Executive Summary
  
   By leveraging predictive modeling and conducting a comprehensive exploratory data analysis, this project successfully optimized the bank's loan approval process, reducing its high 20% default rate. The approach supported profitability, promoted sustainable growth, and ensured compliance with the Equal Credit Opportunity Act (ECOA).

   After addressing missing data and inconsistencies, **5,572 records** were retained for analysis. Key predictors of loan defaults included:
- Debt-to-income ratio
- Delinquency history
- Derogatory records
- Credit line age
- Number of credit lines
- Loan amount

A **credit scoring system** inspired by the FICO methodology was developed to categorize applicants into risk groups, such as **"Poor," "Fair," "Good,"** and **"Excellent."** This system provided additional insights into borrower behavior and lending patterns:

- **Borrowers in the "Excellent" category exhibited the lowest default rates (4.4%) and the highest average loan amounts (over $21,000).**
  - Their financial stability and reliability make them valuable customers.
  - They are ideal candidates for loan approvals with minimal risk.
  - They contribute significantly to the bank’s profitability through larger loan approvals.

- **Borrowers in the "Poor" category had default rates exceeding 50% and received lower average loan amounts.**
  - These individuals posed the highest risk.
  - They necessitate cautious lending practices and stringent approval criteria to safeguard the bank’s financial interests.

After evaluating multiple machine learning algorithms, the **Random Forest model** demonstrated the best performance. During the test phase it achieved:
- **Accuracy**: 92.11%
- **Sensitivity**: 91.51%
- **Precision**: 73.48%

The predictive model demonstrated the ability to significantly reduce the current **20%** default rate to an estimated **2.11%** default rate. Comparisons between historical and model-driven scenarios highlighted:
- **Reductions in defaulted loan amounts** from **$18,860,200 to $491,100**.
- **Decreases in definitive losses** by **$1,321,568 (98%)**.
- **Decreases in recoverable losses** by **$17,047,532 (97%)**.
- **Adjusted total loan issuance from $104,783,000 to $84,764,800**, reflecting more prudent lending decisions.

By integrating this model into its approval process, alongside insights from the credit scoring system, the bank can:
- Strengthen its risk management capabilities.
- Minimize financial losses.
- Offer more competitive lending terms to reliable borrowers.

This initiative positions the bank for long-term growth, improved profitability, and enhanced customer trust.

---

## Handling Missing Values

### 1. Overview

Handling missing values is a critical step in ensuring the quality and reliability of the dataset used for analysis and modeling. Missing data, if left unaddressed, can introduce bias, reduce statistical power, or lead to inaccurate conclusions. This section outlines the nature of the missing data in the dataset, the diagnostic tests performed, and the methods implemented to address missing values effectively.

---

### 2. Missing Data Analysis

The original dataset contained **13 variables**, of which **11 variables had missing values**, accounting for more than **5%** of observations in most cases. The figure below show the percentage of missing values for each variable and the proportion of records by number of missing values: 

<div align="center">
  <img width="600" alt="image" src="https://github.com/user-attachments/assets/21c0e536-d8e2-444f-9904-28a289fd1977" />
</div>

**Key Observations:**
- **56.4%** of the records are complete cases.
- Removing all rows with missing data would reduce the dataset to **3,364 observations** (including only **300 defaults**), which is not ideal due to the significant loss of information.
- A threshold of **4 or more missing values per row** was set to remove rows with excessive missing data, resulting in **5,621 records** retained for further analysis. This approach minimized data loss while preserving data quality.

---

### 3. Diagnosing Missing Data Mechanisms

Understanding the nature of missingness is critical for selecting appropriate handling methods. Missing data mechanisms were classified as follows:
- **MCAR (Missing Completely At Random):** Missingness is entirely random and independent of observed or unobserved variables.
- **MAR (Missing At Random):** Missingness is related to observed data but not to the unobserved data itself.
- **MNAR (Missing Not At Random):** Missingness depends on the unobserved data.

**Diagnostic Steps:**
1. **Little’s MCAR Test:**
<div align="center">
  <img width="296" alt="image" src="https://github.com/user-attachments/assets/a0238b27-73da-4d39-a5d2-81718f104083" />
</div>

   - The p-value obtained was **< 0.05**, indicating that the missing data is **not MCAR**.

2. **Logistic Regression for MAR:**

<div align="center">
  <img src="https://github.com/user-attachments/assets/7e5af1c6-728a-4d0c-89f1-c183c335a30f" alt="image" />
</div>

   - Logistic regression models were used to assess whether missingness in one variable could be predicted by other observed variables. Significant predictors indicated the missingness followed the MAR mechanism.

3. **Missing Data Patterns:**
<div align="center">
  <img width="498" alt="image" src="https://github.com/user-attachments/assets/3c62f436-1d62-42d9-bba2-fd2a79f0f568" />
</div>

   - Correlation heatmaps revealed moderate correlations between missing values in variables like `DEROG`, `DELINQ`, and `NINQ`, suggesting shared underlying factors influencing missingness.

---

### 4. Methods for Handling Missing Values

Given that the missing data followed the MAR mechanism, sophisticated imputation methods were explored. These included:

1. **Simple Imputation (Mean/Median):**
   - Straightforward but risks underestimating variability and distorting relationships between variables.

2. **Predictive Imputation (Single Imputation):**
   - Techniques like K-Nearest Neighbors (KNN) and Random Forest were employed to predict missing values based on observed variables.

3. **Multiple Imputation:**
   - This approach fills missing values multiple times, generating plausible datasets that capture variability and uncertainty.

---

### 5. Performance Evaluation

To evaluate the imputation methods, a complete-case dataset was subjected to 20% artificially induced missingness. The imputed values were compared against the true values using **Root Mean Squared Error (RMSE)** as the performance metric. For **MICE**, the RMSE was averaged across all five imputed datasets.

**Results:**

| **Variable**  | **MICE (PMM)** | **MICE (Random Forest)** | **MissForest** | **KNN** |
|---------------|----------------|--------------------------|----------------|---------|
| `DEBTINC`     | 10.62          | 8.92                    | **5.58**       | 6.85    |
| `YOJ`         | 10.32          | 7.88                    | **4.05**       | 8.17    |
| `MORTDUE`     | 34,738.04      | 28,638.90               | **12,919.87**  | 40,485.32 |
| `VALUE`       | 40,358.41      | 32,277.25               | **16,785.01**  | 43,338.08 |
| `DEROG`       | 0.82           | 0.71                    | **0.51**       | 0.64    |
| `DELINQ`      | 1.11           | 0.93                    | **0.65**       | 0.80    |
| `CLAGE`       | 108.63         | 82.58                   | **44.10**      | 76.23   |
| `NINQ`        | 2.04           | 1.72                    | **0.95**       | 1.35    |

**Key Findings:**
- **MissForest** consistently outperformed other methods, achieving the lowest RMSE across most variables.
- MICE with Random Forest was the second-best method, but its performance lagged behind MissForest, especially for variables with a high percentage of missing values.
- KNN imputation showed the highest RMSE, indicating less accurate imputations.

---
### 6. Final Imputation

Using the **MissForest** algorithm, missing values in the dataset were imputed. This method effectively preserved the overall data structure and maintained consistency with original variable distributions. For example:
- The mean of `DEBTINC` changed marginally from **34.075** to **34.586**, demonstrating that the imputed values aligned closely with the original data.

Additionally, **49 records** were removed where the **Credit Line Age (CLAGE)** was greater than zero while the **Number of Credit Lines (CLNO)** was zero. These records represented logical inconsistencies and were excluded to maintain data integrity.

The resulting imputed dataset contains **5,572 records**, retaining a ratio of default and non-default cases very similar to the original. This ensured that the analysis and modeling were based on high-quality data with minimal bias.

---
### 7. Conclusion

Handling missing values improved the quality and reliability of the dataset. By using **MissForest**, the imputed dataset retained its integrity, enabling accurate predictive modeling and actionable insights for optimizing loan approvals. This step laid the foundation for robust analytics and informed decision-making in subsequent stages of the project.

---
## Credit Score Methodology

### 1. Overview
To assess borrower risk effectively, a custom credit scoring system was developed inspired by the FICO model. This system was tailored specifically for this project and is based on the dataset's key financial variables. It assigns weighted scores to critical financial factors, calculates a raw credit score for each applicant, and scales the results to the FICO range of **300 to 850** for intuitive interpretation.

---

### 2. Factors and Weights
The scoring system incorporates five categories, aligned with industry standards, each weighted according to its relative importance in predicting credit risk:

| **Category**          | **Dataset Variable**            | **Weight (%)** | **Logic**                                                                                      |
|------------------------|---------------------------------|----------------|-----------------------------------------------------------------------------------------------|
| **Payment History**    | `DELINQ` (Delinquencies)       | 35%            | Measures missed payments, with penalties for frequent delinquencies.                         |
| **Amounts Owed**       | `DEBTINC` (Debt-to-Income Ratio)| 30%           | Evaluates debt burden relative to income, penalizing high debt levels.                        |
| **Credit History**     | `CLAGE` (Credit Line Age)      | 15%            | Rewards longer credit histories, which signal financial stability.                            |
| **Ownership Ratio**    | `OWNERSHIP` (Equity / Property Value) | 10%   | Assesses borrower equity as a percentage of property value, emphasizing financial stake.      |
| **New Credit**         | `NINQ` (Number of Inquiries)   | 10%            | Penalizes frequent credit inquiries, indicating credit-seeking behavior.                     |

---

### 3. Scoring Rules and Thresholds
Each factor was binned into categories reflecting varying levels of financial risk. Points were assigned to each bin, scaled so that the maximum raw score aligns with the relative importance of the factor.

**Payment History (DELINQ)**
- **Rationale**: Frequent missed payments increase default risk. Delinquencies reflect an applicant’s ability to manage debt.
- **Thresholds**:
  - **0 delinquencies**: 50 points
  - **1 delinquency**: 30 points
  - **2+ delinquencies**: 10 points

**Amounts Owed (Debt-to-Income Ratio, DEBTINC)**
- **Rationale**: High debt-to-income (DTI) ratios signal financial strain and a higher likelihood of default.
- **Thresholds**:
  - **DTI < 20%**: 50 points (low financial strain)
  - **20% ≤ DTI ≤ 35%**: 30 points (moderate financial strain)
  - **DTI > 35%**: 10 points (high financial strain)

**Length of Credit History (CLAGE)**
- **Rationale**: Longer credit histories demonstrate financial stability and responsible borrowing over time.
- **Thresholds**:
  - **CLAGE > 120 months (10 years)**: 50 points
  - **60 ≤ CLAGE ≤ 120 months**: 30 points
  - **CLAGE < 60 months**: 10 points

**Ownership Ratio (OWNERSHIP)**
- **Rationale**: Higher ownership ratios indicate significant borrower equity, reducing default risk.
- **Thresholds**:
  - **OWNERSHIP > 50%**: 50 points
  - **20% ≤ OWNERSHIP ≤ 50%**: 30 points
  - **OWNERSHIP < 20%**: 10 points

**New Credit (Number of Inquiries, NINQ)**
- **Rationale**: Frequent credit inquiries suggest aggressive credit-seeking behavior, which correlates with higher risk.
- **Thresholds**:
  - **0 inquiries**: 50 points
  - **1 inquiry**: 30 points
  - **2+ inquiries**: 10 points

---

### 4. Scaling the Credit Score
The raw score for each borrower was calculated by summing the points from all factors. To convert these raw scores into a range consistent with FICO (300–850), the following formula was applied:

Scaled Score = 300 + ((Raw Score) / (Max Raw Score) × 550)

- **Raw Score**: The total points earned based on the applicant's characteristics.
- **Max Raw Score**: The maximum possible points across all factors (e.g., 250 points).
- **300–850 Scale**: Ensures compatibility with industry-standard credit scoring.

---

### 5. Thresholds for Credit Score Categories
To align with standard FICO categories, the scaled scores were divided into the following risk groups:

| **Category**    | **Score Range** | **Risk Description**                                                                        |
|------------------|-----------------|-------------------------------------------------------------------------------------------|
| **Poor**        | 300–579         | High risk of default; history of missed payments, high debt, short credit history.        |
| **Fair**        | 580–669         | Moderate risk; may have some delinquencies or higher debt but manageable overall.         |
| **Good**        | 670–739         | Low risk; stable financial behavior and responsible borrowing history.                    |
| **Very Good**   | 740–799         | Very low risk; long credit history, minimal debt, and consistent payments.                |
| **Excellent**   | 800–850         | Minimal risk; exemplary credit management and financial stability.                        |

---

### 6. Transparency and Fairness
The custom credit score model:
- **Promotes Transparency**: Each factor and its contribution to the overall score are explicitly defined.
- **Ensures Fairness**: By adhering to objective thresholds and scaling rules, the system complies with ethical and regulatory standards, such as the Equal Credit Opportunity Act (ECOA).

By integrating this model into the loan approval process, the bank gains a powerful, interpretable tool for assessing borrower risk while fostering trust and transparency with applicants.

---

## Insights Deep Dive

This section highlights the key insights extracted from the exploratory data analysis and modeling phases, providing important conclusions for optimizing the loan approval process.

---

### 1. Default Rate Overview
<div align="center">
  <img width="600" alt="image" src="https://github.com/user-attachments/assets/2a59fc67-4d52-4df7-8c64-276801d5f611" />
</div>

- **High Default Rate**: The bank faces a default rate of approximately **20%**, significantly exceeding the industry standard of 1–5%.
- **Loan Security**: Most loans (82.93%) are fully secured, with default rates around 20%. Unsecured loans, though a small portion (<2%), have a much higher default rate of **32.43%**, emphasizing their riskiness.

---
### 2. Loan Characteristics

- **Default Rates Across Loan Amounts**:

   <div align="center">
     <img width="500" alt="image" src="https://github.com/user-attachments/assets/9cbc0ea5-8aa1-4654-9a41-e5ac0978f3c7" />
   </div>   

  - **Average Loan**: $18,846 with a typical range between $11,300 and $23,500.
  - **High Risk**: Smaller loans (<$10,000) and larger loans ($35,000>)tend to have higher default rates (above 20%).
  - **Lowest Risk**: Loans between $10,000–$35,000 show the lowest default rates.
  - **Very Larger Loans**: Loans exceeding $75,000 exhibit extremely elevated default rates, likely due to over-leveraging.

---

### 3. Behavioral Risk Indicators

- **Debt-to-Income Ratio (DEBTINC)**:

   <div align="center">
     <img width="500" alt="image" src="https://github.com/user-attachments/assets/0e8d5a14-23bf-4896-b720-4d551b2f787e" />
   </div>

  - Borrowers with **DEBTINC > 50%** have a **100% default rate**, indicating extremely high risk.
  - Borrowers with **DEBTINC between 0–10%** also show **default rates above 50%**, possibly due to other factors like unemployment.
  - The safest range is **10–30%**, where default rates are lowest, making this the ideal lending zone.
  - Borrowers outside this range, particularly those above 50%, pose significant financial risks.

- **Number of Inquiries (NINQ)**:
  
     <div align="center">
     <img width="500" alt="image" src="https://github.com/user-attachments/assets/d286ff71-5bcf-4665-8ad6-e080c0bc2f3c" />
   </div>

  - Higher NINQ correlates with higher default rates.
  - Borrowers with **12+ inquiries** face a **100% default rate**, indicating extreme financial instability.
  - A generally linear trend exists, except for inquiries between **7–11**, where the pattern slightly deviates.

- **Derogatory Marks (DEROG)**:
  
   <div align="center">
        <img width="500" alt="image" src="https://github.com/user-attachments/assets/387061fd-5718-429e-8900-1a565acf9879" />
      </div>

  - Default rates increase with more derogatory marks.
  - At **2+ derogatory marks** there are more defaulters than non-defaulters.
  - For those with **5+ marks**, the **default rate reaches 100%**, making derogatory marks a strong predictor of risk.

- **Delinquencies (DELINQ)**:
  
   <div align="center">
     <img width="500" alt="image" src="https://github.com/user-attachments/assets/a9a22081-49b1-408a-a086-28d307739b5e" />
   </div>

  - Borrowers with **0 delinquencies** have a low **13.8% default rate**, reflecting financial reliability.
  - With **1 delinquency**, the default rate rises to **27.1%**, doubling to **42.9%** with **2 delinquencies**.
  - Beyond **3 delinquencies**, default rates exceed **50%**, reaching **81.1%** for **5 delinquencies**.
  - Borrowers with **6+ delinquencies** face a **100% default rate**, highlighting delinquencies as a critical risk factor.

- **Credit Line Age (CLAGE)**:

   <div align="center">
     <img width="500" alt="image" src="https://github.com/user-attachments/assets/4ade5d0a-da33-4b3f-b266-049f0a912e7f" />
   </div>

  - Borrowers with **shorter credit histories (<180 months)** show higher default rates.
  - Borrowers with **longer credit histories (>180 months)** are less likely to default.

---

### 4. Comparative Analysis of Loan Defaults
- **Recoverable vs Definitive Losses**:

    <div align="center">
     <img width="500" alt="image" src="https://github.com/user-attachments/assets/f801e60e-314d-47f3-9d90-3ff48a2cde51" />
   </div>

  - **92.89% of defaulted loans** are recoverable through collateral, reducing the bank’s long-term exposure.
  - **Definitive losses** represent only **7.11%** of total defaults, showing that a strong collateral policy mitigates most financial risks.

---
### 5. Credit Score Insights

- **Default Rates Across Credit Score Categories**:

     <div align="center">
     <img width="700" alt="image" src="https://github.com/user-attachments/assets/dce9632a-fd1a-4dc1-b5f4-9fc2bbc8120a" />
   </div>

  - Borrowers in the **Poor credit category (300–579)** have the highest default rate, exceeding **50%**.
  - Default rates drop significantly in the **Fair (26%)**, **Good (11.7%)**, **Very Good (5%)**, and **Excellent (4.4%)** categories.
  - The majority of applicants fall into the **Fair (30.4%)** and **Good (34.8%)** categories, while only **1.6%** belong to the **Excellent** group.

- **Default Rate and Average Loan Amount vs Credit Score**:

   <div align="center">
     <img width="500" alt="image" src="https://github.com/user-attachments/assets/36792e0c-e613-4cb8-ac15-383550306b58" />
   </div>

   - **Default Rates**:
     - Scores **below 500** have high default rates; restricting loans to **600+** reduces defaults to **<15%**.
     - Beyond **730**, default rates drop below **5%**, indicating minimal risk.

   - **Loan Amounts**:
     - Average loan amounts remain stable (~$20,000) when scores **below 700** are considered.
     - Scores **above 730** see loan amounts rise, exceeding **$30,000** for scores over **800**.

   - **730 Threshold**:
     - Serves as a critical benchmark for reduced risk and larger loan approvals.


- **Default Risk and Average Loan Amount by Credit Score Category**:

   <div align="center">
     <img width="600" alt="image" src="https://github.com/user-attachments/assets/7293d3e5-3350-4c6a-a517-6ef675b661b4" />
   </div>

  - Borrowers in the **Poor** category have a default rate around **50%** and the lowest average loan amounts (below $18,000).
  - **Fair category** borrowers have a default rate of **30%**, with average loan amounts slightly higher than **Poor** borrowers.
  - Default risk continues to decline across the **Good (12%)**, **Very Good (5%)**, and **Excellent (4.4%)** categories, with average loan amounts peaking for **Excellent** borrowers (over $21,000).


- **Impact of Credit Score Thresholds on Default Rate, Loan Amount, and Profit**:

     <div align="center">
     <img width="700" alt="image" src="https://github.com/user-attachments/assets/3215984c-ec7a-45b2-a0bb-f745cdea2df2" />
   </div>

  - Increasing the credit score threshold reduces default rates, particularly beyond the **732 threshold**, where default rates drop below **5%**.
  - Higher thresholds result in fewer loans being issued, reducing the total loan amount and profit.
  - A trade-off exists: while higher thresholds decrease risk, they also lower profitability due to fewer approved loans.

---
## Predictive Modeling Insights

The predictive modeling phase provided valuable insights into the bank's loan approval and default prediction process. The dataset was partitioned into training (70%), validation (20%), and testing (10%) sets to ensure robust model evaluation.By evaluating several machine learning algorithms, the strengths and limitations of each model were identified, allowing for the selection of the most effective strategy for optimizing loan approval processes.

---

### 1. Model Performance and Selection

| **Model**            | **Accuracy** | **Sensitivity** | **Specificity** | **Precision** | **AUC**  |
|-----------------------|--------------|-----------------|-----------------|---------------|----------|
| Logistic Regression   | 78.73%      | 73.66%          | 80.00%          | 48.10%        | 0.8460   |
| Decision Tree         | 85.10%      | 55.36%          | 92.58%          | 65.26%        | 0.7595   |
| KNN                   | 87.97%      | 40.63%          | 99.88%          | **98.91%**    | 0.9216   |
| Random Forest         | **91.38%**  | **92.86%**      | **91.01%**      | 72.22%        | **0.9711** |

#### Key Insights:
- **Random Forest** was selected as the best model for its balance of **high sensitivity (92.86%)**, **precision (72.22%)**, and **AUC (0.9711)**.
- While **KNN** had the highest **precision (98.91%)**, its poor sensitivity (40.63%) made it less effective for capturing defaulters.
- Random Forest ensures reliable predictions with minimal missed defaults, making it optimal for managing credit risk.
- During the test set the model achieved a **Sensitivity of 91.51%** and **Precision of 73.48%**.

---

### 2. SHAP Analysis: Enhancing Model Interpretability

**SHAP (Shapley Additive Explanations)** was used to improve the interpretability of the **Random Forest model** by quantifying each feature's influence on individual predictions. This ensures transparency and compliance with **ECOA regulations**.

- **How SHAP Works**:
  - Derived from **Shapley values** in game theory, SHAP calculates each feature's contribution to predictions by evaluating all possible feature combinations.
  - SHAP values indicate whether a feature pushes a prediction towards default (class 1) or non-default (class 0).

This approach provides clear insights into how features influence predictions, making the model more transparent and reliable for decision-making.

 **SHAP Value Interpretation**:
- For individual predictions, SHAP values clearly show how features push the default probability higher or lower.

<div align="center">
  <img width="1600" alt="Screenshot 2025-01-23 at 5 53 38 PM" src="https://github.com/user-attachments/assets/67a3b120-c690-4d15-a0be-521e0c3e75ea" />
</div>

   - Example:
     - **Row 1578 (Default)**: High DEBTINC and short CLAGE contributed positively to the predicted probability of default.
     - **Row 2370 (Non-Default)**: Longer CLAGE and no delinquencies reduced the predicted probability of default.

### 3. SHAP vs Model Predictions

<div align="center">
  <img width="600" alt="image" src="https://github.com/user-attachments/assets/ca555818-4d48-4911-b333-7a8710d3c60b" />
</div>

- The plot compares **Random Forest model probabilities (blue)** with **SHAP-calculated probabilities (red)** for 200 records.
- Probabilities align closely, with minor discrepancies at higher values where SHAP shows sharper spikes.
- A **Mean Absolute Error (MAE) of 3.23%** confirms SHAP's accuracy in representing the model's predictions.

With the integration of **SHAP** analysis, the Random Forest model not only excels in predictive performance but also provides a transparent framework for understanding the factors driving loan default risk.

---
### 4. Impact of Predictive Model on Loan Approvals and Risk Management

As it was discussed earlier, the predictive model enhances decision-making by providing clear insights into loan default probabilities, enabling accurate risk assessment, improved profitability, and transparent loan approvals. This section highlights the model's key improvements achieved.

---

#### Actual vs Estimated Default Rate with Model-Based Approvals

<div align="center">
  <img width="600" alt="image" src="https://github.com/user-attachments/assets/ced1649b-347d-44c2-b69d-32970a02c3ac" />
</div>


- **Historical Default Rate**: 19.76%.
- **Estimated Model-Based Default Rate**: 2.11% (an **89% reduction**).
- **Key Insight**: The model's predictions significantly lower financial risk and reduce defaults.
---

#### Loans Issued vs Defaulted (Pre-Model vs Post-Model)

<div align="center">
  <img width="700" alt="image" src="https://github.com/user-attachments/assets/091031fa-1c2d-4ef6-82bf-2931c0ed789d" />
</div>


- **Pre-Model**:
  - **Total Loans Issued**: $104,783,000.
  - **Defaulted Loans**: $18,860,200.
- **Post-Model**:
  - **Total Loans Issued**: $84,764,800.
  - **Defaulted Loans**: $491,100 (a **97% reduction**).
- **Insight**: While total loans issued decreased by **19%**, the model drastically reduced defaulted loan amounts, enhancing portfolio quality and financial stability.

---

#### Definitive vs Recoverable Losses (Pre-Model vs Post-Model)

<div align="center">
  <img width="700" alt="image" src="https://github.com/user-attachments/assets/950a808d-17b1-4228-a56d-0b32df22e894" />
</div>

- **Pre-Model**:
  - **Definitive Losses**: $1,352,398.
  - **Recoverable Losses**: $17,507,802.
- **Post-Model**:
  - **Definitive Losses**: $30,830 (a **98% reduction**).
  - **Recoverable Losses**: $460,270 (a **97% reduction**).
- **Insight**: The model significantly minimizes both definitive and recoverable losses, improving resource allocation and overall portfolio quality.

---

### 5. Conclusion

The predictive model demonstrates exceptional capability in reducing default rates and financial losses while ensuring smarter loan approvals. It enables the bank to target low-risk applicants with larger loans and better terms, fostering portfolio stability and profitability in real-world scenarios.

## Recommendations

Based on the insights gained from the analysis and modeling phases, the following recommendations are proposed:

---

### 1. **Transparent Client Education**
- Educate clients on how their **credit scores impact loan approvals**.
- Provide **financial health resources** to help clients improve their credit scores, ensuring better loan opportunities.
- Enhance transparency and build trust by explaining how loan approval decisions are made.

---

### 2. **Integrate the Predictive Model into Lending Processes**
- Incorporate the predictive model into **loan approval workflows** to identify and filter high-risk applicants, ensuring **data-driven decision-making**.
- Leverage **SHAP methodology** to make model-driven decisions more transparent, offering clear explanations for loan approvals or denials.
- Ensure compliance with **Equal Credit Opportunity Act (ECOA)** regulations by providing clients with justifiable reasons for adverse decisions.

---

### 3. **Focus on Low-Risk, High-Value Clients**
- Prioritize applicants with **credit scores over 730**, who demonstrate financial stability and reliability.
- Implement **risk-based pricing** to:
  - Offer **competitive interest rates** to attract and retain low-risk clients.
  - Adjust rates for higher-risk borrowers to reflect their increased likelihood of default.
- Use the predictive model to generate **targeted pre-approvals** for clients with strong credit profiles.
- Offer **exclusive benefits** to high-value clients, such as:
  - **Higher loan limits**.
  - **Loyalty rewards** programs.
  - Personalized services to ensure long-term engagement and satisfaction.

---

### 4. **Careful Approach to Loans Below $10,000**
- Enforce stricter approval criteria for smaller loans (<$10,000) due to their higher default risk.
- Apply **higher interest rates** to manage financial exposure while maintaining profitability.
- Regularly monitor and assess default risk trends associated with these loans to ensure sustainable lending practices.

---

By adopting these recommendations, the bank can significantly improve its loan approval processes, reduce default rates, and enhance profitability. Combining predictive analytics with transparency and targeted strategies will ensure long-term success in managing credit risk.

---

## Tools and Technologies

### Programming Language
- **R**: Used for data preprocessing, analysis, modeling, and visualization.

### Libraries and Packages

- **Data Preprocessing and Imputation**: 
  - `dplyr`, `tidyr`, `mice`, `missForest`, `VIM`, `naniar`, `fastDummies`
- **Visualization**: 
  - `ggplot2`, `ggcorrplot`, `plotly`, `gridExtra`, `rpart.plot`
- **Feature Selection**: 
  - `Boruta`
- **Modeling and Evaluation**: 
  - `caret`, `rpart`, `randomForest`, `class`, `pROC`, `stats`, `car`
- **Model Interpretation**: 
  - `fastshap`

