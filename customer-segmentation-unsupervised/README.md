

                                      Advanced Customer Segmentation Using Clustering Algorithms
 
 1️⃣ Project Title: 
 Advanced Customer Segmentation and Clustering Analysis on Telecom Churn Dataset

 2️⃣ Problem Statement:  
Telecom companies face challenges in understanding customer behavior and identifying high-value or high-risk customers.  
The goal of this project is to perform advanced customer segmentation using unsupervised clustering algorithms to:

- Identify meaningful customer groups
- Detect high-value customers
- Analyze churn-related behavioral patterns
- Provide business insights for decision-making


3️⃣ Dataset Description:

Dataset Used: **churn-bigml.csv**

The dataset contains telecom customer data including:

- Account Length
- State
- Area Code
- International Plan
- Voice Mail Plan
- Total Day Minutes & Charges
- Total Evening Minutes & Charges
- Total Night Minutes & Charges
- Total International Minutes & Charges
- Customer Service Calls
- Churn (Target variable)

The dataset is used for clustering and segmentation analysis.

 4️⃣ Algorithms Used:

The following clustering algorithms were implemented and compared:

### 🔹 KMeans Clustering
- Partition-based clustering
- Used PCA for dimensionality reduction before clustering
- Efficient for large datasets

### 🔹 DBSCAN (Density-Based Clustering)
- Density-based clustering
- Detects noise and outliers
- Does not require predefined number of clusters

### 🔹 Hierarchical Clustering
- Agglomerative approach
- Builds cluster hierarchy
- Useful for visualizing cluster relationships using dendrogram



5️⃣ How to Run the Project:

 Step 1: Install Dependencies

```bash:
pip install -r requirements.txt

Key Results:
 Number of Clusters Found
KMeans: 4 Clusters
DBSCAN: 3 Clusters + Noise Points
Hierarchical: 4 Clusters

Best Algorithm:
KMeans performed best based on:
Clear cluster separation
Balanced distribution
Better interpretability
Lower inertia score

Business Insights:
Identified high-value customers with high call charges and long account tenure.
Detected potential churn-risk customers with high customer service calls.
Segmented low-usage customers for targeted marketing campaigns.
Helped in understanding behavioral patterns across different customer groups.


7️⃣ Sample Visualizations:
Add screenshots of:
PCA 2D Cluster Plot (KMeans)
DBSCAN Cluster Plot
Hierarchical Dendrogram
CLV Distribution Chart
Feature Importance Bar Graph 