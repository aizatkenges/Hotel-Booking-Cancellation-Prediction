# Hotel Booking Cancellation Prediction

## Project overview

This project analyzes hotel booking data through four coursework notebooks. It covers exploratory data analysis, feature engineering, cancellation prediction, and booking segmentation.

The notebooks remain separate and preserve their original code. 

## Dataset

The dataset contains 119,390 hotel booking records and 32 columns. The target variable for classification is `is_canceled` (`1` = canceled, `0` = not canceled).

**Source:** [Hotel Booking Demand on Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

Download `hotel_bookings.csv` and place it in the repository root. The notebooks use this filename with a relative path. 

## Notebooks

| Notebook | Topics |
|---|---|
| `01_SIS1_EDA_sampling_feature_engineering.ipynb` | Exploratory data analysis, sampling strategies, feature engineering and selection, PCA, and a classification workflow. |
| `02_SIS2_feature_analysis_PCA_outliers.ipynb` | Feature scaling, distance and similarity measures, feature analysis, PCA, and outlier detection. |
| `03_SIS4_cancellation_classification.ipynb` | Logistic Regression, Decision Tree, and KNN; cross-validation, hyperparameter search, ANOVA, feature selection, and classification metrics. |
| `04_SIS5_clustering_ensembles.ipynb` | K-means, hierarchical clustering, DBSCAN, Random Forest, AdaBoost, Gradient Boosting, and model tuning. |

## Business findings

- **Sampling affects the estimated cancellation rate.** The full dataset had a cancellation rate of 37.04%. Stratified sampling preserved this rate, while a convenience sample of bookings with the longest lead times had a rate of 66.99%. This shows why a sample used for planning should reflect the broader booking population.

    In the full dataset, about 37 out of every 100 bookings were canceled.
  This is the overall cancellation rate in this dataset.

    In the selected group of bookings made far ahead of arrival, about 67 out
  of every 100 were canceled. This rate applies only to that group, not to all
  bookings. It suggests that lead time may help identify bookings with a
  higher cancellation risk. It does not prove that booking earlier causes
  cancellations.
- **Some booking attributes may help prioritize review.** Feature-selection methods repeatedly highlighted `lead_time` and `total_of_special_requests`. Previous cancellations, booking changes, and required parking spaces were also selected by some methods. 
- **Model choice depends on the decision being supported.** In SIS 4, the tuned Decision Tree achieved a higher test ROC-AUC than the default tree (0.920 vs. 0.844), while its F1-score was slightly lower (0.795 vs. 0.801). A hotel would need to weigh missed cancellations against unnecessary follow-up when choosing a model and prediction threshold.
- **Clusters need further validation before operational use.** K-means with `k = 3` had the highest silhouette score among the tested values. Cluster sizes and booking profiles should be examined to determine whether the resulting groups are useful to hotel staff.

These analyses describe historical patterns. 

## Methods and skills

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, SciPy, statsmodels, exploratory data analysis, sampling, feature engineering, feature selection, PCA, classification, cross-validation, hyperparameter tuning, model evaluation, K-means, hierarchical clustering, DBSCAN, ensemble learning.

## Run the notebooks

Run Jupyter from the repository root so the notebooks can find `hotel_bookings.csv`:

