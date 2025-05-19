# Predicting Forest Fires with Machine Learning

Project Repo: [https://github.com/devansha5/Forest-Fire-Prediction.git](https://github.com/devansha5/Forest-Fire-Prediction.git)

Video Demo: [https://drive.google.com/file/d/1CQkj\_t9LROlQXq3aq9SI7zfWAHQoDoWh/view?usp=drivesdk](https://drive.google.com/file/d/1CQkj_t9LROlQXq3aq9SI7zfWAHQoDoWh/view?usp=drivesdk)

**Summary:**
Wildfires pose a growing threat to ecosystems and communities. This project builds a two-stage machine learning pipeline to **predict whether a major fire (≥ 1 ha) will occur** and **estimate the burned area size**, using meteorological data from the UCI Forest Fires dataset. By combining classification and regression models, we aim to give firefighting agencies both early-warning and severity estimates to optimize resource allocation and preventive measures.

---

⭐ **Key Features**

* **Two-Stage Pipeline**

  * **Classification:** Forecast if a fire ≥ 1 ha will start
  * **Regression:** Estimate the burned area in hectares
* **High Performance**

  * 88 % classification accuracy on large fires
  * R² = 0.67 in burned-area regression
* **Robust Feature Engineering**

  * Log-transform and binarize targets
  * Interaction terms (e.g., temperature × wind)
  * Custom drought index
* **Hyperparameter Tuning & Validation**

  * GridSearchCV (5-fold)
  * F1, ROC-AUC for classification
  * R², RMSE, MAE for regression
* **Interpretability**

  * Feature-importance analysis highlights top drivers (temperature, humidity, wind, ISI)
* **Reproducible**

  * Clear setup instructions
  * Jupyter notebook walkthrough

---

🚀 **Executive Summary**

* **Problem:** Wildfires are increasingly destructive; need early and accurate forecasting
* **Data:** UCI Forest Fires (Portugal; 517 samples; meteorology & fire-weather indices)
* **Approach:**

  1. **Preprocess:** Clean data, encode seasons, log-transform `area`, train/test split (80/20)
  2. **Models:**

     * **Classification:** Logistic Regression (baseline), Random Forest
     * **Regression:** Linear Regression (baseline), Random Forest
  3. **Tune & Evaluate:** GridSearchCV, stratified splits, appropriate metrics
* **Results:**

  * **Classification:** Random Forest → 0.88 accuracy, 0.91 ROC-AUC
  * **Regression:** Random Forest → R² 0.67, RMSE 0.96, MAE 0.64
* **Impact:** Enables agencies to pre-position resources, issue targeted alerts, and plan controlled burns

---

🏁 **Getting Started**

1. **Clone the repo:**

   ```bash
   git clone https://github.com/devansha5/Forest-Fire-Prediction.git
   cd Forest-Fire-Prediction
   ```
2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
3. **Run the analysis:**

   ```bash
   jupyter notebook Forest_Fire_Prediction.ipynb
   ```
4. **Explore results:**

   * Classification vs. regression metrics
   * Feature-importance and distribution plots in `reports/figures/`

---

🔧 **Technologies Used**

* Python (pandas, NumPy, scikit-learn, Matplotlib)
* Jupyter Notebook
* UCI Forest Fires dataset

---

💡 **Why This Matters**
By forecasting **both** the likelihood and severity of fires, this work delivers a comprehensive early-warning system. Fire management teams can:

* Pre-deploy crews and equipment
* Trigger high-risk alerts when thresholds are met
* Optimize prevention strategies and resource allocation

---

📚 **Acknowledgments & References**
Cortez & Morais (2007). A data mining approach to predict forest fires.
NASA Earth Observatory (2024). Wildfires in California.
CAL FIRE (2024). California Wildfire Statistics.
Liu et al. (2010). Trends in global wildfire potential.
Sayad et al. (2019). Predictive modeling of wildfires with ML.

---

🤝 **Contributing**

1. Fork the repo
2. Create a feature branch
3. Submit a pull request

---

📄 **License**
This project is licensed under the MIT License.
