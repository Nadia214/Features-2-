# Features-2-
## 🧠 Feature Optimization Techniques — With Regression Focus

---

### 1️⃣ Feature Selection

Choosing a subset of the original features that are relevant to the target variable.

#### 🔹 A. Filter Methods (Statistical; model-free)

| Method                 | Use Case                                            | Regression? | Notes                                                    |
| ---------------------- | --------------------------------------------------- | ----------- | -------------------------------------------------------- |
| VarianceThreshold      | Remove low-variance features                        | ✅           | Simple, preprocessing step                               |
| SelectKBest            | Score-based selection                               | ✅/❌         | Use `f_regression` (✅) or `chi2` (❌ classification only) |
| SelectPercentile       | Top % features                                      | ✅/❌         | Based on scoring function                                |
| Correlation Threshold  | Drop weakly correlated or highly collinear features | ✅           | Based on Pearson/Spearman correlation                    |
| Mutual Information     | Non-linear dependency                               | ✅           | Use `mutual_info_regression`                             |
| ANOVA F-test           | Feature-class association                           | ❌           | For classification only                                  |
| **mRMR**               | Max relevance, min redundancy                       | ✅           | Use `pymrmr`, discretizes features                       |
| **Feature Clustering** | Reduce correlated redundancy                        | ✅           | Cluster similar features and keep representative         |

#### 🔹 B. Wrapper Methods (Model-based search)

| Method                              | Use Case                     | Regression? | Notes                             |
| ----------------------------------- | ---------------------------- | ----------- | --------------------------------- |
| RFE (Recursive Feature Elimination) | Recursive pruning            | ✅           | Needs a regressor (e.g., SVR, RF) |
| RFECV                               | RFE + cross-validation       | ✅           | Finds optimal #features           |
| Forward Selection                   | Adds one-by-one              | ✅           | Greedy, slow on large sets        |
| Backward Elimination                | Starts with all, removes     | ✅           | Inverse of forward                |
| SFS (Sequential FS)                 | Generalized forward/backward | ✅           | Can control scoring metric        |

#### 🔹 C. Embedded Methods (Feature selection inside training)

| Method              | Use Case                                            | Regression? | Notes                               |
| ------------------- | --------------------------------------------------- | ----------- | ----------------------------------- |
| Lasso Regression    | Regularization-based                                | ✅           | L1 penalty forces sparsity          |
| ElasticNet          | L1 + L2 penalty                                     | ✅           | Better for correlated features      |
| Ridge Regression    | L2 regularization                                   | ✅           | Coefficients shrink but not to zero |
| Tree-based models   | Use splits to derive importances                    | ✅           | RF, XGBoost, LightGBM               |
| Logistic Regression | Classification only                                 | ❌           | Not applicable for regression       |
| SelectFromModel     | Use estimator's `.coef_` or `.feature_importances_` | ✅           | Model-agnostic selector             |

#### 🔹 D. Advanced / Hybrid Methods

| Method                  | Use Case                         | Regression? | Notes                              |
| ----------------------- | -------------------------------- | ----------- | ---------------------------------- |
| Boruta                  | All-relevant feature selection   | ✅           | RandomForest-based wrapper         |
| SHAP-based selection    | Keep top features by SHAP values | ✅           | Model-agnostic, local & global     |
| Permutation Importance  | Drop-shuffle-test                | ✅           | Interpretation-friendly            |
| Genetic Algorithms      | Evolution-based search           | ✅           | Costly, used in AutoML or research |
| **Stability Selection** | Robust under sampling noise      | ✅           | Combines Lasso + bootstrapping     |

---

### 2️⃣ Feature Reduction

Reduce dimensionality by transforming features into a smaller set of derived values.

#### 🔹 A. Unsupervised Reduction

| Method       | Use Case                            | Regression? | Notes                          |
| ------------ | ----------------------------------- | ----------- | ------------------------------ |
| PCA          | Projects to max variance directions | ✅           | Linear, fast                   |
| Kernel PCA   | Non-linear projection               | ✅           | Expensive but powerful         |
| t-SNE        | Visualization only                  | ⚠️          | Don't use for modeling         |
| UMAP         | Visualization                       | ⚠️          | Faster alt to t-SNE            |
| TruncatedSVD | Sparse PCA                          | ✅           | Faster for sparse matrices     |
| ICA          | Independent signal separation       | ✅           | Often used in EEG, signal data |
| Autoencoders | Nonlinear compression               | ✅           | Deep learning-based            |

#### 🔹 B. Supervised Reduction

| Method                                  | Use Case                    | Regression? | Notes                               |
| --------------------------------------- | --------------------------- | ----------- | ----------------------------------- |
| PLSR (Partial Least Squares Regression) | Preserve X–y correlation    | ✅           | Good for few samples, many features |
| LDA (Linear Discriminant Analysis)      | Maximize class separability | ❌           | Classification only                 |
| Supervised PCA                          | PCA guided by target        | ✅           | Improves upon PCA                   |

---

### 3️⃣ Feature Importance Interpretation

Use to interpret or explain model behavior — not for selection directly.

| Method                       | Use Case                     | Regression? | Notes                          |
| ---------------------------- | ---------------------------- | ----------- | ------------------------------ |
| Coefficients (Linear Models) | Sign & magnitude matter      | ✅           | Only for linear models         |
| Tree-based Importances       | Based on splits              | ✅           | `.feature_importances_`        |
| Permutation Importance       | Shuffle-drop-test            | ✅           | Works on any model             |
| SHAP Values                  | Local + global explanations  | ✅           | Best for interpretability      |
| LIME                         | Local explanations           | ✅           | One prediction at a time       |
| Drop Column Importance       | Retrain without one feature  | ✅           | Accurate but slow              |
| Partial Dependence Plots     | Feature effect visualization | ✅           | Not for selection, for insight |

---

### ✅ Practical Tips

* ✅ Combine methods: Use `SelectKBest ➝ Lasso ➝ SHAP` for robust pipelines.
* ✅ Use feature clustering or mRMR before tree-based models to reduce redundancy.
* ⚠️ Avoid t-SNE, UMAP, or LDA for regression tasks.
* 🧪 Use SHAP or Permutation Importance to **validate** feature selection decisions.

---
1. Linear Models (sklearn.linear_model)

| Model                          | Use Case            |
| ------------------------------ | ------------------- |
| **LinearRegression**           | Regression          |
| **Ridge**                      | Regression          |
| **Lasso**                      | Regression          |
| **ElasticNet**                 | Regression          |
| **BayesianRidge**              | Regression          |
| **HuberRegressor**             | Regression (robust) |
| **PassiveAggressiveRegressor** | Regression          |
| **TheilSenRegressor**          | Regression (robust) |
| **RANSACRegressor**            | Regression (robust) |
| **SGDRegressor**               | Regression          |

---

2. Tree-Based Models (sklearn.tree and sklearn.ensemble)

| Model                             | Use Case   |
| --------------------------------- | ---------- |
| **DecisionTreeRegressor**         | Regression |
| **RandomForestRegressor**         | Regression |
| **ExtraTreesRegressor**           | Regression |
| **GradientBoostingRegressor**     | Regression |
| **AdaBoostRegressor**             | Regression |
| **HistGradientBoostingRegressor** | Regression |



3. Support Vector & Neighbors

| Model                              | Use Case   |
| ---------------------------------- | ---------- |
| **SVR** (Support Vector Regressor) | Regression |
| **KNeighborsRegressor**            | Regression |


---
4. Neural Network

| Model            | Use Case   |
| ---------------- | ---------- |
| **MLPRegressor** | Regression |


---
5. Other Popular Libraries

| Model             | Use Case   |
| ----------------- | ---------- |
| **XGBRegressor**  | Regression |
| **LGBMRegressor** | Regression |


---


