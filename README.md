# ML Lab Programs — Jupyter Notebook

## Included experiments
1. `01_Find_S_Algorithm.ipynb` — Find-S using custom EnjoySport data.
2. `02_Linear_Regression.ipynb` — Multiple Linear Regression using California Housing (20,640 observations).
3. `03_Polynomial_Regression.ipynb` — Degree-2 Polynomial Regression using California Housing (20,640 observations).
4. `04_Logistic_Regression.ipynb` — Logistic Regression using Breast Cancer Wisconsin Diagnostic (569 observations, 30 features).
5. `05_SVM_Classification.ipynb` — SVM classification using Breast Cancer Wisconsin Diagnostic.
6. `06_KNN_Classification.ipynb` — KNN classification using Breast Cancer Wisconsin Diagnostic.

Matching `.py` files are included.

## Dataset sources
- Find-S: custom educational dataset.
- Linear/Polynomial Regression: California Housing loaded through scikit-learn; first run may need internet to download/cache.
- Logistic Regression, SVM and KNN: Breast Cancer Wisconsin Diagnostic dataset built into scikit-learn; no manual Kaggle download is required.

## Run
1. Extract the ZIP.
2. Open Jupyter Notebook/JupyterLab in this folder.
3. Open a `.ipynb` notebook and choose **Kernel → Restart Kernel and Run All Cells**.
4. If libraries are missing, run `%pip install -r requirements.txt` in a notebook cell.

7. **Decision Tree Classification** — Breast Cancer Wisconsin Diagnostic dataset (569 rows, 30 features; built into scikit-learn).
   - `07_Decision_Tree_Classification.ipynb`
   - `07_Decision_Tree_Classification.py`

8. **C4.5 Decision Tree Classification** — Breast Cancer Wisconsin Diagnostic dataset (569 rows, 30 features; built into scikit-learn). Uses Chefboost for the C4.5 gain-ratio algorithm.
   - `08_C45_Decision_Tree_Classification.ipynb`
   - `08_C45_Decision_Tree_Classification.py`
   - Additional dependency: `chefboost`

9. **CART Decision Tree Classification** — Breast Cancer Wisconsin Diagnostic dataset (569 rows, 30 features; built into scikit-learn).
   - `09_CART_Decision_Tree_Classification.ipynb`
   - `09_CART_Decision_Tree_Classification.py`

10. **Markov Chain** — custom weather-state sequence (30 observations; no download required).
    - `10_Markov_Chain.ipynb`
    - `10_Markov_Chain.py`
