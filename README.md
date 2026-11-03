# California House Price Predictor

Predicts median house values in California using a tuned KNN regression model, deployed as a Streamlit app.

**Live app:** https://house-price-app-dyrde46e9em7dupmir3woy.streamlit.app/

## What it does

- Loads the California Housing dataset from scikit-learn
- Scales features with a `ColumnTransformer` (`StandardScaler`)
- Splits data 80/20 into train and test sets
- Tunes a `KNeighborsRegressor` with `GridSearchCV` (5-fold CV, R² score)
- Saves the trained pipeline as `best_knn_model.pkl`
- Serves predictions in a Streamlit app

## Results

- Best parameters: ...
- Best CV R²: ...
- Test R²: ...

## Files

- `Assignment9_Model_Deployment.ipynb`: the full workflow (Colab notebook)
- `best_knn_model.pkl`: the saved model
- `app.py`: the Streamlit app
- `requirements.txt`: dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
