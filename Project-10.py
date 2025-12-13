import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import time


# --- 1. Data Loading and Tuning Function ---

@st.cache_resource
def run_tuning_tools():
    """Loads data and runs GridSearchCV and RandomizedSearchCV for SVM hyperparameter optimization."""

    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # --- Define Parameter Space for SVM ---
    param_grid = {
        'C': [0.1, 1, 10, 100],  # Regularization parameter
        'gamma': [1, 0.1, 0.01, 0.001],  # Kernel coefficient
        'kernel': ['rbf', 'linear']  # Type of kernel
    }
    # Total combinations for Grid Search: 4 * 4 * 2 = 32 fits

    # --- 1. GridSearchCV (The Exhaustive Tool) ---
    st.info("Running GridSearchCV (32 model fits)...")
    start_grid = time.time()
    grid_search = GridSearchCV(
        SVC(random_state=42),
        param_grid,
        refit=True,
        verbose=0,
        cv=5,  # 5-fold cross-validation
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    end_grid = time.time()

    grid_result = {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,  # Cross-validation score
        "test_accuracy": accuracy_score(y_test, grid_search.predict(X_test)),
        "time": end_grid - start_grid
    }

    # --- 2. RandomizedSearchCV (The Efficient Tool) ---
    st.info("Running RandomizedSearchCV (Random 10 model fits)...")
    start_rand = time.time()
    rand_search = RandomizedSearchCV(
        SVC(random_state=42),
        param_grid,  # Uses the same grid for demonstration, but samples from it
        n_iter=10,  # Only try 10 random combinations
        refit=True,
        verbose=0,
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    rand_search.fit(X_train, y_train)
    end_rand = time.time()

    rand_result = {
        "best_params": rand_search.best_params_,
        "best_score": rand_search.best_score_,
        "test_accuracy": accuracy_score(y_test, rand_search.predict(X_test)),
        "time": end_rand - start_rand
    }

    return grid_result, rand_result


# Run the tuning tools
grid_results, rand_results = run_tuning_tools()

# --- 2. Streamlit Application UI ---

st.set_page_config(page_title="Day 10: Hyperparameter Tuning")

st.title("⚙️ Day 10: Hyperparameter Tuning Optimizers")
st.subheader("Automated Model Optimization with GridSearchCV & RandomizedSearchCV")
st.markdown(
    "These Scikit-learn tools automate the search for the best model settings, treating optimization as a pipeline utility.")

if grid_results is not None:
    st.write("### 🥇 1. GridSearchCV: The Exhaustive Search Tool")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Best CV Score Found", value=f"{grid_results['best_score'] * 100:.2f}%")
        st.metric(label="Total Time Taken (s)", value=f"{grid_results['time']:.2f}s")

    with col2:
        st.metric(label="Final Test Accuracy", value=f"{grid_results['test_accuracy'] * 100:.2f}%")
        st.code(f"Best Params: {grid_results['best_params']}")

    st.markdown("---")

    st.write("### 🥈 2. RandomizedSearchCV: The Efficient Search Tool")

    col3, col4 = st.columns(2)

    with col3:
        st.metric(label="Best CV Score Found", value=f"{rand_results['best_score'] * 100:.2f}%")
        st.metric(label="Total Time Taken (s)", value=f"{rand_results['time']:.2f}s")

    with col4:
        st.metric(label="Final Test Accuracy", value=f"{rand_results['test_accuracy'] * 100:.2f}%")
        st.code(f"Best Params: {rand_results['best_params']}")

    st.markdown("---")

    st.write("### 💡 Conclusion (The Tool's Value)")
    st.info(f"""
    **Grid Search** was **{grid_results['time'] / rand_results['time']:.1f}x slower** but guaranteed the optimal settings. 
    **Random Search** was much faster and achieved a very competitive score, demonstrating its value when dealing with large datasets and complex models.
    """)

