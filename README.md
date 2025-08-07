# ChurnGuard: ANN-Powered Customer Churn Prediction System 🤖

## 1. 📝 Project Overview

**ChurnGuard** is an end-to-end data science project designed to predict customer churn in the banking sector. The system leverages an Artificial Neural Network (ANN) to analyze customer data and classify individuals based on their likelihood of leaving the bank.

The project encompasses the entire machine learning lifecycle:
-   **Data Preprocessing:** Cleaning and transforming raw data into a model-compatible format.
-   **Model Engineering:** Building, training, and evaluating a neural network using TensorFlow and Keras.
-   **Inference & Deployment:** Persisting the trained model and preprocessors, and serving them through an interactive web application built with Streamlit.

---

## 2. 🏗️ System Architecture

The project is structured into three primary components: model training, prediction pipeline, and a user-facing web interface.

1.  **Model Training (`churn_ann_1.ipynb`)**
    -   The initial dataset (`Churn_Modelling.csv`) is loaded and subjected to rigorous preprocessing.
    -   Categorical features (`Gender`, `Geography`) are numerically encoded using Scikit-learn's `LabelEncoder` and `OneHotEncoder`.
    -   Numerical features are standardized using `StandardScaler` to ensure uniform scale and prevent feature dominance.
    -   An ANN model is constructed and trained on this preprocessed data.
    -   The resulting trained model (`model.h5`) and preprocessing objects (`.pkl` files) are serialized and saved to disk.

2.  **Prediction Pipeline (`loading.ipynb`, `web_app.py`)**
    -   For inference, the serialized model and preprocessors are loaded from disk.
    -   New input data undergoes the exact same transformation pipeline (encoding and scaling) as the training data to ensure consistency.
    -   The transformed data is then fed into the loaded model to generate a churn probability.

3.  **Web Application (`web_app.py`)**
    -   A user-friendly interface powered by Streamlit allows for the input of customer attributes.
    -   The backend of the web app executes the prediction pipeline in real-time.
    -   The final prediction, including the churn probability, is displayed to the end-user.

---

## 3. 🧠 Model Details

The core of this project is a Sequential Artificial Neural Network built with the Keras API.

-   **Architecture:**
    -   **Input Layer:** `Dense` layer with 64 neurons and `ReLU` activation function.
    -   **Hidden Layer:** `Dense` layer with 32 neurons and `ReLU` activation.
    -   **Output Layer:** `Dense` layer with 1 neuron and `Sigmoid` activation, which outputs a probability score between 0 and 1.

-   **Compilation & Training:**
    -   **Optimizer:** `Adam` optimizer with a learning rate of `0.01`.
    -   **Loss Function:** `binary_crossentropy`, suitable for binary classification problems.
    -   **Monitoring:** Model performance during training is tracked using `TensorBoard` for visualization of metrics like accuracy and loss over epochs.

---

## 4. 💻 Technology Stack

-   **Core Libraries:**
    -   **TensorFlow & Keras:** For building and training the neural network.
    -   **Scikit-learn:** For data preprocessing (`StandardScaler`, `LabelEncoder`, `OneHotEncoder`).
    -   **Pandas:** For data manipulation and management.
    -   **NumPy:** For numerical operations.

-   **Web Interface & Deployment:**
    -   **Streamlit:** For creating the interactive web application.
    -   **Pickle:** For serializing and deserializing Scikit-learn preprocessor objects.
    -   **HDF5 (`.h5`):** For saving and loading the trained Keras model.

---

## 5. 📁 Repository Structure

```
.
├── churn_ann_1.ipynb         # Jupyter notebook for data preprocessing and model training.
├── loading.ipynb             # Jupyter notebook for testing the loaded model and pipeline.
├── web_app.py                # The main Streamlit application script.
├── model.h5                  # The saved, trained neural network model.
├── label_encoder_gender.pkl  # Saved LabelEncoder for the 'Gender' feature.
├── onehot_encoder_geo.pkl    # Saved OneHotEncoder for the 'Geography' feature.
├── scaler.pkl                # Saved StandardScaler for numerical features.
├── requirements.txt          # List of required Python packages.
└── README.md                 # This documentation file.
```

---

## 6. 🚀 Execution Instructions

### Prerequisites
-   Python 3.8+
-   pip package manager

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/pragyan2905/ChurnGuard.git](https://github.com/pragyan2905/ChurnGuard.git)
    cd ChurnGuard
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    *(Ensure you have a `requirements.txt` file with packages like `tensorflow`, `scikit-learn`, `pandas`, `streamlit`)*
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application
To launch the interactive web interface, run the following command:
```bash
streamlit run web_app.py
```
Navigate to the local URL provided by Streamlit in your web browser.
