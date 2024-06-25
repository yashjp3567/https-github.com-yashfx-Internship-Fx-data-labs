#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
import numpy as np

# Load the model from the file
model = joblib.load(r"C:\Users\hp\Downloads\credit_card_fraud_model6.pkl")


# In[3]:


def predict_fraud(features):
    """
    Predict if a credit card transaction is fraudulent.

    Parameters:
    - features (list or numpy array): The feature values for the transaction.

    Returns:
    - int: 1 if the transaction is predicted to be fraudulent, 0 otherwise.
    """
    # Ensure features are in the correct format (2D array)
    features = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    
    return prediction[0]

if __name__ == "__main__":
    # Example feature set for a single transaction
    example_features = [0.1, -1.2, 3.4, 4.5, -0.5, 2.3, -1.1, 0.4, 1.2, -0.3,
                        0.1, -1.2, 3.4, 4.5, -0.5, 2.3, -1.1, 0.4, 1.2, -0.3,
                        0.1, -1.2, 3.4, 4.5, -0.5, 2.3, -1.1, 0.4, 1.2, -0.3] # Replace with actual feature values

    # Predict fraud
    result = predict_fraud(example_features)
    
    # Print the result
    print("Fraudulent transaction" if result == 1 else "Legitimate transaction")


# In[ ]:




