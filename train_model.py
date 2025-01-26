import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

# Dataset: Study Hours (X) vs Grade (y)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])  # Study hours
y = np.array([40, 45, 50, 55, 60, 65, 70, 75, 80, 85])  # Grades

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open('study_grade_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved as 'study_grade_model.pkl'.")
