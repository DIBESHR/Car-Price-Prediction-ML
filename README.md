Car Price Prediction using Machine Learning
Overview
This project predicts the selling price of a used car using Machine Learning techniques. The project uses the Linear Regression algorithm to analyze car-related features and estimate the selling price.
The project demonstrates the complete Machine Learning workflow including:
•	Data preprocessing
•	Label Encoding
•	Feature Scaling
•	Train-Test Split
•	Linear Regression
•	Model Evaluation using MAE
•	Data Visualization using matplotlib
•	Search-based car prediction system
________________________________________
Technologies Used
•	Python
•	pandas
•	matplotlib
•	scikit-learn
•	Google Colab / Jupyter Notebook
________________________________________
Machine Learning Concepts Used
•	Supervised Learning
•	Regression
•	Data Preprocessing
•	Label Encoding
•	StandardScaler
•	Train-Test Split
•	Model Evaluation
•	Data Visualization
________________________________________
Dataset Information
The dataset contains information about used cars.
Features Used
Feature	Description
Car_Name	Name of the car
Year	Manufacturing year
Present_Price	Current showroom price
Kms_Driven	Distance driven
Fuel_Type	Fuel type
Seller_Type	Dealer or Individual
Transmission	Manual or Automatic
Owner	Number of previous owners
Target Variable
Target	Description
Selling_Price	Selling price of the car
________________________________________
Project Workflow
Load Dataset
→ Data Preprocessing
→ Label Encoding
→ Train-Test Split
→ Feature Scaling
→ Train Linear Regression Model
→ Predict Prices
→ Evaluate Model using MAE
→ Visualize Results
→ Search Car and Predict Price
________________________________________
Libraries Installation
Install required libraries using:
pip install pandas matplotlib scikit-learn
________________________________________
How to Run the Project
1.	Download the dataset
2.	Open Google Colab or Jupyter Notebook
3.	Upload the dataset
4.	Copy and run the Python code
5.	Enter a car name from the dataset
6.	View predicted selling price
________________________________________
Example Output
Enter Car Name : swift
Output:
CAR DETAILS

Car Name : swift
Year : 2014
Present Price : 6.87
KM Driven : 42450
Fuel Type : Diesel

Predicted Selling Price : 4.58 Lakhs
________________________________________
Model Evaluation
The project uses:
Mean Absolute Error (MAE)
MAE measures the average prediction error.
Lower MAE indicates better prediction performance.
________________________________________
Visualization
Matplotlib scatter plot is used to compare:
•	Actual Prices
•	Predicted Prices
________________________________________
Advantages
•	Beginner-friendly ML project
•	Uses real-world dataset
•	Covers complete ML workflow
•	Easy to understand and implement
________________________________________
Future Improvements
•	Use larger datasets
•	Improve accuracy
•	Try advanced ML algorithms
•	Build web application using Flask or Streamlit
________________________________________
Author
Dibesh R
________________________________________
Conclusion
This project demonstrates how Machine Learning can be used to predict car selling prices using Linear Regression. It provides practical understanding of preprocessing, training, prediction, evaluation, and visualization in Machine Learning.
