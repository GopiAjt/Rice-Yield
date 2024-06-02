Data Visualization and Rice Yield Prediction Tool

This Python application provides a graphical user interface (GUI) for visualizing data and predicting rice yield based on various factors. It utilizes Tkinter for the GUI and Matplotlib for plotting.

Features
Load data from a CSV file
Display scatter plots, line plots, histograms, pair plots, and correlation heatmaps
Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE) for Linear Regression and Random Forest Regression models
Predict rice yield using the trained regression models

Regression Models

Linear Regression:
Linear regression is a simple yet powerful technique used for modeling the relationship between a dependent variable and one or more independent variables. In this application, we utilize linear regression to understand how various factors such as area, average rainfall, pesticides usage, average temperature, humidity, and pH level affect rice yield. The model learns the linear relationship between these features and the yield, allowing us to make predictions based on new input data.

Random Forest Regression:
Random Forest Regression is an ensemble learning method that constructs multiple decision trees during training and outputs the average prediction of the individual trees. This technique is particularly useful for handling nonlinear relationships and capturing complex interactions between features. In our application, we employ a Random Forest Regression model to predict rice yield based on the same set of features used in the linear regression model. By leveraging the collective wisdom of multiple decision trees, the Random Forest model offers improved accuracy and robustness compared to a single decision tree.

Predicting Rice Yield:
Once the models are trained using the provided dataset, the application allows users to input values for various features such as area, average rainfall, pesticides usage, etc. Based on these inputs, the trained regression models predict the expected yield of rice per hectare (hg/ha_yield). Users can explore different scenarios by adjusting the input values and observing how changes in environmental factors impact rice production. This predictive capability empowers stakeholders in the agriculture sector to make informed decisions and optimize resource allocation for maximizing yield.

Requirements:
Python 3.x 
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Tkinter

Interpretation:
Lower MSE and MAE values indicate better performance.
Among the three models, the ANN (Artificial Neural Network) has the lowest MSE and MAE, suggesting it is the most accurate in predicting the rice yield.
The Linear Regression model performs slightly worse than the ANN but better than the Random Forest in terms of MSE.
The Random Forest model has slightly higher errors than both the ANN and Linear Regression in terms of both MSE and MAE.

Summary:
The ANN model appears to be the best-performing model for your dataset, as it has the lowest errors.
The Linear Regression model also performs reasonably well but not as well as the ANN.
The Random Forest model, while useful in many scenarios, seems to be the least accurate for this particular dataset in terms of both MSE and MAE.
