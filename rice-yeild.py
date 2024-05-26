import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load data from a CSV file
try:
    data = pd.read_csv("Rice-Yield.csv")
except FileNotFoundError:
    print("Error: The file 'Rice-Yield.csv' was not found.")
    exit()
except pd.errors.EmptyDataError:
    print("Error: The file 'Rice-Yield.csv' is empty.")
    exit()
except pd.errors.ParserError:
    print("Error: The file 'Rice-Yield.csv' could not be parsed.")
    exit()

# Select relevant columns
features = ['Area', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'humidity', 'ph']
target = 'hg/ha_yield'

# Validate if columns exist in the dataset
for column in features + [target]:
    if column not in data.columns:
        print(f"Error: Column '{column}' not found in the dataset.")
        exit()

# Split the data into features (X) and target (y)
X = data[features]
y = data[target]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
modelLR = LinearRegression()
modelLR.fit(X_train, y_train)

modelRFR = RandomForestRegressor(n_estimators=100, random_state=42)
modelRFR.fit(X_train, y_train)

# Evaluate the model
y_predLR = modelLR.predict(X_test)
y_predRFR = modelRFR.predict(X_test)

mseL = mean_squared_error(y_test, y_predLR)
maeL = mean_absolute_error(y_test, y_predLR)
mseR = mean_squared_error(y_test, y_predRFR)
maeR = mean_absolute_error(y_test, y_predRFR)

# Function to generate scatter plots
def generate_scatter_plots():
    for feature in features:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=data[feature], y=data[target])
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.title(f"{feature} vs. {target}")
        canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def generate_line_plots():
    for feature in features:
        plt.figure(figsize=(6, 4))
        sns.lineplot(x=data.index, y=data[feature])  # Assuming the index is suitable for x-axis
        plt.xlabel("Index")  # Update x-axis label accordingly
        plt.ylabel(feature)
        plt.title(f"{feature} Line Plot")
        canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Function to generate histograms
def generate_histograms():
    for feature in features:
        plt.figure(figsize=(6, 4))
        sns.histplot(data[feature], kde=True)
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {feature}")
        canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Function to generate pair plot
def generate_pair_plot():
    plt.figure(figsize=(8, 6))
    sns.pairplot(data[features + [target]])
    canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Function to generate correlation heatmap
def generate_correlation_heatmap():
    plt.figure(figsize=(8, 6))
    correlation_matrix = data[features + [target]].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Function to generate all plots
def generate_plots():
    print("ganarating plots pleaase wait!")
    # generate_scatter_plots()
    generate_line_plots()
    generate_histograms()
    generate_pair_plot()
    generate_correlation_heatmap()

# Create the main application window
root = tk.Tk()
root.title("Data Visualization")

# Create a canvas and scrollbar
canvas = tk.Canvas(root)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill="y")

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Create a frame for the plots
plot_frame = ttk.Frame(canvas)
canvas.create_window((0, 0), window=plot_frame, anchor="nw")

# Create Message widgets to display MSE and MAE
mse_message = tk.Message(root, text=f"MSE Linear Regression: {mseL}, MSE Random Forest: {mseR}")
mse_message.pack()

mae_message = tk.Message(root, text=f"MAE Linear Regression: {maeL}, MAE Random Forest: {maeR}")
mae_message.pack()


# Generate plots upon starting the application
generate_plots()

# Run the application
root.mainloop()