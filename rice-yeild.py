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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError("Error: The file 'Rice-Yield.csv' was not found.")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("Error: The file 'Rice-Yield.csv' is empty.")
    except pd.errors.ParserError:
        raise pd.errors.ParserError("Error: The file 'Rice-Yield.csv' could not be parsed.")

def validate_columns(data, columns):
    for column in columns:
        if column not in data.columns:
            raise ValueError(f"Error: Column '{column}' not found in the dataset.")

def preprocess_data(data, features, target):
    X = data[features]
    y = data[target]
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_models(X_train, y_train):
    modelLR = LinearRegression()
    modelLR.fit(X_train, y_train)
    
    modelRFR = RandomForestRegressor(n_estimators=100, random_state=42)
    modelRFR.fit(X_train, y_train)
    
    modelANN = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    modelANN.compile(optimizer='adam', loss='mse')
    modelANN.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    return modelLR, modelRFR, modelANN

def evaluate_models(modelLR, modelRFR, modelANN, X_test, y_test):
    y_predLR = modelLR.predict(X_test)
    y_predRFR = modelRFR.predict(X_test)
    y_predANN = modelANN.predict(X_test).flatten()
    
    mseL = mean_squared_error(y_test, y_predLR)
    maeL = mean_absolute_error(y_test, y_predLR)
    mseR = mean_squared_error(y_test, y_predRFR)
    maeR = mean_absolute_error(y_test, y_predRFR)
    mseA = mean_squared_error(y_test, y_predANN)
    maeA = mean_absolute_error(y_test, y_predANN)
    
    return mseL, maeL, mseR, maeR, mseA, maeA

def generate_plots(plot_frame, data, features, target):
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
            plt.close()

    def generate_line_plots():
        for feature in features:
            plt.figure(figsize=(6, 4))
            sns.lineplot(x=data.index, y=data[feature])
            plt.xlabel("Index")
            plt.ylabel(feature)
            plt.title(f"{feature} Line Plot")
            canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            plt.close()

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
            plt.close()

    def generate_pair_plot():
        plt.figure(figsize=(8, 6))
        sns.pairplot(data[features + [target]])
        canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        plt.close()

    def generate_correlation_heatmap():
        plt.figure(figsize=(8, 6))
        correlation_matrix = data[features + [target]].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        plt.close()

    generate_line_plots()
    generate_histograms()
    generate_pair_plot()
    generate_correlation_heatmap()

def main():
    file_path = "Rice-Yield.csv"
    features = ['Area', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'humidity', 'ph']
    target = 'hg/ha_yield'

    # Load and preprocess data
    try:
        data = load_data(file_path)
        validate_columns(data, features + [target])
        X_train, X_test, y_train, y_test = preprocess_data(data, features, target)
    except Exception as e:
        print(e)
        return

    # Train and evaluate models
    modelLR, modelRFR, modelANN = train_models(X_train, y_train)
    mseL, maeL, mseR, maeR, mseA, maeA = evaluate_models(modelLR, modelRFR, modelANN, X_test, y_test)

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
    mse_message = tk.Message(root, text=f"MSE Linear Regression: {mseL:.2f},MAE Linear Regression: {maeL:.2f}", font=("Helvetica", 14))
    mse_message.pack()

    mae_message = tk.Message(root, text=f"MSE Random Forest: {mseR:.2f}, MAE Random Forest: {maeR:.2f}", font=("Helvetica", 14))
    mae_message.pack()

    mae_message = tk.Message(root, text=f"MSE ANN: {mseA:.2f}, MAE ANN: {maeA:.2f}", font=("Helvetica", 14))
    mae_message.pack()
    # Generate plots
    generate_plots(plot_frame, data, features, target)

    # Run the application
    root.mainloop()

if __name__ == "__main__":
    main()
