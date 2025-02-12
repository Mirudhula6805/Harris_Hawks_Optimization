import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import time

class HarrisHawksOptimizer:
    def __init__(self, num_hawks, max_iter, bounds, fitness_function):
        self.num_hawks = num_hawks
        self.max_iter = max_iter
        self.bounds = bounds
        self.fitness_function = fitness_function
        self.hawks = self.initialize_hawks()
        self.convergence_curve = []
        self.generations = []

    def initialize_hawks(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.num_hawks, len(self.bounds)))

    def optimize(self):
        best_hawk = None
        best_fitness = float('inf')
        
        for iteration in range(self.max_iter):
            generation_fitness = []
            print(f"\nIteration {iteration + 1}:")
            
            for i in range(self.num_hawks):
                fitness = self.fitness_function(self.hawks[i])
                generation_fitness.append(fitness)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_hawk = self.hawks[i]
            
            self.convergence_curve.append(best_fitness)
            self.generations.append(generation_fitness)
            
            for i in range(self.num_hawks):
                self.hawks[i] += np.random.uniform(-1, 1, len(self.bounds)) * (best_hawk - self.hawks[i])
                self.hawks[i] = np.clip(self.hawks[i], self.bounds[0], self.bounds[1])

            print(f"Best Fitness in this iteration: {best_fitness}")
        
        return best_hawk, best_fitness

def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def hho_fitness_function(params):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    return mean_squared_error(y_test, y_pred)

# Load Heart Disease dataset from CSV
df = pd.read_csv("heart.csv")  # Update with the correct file path if needed

# Split dataset into features (X) and target (y)
X = df.drop(columns=["target"])  # Assuming "target" is the column you're predicting
y = df["target"]

# Preprocess the dataset
X_train, X_test, y_train, y_test = preprocess_data(X, y)

# User input
num_hawks = int(input("Enter the number of hawks: "))
max_iter = int(input("Enter the number of iterations: "))

# Measure time for HHO optimization
start_time_hho = time.time()

# Run HHO optimizer
optimizer = HarrisHawksOptimizer(num_hawks=num_hawks, max_iter=max_iter, bounds=[-10, 10], fitness_function=hho_fitness_function)
best_solution, best_value = optimizer.optimize()

end_time_hho = time.time()
hho_time_taken = end_time_hho - start_time_hho

# Compare with baseline methods
# Measure time for Linear Regression
start_time_lr = time.time()
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lr = lin_reg.predict(X_test)
lin_mse = mean_squared_error(y_test, y_pred_lr)
lin_r2 = r2_score(y_test, y_pred_lr)
end_time_lr = time.time()
lr_time_taken = end_time_lr - start_time_lr

# Measure time for Ridge Regression
start_time_ridge = time.time()
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, y_train)
y_pred_ridge = ridge_reg.predict(X_test)
ridge_mse = mean_squared_error(y_test, y_pred_ridge)
ridge_r2 = r2_score(y_test, y_pred_ridge)
end_time_ridge = time.time()
ridge_time_taken = end_time_ridge - start_time_ridge

# Calculate accuracy for comparison
y_pred_lr_class = (y_pred_lr >= 0.5).astype(int)  # Assuming a threshold of 0.5 for binary classification
y_pred_ridge_class = (y_pred_ridge >= 0.5).astype(int)
hho_accuracy = accuracy_score(y_test, y_pred_lr_class)  # Using LR accuracy for HHO in this case
lr_accuracy = accuracy_score(y_test, y_pred_lr_class)
ridge_accuracy = accuracy_score(y_test, y_pred_ridge_class)

# Plot convergence curve
plt.figure(figsize=(8, 5))
plt.plot(optimizer.convergence_curve, color='blue', label='HHO Convergence')
plt.xlabel('Iterations')
plt.ylabel('Best Fitness')
plt.title('Harris Hawks Optimization Convergence')
plt.legend()
plt.grid()
plt.show()

# Show fitness per generation with colors
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(optimizer.generations)))  # Using viridis colormap for different colors

for i, (gen, color) in enumerate(zip(optimizer.generations, colors)):
    plt.scatter([i] * len(gen), gen, color=color, alpha=0.6, label=f'Generation {i+1}' if i == 0 else "")

plt.xlabel('Iterations')
plt.ylabel('Fitness Values')
plt.title('Fitness per Generation with Colors')
sm = plt.cm.ScalarMappable(cmap='viridis')
sm.set_array([])
plt.colorbar(sm, ax=plt.gca(), label='Generation Index')  # Add colorbar for reference
plt.grid()
plt.show()

# Log fitness values per generation into a DataFrame
generations_df = pd.DataFrame(optimizer.generations)
generations_df.index = [f"Generation {i+1}" for i in range(len(optimizer.generations))]
print("\nFitness values per generation:")
print(generations_df)

# Print results
print("\nBest Solution:", best_solution)
print("Best Value (MSE):", best_value)
print("Linear Regression MSE:", lin_mse, "R2 Score:", lin_r2)
print("Ridge Regression MSE:", ridge_mse, "R2 Score:", ridge_r2)

# Performance comparison
methods = ['HHO', 'Linear Regression', 'Ridge Regression']
mse_values = [best_value, lin_mse, ridge_mse]
plt.figure(figsize=(8, 5))
plt.bar(methods, mse_values, color=['#4B0082','#9932CC', '#DDA0DD'])
plt.xlabel('Methods')
plt.ylabel('Mean Squared Error')
plt.title('Comparison of Optimization Methods')
plt.grid()
plt.show()

# Time comparison
methods_time = ['HHO', 'Linear Regression', 'Ridge Regression']
time_values = [hho_time_taken, lr_time_taken, ridge_time_taken]
plt.figure(figsize=(8, 5))
plt.bar(methods_time, time_values, color=['purple', 'cyan', 'orange'])
plt.xlabel('Methods')
plt.ylabel('Time (seconds)')
plt.title('Time Taken by Each Method')
plt.grid()
plt.show()

# Accuracy comparison
methods_accuracy = ['HHO', 'Linear Regression', 'Ridge Regression']
accuracy_values = [hho_accuracy * 100, lr_accuracy * 100, ridge_accuracy * 100]  # Convert accuracy to percentage

# Plot the accuracies
plt.figure(figsize=(8, 5))
plt.bar(methods_accuracy, accuracy_values, color=['#6B8E23', '#B22222', '#228B22'])
plt.xlabel('Methods')
plt.ylabel('Accuracy (%)')
plt.title('Comparison of Accuracies Across Methods')
plt.ylim([0, 100])  # Set y-axis limit to 100% for clarity
plt.grid()
plt.show()

# Print computation times
print("\nComputation Time (seconds):")
print(f"HHO Time Taken: {hho_time_taken:.4f} seconds")
print(f"Linear Regression Time Taken: {lr_time_taken:.4f} seconds")
print(f"Ridge Regression Time Taken: {ridge_time_taken:.4f} seconds")

# Print accuracy comparison
print(f"\nHHO Accuracy: {hho_accuracy * 100:.2f}%")
print(f"Linear Regression Accuracy: {lr_accuracy * 100:.2f}%")
print(f"Ridge Regression Accuracy: {ridge_accuracy * 100:.2f}%")


