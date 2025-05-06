import pandas
from sklearn.linear_model import LinearRegression

print("Simple Linear Regression Predictor - By Aaditya Awati.\n")

dataset_name = input("Please Enter Name Of Dataset -> ")
dataset = pandas.read_csv(dataset_name)

dependent_variable, independent_variable = list(dataset.to_dict().keys())[0], list(dataset.to_dict().keys())[1]

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

print("The Model has been trained ...\n")

print(f"Dependent Variable -> {dependent_variable}")
print(f"Value to be Predicted -> {independent_variable}\n")

test_value = int(input(f"Please Enter The Dependent Value({dependent_variable}) -> "))
test_value = [[test_value]]

predicted_value = linear_regressor.predict(test_value)[0]
print(f"The Predicted Value is {predicted_value}.")
