
    
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# This function will take in all the feature data X
# as well as the current coefficient and bias values
# It should multiply all the feature value by their associated 
# coefficient and add the bias. It should then return the predicted 
# y values
def hypothesis(X, coefficients, bias):
    return np.dot(X,coefficients)+bias

def calculateRSquared(bias, coefficients,X, Y):
    
    predictedY = hypothesis(X, coefficients, bias)
    
    avgY = np.average(Y)
    totalSumSq = np.sum((avgY - Y)**2)
    
    sumSqRes = np.sum((predictedY - Y)**2)
    
    r2 = 1.0-(sumSqRes/totalSumSq)
    
    return r2
    


        
    

def gradient_descent(bias, coefficients, alpha, X, Y, max_iter):
    m = X.shape[0]
    length = len(X)
    
    # array is used to store change in cost function for each iteration of GD
    errorValues = []
    finalcost=[]
    finallambda=[]
    finalbias=[]
    for num in range(0, max_iter):
        predictedY = hypothesis(X, coefficients, bias)
        loss = predictedY - Y
        cost = 1 / (2 * m) * np.sum(np.square(loss))
        finalcost.append(cost)
        #print('cost', cost)
        #print (len(coefficients))
        for i in range (len(coefficients)):
            coefficients[i] =coefficients[i] - alpha * (1 / m) * np.sum(np.dot(X[:,i], loss))
        #coefficients = coefficients - alpha * (1 / m) * np.sum(np.dot(X[:,[i]], loss))
        #print(coefficients)
        finallambda.append(coefficients)
        #print('lambda', coefficients)
        bias = bias - alpha * (1 / m) * np.sum(loss)
        finalbias.append(bias)
        #print('bias', bias)

        # TODO: 
        # Calculate predicted y values for current coefficient and bias values 
        # calculate and update bias using gradient descent rule
        # Update each coefficient value in turn using gradient descent rule

    
    # calculate R squared value for current coefficient and bias values
    rSquared = calculateRSquared(bias, coefficients,X, Y)
    print ("Final R2 value is ", rSquared)

    # plot the cost for each iteration of gradient descent
    plt.plot(finalcost)
    plt.show()
    
    return bias, coefficients



# Complete this function for part 2 of the exercise. 

def calculateTestR2(bias, coefficients, testFile):
    df = pd.read_csv("testData.csv")
    df = df.dropna()
    print(df.shape)

    data = df.values

    # Seperate teh features from the target feature
    Y = data[:, -1]
    X = data[:, :-1]

    # Standardize each of the features in the dataset.
    for num in range(len(X[0])):
        feature = data[:, num]
        feature = (feature - np.mean(feature)) / np.std(feature)
        X[:, num] = feature

    predictedY = hypothesis(X, coefficients, bias)

    avgY = np.average(Y)
    totalSumSq = np.sum((avgY - Y) ** 2)

    sumSqRes = np.sum((predictedY - Y) ** 2)

    r2 = 1.0 - (sumSqRes / totalSumSq)

    return r2

def multipleLinearRegression(X, Y):

    # set the number of coefficients (weights) equal to the number of features
    #complete this line of code
    coefficients = np.zeros(X.shape[1])
    print ('coefficients',coefficients)
    bias = 0.0
    
    alpha = 0.1 # learning rate
    
    max_iter=10

    # call gredient decent, and get intercept(=bias) and coefficents
    bias, coefficients = gradient_descent(bias, coefficients, alpha, X, Y, max_iter)
    
    return bias, coefficients
    
    
    
def main():
    
    df = pd.read_csv("trainingData.csv")
    df = df.dropna()
    print (df.shape)
    
    data = df.values

     
    # Seperate teh features from the target feature    
    Y = data[:, -1]
    X = data[:, :-1]
    
    # Standardize each of the features in the dataset. 
    for num in range(len(X[0])):
        feature = data[:, num]
        feature = (feature - np.mean(feature))/np.std(feature)
        X[:, num] = feature
     
    # run regression function and return bias and coefficients (weights)
    bias, coefficients = multipleLinearRegression(X, Y)
    
    # Enable code if you have a test set  (as in part 2)

    print('Test R2 Value ')
    testFile = "testData.csv"
    print('Final R2 value for test data is ',calculateTestR2(bias, coefficients, testFile))
    


main()
