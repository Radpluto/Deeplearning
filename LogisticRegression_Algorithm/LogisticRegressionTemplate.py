
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing



# TODO: 1 
# Complete the logistic function below 
# The objective of this function is to take in a NumPy array 
# and to pass it through the sigmoid function and return the results. 
def logistic(x):
  return 1/(1+np.exp(-x))




# TODO: 2 
# This function takes in the training feature data, the current coefficients and the bias 
# and should return the value predicted by our logistic classifier. 
# This will be very similar to what you wrote for the MLR exercise.
# It will also call the logistic function that you wrote above 

def hypothesisLogistic(X, coefficients, bias):

    # # array of zeros. Length is same as number of training rows.
    # predictedValues = np.zeros(X.shape[0])
    #
    # # for each feature multiple the X training data by the appropriate
    # # coefficient and add to to predictedvalues
    # for num in range(len(coefficients)):
    #     predictedValues += coefficients[num] * X[:, num]
    #
    predictedValues = np.dot(X,coefficients)+bias
    logisticPredicitons = logistic(predictedValues)

    return logisticPredicitons




def gradient_descent_log(bias, coefficients, alpha, X, Y, max_iter):

    length = len(Y)
    
    # array is used to store change in cost function for each iteration of GD
    errorValues = []
    
    for num in range(0, max_iter):
        
        # Calculate predicted y values for current coefficient and bias values 
        predictedY = hypothesisLogistic(X, coefficients, bias)
        

        # calculate gradient for bias
        biasGrad =    (1.0/length) *  (np.sum( predictedY - Y))
        
        #update bias using GD update rule
        bias = bias - (alpha*biasGrad)
        
        # for loop to update each coefficient value in turn
        for coefNum in range(len(coefficients)):
            
            # calculate the gradient of the coefficient
            gradCoef = (1.0/length)* (np.sum( (predictedY - Y)*X[:, coefNum]))
            
            # update coefficient using GD update rule
            coefficients[coefNum] = coefficients[coefNum] - (alpha*gradCoef)
        
        # TODO 3: 
        # Calculate the average cross entropy error for the current iteration of GD

        crossEntropyError = (1.0/length) * (np.sum(-Y*(np.log(predictedY))-(1-Y)*np.log(1-predictedY)))
        errorValues.append(crossEntropyError)

    
    # plot the cost for each iteration of gradient descent
    plt.plot(errorValues)
    plt.show()
    
    return bias, coefficients




def calculateAccuracy(bias, coefficients, X_test, y_test):
    
    # TODO 4:
    # Insert code here to calculate final accuracy of your model on the test set
    Y_pred =hypothesisLogistic(X_test, coefficients, bias)
    # pred = [for ]
    correctly_classified = 0
    # counter
    count = 0
    for count in range(np.size(Y_pred)):

        if y_test[count] == round(Y_pred[count]):
            correctly_classified = correctly_classified + 1

        count = count + 1

    print("Accuracy on test set by our model       :  ", (
            correctly_classified / count) * 100)



#print ("Final Accuracy: TODO: ")
    

def logisticRegression(X_train, y_train, X_test, y_test):

    # set the number of coefficients equal to the number of features
    coefficients = np.zeros(X_train.shape[1])
    bias = 0.0
   
    alpha = 0.001 # learning rate
    
    max_iter=200

    # call gredient decent, and get intercept(bias) and coefficents
    bias, coefficients = gradient_descent_log(bias, coefficients, alpha, X_train, y_train, max_iter)
    
    calculateAccuracy(bias, coefficients, X_test, y_test)
    
    

def main():
    
    digits = datasets.load_digits()
    
    # Load the feature data and the class labels
    X_digits = digits.data
    y_digits = digits.target
    
    # The logistic regression model will differentiate between two digits
    # Below we set this to 1 and 7 but you can change these values
    # Code allows you specify the two digits and extract the images 
    # related to these digits from the dataset
    indexD1 = y_digits==1
    indexD2 = y_digits==7
    allindices = indexD1 | indexD2
    X_digits = X_digits[allindices]
    y_digits = y_digits[allindices]
 

    # We need to make sure that we conveert the labels to 
    # 0 and 1 otherwise our cross entropy won't work 
    # Remember we will compare our predicted values between 0 and 1
    # with actual values which should be either 0 or 1
    lb = preprocessing.LabelBinarizer()
    y_digits = lb.fit_transform(y_digits)
    y_digits  =y_digits.flatten()

    n_samples = len(X_digits)

    
    # Seperate data in training and test
    # Training data 
    X_train = X_digits[:int(.7 * n_samples)]
    y_train = y_digits[:int(.7 * n_samples)]
    
    # Test data
    X_test = X_digits[int(.7 * n_samples):]
    y_test = y_digits[int(.7 * n_samples):]

   
    logisticRegression(X_train, y_train, X_test, y_test)
    
    
  

main()
