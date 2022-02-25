import numpy as np
import csv

def import_data(test_X_file_path, test_Y_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    test_Y = np.genfromtxt(test_Y_file_path, delimiter=',', dtype=np.float64)
    return test_X, test_Y

def compute_gradient_of_cost_function(X, Y, W):
    y_predicted=np.matmul(X,W)-Y
    y_predicted=np.matmul(np.transpose(y_predicted),X)
    y_predicted=y_predicted/len(Y)
    return np.transpose(y_predicted)


def optimize_weights_using_gradient_descent(X, Y, W, num_iterations, learning_rate):
  
    for i in range(num_iterations):
            W=W-learning_rate*compute_gradient_of_cost_function(X,Y,W)
            cost=compute_cost(X,Y,W)
            print(i,cost)
    return W

def compute_cost(X, Y, W):
    y_predicted=np.matmul(X,W)
    Y=Y-y_predicted
    Y=np.square(Y)
    return ((np.sum(Y))/(2*(len(Y))))    

def train_model():
    X, Y = import_data("./train_X_lr.csv","./train_Y_lr.csv")
    #insert 1s row in X
    a=np.ones((X.shape[0],1),np.float64)
    X=np.hstack((a,X))
    Y=Y.reshape(len(X),1)
    #on basis of initial W=[0s] and learning rate=0.00021457 and iteratons= nearly 11 crore
    W=np.array([[-3644.048346708415],[0.5904458396592063],[67.94140067272193],[42.31264892442709],[28.252418102605578]])
    #W=np.zeros((X.shape[1],1))
    W=optimize_weights_using_gradient_descent(X,Y,W,500000,0.00021457999999)
    return W

def save_model(W,file_name):
    with open(file_name,'w') as file_x:
        wr=csv.writer(file_x)
        wr.writerows(W)
        file_x.close()

if __name__ == "__main__":
    W=train_model()
    save_model(W,"WEIGHTS_FILE.csv")