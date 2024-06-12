import numpy as np
import matplotlib.pyplot as plt
import data

def binlogreg_train(X, Y_, param_niter: int = 30000, param_delta = 0.01):
    n, d = X.shape

    w: float = np.random.randn(d)
    b = 0.0

    # gradijentni spust (param_niter iteracija)
    for i in range(param_niter):
        # klasifikacijske mjere
        scores = np.dot(X, w) + b  # N x 1

        # vjerojatnosti razreda c_1
        probs = 1 / (1 + np.exp(-scores)) # N x 1

        # gubitak
        loss = -np.mean(Y_ * np.log(probs) + (1 - Y_) * np.log(1 - probs)) # dodao sam epsilon kako nebi doslo do dijeljenja s 0

        # dijagnostički ispis
        #if i % 10 == 0:
        #    print("iteration {}: loss {}".format(i, loss))

        # derivacije gubitka po klasifikacijskim mjerama
        dL_dscores = probs - Y_  # N x 1

        # gradijenti parametara
        grad_w = np.dot(dL_dscores.T, X) / n  # D x 1
        grad_b = np.sum(dL_dscores.T) / n # 1 x 1

        # poboljšani parametri
        w -= param_delta * grad_w
        b -= param_delta * grad_b

    return w, b

def binlogreg_classify(X, w, b):
    scores = np.dot(X, w) + b
    
    # koristio sam sigmoidalnu funkciju
    return 1 / (1 + np.exp(-scores))

def binlogreg_decfun(w,b):
    def classify(X):
      return binlogreg_classify(X, w, b)
    return classify

def myDummyDecision(X):
    return X[:,0] + X[:,1] - 5

if __name__=="__main__":
    #np.random.seed(100)
    # instantiate the dataset
    X, Y_ = data.sample_gauss_2d(2, 100)

    # train the logistic regression model
    w, b = binlogreg_train(X, Y_)
    
    # evaluate the model on the train set
    classify = binlogreg_decfun(w, b)
    probs = classify(X)

    # recover the predicted classes Y
    Y = probs > 0.5

    # evaluate and print performance measures
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print (f'accuracy: {accuracy}, recall: {recall}, precision: {precision}, AP: {AP}')

    # graph the decision surface
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(classify, bbox, offset=0.5)
    
    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()
