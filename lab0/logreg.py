import numpy as np
import matplotlib.pyplot as plt
import data

def logreg_train(X, Y_, param_niter: int = 30000, param_delta = 0.01):
    n, d = X.shape
    nClasses = max(Y_) + 1

    w: float = np.random.randn(d, nClasses)
    b = np.zeros(nClasses)

    Y_onehot = np.zeros((n, nClasses))
    Y_onehot[np.arange(n), Y_] = 1

    # gradijentni spust (param_niter iteracija)
    for i in range(param_niter):
        # eksponencirane klasifikacijske mjere
        # pri računanju softmaksa obratite pažnju
        # na odjeljak 4.1 udžbenika
        # (Deep Learning, Goodfellow et al)!
        scores = np.dot(X, w) + b    # N x C
        expscores = np.exp(scores) # N x C
        
        # nazivnik sofmaksa
        sumexp = np.sum(expscores, axis=1, keepdims=True)    # N x 1

        # logaritmirane vjerojatnosti razreda 
        probs = expscores / sumexp    # N x C
        logprobs = np.log(probs)  # N x C

        # gubitak
        loss = -np.sum(logprobs)     # scalar
        
        # dijagnostički ispis
        #if i % 10 == 0:
        #    print("iteration {}: loss {}".format(i, loss))

        # derivacije komponenata gubitka po mjerama
        dL_ds = probs - Y_onehot     # N x C

        # gradijenti parametara
        grad_W = np.dot(X.T, dL_ds) / n    # C x D (ili D x C)
        grad_b = np.sum(dL_ds, axis=0) / n    # C x 1 (ili 1 x C)

        # poboljšani parametri
        w -= param_delta * grad_W
        b -= param_delta * grad_b

    return w, b

def logreg_classify(X, w, b):
    # eksponencirane klasifikacijske mjere
    # pri računanju softmaksa obratite pažnju
    # na odjeljak 4.1 udžbenika
    # (Deep Learning, Goodfellow et al)!
    scores = np.dot(X, w) + b    # N x C
    expscores = np.exp(scores) # N x C
    
    # nazivnik sofmaksa
    sumexp = np.sum(expscores, axis=1, keepdims=True)    # N x 1

    return expscores / sumexp

def logreg_decfun(w,b):
    def decfun(X):
        return np.argmax(logreg_classify(X, w, b), axis=1)
    
    return decfun

def myDummyDecision(X):
    return X[:,0] + X[:,1] - 5

def eval_perf_multi(Y, Y_):
    n = np.max(Y_) + 1
    
    # inicijaliziraj konfuzijsku matricu
    # retci su stvarni razredi, dok su stupci predikcija
    confusion_matrix = np.zeros((n, n))

    for predicted, actual in zip(Y, Y_):
        confusion_matrix[actual, predicted] += 1

    # (TP + TN) / (TP + TN + FP + FN) = (zbroj na dijagonali) / ukupan zbroj
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    precision = np.zeros(n)
    recall = np.zeros(n)

    for i in range(n):
        precision[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])
        recall[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])

    return accuracy, precision, recall

def graph_data(X, Y_, Y, special=[]):
    correct = []
    incorrect = []

    for i in range(Y_.shape[0]):
        if Y_[i] == np.argmax(Y[i]):
            correct.append(i)
        else:
            incorrect.append(i)
     
    sizes = np.repeat(20, len(Y_))
    sizes[special] = 40

    plt.scatter(X[correct, 0], X[correct, 1], c=['gray' if y == 0 else 'white' for y in Y_[correct]],
                edgecolors='black', marker='o', s=sizes[correct])
     
    plt.scatter(X[incorrect, 0], X[incorrect, 1], c=['gray' if y == 0 else 'white' for y in Y_[incorrect]],
                edgecolors='black', marker='s', s=sizes[incorrect])


if __name__=="__main__":
    #np.random.seed(100)
    # instantiate the dataset
    X, Y_ = data.sample_gauss_2d(3, 100)

    # train the logistic regression model
    w, b = logreg_train(X, Y_)
    
    # evaluate the model on the train set
    probs = logreg_classify(X, w, b)

    # recover the predicted classes Y
    Y = probs > 0.5

    # evaluate and print performance measures
    accuracy, recall, precision = eval_perf_multi(Y, Y_)
    print (f'accuracy: {accuracy}, recall: {recall}, precision: {precision}')

    # graph the decision surface
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(logreg_decfun(w, b), bbox, offset=0)
    
    # graph the data points
    data.graph_data(X, Y_, np.argmax(Y, axis=1), special=[])

    # show the plot
    plt.show()
