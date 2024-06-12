import numpy as np
import matplotlib.pyplot as plt

class Random2DGaussian:
    def __init__(self, minx: int = 0, maxx: int = 10, miny: int = 0, maxy: int = 10):
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
        self.mean = np.random.random_sample(2) * [self.maxx - self.minx, self.maxy - self.miny]
        self.cov_matrix = [[Random2DGaussian.calculate_eigen_value(maxx - minx), 0], [0, Random2DGaussian.calculate_eigen_value(maxy - miny)]]
        self.alpha = np.random.random_sample() * 2 * np.pi
        self.rotational_matrix = [[np.cos(self.alpha), -np.sin(self.alpha)],
                                  [np.sin(self.alpha), np.cos(self.alpha)]]

        self.sigma_matrix = np.dot(np.dot(np.transpose(self.rotational_matrix), self.cov_matrix), self.rotational_matrix)

    def get_sample(self, size: int = 100) -> np.array:
        return np.random.multivariate_normal(self.mean, self.sigma_matrix, size)

    @staticmethod
    def calculate_eigen_value(ran: int):
        return (np.random.random_sample() * ran / 5) ** 2


def eval_perf_binary(Y, Y_):
    TP = np.sum(np.logical_and(Y == 1, Y_ == 1))
    FP = np.sum(np.logical_and(Y == 1, Y_ == 0))
    TN = np.sum(np.logical_and(Y == 0, Y_ == 0))
    FN = np.sum(np.logical_and(Y == 0, Y_ == 1))

    return (TP + TN) / (TP + FP + TN + FN), TP / (TP + FP), TP / (TP + FN)

def eval_AP(Yr):
    tp = np.sum(Yr)
    fp = len(Yr) - tp

    prec = []

    for c in Yr:
        if c:
            prec.append(tp / (tp + fp))

        tp -= c
        fp -= not c

    return np.sum(prec) / np.sum(Yr)

def sample_gauss_2d(C, N):
  Gs=[]
  Ys=[]
  for i in range(C):
    Gs.append(Random2DGaussian())
    Ys.append(i)

  X = np.vstack([G.get_sample(N) for G in Gs])
  Y_= np.hstack([[Y]*N for Y in Ys])
  
  return X,Y_

def generate_random_colors(n):
    return [(np.random.random() * 0.5, np.random.random() * 0.5, np.random.random() * 0.5) for i in range(n)]

def graph_data(X, Y_, Y, special=[]):
    correct = Y_ == Y
    incorrect = Y_ != Y
     
    sizes = np.repeat(20, len(Y_))
    sizes[special] = 40

    colors = ['gray', 'white'] + generate_random_colors(np.argmax(Y_) - 1)

    plt.scatter(X[correct, 0], X[correct, 1], c=[colors[y] for y in Y_[correct]],
                edgecolors='black', marker='o', s=sizes[correct])
     
    plt.scatter(X[incorrect, 0], X[incorrect, 1], c=[colors[y] for y in Y_[incorrect]],
                edgecolors='black', marker='s', s=sizes[incorrect]) 
    
def graph_surface(function, rect, offset=0.5, width=256, height=256):
    lsw = np.linspace(rect[0][1], rect[1][1], width) 
    lsh = np.linspace(rect[0][0], rect[1][0], height)
    xx0,xx1 = np.meshgrid(lsh, lsw)
    grid = np.stack((xx0.flatten(),xx1.flatten()), axis=1)

    #get the values and reshape them
    values=function(grid).reshape((width,height))
    
    # fix the range and offset
    delta = offset if offset else 0
    maxval=max(np.max(values)-delta, - (np.min(values)-delta))
    
    # draw the surface and the offset
    plt.pcolormesh(xx0, xx1, values, 
        vmin=delta-maxval, vmax=delta+maxval)
        
    if offset != None:
        plt.contour(xx0, xx1, values, colors='black', levels=[offset])
