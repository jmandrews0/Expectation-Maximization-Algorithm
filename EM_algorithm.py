import plot_data_and_gaussians as P
import numpy as np
import matplotlib.pyplot as plt

def calcDistances(D, mu):
    dist = np.zeros((D.shape[0], mu.shape[0]))
    for k in range(mu.shape[0]):
        diff = D - np.repeat([mu[k]], D.shape[0], axis=0)
        dist[:,k] = np.sqrt(np.matmul(diff,diff.transpose()).diagonal())
    return dist

def assignPoints(dist):
    assignment = np.zeros(dist.shape)
    counts = np.zeros(dist.shape[1])
    for i in range(dist.shape[0]):
        minIndex = np.where(dist[i] == np.amin(dist[i]))[0]
        assignment[i,minIndex] = 1
        counts[minIndex] += 1
    return assignment, counts
        
def sum_squared_err(D, mu, assignment):
    sum2err = 0
    for i in range(len(D)):
        sum2err += np.square(np.linalg.norm(D[i] - np.matmul(assignment[i],mu)))
    return sum2err / len(D)


def k_means(D, K, init_method, epsilon, niterations, plotflag, RSEED=123):
    
    # initialize problem
    errorPlot = []
    N = len(D)
    converged = False
    
    # randomly select k mean vectors
    mu = np.zeros((K,2))
    for i in range(K):
        mu[i] = D[np.random.randint(N)]

    while not converged:
        # assign each of the N data vectors to the clusters
        dist = calcDistances(D, mu)
        assignment, counts = assignPoints(dist)
        #print("assignment: ", assignment.shape)
        #print("counts\n", counts)
        
        # compute new means
        for k in range(K):
            mask = np.repeat([assignment[:,k]], 2, axis=0)
            mu[k] = np.sum((D*mask.transpose()), axis=0) / counts[k]
        #print("mu: ", mu)
        
        # check for convergence
        dist = calcDistances(D, mu)
        assignment2, counts2 = assignPoints(dist)
        change = assignment - assignment2
        if np.count_nonzero(change) == 0:
            converged = True
            
        # add to error plot data
        errorPlot.append(sum_squared_err(D, mu, assignment))
    
    return mu, assignment, errorPlot


def k_means_N_times(D, K, N, init_method, epsilon, niterations, plotflag):
    
    mu, assignment , plot = k_means(D, K, init_method, epsilon, niterations, plotflag)
    for i in range(N):
        #print("iteration: ", i+1)
        mu1, assignment1 , plot1 = k_means(D, K, init_method, epsilon, niterations, plotflag)
        if plot1[-1] < plot[-1]:
            mu = np.array(mu1)
            assignment = np.array(assignment1)
            plot = np.array(plot1)
    return mu, assignment, plot
    
if __name__ == "__main__":
    data = np.genfromtxt("dataset1.txt")
    k_means(data, 2, 1, 0.01, 10, True)