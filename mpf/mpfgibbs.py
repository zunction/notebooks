import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import time

def initializestates(v, p):
    """
    Initializes a v by 1 array of initial states
    Input:
    - v: (int) number of neurons in the system.
    - p: (probability) probability of getting a 1.
    """
    initialState = np.random.binomial(1, p, v)
    initialState = initialState.reshape(1, v)
    return initialState


def initializeW(v):
    """
    Initializes a v by v matrix of values. For now we can think of it
    in the scenario of a Ising model where W describes the interaction
    between the different nodes.
    Input:
    - v: (int) number of neurons in the system.
    """
    W = np.random.normal(0, 1, (v, v))
    W = np.triu(W, 1)+np.triu(W, 1).T
    # To save and load W matrix
    np.save('W.dat', W)
    # W = np.load('W.dat')
    print ('Initialized W matrix: \n', W)
    return W


def initializeb(v):
    """
    Initializes a 1 by v array of values. For now we can think of it
    some form of bias in the system.
    Input:
    - v: (int) number of neurons in the system.
    """
    b = np.zeros((1,v))
    # W = np.random.normal(0, 1, (v,v))
    # W = 0.5 * (W + np.transpose(W))
    # W = W - np.diag(W)

    # To save and load W matrix
    # np.save('W.dat', W)
    # W = np.load('W.dat')
    print ('Initialized bias: ', b)
    return b

def sigmoid(x):
    """
    Takes in a vector x and returns its sigmoid activation.
    Input:
    - x: a numpy array
    """
    return 1/(1 + np.exp(-x))


def single_unit_update(initialState, W, b, v):
    """
    Returns the new states and the state of the vth vertex that has been updated conditioned on the other units
    Input:
    - initialState: a numpy array of binary values denoting the initial state of the nodes.
    - W: a 2d numpy array of values that the prior distribution is based from.
    - b: a (1, v) numpy array of bias
    - v: (int) the state of the vertex to be updated.
    """
    stateSize = initialState.shape
    newState = np.zeros(stateSize) + initialState
    prob = sigmoid(initialState.dot(W) + b)
    newState[0, v] = np.random.binomial(1, prob[0, v], 1)
    return newState, newState[0, v]


def rand_gibbs_sample(initialState, W, b, n):
    """
    Does a random scan Gibbs sampling n times with a given initial state, weight matrix W and bias b.
    Input:
    - initialState: a numpy array of binary values denoting the initial state of the nodes.
    - W: a 2d numpy array.
    - b: a (1, v) numpy array of bias
    - n: (int) number of samples to be generated.
    """
    for i in range(n):
        s = np.random.randint(0, v)
        initialState, vertexState = single_unit_update(initialState, W, b, s)
    return initialState


def burnin(initialState, W, b):
    """
    Performs burn in of 10000 x v iterations.
    Input:
    - initialState: a numpy array of binary values denoting the initial state of the nodes.
    - W: a 2d numpy array.
    - b: a (1, v) numpy array of bias
    """
    v = W.shape[0]
    burnin_state = rand_gibbs_sample(initialState, W, b, 10000 * v)
    print ('Burn-in state: ', burnin_state)
    return burnin_state


def mixin_gibbs_sample(initialState, W, b, n, m, savesamples = 'True'):
    """
    Does a random scan Gibbs sampling n * m times with a given initial state and weight matrix W and
    stores a sample every m iterations.
    Input:
    - initialState: a numpy array of binary values denoting the initial state of the nodes.
    - W: a 2d numpy array.
    - n: (int) number of samples to be drawn.
    - m: (int) number of iterations before a sample is drawn.
    - savedate: (bool) save samples as 'samples.dat.npy' if True and does not save if false.
    """
    tic = time.time()

    v = W.shape[0]
    sample = np.zeros((n, initialState.shape[1]))
    for i in range(n):
        initialState = rand_gibbs_sample(initialState, W, b, m)
        sample[i] = initialState
    if savesamples == "True":
        np.save('gibbs-sample.dat', sample)
        print ('Samples are saved as "gibbs-sample.dat.npy"')
    elif savesamples == "False":
        print ('Samples were not saved. Run np.save("gibbs-sample.dat", sample) to save them. ')
    else:
        raise ValueError("savesamples must be 'True' or 'False'")

    toc = time.time()
    print ('Time taken to create %d samples is %f minutes' % (n, (toc - tic)/60))
    return sample

def makesamples(initialState, W, b, n, m, savesamples = 'True'):
    """
    Make samples.
    Input:
    - initialState: a numpy array of binary values denoting the initial state of the nodes.
    - W: a 2d numpy array.
    - n: (int) number of samples to be drawn.
    - m: (int) number of iterations before a sample is drawn.
    - savedate: (bool) save samples as 'samples.dat.npy' if True and does not save if false.
    """
    b = burnin(initialState, W, b)
    samples = mixin_gibbs_sample(b, W, b, n, m, savesamples)
    return samples


if __name__ == "__main__":
    v = 16
    p = 0.5
    t = 10000
    initialState = initializestates(v, p)
    W = initializeW(v)
    b = initializeb(v)
    makesamples(initialState, W, b, 50000, 100)
