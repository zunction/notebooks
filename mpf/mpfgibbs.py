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
    np.random.seed(0)
    initialState = np.random.binomial(1, p, v)
    print ('Initial states: ',initialState)
    return initialState


def initializeW(v):
    """
    Initializes a v by v matrix of values. For now we can think of it
    in the scenario of a Ising model where W describes the interaction
    between the different nodes.
    Input:
    - v: (int) number of neurons in the system.
    """
    W = np.random.normal(0, 1, (v,v))
    W = 0.5 * (W + np.transpose(W))
    W = W - np.diag(W)

    # To save and load W matrix
    np.save('W.dat', W)
    # W = np.load('W.dat')
    print ('Initialized W matrix: ', W)
    return W


def sigmoid(x):
    """
    Takes in a vector x and returns its sigmoid activation.
    Input:
    - x: a numpy array
    """
    return 1/(1 + np.exp(-x))


def single_unit_update(initialState, W, v):
    """
    Returns the new states and the state of the vth vertex that has been updated conditioned on the other units
    Input:
    - initialState: a numpy array of binary values denoting the initial state of the nodes.
    - W: a 2d numpy array of values that the prior distribution is based from.
    - v: (int) the state of the vertex to be updated.
    """
    stateSize = initialState.shape[0]
    newState = initialState
#     Here we see that to update a single vertex state we only use the weights Wij for i not
#     equal to j and hence the reason to set the diagonals to be zero earlier. But since
#     we did not we have to kill off the diagonals of W here.
    prob = sigmoid((W - (W * np.eye(stateSize))).dot(initialState))
    newState[v] = np.random.binomial(1, prob[v], 1)
#     print (initialState[n], newState[n])
    return newState, newState[v]


def gibbs_sample(initialState, W):
    """
    Returns the new state of the network after updating all v units systematically, given an initialized state
    of the network and weight matrix W.
    Input:
    - initialState: a numpy array of binary values denoting the initial state of the nodes.
    - W: a 2d numpy array.
    """
#     print ('initialState:', initialState)
    stateSize = initialState.shape
    newState = np.zeros(stateSize)
    for i in range(stateSize[0]):
#         print ('Changing the state for unit %d...'% i)
        initialState, vertexState = single_unit_update(initialState, W, i)
#         print ('Old unit state is %d, new unit state is %d'% (initialState[i], unitState))
        newState[i] = vertexState
#     print ('newState:', newState)
    return newState


def multi_gibbs_sample(initialState, W, n):
    """
    Performs Gibbs sampling n times with a given initial state and weight matrix W
    and stores each sample as a row.
    Input:
    - initialState: a numpy array of binary values denoting the initial state of the nodes.
    - W: a 2d numpy array.
    - n: (int) number of samples to be drawn.
    """
    stateSize = initialState.shape[0]
    sample = np.zeros((n, stateSize))
    for i in range(n):
        sample[i, :] = gibbs_sample(initialState, W)
    return sample

def rand_gibbs_sample(initialState, W, n):
    """
    Does a random scan Gibbs sampling n times with a given initial state and weight matrix W.
    - initialState: a numpy array of binary values denoting the initial state of the nodes.
    - W: a 2d numpy array.
    - n: (int) number of samples to be drawn.
    """
#     v = W.shape[0]
#     sample = np.zeros((n, initialState.shape[0]))
    for i in range(n):
        s = np.random.randint(0, v)
        initialState, vertexState = single_unit_update(initialState, W, s)
#         sample[i, :] = gibbs_sample(initialState, W)
    return initialState


def burnin(initialState, W, t):
    """
    Burn-in time for Gibbs sampling.
    Input:
    - initialState: (numpy array) v by 1 array of initialState.
    - v: (int) number of neurons in the system.
    - t: (time/iterations) As a figure of rule, we use t = 10000 multiplied
    by the number of neurons.
    """
    v = initialState.shape[0]
    burnin_state = rand_gibbs_sample(initialState, W, t * v)
    print ('Burn-in state: ', burnin_state)
    return burnin_state


def mixin_gibbs_sample(initialState, W, n, m, savesamples = 'True'):
    """
    Does a random scan Gibbs sampling n * m times with a given initial state and weight matrix W and
    stores a sample every m iterations.
    - initialState: a numpy array of binary values denoting the initial state of the nodes.
    - W: a 2d numpy array.
    - n: (int) number of samples to be drawn.
    - m: (int) number of iterations before a sample is drawn.
    - savedate: (bool) save samples as 'samples.dat.npy' if True and does not save if false.
    """
    tic = time.time()

    v = W.shape[0]
    sample = np.zeros((n, initialState.shape[0]))
    for i in range(n):
        sample[i, :] = rand_gibbs_sample(initialState, W, m)
    if savesamples == "True":
        np.save('sample.dat', sample)
        print ('Samples are saved as "sample.dat.npy"')
    elif savesamples == "False":
        print ('Samples were not saved. Run np.save("sample.dat", sample) to save them. ')
    else:
        raise ValueError("savesamples must be 'True' or 'False'")

    toc = time.time()
    print ('Time taken to create %d samples is %f minutes' % (n, (toc - tic)/60))
    return sample


if __name__ == "__main__":
    v = 16
    p = 0.5
    t = 10000
    initialState = initializestates(v, p)
    W = initializeW(v)
    burnin_state = burnin(initialState, W, t)
    sample = mixin_gibbs_sample(burnin_state, W, 50000, 100)
