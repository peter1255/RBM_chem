import numpy as np
import math
import random

NUM_SAMPLES = 10000
delta = 1.0
means = [0.96, 0.96, 104.4]
sigmas = [0.023, 0.023, 2.97]

def equation(x):
    energy = ((x[0]-means[0])/sigmas[0])**2 + ((x[1]-means[1])/sigmas[1])**2 + ((x[2]-means[2])/sigmas[2])**2 + \
             10 * (((x[0]-means[0])/sigmas[0])**2) * (((x[1]-means[1])/sigmas[1])**2) * (((x[2]-means[2])/sigmas[2])**2)

    #multivariate gaussian

    #four pointed star
    #energy = 100000 * ((x[0]-means[0])/sigmas[0])**4 * ((x[1]-means[1])/sigmas[1])**4 * ((x[2]-means[2])/sigmas[2])**4 + \
    #         ((x[0] - means[0]) / sigmas[0])**2 + ((x[1]-means[1])/sigmas[1])**2 + ((x[2]-means[2])/sigmas[2])**2

    #weird circle
    #energy = ((x[0]-means[0])/sigmas[0])**2 + ((x[1]-means[1])/sigmas[1])**2 + ((x[2]-means[2])/sigmas[2])**2 + \
    #         1.8 * np.sqrt(((x[0]-means[0])/sigmas[0])**2 + ((x[1]-means[1])/sigmas[1])**2)*((x[2]-means[2])/sigmas[2])

    # positive corr
    #energy = ((x[0]-means[0])/sigmas[0])**2 + ((x[1]-means[1])/sigmas[1])**2 + ((x[2]-means[2])/sigmas[2])**2 - \
    #         ((x[0]-means[0])/sigmas[0])*((x[1]-means[1])/sigmas[1]) - ((x[1]-means[1])/sigmas[1])*((x[2]-means[2])/sigmas[2])

    # negative corr
    #energy = ((x[0]-means[0])/sigmas[0])**2 + ((x[1]-means[1])/sigmas[1])**2 + ((x[2]-means[2])/sigmas[2])**2 + \
    #         ((x[0]-means[0])/sigmas[0]) * ((x[1]-means[1])/sigmas[1]) + ((x[1]-means[1])/sigmas[1])*((x[2]-means[2])/sigmas[2])
    return energy


def keep_or_not(vec, prev_energy):
    kBT = 1
    current_energy = equation(vec)
    if current_energy <= prev_energy:
        return True
    probs = math.exp((prev_energy - current_energy) / kBT)
    if random.random() < probs:
        return True
    else:
        return False

def simulate():
    table = np.zeros([NUM_SAMPLES, 3], dtype=float)
    vec = np.random.normal(loc=0, scale=1, size=[3])*np.array([0.023, 0.023, 2.97])+np.array([0.96, 0.96, 104.4])
    tried=0
    acc=0
    while acc < NUM_SAMPLES:
        energy = equation(vec)
        new_x = vec + np.random.normal(loc=0, scale=delta, size=[3])*np.array([0.023, 0.023, 2.97])
        if keep_or_not(new_x, energy):
            table[acc] = new_x
            vec = new_x
            acc+=1
        tried+=1
    acceptance_rate = acc/tried
    print("PERCENT CONFS ACCEPTED: {:.2f}%".format(acceptance_rate * 100))
    np.savetxt("h2o_geom_from_eq6.dat", table, delimiter='\t', fmt='%.18f')

def generate_2D_mixture(num_samples,
                         mean=0.0,
                         scale=np.sqrt(2.0) / 2.0):
    ''' Creates a dataset containing 2D data points from a random mixtures of
        two independent Laplacian distributions.

    :Info:
        Every sample is a 2-dimensional mixture of two sources. The sources
        can either be super_gauss or sub_gauss. If x is one sample generated
        by mixing s, i.e. x = A*s, then the mixing_matrix is A.

    :Parameters:
        num_samples: The number of training samples.
                    -type: int

        mean:        The mean of the two independent sources.
                    -type: float

        scale:       The scale of the two independent sources.
                    -type: float

    :Returns:
        Data and mixing matrix
       -type: list of numpy arrays ([num samples, 2], [2,2])

    '''
    source = np.concatenate((np.random.laplace(mean, scale, num_samples),
                             np.random.laplace(mean, scale, num_samples))).reshape(num_samples, 2, order='F')
    mixing_matrix = np.array([[np.sqrt(2)/3, np.sqrt(2)/2],[-np.sqrt(2)/3, np.sqrt(2)/2]]) #np.random.rand(2, 2) - 0.5
    mixture = np.dot(source, mixing_matrix.T)
    return mixture, mixing_matrix

def generate_2D_linear(num_samples,
                       mean=0.0,
                       scale=np.sqrt(2.0) / 2.0):
    ''' Creates a dataset containing 2D data points from a random mixtures of
        two independent Laplacian distributions.

    :Info:
        Every sample is a 2-dimensional mixture of two sources. The sources
        can either be super_gauss or sub_gauss. If x is one sample generated
        by mixing s, i.e. x = A*s, then the mixing_matrix is A.

    :Parameters:
        num_samples: The number of training samples.
                    -type: int

        mean:        The mean of the two independent sources.
                    -type: float

        scale:       The scale of the two independent sources.
                    -type: float

    :Returns:
        Data and mixing matrix
       -type: list of numpy arrays ([num samples, 2], [2,2])

    '''
    x = np.random.laplace(mean, scale, num_samples)
    y = x
    x = x + np.random.normal(mean,1.0,num_samples)
    y = y + np.random.normal(mean,1.0,num_samples)
    source = np.concatenate((x, y)).reshape(num_samples, 2, order='F')
    mixing_matrix = np.array([[np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, np.sqrt(2) / 2]])  # np.random.rand(2, 2) - 0.5
    mixture = np.dot(source, mixing_matrix.T)
    return source, mixing_matrix

def generate_3D_mixture(num_samples,
                         mean=0.0,
                         scale=np.sqrt(2.0) / 2.0):
    source,_ = generate_2D_mixture(num_samples, mean, scale)
    x = source[:,0]
    y = source[:,1]
    z = np.random.normal(mean,1.0,num_samples)
    source = np.concatenate((x, y, z)).reshape(num_samples, 3, order='F')
    mixing_matrix = np.array([[1,0,0],[0, 1, 0], [0, 0, 1]]) #np.random.rand(2, 2) - 0.5
    mixture = np.dot(source, mixing_matrix.T)
    return source, mixing_matrix


if __name__ == '__main__':
    confs, mixing_matrix = generate_3D_mixture(10000, mean=0, scale=1)
    confs -= np.average(confs, 0)
    confs /= np.std(confs, 0)
    confs = confs * sigmas + means
    np.savetxt("3D_laplace_eigenmode1.txt", confs)
    print("confs generated")


