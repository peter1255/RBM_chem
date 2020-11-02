from itertools import islice
import pandas as pd
import numpy as np
from math import floor

input_file = "h2o_geom_from_scf.dat"
output_file = "h2o_scf.dat"


# distance between two points
def distance(c1, c2):
    return np.linalg.norm(c2 - c1)

# angle between three points, where b is the vertex
def angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

def main():
    bonds_1 = []
    bonds_2 = []
    angles_1 = []
    n = 0
    num_lines = floor(sum(1 for line in open(input_file)) / 5)
    while True:
        df = pd.read_csv(input_file, delimiter='\t')
        for i in range(num_lines-1):
            o1 = np.asarray(df.iloc[5 * i + 1, 0].split()[1:], dtype=np.float32)
            h1 = np.asarray(df.iloc[5 * i + 2, 0].split()[1:], dtype=np.float32)
            h2 = np.asarray(df.iloc[5 * i + 3, 0].split()[1:], dtype=np.float32)

            bonds_1.append(distance(o1, h1))
            bonds_2.append(distance(o1, h2))
            angles_1.append(angle(h1, o1, h2))

            if i % 100 == 0:
                print('\r>> Calculating bonds and angles for conf # %d' % (i), flush=True)

        print("Finished calculations, writing file.")
        df1 = pd.DataFrame(data=[bonds_1, bonds_2, angles_1]).transpose()
        df1.to_csv(output_file, sep='\t', header=False)
        print("Finished writing.")
        break

def new_main():
    coord = np.genfromtxt(input_file)
    n_samples = int(coord.shape[0]/3)
    internal_coord = np.zeros((n_samples,3))
    for ii in range(n_samples):
        b1 = distance(coord[ii*3], coord[ii*3+1])
        b2 = distance(coord[ii*3+1], coord[ii*3+2])
        a1 = angle(coord[ii*3], coord[ii*3+1], coord[ii*3+2])
        internal_coord[ii] = np.array([b1,b2,a1])
    np.savetxt(output_file, internal_coord)
    print(np.mean(internal_coord, axis=0))
    print(np.std(internal_coord, axis=0))


if __name__ == '__main__':
    new_main()





