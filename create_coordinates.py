import numpy as np
import pandas as pd
import math

input_file = "gbrbm_mc_output.txt"
output_file = "h2o_gbrbm.xyz"

initial_o = np.array([0.53890, 0.65566, 0.00000])

def main():
    bonds_and_angles = np.loadtxt(input_file)
    f = open(output_file, "w+")
    counter=0
    for line in bonds_and_angles:
        f.write("3 \nconf {}\n".format(counter))
        counter+=1
        b1 = line[0]
        b2 = line[1]
        a1 = line[2]
        angle = math.radians(a1)
        wha = math.sin(angle)
        o_pos = initial_o
        h1_pos = initial_o + np.array([-b1, 0, 0])
        h2_pos = initial_o + np.array([-b2*math.cos(math.radians(a1)), b2*math.sin(math.radians(a1)), 0])
        xyz = np.array([o_pos,h1_pos,h2_pos])
        conf = pd.DataFrame(xyz, index=["O", "H", "H"])
        conf.to_csv(f, sep="\t", header=False, line_terminator='\n')
    f.close()

if __name__ == '__main__':
    main()


