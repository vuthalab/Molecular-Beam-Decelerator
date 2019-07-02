import os
os.chdir("/home/james/Desktop/Molecular-Beam-Decelerator/Molecular_beam_slowing/Magnetic Field Plot/Code")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from collections import Iterable
from textfile_functions import *


## Generating Axis Coordinates 
x = np.linspace(-2.5, 2.5, num = 51)    
y = np.linspace(-2.5, 2.5,num = 51)    
z = np.linspace(0, 20, num = 41)

points = []
for i in range(len(x)):
    for j in range(len(y)):
        for k in range(len(z)):
            points.append((x[i], y[j], z[k]))


## Analyzing Data
def flatten_list(list,ignore_types=(str)): 
    """
    Takes a nested list of any dimension and recursively "flattens" it out 
    into a 1 dimension list.
    
    @type list: List Object
    @type ignore_types= (str): Ignores all string inputs
    @rtype: None
    """
    for item in list:
        if isinstance(item, Iterable) and not isinstance(item, ignore_types):
            yield from flatten_list(item,ignore_types=(str))
        else:
            yield item
            

def generator_to_list(gen):
    """
    Converts an abstract generator object into a list containing the same 
    elements as the generator.
    
    @type gen: Generator object
    @rtype: List[items]
    """
    temp_list = []
    for item in gen:
        temp_list.append(item)
    return temp_list
    
    
def list_to_numpy_matrix(xdim, ydim, bvalues):
    """
    Converts a list of b field values into a xdim*ydim numpy matrix.
    
    @type xdim: List
    @type ydimL List
    @type bvalues: List
    @rtype: numpy array
    """
    dim_list = []
    for i in range(len(xdim)):
        b_list = []
        for j in range(len(ydim)):
            b_list.append(bvalues.pop(0))
        temp_np = np.array(b_list)
        dim_list.append(temp_np)
    return np.array(dim_list)
    ##
    
#Put in the name of your text file here.
mag_values = load_txtfile_list("bnorm_actual.txt")
#print(mag_values)
print(len(mag_values))

        ##

# Creates nested list of b-field values. 
bfields = [] #change to lst

z_len = []
for i in range(len(z)):
    bfields.append([]) #change to lst
    z_len.append(i)

loop_range = len(mag_values) // len(z)
for i in range(loop_range):
    for i in z_len:
        popped_value = mag_values.pop(0)
        bfields[i].append(popped_value)


# Bfield values at some Z.
start = list_to_numpy_matrix(x, y, bfields[0])
midpoint = list_to_numpy_matrix(x, y, bfields[20])
end = list_to_numpy_matrix(x, y, bfields[40])

            
## Plotting Figures
# When the meshgrid is symmetric, ie size(x) = size(y), they can not all be named X,Y,Z
# for each meshgrid, this the names are written to be cyclic and contain the dimensions of the grid

XY, YX = np.meshgrid(x, y, indexing='ij', sparse=True)
XZ, ZX = np.meshgrid(x, z, indexing='ij', sparse=True)
YZ, ZY = np.meshgrid(y, z, indexing='ij', sparse=True)
XYZ, YZX, ZXY = np.meshgrid(x, y, z, indexing='ij', sparse=True)

fig = plt.figure("Z = 0")
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(XY, YX, start,cmap='coolwarm')
ax.set_title('surface')
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("|B| [T]")

fig = plt.figure("Z = 10")
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(XY, YX, midpoint, cmap='coolwarm')
ax.set_title('surface')
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("|B| [T]")

fig = plt.figure("Z = 20")
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(XY, YX, end, cmap='coolwarm')
ax.set_title('surface')
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("|B| [T]")

plt.show()




