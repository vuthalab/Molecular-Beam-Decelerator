"""
Useful class objects and helper functions
"""
import numpy as np
from numpy import linalg as lg
from numpy import pi,sin,cos,tan,sqrt
from sympy.physics.wigner import wigner_3j,wigner_6j
import sympy as sy
from copy import deepcopy


## Class Modules

class MolecularState():
    """
    A class object which represents the molecular state of a molecule based on
    its internal conditions.
    """
    def __init__(self, N=0, S=1/2, I=1/2):
        """
        Initializes a new Molecular class with the given parameters.
        """
        self.N = N
        self.mN = N
        self.S = S
        self.mS = S
        self.I = I
        self.mI = I
        self.p = (-1)**self.N

    def __repr__(self):
        """
        Returns a string representation of the MolecularState object.

        @type MolecularState
        @rtype: string
        """
        attribs = [str(s) for s in [self.N, self.mN,self.S, self.mS, self.I, self.mI]]
        string = ','.join([str(s) for s in attribs])
        return "|" + string + "; " + "mF=" + str(self.mF()) + ">"

    def mF(self):
        return self.mN + self.mS + self.mI


def sublevel_expand(basis):
    """
    Creates all possible permuations from individual elements in our basis.

    @type basis
    @rtype: List[Basis Vectors]
    """
    newbasis = []
    for ket in basis:
        for mN in np.arange(-ket.N,ket.N+1,1):
            for mS in np.arange(-ket.S,ket.S+1,1):
                for mI in np.arange(-ket.I,ket.I+1,1):
                    newket = deepcopy(ket)
                    newket.mN = mN
                    newket.mS = mS
                    newket.mI = mI
                    newbasis.append(newket)
    return newbasis


class ExcitedMolecularState():
    """
    A new class obeject representing a Molecule in a particular orientation
    in it's ground state.
    """
    def __init__(self,J=1/2,I=1/2):
        """
        @type self: ExcitedMolecularState Object
        @type N: float (Angular Momentum)
        @rtype S: float (Electron Spin)
        @type I: float (Intrinsic spin)
        """
        self.J=J
        self.mJ = J
        self.I=I
        self.mI = I

    def __repr__(self):
        attribs = [str(s) for s in [self.J,self.mJ,self.I,self.mI]]
        string = ','.join([str(s) for s in attribs])
        return "|" + string + "; " + "mF=" + str(self.excited_mF()) + ">"

    def excited_mF(self):
        return self.mJ + self.mI

    def excited_F(self):
        return self.J + self.I

def excited_sublevel_expand(basis):
    """
    Creates all possible permuations from individual elements in our basis.

    @type basis
    @rtype: List[Basis Vectors]
    """
    newbasis = []
    for ket in basis:
        for mJ in np.arange(-ket.J, ket.J+1, 1):
            for mI in np.arange(-ket.I, ket.I+1, 1):
                newket = deepcopy(ket)
                newket.mJ = mJ
                newket.mI = mI
                newbasis.append(newket)
    return newbasis



class Energy_Curve:
    """
    A energy curve object which will be used to label each energy curve.
    """
    def __init__(self, energy_values, label):
        """
        Initializes a new energy curve with set energy values and a label

        @type self: energy_curve object
        @type var_1: Array
        @type var_2: Array
        @type energy_values: Array
        @type label: String
        @rtype: None
        """
        self.energy_values = energy_values
        self.label = label

    def __repr__(self):
        """
        Returns the string representation of this Energy curve.

        @type self: Energy_curve
        @rtype: string
        """
        return self.label

    def get_energy_values(self):
        """
        Returns the Energy level values in this curve.

        @typse self: Energy_surface
        @rtype: array
        """
        return self.energy_values
