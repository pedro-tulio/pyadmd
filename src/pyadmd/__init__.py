"""
The Adaptive Molecular Dynamics with Excited Normal Modes (aMDeNM) method applies a kinetic excitation of normal modes (NMs)
to enhance molecular dynamics simulations sampling. This technique consists in injecting additional atomic velocities along
a combinations of NM vectors, creating an effective coupling between slow and fast molecular motions. The motions described
by preselected directions of low-frequency NMs are dynamically adjusted throughout the simulation. By coupling low-frequency
NM excitation with adaptive directional adjustments, aMDeNM facilitates extensive exploration of the energy landscape,
overcoming the constraints of fixed, rectilinear displacements and alleviating structural stresses and environmental resistance.
Importantly, aMDeNM requires only an initial structure without the need to specify predefined target states, distinguishing
it from many biased sampling techniques that rely on predefined target conformations.
"""

__version__ = "3.1.0"
