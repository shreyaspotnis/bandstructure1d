import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt


def calculate_band_structure(n_bands, n_q, lattice_depth):
    """Calculate the band structure for a given depth of the potential.

    Inputs:
        n_bands - Number of bands to include in the calculation. The higher this
        number, the more accurate your results. But note that the calculation
        involves diagonalizing a (2M+1)*(2M+1) matrix and the time required
        scales as n_bands^3. For the first 5-6 bands, n_bands=20 is sufficient.

        n_q - Number of quasi-momentum to calculate the band structure for. The
        more this number, the finer your plot will look. Note that the
        quasimomentum go from -1 recoil momentum to +1 recoil momentum

        lattice_depth - depth of your lattice divided by the recoil energy.

    Outputs:
        Q - numpy array of all the quasimomentum (in units of the reciprocal
        lattice vector) for which the band energies are calculated.

        E - 2D numpy array E[q,n] would be the energy of the particle in the
        nth band with q quasimomentum.

        E_free - Energy if there wasn't any lattice (free particle energy =
        q^2)

        u - 3D numpy array u[q,:,n] would give you the eignenvector
        corresponding to nth band with q quasimomentum. u is in the k basis.

        u_free - corresponding free particle eigenstates.
    """

    # create an array of quasimomenta in the first brilluoin zone
    Q = np.arange(-n_q*1.0, n_q*1.0) / n_q

    H = np.zeros((2*n_bands+1, 2*n_bands+1))
    H_free = np.zeros((2*n_bands+1, 2*n_bands+1))
    E = np.zeros((2*n_q, 2*n_bands+1))
    E_free = np.zeros((2*n_q, 2*n_bands+1))

    u_free = np.zeros((2.0*n_q, 2.0*n_bands+1, 2.0*n_bands+1), complex)
    u = np.zeros((2.0*n_q, 2.0*n_bands+1, 2.0*n_bands+1), complex)

    for i, q in enumerate(Q):
        # fill up the hamiltonian
        kinetic = (q+2.0*np.linspace(-n_bands, n_bands, 2*n_bands+1)) ** 2
        potential = lattice_depth*np.ones(2.0*n_bands)/4.0
        H_free = np.diag(kinetic, 0)
        H = H_free+np.diag(potential, 1)+np.diag(potential, -1)

        # diagonalize the hamiltonian
        energies, eigvectors = lin.eigh(H)
        energies_free, eigvectors_free = lin.eigh(H_free)

        # store all eigenvalues and eigen-energies
        E[i, :] = energies
        E_free[i, :] = energies_free
        u[i, :] = eigvectors
        u_free[i, :] = eigvectors_free

    return (Q, E, E_free, u, u_free)


def plot_bands(n_plot, Q, E, E_free, show=True):
    """ Plots bands given the band structure.

    Input:
        n_plot - number of bands to plot
        Q - an array of all quasi-momenta q
        E - band structure energies calculated using band_structure_calculate
        E_free - corresponding free particle energies.
        show - whether to show the plot or not (use it if you'd rather print
        it.)

    Output:
        none
    """
    plt.plot(Q, E[:, 0:n_plot], Q, E_free[:, 0:3])
    plt.xlabel('$q/\hbar k$')
    plt.ylabel('$E/E_r$')
    plt.xlim(Q.min(), Q.max())
    if show:
        plt.show()
