import kwant
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi

# ========== Parameters ==========
t = 0.493               # Hopping (eV)
E_up = 9.5913           # Spin-up onsite (eV)
E_down = 7.3639         # Spin-down onsite (eV)
E_fermi = 10.0579       # Fermi level (eV)

# Pauli matrices for spin calculations
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# ========== MTJ Geometry ==========
fm1_length = 12         # FM1 region (sites 0-11)
barrier_length = 6      # Insulating region (sites 12-17)
fm2_length = 6          # FM2 region (sites 18-23)
lat = kwant.lattice.chain(norbs=2)  # 1D chain with spin

def make_mtj(theta=0):
    """Create MTJ with specified geometry and magnetization angle theta"""
    syst = kwant.Builder()
    
    # Rotation matrix for magnetization angle
    rot = np.array([[cos(theta/2), -sin(theta/2)],
                   [sin(theta/2), cos(theta/2)]])
    
    # Onsite potentials
    def fm_potential(site):
        return rot @ np.array([[E_up, 0], [0, E_down]]) @ rot.T
    
    def barrier_potential(site):
        return np.array([[E_fermi, 0], [0, E_fermi]])  # Insulator
    
    # Build system
    # FM1 region (sites 0-11)
    syst[(lat(x) for x in range(fm1_length))] = fm_potential
    
    # Barrier region (sites 12-17)
    syst[(lat(x) for x in range(fm1_length, fm1_length + barrier_length))] = barrier_potential
    
    # FM2 region (sites 18-23)
    syst[(lat(x) for x in range(fm1_length + barrier_length, 
                              fm1_length + barrier_length + fm2_length))] = fm_potential
    
    # Hopping
    syst[lat.neighbors()] = -t * np.eye(2)

    # Leads
    lead_left = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lead_left[lat(0)] = np.array([[E_up, 0], [0, E_down]])
    lead_left[lat.neighbors()] = -t * np.eye(2)
    
    lead_right = kwant.Builder(kwant.TranslationalSymmetry((1,)))
    lead_right[lat(0)] = rot @ np.array([[E_up, 0], [0, E_down]]) @ rot.T
    lead_right[lat.neighbors()] = -t * np.eye(2)

    syst.attach_lead(lead_left)
    syst.attach_lead(lead_right)

    return syst.finalized()

def calculate_stt(syst, energy=E_fermi):
    """Calculate spin transfer torque components"""
    # Get all wave functions
    wf = kwant.wave_function(syst, energy)
    
    # Check if we have any incoming modes
    if len(wf(0)) == 0:
        return 0.0, 0.0  # No propagating modes
    
    # Get first propagating mode from left lead
    psi = wf(0)[0]
    
    # Find the index of the last site in FM2
    site_index = -1
    site = syst.sites[site_index]
    
    # Get wavefunction components for this site (2 components per site)
    start_idx = 2 * (len(syst.sites) + site_index) if site_index < 0 else 2 * site_index
    psi_site = psi[start_idx:start_idx+2]
    
    # Verify we have wavefunction components
    if len(psi_site) != 2:
        return 0.0, 0.0
    
    # Reshape to column vector
    psi_col = psi_site.reshape((2, 1))
    psi_dag = psi_col.conj().T
    
    # Get the Hamiltonian at this site
    site_id = syst.id_by_site[site]
    h = syst.hamiltonian(site_id, site_id)
    
    # Estimate theta from the Hamiltonian
    theta = np.arctan2(h[0,1].real, h[0,0].real - h[1,1].real)
    m = np.array([sin(theta), 0, cos(theta)])  # magnetization vector
    
    # Calculate spin density S = ψ⁺ σ ψ
    Sx = (psi_dag @ sigma_x @ psi_col).item()
    Sy = (psi_dag @ sigma_y @ psi_col).item()
    Sz = (psi_dag @ sigma_z @ psi_col).item()
    S = np.array([Sx, Sy, Sz])
    
    # Spin transfer torque is T = m × S
    torque = np.cross(m, S)
    
    # Return torque components
    T_perp = torque[1].real  # y-component (in-plane)
    T_parallel = (torque[0] * cos(theta) - torque[2] * sin(theta)).real  # out-of-plane
    
    return T_perp, T_parallel

# ========== Main Execution ==========
if __name__ == '__main__':
    # Calculate STT vs angle
    angles = np.linspace(0, pi, 50)
    T_perp = []
    T_parallel = []
    transmissions = []
    
    for theta in angles:
        syst = make_mtj(theta)
        t_perp, t_parallel = calculate_stt(syst)
        T_perp.append(t_perp)
        T_parallel.append(t_parallel)
        transmissions.append(kwant.smatrix(syst, E_fermi).transmission(0, 1))
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(np.degrees(angles), transmissions, 'b-', linewidth=2)
    plt.xlabel('Magnetization Angle θ (degrees)')
    plt.ylabel('Transmission')
    plt.title('TMR Effect in MTJ')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(np.degrees(angles), T_perp, 'r-', linewidth=2)
    plt.xlabel('Magnetization Angle θ (degrees)')
    plt.ylabel('Torque (ħ/2e)')
    plt.title('In-plane (Slonczewski) Spin Transfer Torque')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(np.degrees(angles), T_parallel, 'g-', linewidth=2)
    plt.xlabel('Magnetization Angle θ (degrees)')
    plt.ylabel('Torque (ħ/2e)')
    plt.title('Out-of-plane (Field-like) Spin Transfer Torque')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(np.degrees(angles), T_perp, 'r-', label='In-plane')
    plt.plot(np.degrees(angles), T_parallel, 'g-', label='Out-of-plane')
    plt.xlabel('Magnetization Angle θ (degrees)')
    plt.ylabel('Torque (ħ/2e)')
    plt.title('Both Torque Components')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()