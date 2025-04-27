import numpy as np
import kwant
import matplotlib.pyplot as plt

# Constants from previous code
h0 = -7.51
hz = -1.54 / 2
t = 25.45

# Create the lattice (a 1D chain)
lat = kwant.lattice.chain()

# Onsite function for FM1 (fixed ↑ along z)
def fm1_onsite(site):
    return np.array([
        [h0 + hz, 0],
        [0, h0 - hz]
    ])

# Onsite function for FM2 (rotated magnetization angle θ in degrees)
def fm2_onsite(site, theta_deg):
    theta = np.radians(theta_deg)
    hz_matrix = np.array([
        [h0 + hz, 0],
        [0, h0 - hz]
    ])
    # Spin rotation around y-axis
    U = np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2),  np.cos(theta / 2)]
    ])
    return U @ hz_matrix @ U.T

# Onsite function for the non-magnetic barrier
def barrier_onsite(site):
    return np.eye(2) * (-5.0)

# Hopping matrix
hopping = -t * np.eye(2)

# Function to build MTJ with FM2 rotated by θ
def make_stt_mtj(L=25, barrier_range=(14, 20), fm2_angle=30):
    syst = kwant.Builder()
    
    for i in range(L):
        site = lat(i)

        if barrier_range[0] <= i < barrier_range[1]:
            syst[site] = barrier_onsite(site)
        elif i < barrier_range[0]:
            syst[site] = fm1_onsite(site)
        else:
            # Use lambda to defer evaluation with angle
            syst[site] = lambda site, angle=fm2_angle: fm2_onsite(site, angle)
        
        if i > 0:
            syst[lat(i), lat(i - 1)] = hopping
            
    # Define the leads
    lead = kwant.Builder(kwant.TranslationalSymmetry([-1]))
    lead[lat(0)] = fm1_onsite(lat(0))
    lead[lat(0), lat(1)] = hopping
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    return syst.finalized()

# Function to compute transmission at energy E
def compute_transmission(system, energy):
    smatrix = kwant.smatrix(system, energy)
    return smatrix.transmission(1, 0)

# Transmission calculation for different angles θ
angles = np.linspace(0, 180, 181)  # Angles from 0° to 180°
transmissions = []

for angle in angles:
    # Build MTJ with FM2 rotated by angle θ
    stt_mtj = make_stt_mtj(fm2_angle=angle)
    
    # Compute transmission at E = 0 for each angle
    T = compute_transmission(stt_mtj, energy=0.0)
    transmissions.append(T)
#making file 
with open("transmission_magnetization.dat", "w") as f:
    print(T)
# Plot transmission vs magnetization angle θ
plt.figure(figsize=(8, 6))
plt.plot(angles, transmissions, label='Transmission at E = 0')
plt.xlabel(r'$\theta$ (degrees)')
plt.ylabel('Transmission')
plt.title('Transmission vs Magnetization Angle $\theta$')
plt.grid(True)
plt.legend()
plt.show()

# Compute the numerical derivative of transmission with respect to angle θ
dT_dtheta = np.gradient(transmissions, angles)

# STT is proportional to the derivative of the transmission
STT = dT_dtheta  # You can add a proportionality constant if needed

# Plot STT as a function of magnetization angle θ
plt.figure(figsize=(9, 4))
plt.plot(angles, STT, label='Spin Transfer Torque (STT)', color='r')
plt.xlabel(r'$\theta$ (degrees)')
plt.ylabel('STT')
plt.title('Spin Transfer Torque vs Magnetization Angle $\theta$')
plt.grid(False)
plt.legend()
plt.show()
