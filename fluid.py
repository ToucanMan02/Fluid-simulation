#Note for reader: 
#This is a model for a simple 2D fluid simulation using Smoothed Particle Hydrodynamics (SPH). 
# The code simulates the behavior of fluid particles under the influence of forces such as pressure, viscosity, and external forces (in this case gravity). 
#A special thanks to MACHINE LEARNING AND SIMULATIONS on youtube for his tutorial on how to implement this code.
# link to video: https://www.youtube.com/watch?v=-0m05gzk8nk


import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from tqdm import tqdm




##### This section is for all the constants used later in the code #####
MAX_PARTICLES = 120
DOMAIN_WIDTH = 200
DOMAIN_HEIGHT = 50

PARTICLE_MASS = 1.0
ISOTROPIC_EXPONTENT = 20
BASE_DENSITY = 1
SMOOTHING_LENGTH = 5
DYNAMIC_VISCOSITY = 0.5
DAMPING_COEFFICIENT = 0.5
CONSTANT_FORCE = np.array([0.0, -1.0])

TIME_STEP_LENGTH = 0.001
N_TIME_STEPS = 2500
ADD_PARTICLE_EVERY = 50
FIGURE_SIZE = (6, 12)
PLOT_EVERY = 6
SCATTER_DOT_SIZE = 1800

DOMAIN_X_LIM = np.array([SMOOTHING_LENGTH, DOMAIN_WIDTH - SMOOTHING_LENGTH])
DOMAIN_Y_LIM = np.array([SMOOTHING_LENGTH, DOMAIN_HEIGHT - SMOOTHING_LENGTH])

NORMALIZATION_DENSITY = (315 * PARTICLE_MASS) / (64 * np.pi * SMOOTHING_LENGTH**9)
NORMALIZATION_PRESSURE_FORCE = -(45 * PARTICLE_MASS / (np.pi * SMOOTHING_LENGTH**6))
NORMALIZATION_VISCUS_FORCE = (
    45 * DYNAMIC_VISCOSITY * PARTICLE_MASS / (np.pi * SMOOTHING_LENGTH**6)
)
#######################################################################

##### the main function that runs the simulation #####

def main():
    ###### variables used later#####
    n_particles = 1

    positions = np.zeros((MAX_PARTICLES, 2), dtype=float)
    velocities = np.zeros((MAX_PARTICLES, 2), dtype=float)
    forces = np.zeros((MAX_PARTICLES, 2), dtype=float)
    #################################

    ##### important to note plt is used here as opposed to later so it doesnt create a new window every frame instead creates one window that changes the frames#####
    plt.figure(figsize=FIGURE_SIZE)
    ################################
    

    for iter in tqdm(range(N_TIME_STEPS)):
        if iter % ADD_PARTICLE_EVERY == 0 and n_particles + 3 <= MAX_PARTICLES:
            new_positions = np.array([ #this makes new particles at the top of the screen with some variation between them, change the constants to change how far apart you want them to be
                [10 + np.random.rand(), DOMAIN_Y_LIM[1]],
                [15 + np.random.rand(), DOMAIN_Y_LIM[1]],
                [20 + np.random.rand(), DOMAIN_Y_LIM[1]],
            ], dtype=float)

            new_velocities = np.array([ #now each new particle has the same inital downward velocity
                [-3.0, -15.0],
                [-3.0, -15.0],
                [-3.0, -15.0],
            ], dtype=float)

            #updatate the positions and velocities arrays with the new particles, add particles
            positions[n_particles:n_particles+3] = new_positions
            velocities[n_particles:n_particles+3] = new_velocities
            n_particles += 3

        pos = positions[:n_particles] #only getting data on "real" particles, not the empty ones in the array
        vel = velocities[:n_particles]

        neighbors_ids, distances = neighbors.KDTree(pos).query_radius( #this function returns the indices of the neighboring particles within the smoothing length and their distances from the current particle. 
            pos,
            SMOOTHING_LENGTH,
            return_distance=True,
            sort_results=True
        )

        densities = np.zeros(n_particles, dtype=float) #array that will store densities for each particle


        ##### using the distance found earlier to calculate density for each particle based on the equation from the video
        for i in range(n_particles): #double loop as each particle as the particles are in a list and the neighbors distance relative to that particle are also a list
            for j_in_list, j in enumerate(neighbors_ids[i]):
                densities[i] += NORMALIZATION_DENSITY * (
                    SMOOTHING_LENGTH**2 - distances[i][j_in_list]**2
                )**3
        #############################################


        pressures = ISOTROPIC_EXPONTENT * (densities - BASE_DENSITY) #following equation to find pressure

        f = np.zeros((n_particles, 2), dtype=float) #force vector for each particle

        neighbors_ids = [np.delete(x, 0) for x in neighbors_ids] #removing the first element of each list in neighbors_ids because the first element is the particle itself as that would lead to errors
        distances = [np.delete(x, 0) for x in distances] #same here just for distances

        for i in range(n_particles): #get each particle and neighbors
            for j_in_list, j in enumerate(neighbors_ids[i]): #for each neighbor follow the equations for force
                r = distances[i][j_in_list] #distance between the particle and its neighbor
                if r == 0: #dont calculate if r = 0 as that would lead to divisiion by zero
                    continue
                #pressure force
                f[i] += NORMALIZATION_PRESSURE_FORCE * (
                    -(pos[j] - pos[i]) / r
                    * (pressures[i] + pressures[j]) / (2 * densities[j])
                    * (SMOOTHING_LENGTH - r)**2
                )
                #viscosity force
                f[i] += NORMALIZATION_VISCUS_FORCE * (
                    (vel[j] - vel[i]) / densities[j]
                    * (SMOOTHING_LENGTH - r)
                )

        f += CONSTANT_FORCE * densities[:, None] #gravity times density to get fg

        safe_densities = np.maximum(densities, 1e-12) #avoid /0
        
        #####forward euler#####
        vel += TIME_STEP_LENGTH * (f / safe_densities[:, None]) 
        pos += TIME_STEP_LENGTH * vel
        #########################

        #make sure it stays in the box
        out_of_left_boundary = pos[:, 0] < DOMAIN_X_LIM[0]
        out_of_right_boundary = pos[:, 0] > DOMAIN_X_LIM[1]
        out_of_bottom_boundary = pos[:, 1] < DOMAIN_Y_LIM[0]
        out_of_top_boundary = pos[:, 1] > DOMAIN_Y_LIM[1]

        vel[out_of_left_boundary, 0] *= -DAMPING_COEFFICIENT
        vel[out_of_right_boundary, 0] *= -DAMPING_COEFFICIENT
        vel[out_of_bottom_boundary, 1] *= -DAMPING_COEFFICIENT
        vel[out_of_top_boundary, 1] *= -DAMPING_COEFFICIENT

        pos[out_of_left_boundary, 0] = DOMAIN_X_LIM[0]
        pos[out_of_right_boundary, 0] = DOMAIN_X_LIM[1]
        pos[out_of_bottom_boundary, 1] = DOMAIN_Y_LIM[0]
        pos[out_of_top_boundary, 1] = DOMAIN_Y_LIM[1]
        ############################################
       
       #update main guys#
        positions[:n_particles] = pos
        velocities[:n_particles] = vel
        forces[:n_particles] = f

        #plotting the particles every PLOT_EVERY frames, change this to plot more or less often
        if iter % PLOT_EVERY == 0:
            plt.scatter(
                positions[:n_particles, 0],
                positions[:n_particles, 1],
                s=SCATTER_DOT_SIZE
            )
            plt.xlim(0, DOMAIN_WIDTH)
            plt.ylim(0, DOMAIN_HEIGHT)
            plt.tight_layout()
            plt.pause(0.0001)
            plt.clf()
        ##################################

    
if __name__ == "__main__": #calling the main function to run the simulation
    main()
