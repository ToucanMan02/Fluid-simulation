import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from tqdm import tqdm

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

def main():
    n_particles = 1

    positions = np.zeros((MAX_PARTICLES, 2), dtype=float)
    velocities = np.zeros((MAX_PARTICLES, 2), dtype=float)
    forces = np.zeros((MAX_PARTICLES, 2), dtype=float)

    plt.figure(figsize=FIGURE_SIZE)

    for iter in tqdm(range(N_TIME_STEPS)):
        if iter % ADD_PARTICLE_EVERY == 0 and n_particles + 3 <= MAX_PARTICLES:
            new_positions = np.array([
                [10 + np.random.rand(), DOMAIN_Y_LIM[1]],
                [15 + np.random.rand(), DOMAIN_Y_LIM[1]],
                [20 + np.random.rand(), DOMAIN_Y_LIM[1]],
            ], dtype=float)

            new_velocities = np.array([
                [-3.0, -15.0],
                [-3.0, -15.0],
                [-3.0, -15.0],
            ], dtype=float)

            positions[n_particles:n_particles+3] = new_positions
            velocities[n_particles:n_particles+3] = new_velocities
            n_particles += 3

        pos = positions[:n_particles]
        vel = velocities[:n_particles]

        neighbors_ids, distances = neighbors.KDTree(pos).query_radius(
            pos,
            SMOOTHING_LENGTH,
            return_distance=True,
            sort_results=True
        )

        densities = np.zeros(n_particles, dtype=float)

        for i in range(n_particles):
            for j_in_list, j in enumerate(neighbors_ids[i]):
                densities[i] += NORMALIZATION_DENSITY * (
                    SMOOTHING_LENGTH**2 - distances[i][j_in_list]**2
                )**3

        pressures = ISOTROPIC_EXPONTENT * (densities - BASE_DENSITY)

        f = np.zeros((n_particles, 2), dtype=float)

        neighbors_ids = [np.delete(x, 0) for x in neighbors_ids]
        distances = [np.delete(x, 0) for x in distances]

        for i in range(n_particles):
            for j_in_list, j in enumerate(neighbors_ids[i]):
                r = distances[i][j_in_list]
                if r == 0:
                    continue

                f[i] += NORMALIZATION_PRESSURE_FORCE * (
                    -(pos[j] - pos[i]) / r
                    * (pressures[i] + pressures[j]) / (2 * densities[j])
                    * (SMOOTHING_LENGTH - r)**2
                )

                f[i] += NORMALIZATION_VISCUS_FORCE * (
                    (vel[j] - vel[i]) / densities[j]
                    * (SMOOTHING_LENGTH - r)
                )

        f += CONSTANT_FORCE * densities[:, None]

        safe_densities = np.maximum(densities, 1e-12)
        vel += TIME_STEP_LENGTH * (f / safe_densities[:, None])
        pos += TIME_STEP_LENGTH * vel

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

        positions[:n_particles] = pos
        velocities[:n_particles] = vel
        forces[:n_particles] = f

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

if __name__ == "__main__":
    main()