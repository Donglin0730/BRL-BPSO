import copy
import numpy as np
import matplotlib.pyplot as plt
import random
import os


class MultiAgentEnvironment:
    def __init__(self, num_station=20, r=5, map_size=(20, 20, 20), num_special=5, special_n=2, length=2):
        self.num_station = num_station  # Number of base stations
        self.map_size = map_size  # Map size
        self.length = length
        max_height, max_weight, max_depth = self.map_size
        self.map = np.zeros((max_height, max_weight, max_depth))  # 3D space [x,y,z]
        self.r = r  # Base station coverage radius
        self.num_special = num_special  # Number of special points
        self.special_n = special_n
        self.special_points = self.special_points()
        self.special_sum = self.special_sum()
        self.agents_positions = self.init_agent_position()  # Coordinates [x,y,z]
        self.station = self.agents_positions[0]
        self.coverage = self.get_coverage()
        self.image_counter = 0
        self.ans = 0
        self.step_bool = False

    def count_coverage_in_special_areas(self):
        special_coverage_counts = []
        for i in range(self.num_special):
            start_point, end_point = self.special_points[i]
            x0, y0, z0 = start_point
            x1, y1, z1 = end_point
            max_height, max_width, max_depth = self.map_size
            # Initialize coverage count list for current special area
            coverage_count_list = []

            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    for z in range(z0, z1 + 1):
                        # Add current grid's coverage count to the list
                        if x < max_height and y < max_width and z < max_depth:
                            coverage_count_list.append(self.map[x][y][z])

            # Add current special area's coverage count list to result list
            special_coverage_counts.append(coverage_count_list)

        return special_coverage_counts

    def special_points(self):
        # special_points[[[x0,y0,y1][x1,y2,z2]] [x3,y3,z4][x5,y5,z5]] x0 is start x1 is end cube
        special_points = np.empty((0, 2, 3), dtype=int)
        for _ in range(self.num_special):
            # Randomly generate a vertex position for special point
            x = np.random.randint(0, self.map_size[0])
            y = np.random.randint(0, self.map_size[1])
            z = np.random.randint(0, self.map_size[2])

            # Calculate position range occupied by special point
            x_end = min(x + self.length - 1, self.map_size[0] - 1)
            y_end = min(y + self.length - 1, self.map_size[1] - 1)
            z_end = min(z + self.length - 1, self.map_size[2] - 1)

            # Add position range occupied by special point to special points list
            special_points = np.concatenate((special_points, np.array([[(x, y, z), (x_end, y_end, z_end)]])), axis=0)

        return special_points

    # Calculate number of unit cubes occupied by special points
    def special_sum(self):
        sum = 0
        for i in range(self.num_special):
            start_point, end_point = self.special_points[i]
            x0, y0, z0 = start_point
            x1, y1, z1 = end_point
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    for z in range(z0, z1 + 1):
                        sum += 1
        return sum

    def init_agent_position(self):  # agents_positions=[[x1,y1,z1] [x2 y2 z2]]
        # Initialize agent position array
        self.agents_positions = np.empty((0, 3), dtype=int)

        # Randomly generate agent positions
        for i in range(self.num_station):
            while True:
                # Randomly generate coordinates within map boundaries
                array1 = np.random.randint(0, self.map_size[0])
                array2 = np.random.randint(0, self.map_size[1])
                array3 = np.random.randint(0, self.map_size[2])

                # Add new position to agent position array
                new_position = np.array([[array1, array2, array3]])
                self.agents_positions = np.concatenate((self.agents_positions, new_position), axis=0)

                break

                # Return agent position array
        return self.agents_positions

    def get_coverage(self):
        max_width, max_height, max_depth = self.map_size
        count = np.count_nonzero(self.map)  # Count number of non-zero values
        coverage = count / (max_width * max_height * max_depth)
        return coverage  # Calculate coverage rate

    def reset_map(self):
        self.map = np.zeros(self.map_size)  # Reset map state
        return self.map

    # Check if special point is within base station coverage radius r
    def mark_map(self, positions):  # positions are base station coordinates
        self.reset_map()
        for station_index in range(self.num_station):
            x, y, z = positions[station_index]
            for i in range(max(0, x - self.r), min(self.map_size[0], x + self.r + 1)):
                for j in range(max(0, y - self.r), min(self.map_size[1], y + self.r + 1)):
                    for k in range(max(0, z - self.r), min(self.map_size[2], z + self.r + 1)):
                        if ((i - x) ** 2 + (j - y) ** 2 + (k - z) ** 2 <= self.r ** 2):
                            self.map[i][j][k] += 1

    def visualize_map(self, agent_positions):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Visualize agent positions
        for agent_pos in agent_positions:
            ax.scatter(agent_pos[0], agent_pos[1], agent_pos[2], color='b', marker='o')
        # Iterate through each special area
        for i in range(self.num_special):
            start_point, end_point = self.special_points[i]
            x0, y0, z0 = start_point
            x1, y1, z1 = end_point

            # Initialize coordinates and values for storing special area coverage
            x_vals = []
            y_vals = []
            z_vals = []
            coverage_vals = []

            # Iterate through each grid in special area
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    for z in range(z0, z1 + 1):
                        if x < self.map_size[0] and y < self.map_size[1] and z < self.map_size[2]:
                            x_vals.append(x)
                            y_vals.append(y)
                            z_vals.append(z)
                            coverage_vals.append(self.map[x][y][z])

            # Draw scatter plot for special area, set color to blue
            ax.scatter(x_vals, y_vals, z_vals, c='green', marker='o')

        # Visualize special points

        for special_point in self.special_points:
            start_coord = special_point[0]
            end_coord = special_point[1]

            # Calculate cube vertices
            vertices = [
                [start_coord[0], start_coord[1], start_coord[2]],
                [end_coord[0], start_coord[1], start_coord[2]],
                [end_coord[0], end_coord[1], start_coord[2]],
                [start_coord[0], end_coord[1], start_coord[2]],
                [start_coord[0], start_coord[1], end_coord[2]],
                [end_coord[0], start_coord[1], end_coord[2]],
                [end_coord[0], end_coord[1], end_coord[2]],
                [start_coord[0], end_coord[1], end_coord[2]]
            ]

            # Define cube edges
            edges = [
                [vertices[0], vertices[1], vertices[2], vertices[3], vertices[0]],
                [vertices[4], vertices[5], vertices[6], vertices[7], vertices[4]],
                [vertices[0], vertices[4]],
                [vertices[1], vertices[5]],
                [vertices[2], vertices[6]],
                [vertices[3], vertices[7]]
            ]

            # Draw cube edges
            for edge in edges:
                x_values = [vertex[0] for vertex in edge]
                y_values = [vertex[1] for vertex in edge]
                z_values = [vertex[2] for vertex in edge]
                ax.plot(x_values, y_values, z_values, color='r')

                # Visualize base station signal coverage range
        for station_pos in agent_positions:
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 50)
            x_cover = self.r * np.outer(np.cos(u), np.sin(v)) + station_pos[0]
            y_cover = self.r * np.outer(np.sin(u), np.sin(v)) + station_pos[1]
            z_cover = self.r * np.outer(np.ones(np.size(u)), np.cos(v)) + station_pos[2]
            ax.plot_surface(x_cover, y_cover, z_cover, color='#92A5D1', alpha=0.3)

        # Display coverage rate
        total_coverage = self.get_coverage()
        plt.title(f'Total Coverage: {total_coverage:.2%}')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([0, self.map_size[0]])
        ax.set_ylim([0, self.map_size[1]])
        ax.set_zlim([0, self.map_size[2]])
        os.makedirs('map_visualizations', exist_ok=True)
        file_name = f'map_{self.image_counter}.png'
        plt.savefig(os.path.join('map_visualizations', file_name))
        plt.close()

        self.image_counter += 1  # Increment counter

        plt.show()

    # Check if special point coverage requirements are met
    def check_special_points(self, old_position, agent_index, mark_special, mark_special_after):
        count1 = np.count_nonzero(mark_special == True)
        count2 = np.count_nonzero(mark_special_after == True)
        if count1 <= count2:
            return
        else:
            self.agents_positions[agent_index] = old_position
            self.mark_map(self.agents_positions)
            return

    def distance(self, position1, position2):
        # Calculate Euclidean distance between two points
        return np.sqrt((position1[0] - position2[0]) ** 2 + (position1[1] - position2[1]) ** 2 + (
                    position1[2] - position2[2]) ** 2)

    def distance_to_special_area(self, position, start_point, end_point):
        # Calculate distance to special area edge
        x0, y0, z0 = start_point
        x1, y1, z1 = end_point
        px, py, pz = position
        dx = max(0, x0 - px, px - x1)
        dy = max(0, y0 - py, py - y1)
        dz = max(0, z0 - pz, pz - z1)
        distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return distance

    def index_special(self, x, y, z, i):
        x_start, y_start, z_start = self.special_points[i][0]
        # Map 3D coordinates to 1D index
        index = (x - x_start) + (y - y_start) * self.length + (z - z_start) * self.length ** 2 + i * self.length ** 3
        return index

    # Remap 1D index back to 3D coordinates
    def inverse_index_special(self, index):
        # Calculate fourth dimension coordinate i (1D index)
        length_cube = self.special_sum
        i = index // length_cube
        remainder = index % length_cube
        length_square = self.length * self.length
        z = remainder // length_square
        remainder = remainder % length_square
        y = remainder // self.length
        x = remainder % self.length
        x_start, y_start, z_start = self.special_points[i][0]
        return x + x_start, y + y_start, z + z_start

    def step(self, i, action):
        self.ans += 1
        # Record coverage before moving
        coverage_before = self.get_coverage()
        old_position = self.agents_positions[i]
        old_position_copy = copy.deepcopy(old_position)  # Make agent return to original position

        mark_special = np.zeros(self.length ** 3 * self.num_special)

        for j in range(self.num_special):
            start_point, end_point = self.special_points[j]
            x0, y0, z0 = start_point
            x1, y1, z1 = end_point

            # Iterate through each grid in special area
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    for z in range(z0, z1 + 1):
                        if self.map[x][y][z] >= self.special_n:
                            index = self.index_special(x, y, z, j)
                            mark_special[index] = True

        # Find nearest special point
        nearest_distance = float('inf')
        nearest_special_index = -1
        for j in range(self.num_special):
            start_point, end_point = self.special_points[j]
            x0, y0, z0 = start_point
            x1, y1, z1 = end_point
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    for z in range(z0, z1 + 1):
                        if self.map[x][y][z] >= self.special_n:
                            dist = np.linalg.norm(np.array(old_position) - np.array([x, y, z]))
                            if dist < nearest_distance:
                                nearest_distance = dist
                                nearest_special_index = j

        # Reward for moving towards nearest special point
        reward_distance = 1.0 / (nearest_distance + 1)  # Closer distance, higher reward

        reward_special = 0
        special_before = 0
        special = self.count_coverage_in_special_areas()
        for m in special:
            special_before += sum(m)

        # Update position based on action
        new_position = self.update_position(old_position, action)
        self.agents_positions[i] = new_position
        self.mark_map(self.agents_positions)

        mark_special_after = np.zeros(self.length ** 3 * self.num_special)

        # Calculate number of satisfied coverage requirements after base station movement
        sum_after = 0
        for j in range(self.num_special):
            start_point, end_point = self.special_points[j]
            x0, y0, z0 = start_point
            x1, y1, z1 = end_point

            # Iterate through each grid in special area
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    for z in range(z0, z1 + 1):
                        if self.map[x][y][z] >= self.special_n:
                            index = self.index_special(x, y, z, j)
                            mark_special_after[index] = True
                            sum_after += 1

        # Calculate number of satisfied coverage requirements before base station movement
        sum_before = 0
        for index in range(mark_special.size):
            if mark_special[index] == True:
                sum_before = sum_before + 1
                x, y, z = self.inverse_index_special(index)
                self.check_special_points(old_position_copy, i, mark_special, mark_special_after)

        # Calculate coverage after moving
        coverage_after = self.get_coverage()

        # Initialize reward and nearest special area distance
        reward = 0

        reward_special = sum_after - sum_before
        # Update state, calculate additional rewards
        state = self.agents_positions
        reward_coverage = coverage_after - coverage_before

        total_iterations = 1000  # Total number of iterations

        # Calculate iteration ratio
        iteration_ratio = self.ans / total_iterations
        for j in range(self.num_special):
            start_point, end_point = self.special_points[j]
            x0, y0, z0 = start_point
            x1, y1, z1 = end_point

            # Iterate through each grid in special area
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    for z in range(z0, z1 + 1):
                        if self.map[x][y][z] >= self.special_n:
                            index = self.index_special(x, y, z, j)
                            mark_special[index] = True
        count = np.count_nonzero(mark_special == True)

        # step_bool determines if special area is fully covered. When True, special area is considered to meet coverage requirements and should not provide additional rewards.
        if (self.step_bool == False and count == self.special_sum):
            self.step_bool = True

        if (self.step_bool == False):
            reward = iteration_ratio * reward_coverage + (1 - iteration_ratio) * (reward_special + reward_distance)
        else:
            reward = reward_coverage
        # Normalized reward function

        return state, reward, False, coverage_after, self.step_bool, count, mark_special.size

    def calculate_reward(self):
        # Calculate coverage rate of all base stations
        reward = self.get_coverage()
        return reward

    def update_position(self, position, action):  # newposition=[x,y,z]
        # Update position based on action
        new_position = position.copy()
        if action == 0:  # Move up
            new_position[2] += 1
        elif action == 1:  # Move down
            new_position[2] -= 1
        elif action == 2:  # Move left
            new_position[1] -= 1
        elif action == 3:  # Move right
            new_position[1] += 1
        elif action == 4:  # Move forward
            new_position[0] += 1
        elif action == 5:  # Move backward
            new_position[0] -= 1

        # Ensure position is within map boundaries
        new_position[0] = np.clip(new_position[0], 0, self.map_size[0] - 1)
        new_position[1] = np.clip(new_position[1], 0, self.map_size[1] - 1)
        new_position[2] = np.clip(new_position[2], 0, self.map_size[2] - 1)
        return new_position


def discretize_position(position, map_size, num_regions_height, num_regions_width, num_regions_depth):
    """
    Discretize position into region index
    """
    max_height, max_width, max_depth = map_size
    height_per_region = max_height // num_regions_height
    width_per_region = max_width // num_regions_width
    depth_per_region = max_depth // num_regions_depth

    # Calculate index adjusted according to new order
    col = min(position[0] // width_per_region, num_regions_width - 1)  # Corresponds to x
    row = min(position[1] // height_per_region, num_regions_height - 1)  # Corresponds to y
    depth = min(position[2] // depth_per_region, num_regions_depth - 1)  # Corresponds to z

    index = row * num_regions_width * num_regions_depth + col * num_regions_depth + depth
    return index


def index(position, map_size):
    x, y, z = position
    map_width, map_height, map_depth = map_size

    # Map 3D coordinates to 1D index
    index = x + y * map_width + z * map_width * map_height
    return index


# Particle Swarm Optimization parameters
DIMENSIONS = 3  # Spatial dimensions of each particle
MAX_VELOCITY = 5  # Maximum velocity of particles
W = 0.9  # Inertia weight
C1 = 1.5  # Cognitive coefficient
C2 = 1.5  # Social coefficient


# Initialize Q-tables and PSO particles
def initialize_system(env):
    num_states = np.prod(env.map_size)
    num_particles = env.num_station
    num_actions = 6  # Assume 6 actions
    Q_tables = [np.zeros((num_states, num_actions)) for _ in
                range(num_particles)]  # Q_tables[i] represents Q-table of i-th agent
    particles_positions = env.init_agent_position()  # Get initial particle positions
    particles_velocities = np.zeros((num_particles, 3))  # 3-dimensional velocity
    best_personal_positions = particles_positions.copy()  # Initial positions are considered best positions
    best_global_coverage = env.calculate_reward()  # Calculate initial total coverage as best coverage
    best_global_position = random.choice(particles_positions)
    return Q_tables, particles_positions, particles_velocities, best_personal_positions, best_global_position, best_global_coverage


import numpy as np

# Particle Swarm Optimization parameters
MAX_VELOCITY = 5  # Maximum velocity of particles
W = 0.9  # Inertia weight
C1 = 1.5  # Cognitive coefficient
C2 = 1.5  # Social coefficient


def update_PSO_particles(env, particles_positions, particles_velocities, best_positions, best_global_position, w=W,
                         c1=C1, c2=C2):
    N = particles_positions.shape[0]  # Number of particles
    D = particles_positions.shape[1]  # Dimensions of each particle

    # Ensure particle positions and velocities are integers
    particles_positions = np.array(particles_positions, dtype=int)
    particles_velocities = np.array(particles_velocities, dtype=int)

    # Generate random cognitive and social acceleration coefficients for each particle
    r1 = np.random.rand(N, D)
    r2 = np.random.rand(N, D)

    # Calculate new velocities
    new_velocities = w * particles_velocities + \
                     c1 * r1 * (best_positions.astype(int) - particles_positions) + \
                     c2 * r2 * (best_global_position.astype(int) - particles_positions)

    # Convert velocities to integers and limit velocity range
    particles_velocities = np.clip(new_velocities.astype(int), -MAX_VELOCITY, MAX_VELOCITY)

    # Update particle positions
    new_positions = particles_positions + particles_velocities

    # Ensure new positions are within map boundaries
    map_size = np.array(env.map_size)
    new_positions = np.clip(new_positions, 0, map_size - 1)

    # Check special area coverage
    for i in range(N):
        new_pos = new_positions[i]
        old_pos = particles_positions[i]
        # Check if special area coverage meets requirements
        if not is_special_area_satisfied(env, old_pos, new_pos):
            # If not satisfied, keep old position
            new_positions[i] = old_pos

    return new_positions


def is_special_area_satisfied(env, old_pos, new_pos):
    """
    Check if special area coverage meets requirements.
    """
    special_points = env.special_points
    special_n = env.special_n  # Minimum coverage times required for special area

    for special_area in special_points:
        start_point, end_point = special_area
        x0, y0, z0 = start_point
        x1, y1, z1 = end_point

        # Check if old position satisfies special area coverage requirements
        if is_position_in_special_area(old_pos, start_point, end_point) and \
                not is_position_in_special_area(new_pos, start_point, end_point):
            return False

    return True


def is_position_in_special_area(pos, start_point, end_point):
    """
    Check if position is within special area.
    """
    x0, y0, z0 = start_point
    x1, y1, z1 = end_point
    px, py, pz = pos
    return x0 <= px < x1 and y0 <= py < y1 and z0 <= pz < z1


def update_num_regions(current_num_regions, episode, num_episodes, env_map_size):
    # Gradually increase num_regions until it equals env_map_size in the final iteration.
    # Calculate the increment for num_regions per iteration
    increment = (env_map_size[0] - current_num_regions[0]) / num_episodes

    # Update num_regions, ensuring each dimension gradually increases
    new_num_regions = tuple(
        max(1, int(current_dim + increment * (episode + 1))) for current_dim in current_num_regions
    )

    # Ensure num_regions does not exceed env_map_size in the final iteration
    new_num_regions = tuple(
        min(env_map_size[dim], region) for dim, region in enumerate(new_num_regions)
    )

    return new_num_regions


# Main loop
# Use update_num_regions function in main_loop
def main_loop(env, num_episodes=1000, epsilon=0.2, learning_rate=0.5, discount_factor=0.95):
    Q_tables, particles_positions, particles_velocities, best_personal_positions, best_global_position, best_global_coverage = initialize_system(
        env)
    num_regions = (2, 2, 2)  # Initial region division, starting from (2, 2, 2)
    reset_bool = False  # Flag for outputting special area information
    count_max = 0

    for episode in range(num_episodes):
        num_regions = update_num_regions(num_regions, episode, num_episodes, env.map_size)  # Update region division

        # Ensure num_regions equals env_map_size in the final iteration
        if episode == num_episodes - 1:
            num_regions = env.map_size

        env.reset_map()  # Reset environment
        total_reward = np.zeros(env.num_station)

        for station_index in range(env.num_station):
            current_position = particles_positions[station_index]
            state_index = discretize_position(current_position, env.map_size, num_regions[0], num_regions[1],
                                              num_regions[2])  # Discretize position into region index

            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 5)  # Randomly select from 6 actions
            else:
                action = np.argmax(Q_tables[station_index][state_index])  # Select action with highest Q-value

            next_positions, reward, done, coverage, bool, count, size = env.step(station_index,
                                                                                 action)  # Execute action and get feedback
            next_position = next_positions[station_index]

            # Special area coverage Q-table reset
            if (bool == True and reset_bool == False):
                reset_bool = True
                print("Special points covered")
                # Reset environment
                Q_tables, particles_positions, particles_velocities, best_personal_positions, best_global_position, best_global_coverage = initialize_system(
                    env)

            total_reward[station_index] += reward

            next_state_index = discretize_position(next_position, env.map_size, num_regions[0], num_regions[1],
                                                   num_regions[2])

            # Update Q-value
            td_target = reward + discount_factor * np.max(Q_tables[station_index][next_state_index])
            Q_tables[station_index][state_index][action] += learning_rate * (
                        td_target - Q_tables[station_index][state_index][action])

            if count > count_max:
                count_max = count
            # Update PSO individual best position and total coverage
            if coverage > best_global_coverage:
                best_personal_positions[station_index] = next_positions[station_index]
                best_global_coverage = coverage
        if reset_bool == False:
            print("Covered points {}/{}".format(count_max, size))
        env.visualize_map(env.agents_positions)

        # Update PSO particles every 10 episodes
        if episode % 10 == 0:
            max_index = np.argmax(total_reward)
            best_global_position = particles_positions[max_index]
            particles_positions = update_PSO_particles(env, particles_positions, particles_velocities,
                                                       best_personal_positions, best_global_position)

        print("Episode {}: Total Coverage : {:.5f}%".format(episode + 1, 100 * coverage))
    return best_personal_positions, best_global_coverage


# Create environment and run main loop
env = MultiAgentEnvironment()

print("Initial base station positions:", env.agents_positions)
print("Special areas:", env.special_points)

env.visualize_map(env.agents_positions)

best_station_positions, best_coverage = main_loop(env)

print("Best coverage:", best_coverage)
print("Best station positions:", best_station_positions)
special_coverage_counts = env.count_coverage_in_special_areas()
env.visualize_map(best_station_positions)