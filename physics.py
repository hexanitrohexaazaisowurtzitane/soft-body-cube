import numpy as np

class Engine:
    def __init__(self):
        # params
        self.dt = 1.0 / 60.0 # 60 fps
        self.num_substeps = 4
        self.substep_dt = self.dt / self.num_substeps
        self.gravity = np.array([0, -9.81, 0])

        self.volume_forces = {}
        self.volume_compliance = 1e-4
        self.compliance = 1e-3

    def initialize_particles(self, positions, masses):
        self.num_particles = len(positions)

        # pos vector ( 3 * num_particles )
        self.positions = np.array(positions).flatten()
        self.prev_positions = self.positions.copy()

        self.velocities = np.zeros(3 * self.num_particles)

        # mass matrix
        self.masses = np.array(masses)
        self.inv_masses = 1.0/self.masses 

    def initialize_constraints(self, constraints, positions):
        self.constraints   = constraints
        self.rest_lengths  = []
        self.lambda_values = []

        # def rest lengths
        for i, j in constraints:
            pos_i = self.get_particle_position(i)
            pos_j = self.get_particle_position(j)
            rest_length = np.linalg.norm(pos_i - pos_j)
            self.rest_lengths.append(rest_length)
            self.lambda_values.append(0.0) # init multipliers

        # convert to arrays
        self.rest_lengths  = np.array(self.rest_lengths)
        self.lambda_values = np.array(self.lambda_values)

    def initialize_tetrahedra(self, tetrahedra):
        self.tetrahedra = tetrahedra
        self.num_tetrahedra = len(tetrahedra)

        # rest volumes
        self.rest_volumes = []
        for tet in tetrahedra:
            volume = self.get_tetrahedron_volume(tet)
            self.rest_volumes.append(volume)

        self.rest_volumes = np.array(self.rest_volumes)

        # volume gradient ordering
        self.vol_id_order = [
            # faces opposite to vertex x
            [1, 3, 2],  # v0
            [0, 2, 3],  # v1
            [0, 3, 1],  # v2
            [0, 1, 2]   # v3
        ]

    def get_tetrahedron_volume(self, tet):
        # get pos of 4 verts
        p0 = self.get_particle_position(tet[0])
        p1 = self.get_particle_position(tet[1])
        p2 = self.get_particle_position(tet[2])
        p3 = self.get_particle_position(tet[3])

        # edge vectors from p0 to px
        edge1 = p1 - p0
        edge2 = p2 - p0
        edge3 = p3 - p0

        # vol = (edge1*edge2) * edge3 / 6
        cross  = np.cross(edge1, edge2)
        volume = np.dot(cross, edge3) / 6.0

        return volume

    def get_particle_position(self, index):
        start_idx = index * 3   # 3d pos
        return self.positions[start_idx:start_idx+3]

    def set_particle_position(self, index, position):
        start_idx = index * 3
        self.positions[start_idx:start_idx+3] = position

    
    def apply_external_forces(self):

        for i in range(self.num_particles):
            # F = ma -> a = F/m = g, since F = mg
            start_idx = i*3
            self.velocities[start_idx+1] += self.gravity[1] * self.substep_dt

    def predict_positions(self):
        self.prev_positions = self.positions.copy()
        self.positions += self.velocities * self.substep_dt

    def solve_constraints(self):
        # constraints are solved according to the pfd in the desc!
        alpha_tidle = self.compliance / (self.substep_dt ** 2)

        for constraint_idx, (i, j) in enumerate(self.constraints):
            pos_i = self.get_particle_position(i)
            pos_j = self.get_particle_position(j)

            diff = pos_i - pos_j
            current_length = np.linalg.norm(diff)

            # avoid div by 0
            if current_length < 1e-6: continue

            constraint_value = current_length - self.rest_lengths[constraint_idx]

            n = diff / current_length # constraint gradient

            # generalize inverse masses
            w_i = self.inv_masses[i]
            w_j = self.inv_masses[j]
            w_sum = w_i + w_j

            if w_sum < 1e-6: continue # both are fixed


            # update for lagrange multiplier
            delta_lambda = ( -constraint_value - alpha_tidle * self.lambda_values[constraint_idx]) / (w_sum + alpha_tidle)
            self.lambda_values[constraint_idx] += delta_lambda

            # pos corrections
            correction_i =  (delta_lambda * w_i) * n
            correction_j = -(delta_lambda * w_j) * n

            self.set_particle_position(i, pos_i + correction_i)
            self.set_particle_position(j, pos_j + correction_j)

    def solve_grab_constraints(self, constraints, point_3d):
        if not constraints: return

        grab_compliance = 1e-6
        alpha_tilde = grab_compliance / (self.substep_dt ** 2)
        for particle_idx, rest_distance in constraints:
            particle_pos = self.get_particle_position(particle_idx)
            diff = particle_pos - point_3d
            cur_distance = np.linalg.norm(diff)

            if cur_distance < 1e-8: continue

            constraint_value = cur_distance - rest_distance
            n = diff / cur_distance

            w_particle = self.inv_masses[particle_idx]
            if w_particle < 1e-8: continue

            delta_lambda = -constraint_value / (w_particle + alpha_tilde)
            correction = (delta_lambda * w_particle) * n 
            self.set_particle_position(particle_idx, particle_pos + correction)

    def solve_volume_constraints(self):
        self.volume_forces = {}

        alpha = self.volume_compliance / (self.substep_dt ** 2)

        for tet_idx, tet_vert in enumerate(self.tetrahedra):
            gradients = []
            w_sum = 0.0

            for vertex_idx in range(4):
                # 3 vert of the face opposite to current vertex
                opp_face_idx = self.vol_id_order[vertex_idx]

                face_v0 = self.get_particle_position(tet_vert[opp_face_idx[0]])
                face_v1 = self.get_particle_position(tet_vert[opp_face_idx[1]])
                face_v2 = self.get_particle_position(tet_vert[opp_face_idx[2]])

                edge1 = face_v1 - face_v0
                edge2 = face_v2 - face_v0
                face_normal = np.cross(edge1, edge2) / 6.0

                gradients.append(face_normal)

                particle_idx = tet_vert[vertex_idx]
                w_sum += self.inv_masses[particle_idx] * np.dot(face_normal, face_normal)

            if w_sum < 1e-8: continue

            current_volume = self.get_tetrahedron_volume(tet_vert)
            rest_volume    = self.rest_volumes[tet_idx]
            constraint_violation = current_volume - rest_volume

            correction_magnitude = -constraint_violation / (w_sum + alpha)

            # apply corrections to restore volume
            for vertex_idx in range(4):
                particle_idx = tet_vert[vertex_idx]
                gradient = gradients[vertex_idx]

                correction  = correction_magnitude * self.inv_masses[particle_idx] * gradient
                current_pos = self.get_particle_position(particle_idx)
                self.set_particle_position(particle_idx, current_pos + correction)

                volume_force = correction * 50.0
                if particle_idx in self.volume_forces:
                     self.volume_forces[particle_idx] += volume_force
                else: self.volume_forces[particle_idx] = volume_force

    def update_velocities(self):
        self.velocities = (self.positions - self.prev_positions) / self.substep_dt
    
    def add_damping(self, damping=0.99):
        self.velocities *= damping

    def handle_ground_collision(self, ground_y=0.0):
        for i in range(self.num_particles):
            start_idx = i * 3
            if self.positions[start_idx + 1] < ground_y:
                self.positions[start_idx + 1]   = ground_y
                self.velocities[start_idx + 1] *= -0.5 # bounce

    
    def physics_step(self, grab_constraints=None, grab_point=None):

        self.lambda_values.fill(0.0) #reset lagrange

        for substep in range(self.num_substeps):
            self.apply_external_forces()
            self.predict_positions()

            for i in range(1):
                self.solve_constraints()
                if grab_constraints and grab_point is not None:
                    self.solve_grab_constraints(grab_constraints, grab_point)
                self.solve_volume_constraints()
            
            self.handle_ground_collision()
            self.update_velocities()
            self.add_damping(0.998)
