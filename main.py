import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import math

from physics import Engine
from body    import Cube

class SoftPhysics:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width  = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)


        # gl steup
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, width/height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

        self.physics = Engine()

        # camera
        self.cam_pos = np.array([0.0, 0.0, 0.0])
        self.camera_distance = 10.0
        self.camera_angle_x = 0
        self.camera_angle_y = 0

        # mous grab
        self.is_grabbing = False 
        self.grab_constraints  = [] # list (particle_idx, rest_distance) for grabbed particles
        self.grab_point_3d     = np.array([0, 0, 0]) # 3d pos of the mouse
        self.grab_plane_point  = np.array([0, 0, 0]) # intresection with plane
        self.grab_plane_normal = np.array([0, 0, 1])
        

        self.create_cube()

    def create_cube(self):
        cube_config = Cube()

        tetrahedra  = cube_config.get_tetrahedra()
        constraints = cube_config.get_constraints()
        vertices    = cube_config.get_vertices()
        masses      = cube_config.get_masses()

        self.cube_faces = cube_config.get_faces()

        self.physics.initialize_particles(vertices, masses)
        self.physics.initialize_constraints(constraints, vertices)
        self.physics.initialize_tetrahedra(tetrahedra)

        self.num_particles = self.physics.num_particles
        self.constraints   = self.physics.constraints
        self.tetrahedra    = self.physics.tetrahedra

    def get_particle_position(self, particle_idx):
        return self.physics.get_particle_position(particle_idx)
    
    def set_particle_position(self, particle_idx, position):
        self.physics.set_particle_position(particle_idx, position)

    def screen_world_ray(self, screen_x, screen_y):
        model_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
        proj_matrix = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport    = glGetIntegerv(GL_VIEWPORT) 

        gl_y = viewport[3] - screen_y
        
        near_point = gluUnProject(screen_x, gl_y, 0.0, model_matrix, proj_matrix, viewport)
        far_point  = gluUnProject(screen_x, gl_y, 1.0, model_matrix, proj_matrix, viewport)

        # ray
        ray_origin = np.array(near_point)
        ray_dir = np.array(far_point) - np.array(near_point)
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        return ray_origin, ray_dir

    def ray_triangle_intresection(self, ray_origin, ray_dir, v0, v1, v2):
        """
        Ray triangle intresection using Moller algo
        Returns (hit, t, u, v):
        * hit  : is_instresection
        * t    : distance along ray to intresection
        * u, v : barycentric coords
        """

        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(ray_dir, edge2)
        a = np.dot(edge1, h)

        if abs(a) < 1e-8: return False, 0, 0, 0 # ray || triangle

        f = 1.0 / a
        s = ray_origin - v0
        u = f * np.dot(s, h)

        if u < 0.0 or u > 1.0: return False, 0, 0, 0

        q = np.cross(s, edge1)
        v = f * np.dot(ray_dir, q)

        if v < 0.0 or u + v > 1.0: return False, 0, 0, 0

        t = f * np.dot(edge2, q)

        if t > 1e-8: return True, t, u, v
        else: return False, 0, 0, 0

    def find_triangle_hit(self, ray_origin, ray_dir):
        closest_hit = None
        closest_t = float('inf')
        closest_triangle = None 

        for triangle in self.cube_faces:
            v0 = self.get_particle_position(triangle[0])
            v1 = self.get_particle_position(triangle[1])
            v2 = self.get_particle_position(triangle[2])

            hit, t, u, v = self.ray_triangle_intresection(ray_origin, ray_dir, v0, v1, v2)

            if hit and t < closest_t:
                closest_t = t
                hit_point = ray_origin + t * ray_dir
                closest_hit = hit_point
                closest_triangle = triangle

        return closest_hit, closest_triangle

    def ray_plane_intresection(self, ray_origin, ray_dir, point, normal):
        denom = np.dot(normal, ray_dir)
        if abs(denom) < 1e-8: return None

        t = np.dot(normal, point - ray_origin) / denom
        if t >= 0 : return ray_origin + t * ray_dir
        else: return None

    def start_grab(self, screen_x, screen_y):
        ray_origin, ray_dir = self.screen_world_ray(screen_x, screen_y)
        hit_point, hit_triangle = self.find_triangle_hit(ray_origin, ray_dir)

        if hit_point is not None:
            print(f"Hit triangle: {hit_triangle} at {hit_point}")

            self.is_grabbing   = True
            self.grab_point_3d = hit_point.copy()

            self.grab_screen_x = screen_x
            self.grab_screen_y = screen_y

            cam_dir = ray_dir
            self.grab_plane_normal = -cam_dir
            self.grab_plane_point  = hit_point.copy()

            self.grab_init_cam_pos = self.cam_pos.copy()

            self.grab_constraints = []
            for vertex_idx in hit_triangle:
                vertex_pos = self.get_particle_position(vertex_idx)
                d = np.linalg.norm(vertex_pos - hit_point)
                self.grab_constraints.append((vertex_idx, d))
            
            print(f"Generated {len(self.grab_constraints)} constraints")

    def update_grab(self, screen_x, screen_y):
        if not self.is_grabbing: return

        ray_origin, ray_dir = self.screen_world_ray(screen_x, screen_y)
        
        camera_grab = self.grab_point_3d - ray_origin
        grab_dist   = np.dot(camera_grab, ray_dir)

        self.grab_point_3d = ray_origin +ray_dir * grab_dist

    def stop_grab(self):
        self.is_grabbing = False
        self.grab_constraints = []

        if hasattr(self, 'grab_screen_x'): delattr(self, 'grab_screen_x')
        if hasattr(self, 'grab_screen_y'): delattr(self, 'grab_screen_y')
        if hasattr(self, 'grab_init_cam_pos'): delattr(self, 'grab_init_cam_pos')


    def get_cam_forward(self):
        pitch = math.radians(self.camera_angle_x)
        yaw   = math.radians(self.camera_angle_y)

        forward = np.array([
            math.sin(yaw) * math.cos(pitch),
            - math.sin(pitch),
            - math.cos(yaw) * math.cos(pitch)
        ])
        return forward

    def get_cam_right(self):
        forward = self.get_cam_forward()
        world_up = np.array([0, 1, 0])
        right = np.cross(forward, world_up)

        right_length = np.linalg.norm(right)
        if right_length > 1e-6: right = right / right_length
        return right

    def get_cam_up(self):
        forward = self.get_cam_forward()
        right   = self.get_cam_right()
        up = np.cross(right, forward)
        return up

    def cam_movement(self, speed=0.1):
        keys = pygame.key.get_pressed()
        prev_pos = self.cam_pos.copy()

        forward = self.get_cam_forward()
        right   = self.get_cam_right()
        world_up = np.array([0, 1, 0])

        if keys[pygame.K_w]: self.cam_pos += forward * speed
        if keys[pygame.K_s]: self.cam_pos -= forward * speed
        if keys[pygame.K_a]: self.cam_pos -= right   * speed
        if keys[pygame.K_d]: self.cam_pos += right   * speed

        if keys[pygame.K_e]: self.cam_pos += world_up * speed
        if keys[pygame.K_q]: self.cam_pos -= world_up * speed

        cam_moved = not np.allclose(prev_pos, self.cam_pos, atol=1e-6)
        if cam_moved and self.is_grabbing:
            # make the grab point move with camera
            cur_mouse_x, cur_mouse_y = pygame.mouse.get_pos()
            self.update_grab(cur_mouse_x, cur_mouse_y)

    def handle_input(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                # reset sim
                self.stop_grab()
                self.create_cube()
            elif event.key == pygame.K_SPACE:
                # apply upwards impulse
                for i in range(self.num_particles):
                    self.physics.velocities[i * 3 + 1] += 5.0

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 3: # rm
                mouse_x, mouse_y = pygame.mouse.get_pos()
                self.start_grab(mouse_x, mouse_y)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 3: self.stop_grab()

        elif event.type == pygame.MOUSEMOTION:
            if self.is_grabbing:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                self.update_grab(mouse_x, mouse_y)
            elif pygame.mouse.get_pressed()[0]:
                # cam rotation
                dx, dy = event.rel
                self.camera_angle_y += dx * 0.5
                self.camera_angle_x += dy * 0.5
                self.camera_angle_x = max(-90, min(90, self.camera_angle_x))

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # cam
        #glTranslatef(0, 0, -self.camera_distance)
        glRotatef(self.camera_angle_x, 1, 0, 0)
        glRotatef(self.camera_angle_y, 0, 1, 0)

        glTranslatef(
            -self.cam_pos[0],
            -self.cam_pos[1],
            -self.camera_distance - self.cam_pos[2]
        )

        self.draw_ground()

        self.draw_faces()

        self.draw_constraints()

        self.draw_force_vectors()

        if self.is_grabbing: self.draw_grab_lines()

        pygame.display.flip()

    def draw_ground(self):
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_QUADS)
        
        glVertex3f(-15, 0, -15)
        glVertex3f( 15, 0, -15)
        glVertex3f( 15, 0,  15)
        glVertex3f(-15, 0,  15)
        glEnd()

    def draw_force_vectors(self):
        # gravity forces
        glColor3f(0.0, 0.0, 1.0)
        for i in range(self.num_particles):
            pos = self.get_particle_position(i)
            g_force = self.physics.gravity * 0.2

            glBegin(GL_LINES)
            glVertex3f(pos[0], pos[1], pos[2])
            glVertex3f(
                pos[0] + g_force[0],
                pos[1] + g_force[1],
                pos[2] + g_force[2]
            )
            glEnd()
        # volume forces
        glColor3f(1.0, 0.0, 0.0)
        for  particle_idx, volume_force in self.physics.volume_forces.items():
            pos = self.get_particle_position(particle_idx)

            force_magnitude = np.linalg.norm(volume_force)
            if force_magnitude > 0.01:
                glBegin(GL_LINES)
                glVertex3f(pos[0], pos[1], pos[2])
                glVertex3f(
                    pos[0] + volume_force[0],
                    pos[1] + volume_force[1],
                    pos[2] + volume_force[2]
                )
                glEnd()
        # constraint forces
        glColor3f(0.0, 1.0, 0.0)
        for i, j in self.constraints:
            pos_i = self.get_particle_position(i)
            pos_j = self.get_particle_position(j)
            
            diff = pos_j - pos_i
            current_length = np.linalg.norm(diff)
            if current_length < 1e-6:
                continue
            
            # force based on constraint violation
            rest_length = self.physics.rest_lengths[self.constraints.index((i, j))]
            force_magnitude = (current_length - rest_length) * 10.0  # Scale for visibility
            force_direction = diff / current_length
            

            if abs(force_magnitude) > 0.01:
                glBegin(GL_LINES)
                # force on particle i
                glVertex3f(pos_i[0], pos_i[1], pos_i[2])
                glVertex3f(
                    pos_i[0] + force_direction[0] * force_magnitude,
                    pos_i[1] + force_direction[1] * force_magnitude,
                    pos_i[2] + force_direction[2] * force_magnitude
                )
                # force on j (opposite dir)
                glVertex3f(pos_j[0], pos_j[1], pos_j[2])
                glVertex3f(
                    pos_j[0] - force_direction[0] * force_magnitude,
                    pos_j[1] - force_direction[1] * force_magnitude,
                    pos_j[2] - force_direction[2] * force_magnitude
                )
                glEnd()

    def draw_constraints(self):
        glColor3f(0.8, 0.8, 0.8)
        glBegin(GL_LINES)
        for i, j in self.constraints:
            pos_i = self.get_particle_position(i)
            pos_j = self.get_particle_position(j)
            glVertex3f(pos_i[0], pos_i[1], pos_i[2])
            glVertex3f(pos_j[0], pos_j[1], pos_j[2])
        glEnd()

    def draw_grab_lines(self):
        glColor3f(1.0, 0.2, 0.2)
        glBegin(GL_LINES)
        for particle_idx, _ in self.grab_constraints:
            pos = self.get_particle_position(particle_idx)
            glVertex3f(pos[0], pos[1], pos[2])
            glVertex3f(
                self.grab_point_3d[0],
                self.grab_point_3d[1],
                self.grab_point_3d[2]
            )
        glEnd()

    def draw_faces(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0.6, 0.8, 1.0, 0.3)

        glBegin(GL_TRIANGLES)
        for triangle in self.cube_faces:
            for vertex_idx in triangle:
                pos = self.get_particle_position(vertex_idx)
                glVertex3f(pos[0], pos[1], pos[2])
        glEnd()
        glDisable(GL_BLEND)

    def run(self):
        clock = pygame.time.Clock()
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: exit()
                self.handle_input(event)

            self.cam_movement()

            grab_constraints = self.grab_constraints if self.is_grabbing else None
            grab_point       = self.grab_point_3d    if self.is_grabbing else None
            self.physics.physics_step(grab_constraints, grab_point)

            self.render()
            

            clock.tick(60)
        pygame.quit()


engine = SoftPhysics()
engine.run()