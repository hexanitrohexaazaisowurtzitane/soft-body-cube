import numpy as np 

class Cube:
    def __init__(self, size=1.0, height_offset=3.0):
        self.size = size
        self.height_offset = height_offset
        self.particle_mass = 1.0

    def get_vertices(self):
        size = self.size
        vertices = [
            [-size, -size, -size],  # 0: bottom left  back
            [ size, -size, -size],  # 1: bottom right back
            [ size,  size, -size],  # 2: top    right back
            [-size,  size, -size],  # 3: top    left  back
            [-size, -size,  size],  # 4: bottom left  front
            [ size, -size,  size],  # 5: bottom right front
            [ size,  size,  size],  # 6: top    right front
            [-size,  size,  size],  # 7: top    left  front
        ]

        for i in range(len(vertices)):
            vertices[i][1] += self.height_offset

        return vertices

    def get_tetrahedra(self):
        tetrahedra = [
            [0, 1, 2, 4],  # tetra 1
            [2, 4, 6, 1],  # 2
            [1, 4, 5, 6],  # 3
            [2, 3, 4, 7],  # 4
            [2, 4, 6, 7]   # 5
        ]

        print(f"Created {len(tetrahedra)} tetrahedra for volume constraints")
        return tetrahedra

    def get_faces(self):
        faces = [
            (0, 1, 2), (0, 2, 3),  # bottom | y = -size
            (4, 5, 7), (5, 6, 7),  # top    | y = +size
            (0, 4, 7), (0, 7, 3),  # left   | x = -size
            (1, 5, 2), (5, 6, 2),  # right  | x = +size
            (4, 5, 0), (5, 1, 0),  # front  | z = +size
            (3, 2, 7), (2, 6, 7),  # back   | z = -size
        ]
        return faces
    
    def get_masses(self):
        return [self.particle_mass] * 8

    def get_constraints(self):
        
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0), # bottom | y = -size
            (4, 5), (5, 6), (6, 7), (7, 4), # top    | y = +size
            (0, 4), (1, 5), (2, 6), (3, 7), # verticals
        ]

        diagonals = [
            (0, 2), # bottom | y = -size
            (5, 7), # top    | y = +size
            (0, 7), # left   | x = -size
            (2, 5), # right  | x = +size

            (0, 5), # y fance diagonals
            (2, 7),

        ]

        constraints = edges + diagonals
        
        return constraints