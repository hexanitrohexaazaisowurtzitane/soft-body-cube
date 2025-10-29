# soft-body-cube
A real-time soft body physics simulator built with Python, PyOpenGL, and Pygame.
It simulates deformable cube meshes using constraint-based dynamics and tetrahedral volume preservation, complete with interactive mouse grabbing and 3D visualization.
The project is entirely based on the XPBD paper found here: https://matthias-research.github.io/pages/publications/XPBD.pdf
Youtube video: https://www.youtube.com/watch?v=UZNTFbsktHc

<img width="405" height="359" alt="output-onlinepngtools(8)" src="https://github.com/user-attachments/assets/2b1708dc-2428-4e5e-bd91-2989bc8a44f9" />

## Features:
* Real-time 3D soft-body physics
* Tetrahedral volume constraints
* Constraint-based position-based dynamics
* Mouse grabbing and camera



## How It Works
The simulator uses Position based dynamics to achieve a stable and real time soft body motion.

### Position-Based Dynamics (PBD):
Instead of integrating forces directly, PBD predicts particle positions, then iteratively corrects them to satisfy constraints which ensures stability even with large time steps.

### Distance Constraints:
Each pair of connected particles maintains a fixed rest length. After prediction, the solver adjusts their positions so the distance matches the rest length, mimicking elastic connections.

### Volume Constraints:
Each cube is decomposed into tetrahedra. The volume of each tetrahedron is preserved by computing the difference between its current and rest volume and applying corrections to its four vertices. This keeps the soft body from collapsing or expanding unnaturally.
