# SkirtSim: Skirt/Seam Simulation in Taichi


This is a simple cloth simulation that models a circle skirt draping over a cone. It supports different fabric behaviors (denim vs jersey) and a seam effect (regular vs french seam).

The physical simulation follows a standard force-based approach described in [Large Steps in Cloth Simulation](https://dl.acm.org/doi/10.1145/280814.280821) and [Dynamic Deformables, Chapter 10, Thin Shell Forces](https://www.tkim.graphics/DYNAMIC_DEFORMABLES/DynamicDeformables.pdf). For the regular seam simulation, I followed [Computational Design of Skintight Clothing](https://www.researchgate.net/publication/343616934_Computational_design_of_skintight_clothing).

To differentiate fabric behaviors, I set different bending and stretch parameters so the results qualitatively match real-world fabrics. For the French seam finish, I assigned heavier mass to seam-edge vertices to mimic the extra fabric layers that increase mass along French seams.

## Demo
![example](recordings/example.gif)


## Requirements
- Python 3.9+
- taichi >= 1.7
- pywavefront
- numpy

---
## How to Run

1. Clone this repo and open a terminal in the project folder.

2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
    or
    ```
    pip install taichi pywavefront numpy
    ```

3.  Run the simulator:
    ```
    python skirt.py
    ```
### Interactive Controls
- ← / → Orbit camera (rotate around the skirt)
- ↑ / ↓ Zoom in/out
- Use the GUI panel to switch fabric and seam type
- Toggle “Highlight Side Seam” to visualize the side seam
- Click “Stop simulation” to quit

