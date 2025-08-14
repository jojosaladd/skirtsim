import taichi as ti
import numpy as np
import math

from scene import Scene, Init

ti.init(arch=ti.vulkan)  # Use Vulkan for Mac compatibility
#ti.init(arch=ti.cpu, debug=True) ##debug
#ti.init(arch=ti.cuda)

# Scene and simulation settings
obstacle = Scene(Init.CLOTH_CONE)
contact_eps = 1e-2
record = False

## Camera Setting
yaw = 0.0  # Horizontal rotation angle
radius = 5.0  # Distance from center
target = ti.Vector([0.5, 0.2, 0.5])  # Look-at point (center of the skirt)

# Time step settings

dt = 1e-4 # Slower for debugging
substeps = int(1 / 120// dt)

# Gravity

gravity = ti.Vector([0, -9.8, 0])

CHOSEN_FABRIC = "denim"  # Options: "jersey", "denim"
CHOSEN_SEAM_TYPE = "regular"  # Options: "regular", "french", "none"

# Fabric Parameters Data
FABRIC_PROPERTIES = {
    "jersey": {
        "k_su": 3.0, "k_sv": 3.0, "kd": 0.0003, "k_shear": 0.02, "k_bend": 3.5,
        "w_seam": 0.002, "t_seam": 0.001, "E_seam": 1e2, # Regular seam specific
    },
    "denim": {
        "k_su": 20.0, "k_sv": 20.0, "kd": 0.0003, "k_shear": 0.02, "k_bend": 8.0,
        "w_seam": 0.003, "t_seam": 0.007, "E_seam": 2e5, # Regular seam specific
    }
}

k_su = 0.0
k_sv = 0.0
kd = 0.0
k_shear = 0.0
k_bend = 0.0
w_seam = 0.0
t_seam = 0.0
E_seam = 0.0

def update_fabric_parameters(fabric_name):
    global k_su, k_sv, kd, k_shear, k_bend, w_seam, t_seam, E_seam

    if fabric_name in FABRIC_PROPERTIES:
        config = FABRIC_PROPERTIES[fabric_name]
        k_su = config["k_su"]
        k_sv = config["k_sv"]
        kd = config["kd"]
        k_shear = config["k_shear"]
        k_bend = config["k_bend"]
        w_seam = config["w_seam"]
        t_seam = config["t_seam"]
        E_seam = config["E_seam"]
        print(f"Switched to {fabric_name} parameters: su={k_su}, sv={k_sv}, bend={k_bend}") # For debugging

update_fabric_parameters(CHOSEN_FABRIC)

PULL_STRENGTH_FRENCH_SEAM = 0.1

def find_bending_edges(triangles):
    edge_map = dict()
    bending_edges = []

    for tri in triangles:
        a, b, c = tri
        edges = [(a, b, c), (b, c, a), (c, a, b)]
        for u, v, opp in edges:
            edge = tuple(sorted([u, v]))
            if edge not in edge_map:
                edge_map[edge] = [opp]
            else:
                edge_map[edge].append(opp)
                if len(edge_map[edge]) == 2:
                    k, l = edge_map[edge]
                    i, j = edge
                    bending_edges.append([i, j, k, l])
    return np.array(bending_edges, dtype=np.int32)
def side_seam_edges(radius_outer, radius_inner, num_radial, num_circular):
    seam_edges = []
    # front_idx = i * num_radial + 0 # right
    # back_idx = i * num_radial + num_radial // 2  # left
    # side_seam_indices.append(front_idx)
    # side_seam_indices.append(back_idx)
    # Same radial positions
    for i in range(num_circular -1 ):
        front_idx_current = i * num_radial + 0
        front_idx_next = (i + 1) * num_radial + 0

        back_idx_current = i * num_radial + num_radial // 2
        back_idx_next = (i + 1) * num_radial + num_radial // 2

        seam_edges.append([front_idx_current, front_idx_next])  # right side seam edge
        seam_edges.append([back_idx_current, back_idx_next])    # left side seam edge

    return np.array(seam_edges, dtype=np.int32)


    # Generate hollow disk mesh
def generate_disk_mesh(radius_outer, radius_inner, num_radial, num_circular):
    verts = []
    tris = []
    uvs = []

    for i in range(num_circular):
        r = radius_inner + (radius_outer - radius_inner) * i / (num_circular - 1)

        for j in range(num_radial):
            theta = 2 * np.pi * j / num_radial
            x = 0.5 + r * np.cos(theta)
            z = 0.5 + r * np.sin(theta)
            verts.append([x, 0.089, z])

            ## compute uv (rest state, no offset 2.1)
            u = r * np.cos(theta)
            v = r * np.sin(theta)
            uvs.append([u,v])

    for i in range(num_circular - 1):
        ring_start = i * num_radial
        next_ring_start = (i + 1) * num_radial
        for j in range(num_radial):
            a = ring_start + j
            b = ring_start + (j + 1) % num_radial
            c = next_ring_start + j
            d = next_ring_start + (j + 1) % num_radial
            tris.append([a, b, d])
            tris.append([a, d, c])

    return np.array(verts, dtype=np.float32), np.array(tris, dtype=np.int32), np.array(uvs, dtype=np.float32)


# Generate and load hollow disk mesh
verts_np, tris_np, uvs_np = generate_disk_mesh(radius_outer=0.7, radius_inner=0.2, num_radial=64, num_circular=10)
ss_np =side_seam_edges(radius_outer=0.7, radius_inner=0.2, num_radial=64, num_circular=10)

ss_len = len(ss_np)
ss_field = ti.Vector.field(2, dtype=ti.i32, shape=ss_len)  # holds pairs of (i, j)

for idx in range(ss_len):
    ss_field[idx] = ti.Vector([ss_np[idx][0], ss_np[idx][1]])


num_verts = verts_np.shape[0]
num_tris = tris_np.shape[0]
cloth_verts = ti.Vector.field(3, dtype=ti.f32, shape=num_verts)
cloth_velocity = ti.Vector.field(3, dtype=ti.f32, shape=num_verts)
cloth_tris = ti.field(int, shape=num_tris * 3)
cloth_verts.from_numpy(verts_np)
cloth_velocity.fill([0.0, 0.0, 0.0])
cloth_colors = ti.Vector.field(3, dtype=ti.f32, shape=num_verts)
cloth_colors.from_numpy(np.tile(np.array([[1.0, 0.6, 0.6]], dtype=np.float32), (num_verts, 1)))

cloth_tris.from_numpy(tris_np.ravel())
#


## this is to recolor the side seam area to debug
def sideseam_colorguide(num):
    if num == 1:
        for i,j in ss_np:
            cloth_colors[i] = ti.Vector([0.2, 1.0, 0.2])  # green
            cloth_colors[j] = ti.Vector([0.2, 1.0, 0.2])  # green
    else:
        for i,j in ss_np:
            cloth_colors[i] = ti.Vector([1.0, 0.6, 0.6])  # pink
            cloth_colors[j] = ti.Vector([1.0, 0.6, 0.6])  # pink


total_seam_edges = ss_np.shape[0]
seam_edges_ti = ti.Vector.field(2, dtype=ti.i32, shape=total_seam_edges)
seam_edges_ti.from_numpy(ss_np)

rest_seam_lengths_np = np.linalg.norm(
    verts_np[ss_np[:, 0]] - verts_np[ss_np[:, 1]], axis=1
).astype(np.float32)

seam_rest_lengths = ti.field(dtype=ti.f32, shape=total_seam_edges)
seam_rest_lengths.from_numpy(rest_seam_lengths_np)


## uv coord
cloth_uvs = ti.Vector.field(2, dtype=ti.f32, shape=num_verts)
cloth_uvs.from_numpy(uvs_np)

cloth_force    = ti.Vector.field(3, dtype=ti.f32, shape=num_verts)
cloth_mass     = 1.0 / num_verts
cloth_masses = ti.Vector.field(3, dtype=ti.f32, shape=num_verts)

bending_edges_np = find_bending_edges(tris_np)
total_bending_edges = bending_edges_np.shape[0]
bending_edges = ti.Vector.field(4, dtype=ti.i32, shape=total_bending_edges)
bending_edges.from_numpy(bending_edges_np)
rest_angle = ti.field(dtype=ti.f32, shape=len(bending_edges_np))
rest_lengths_np = np.array([np.linalg.norm(verts_np[k] - verts_np[l]) for i, j, k, l in bending_edges_np], dtype=np.float32)
rest_angle.from_numpy(rest_lengths_np)


## pinning for debugging
pinned_mask = np.zeros(num_verts, dtype=np.int32)

for idx in range(num_verts):
    u, v = uvs_np[idx]
    r = np.sqrt(u**2 + v**2)
    if idx <64:  # Pin inner ring for debugging 288 is the outer
        pinned_mask[idx] = 1

pinned = ti.field(dtype=ti.i32, shape=num_verts)
pinned.from_numpy(pinned_mask)

@ti.func
def collision_cone(i):
    cr= .0 # restitution
    mu = .8 # tangential friction
    P = cloth_verts[i]
    # reuse obstacle fields from Scene
    C = obstacle.cone_center[0]
    min_y = obstacle.cone_min_y
    height = obstacle.cone_height
    r_base = obstacle.cone_base_radius


    # compute radial distance in XZ plane
    d = ti.Vector([P.x - C.x, P.z - C.z])
    rd = d.norm()
    frac = (P.y - min_y) / height
    r_lvl = r_base * (1.0 - frac)

    if rd < r_lvl + contact_eps:
        # surface normal of cone
        n = ti.Vector([d.x/rd, r_base/height, d.y/rd]).normalized()
        v_n = cloth_velocity[i].dot(n) * n
        v_t = cloth_velocity[i] - v_n
        cloth_velocity[i] = v_t * mu - cr * v_n

        # push out
        push = (r_lvl + contact_eps) - rd
        P.x += n.x * push
        P.z += n.z * push
        cloth_verts[i] = P

def reset_simulation():
    cloth_verts.from_numpy(verts_np)
    cloth_velocity.fill([0.0, 0.0, 0.0])
    cloth_force.fill([0.0, 0.0, 0.0])
    pinned.from_numpy(pinned_mask)
    update_fabric_parameters(CHOSEN_FABRIC)

@ti.kernel
def initial_mass():
    for i in cloth_masses:
        cloth_masses[i] = cloth_mass

# Simple gravity update
@ti.kernel
def reset():
    for i in cloth_force:
        cloth_force[i] = ti.Vector([0.0, 0.0, 0.0])
        cloth_masses[i] = cloth_mass

@ti.kernel
def handle_collisions():
    for i in range(num_verts):
        if pinned[i] == 0:
            collision_cone(i)


@ti.kernel
def update_g():
    for i in range(num_verts):
        if pinned[i] == 0:  # Only update non-pinned particles
            cloth_force[i] += gravity * cloth_masses[i]

@ti.kernel
def update_x():
    for i in range(num_verts):
        if pinned[i] == 0:
            cloth_verts[i] += cloth_velocity[i] * dt

@ti.kernel
def update_v():
    for i in range(num_verts):
        cloth_velocity[i] += dt * (cloth_force[i] / cloth_masses[i])

@ti.kernel
def update_stretch(current_k_su: ti.f32, current_k_sv: ti.f32):
    for tri_id in range(num_tris):
        ## index
        i = cloth_tris[tri_id * 3 + 0]
        j = cloth_tris[tri_id * 3 + 1]
        k = cloth_tris[tri_id * 3 + 2]

        ## verts, velocity, uvs
        pi, pj, pk = cloth_verts[i], cloth_verts[j], cloth_verts[k]
        ui, uj, uk = cloth_uvs[i], cloth_uvs[j], cloth_uvs[k]

        # delta p1 and p2
        p1 = pj - pi
        p2 = pk - pi

        # delta u1 and u2
        u1 = uj - ui
        u2 = uk - ui

        Dm = ti.Matrix.cols([u1, u2])  # So u1 and u2 are each 2D vectors, like [Δu, Δv]
        Ds = ti.Matrix.cols([p1, p2])  # 3x2 matrix
        F = Ds @ Dm.inverse()          # Deformation gradient: 3x2


        # # Derivatives in world space
        wu = F @ ti.Vector([1.0, 0.0])
        wv = F @ ti.Vector([0.0, 1.0])
        #
        # # Residuals (desired norm is 1.0) from 10.9
        Cu = wu.norm() - 1.0
        Cv = wv.norm() - 1.0
        #

        # # wu_hat and wv_hat are just unit vectors of wu and wv
        wu_hat = wu.normalized() if wu.norm() > 1e-6 else ti.Vector([0.0, 0.0, 0.0])
        wv_hat = wv.normalized() if wv.norm() > 1e-6 else ti.Vector([0.0, 0.0, 0.0])
        #
        a_u = Dm.inverse() @ ti.Vector([1.0, 0.0])
        a_v = Dm.inverse() @ ti.Vector([0.0, 1.0])
        a1u, a2u = a_u[0], a_u[1]
        a1v, a2v = a_v[0], a_v[1]

        # Stretch gradients (∂wu/∂p, ∂wv/∂p)
        dwu_i = -(a1u + a2u)
        dwu_j = a1u
        dwu_k = a2u

        dwv_i = -(a1v + a2v)
        dwv_j = a1v
        dwv_k = a2v

        area = 0.5 * ti.abs((uj - ui).cross(uk - ui))

        fi = -area * (current_k_su * Cu * wu_hat * dwu_i + current_k_sv * Cv * wv_hat * dwv_i)
        fj = -area * (current_k_su * Cu * wu_hat * dwu_j + current_k_sv * Cv * wv_hat * dwv_j)
        fk = -area * (current_k_su * Cu * wu_hat * dwu_k + current_k_sv * Cv * wv_hat * dwv_k)

        if pinned[i] == 0:
            cloth_force[i] += fi
        if pinned[j] == 0:
            cloth_force[j] += fj
        if pinned[k] == 0:
            cloth_force[k] += fk

@ti.kernel
def update_stretch_damping(current_kd: ti.f32):
    for tri_id in range(num_tris):
        i = cloth_tris[tri_id * 3 + 0]
        j = cloth_tris[tri_id * 3 + 1]
        k = cloth_tris[tri_id * 3 + 2]

        # Current positions and velocities
        xi, xj, xk = cloth_verts[i], cloth_verts[j], cloth_verts[k]
        vi, vj, vk = cloth_velocity[i], cloth_velocity[j], cloth_velocity[k]

        # UV coordinates (rest state)
        ui, uj, uk = cloth_uvs[i], cloth_uvs[j], cloth_uvs[k]

        # Build rest matrix Dm and invert it
        Dm = ti.Matrix.cols([uj - ui, uk - ui])  # 2x2
        Dm_inv = Dm.inverse()

        # Build current matrix Ds (3x2)
        dx1 = xj - xi
        dx2 = xk - xi
        Ds = ti.Matrix.cols([dx1, dx2])

        # Deformation gradient
        F = Ds @ Dm_inv

        wu = F @ ti.Vector([1.0, 0.0])
        wv = F @ ti.Vector([0.0, 1.0])

        wu_hat = wu.normalized() if wu.norm() > 1e-5 else ti.Vector([0.0, 0.0, 0.0])
        wv_hat = wv.normalized() if wv.norm() > 1e-5 else ti.Vector([0.0, 0.0, 0.0])

        # Gradients of stretch constraint (Eq. 10.24 style)
        dCdu_i = -wu_hat
        dCdu_j = wu_hat * Dm_inv[0, 0]
        dCdu_k = wu_hat * Dm_inv[0, 1]

        dCdv_i = -wv_hat
        dCdv_j = wv_hat * Dm_inv[1, 0]
        dCdv_k = wv_hat * Dm_inv[1, 1]

        # Project velocities onto constraint gradients
        Cdot_u = dCdu_i.dot(vi) + dCdu_j.dot(vj) + dCdu_k.dot(vk)
        Cdot_v = dCdv_i.dot(vi) + dCdv_j.dot(vj) + dCdv_k.dot(vk)

        fd_i = -current_kd * (dCdu_i * Cdot_u + dCdv_i * Cdot_v)
        fd_j = -current_kd * (dCdu_j * Cdot_u + dCdv_j * Cdot_v)
        fd_k = -current_kd * (dCdu_k * Cdot_u + dCdv_k * Cdot_v)

        if pinned[i] == 0:
            cloth_force[i] += fd_i
        if pinned[j] == 0:
            cloth_force[j] += fd_j
        if pinned[k] == 0:
            cloth_force[k] += fd_k

@ti.kernel
def update_shear(current_k_shear: ti.f32):
    for tri_id in range(num_tris):
        i = cloth_tris[tri_id * 3 + 0]
        j = cloth_tris[tri_id * 3 + 1]
        k = cloth_tris[tri_id * 3 + 2]

        pi, pj, pk = cloth_verts[i], cloth_verts[j], cloth_verts[k]
        ui, uj, uk = cloth_uvs[i], cloth_uvs[j], cloth_uvs[k]

        u1 = uj - ui
        u2 = uk - ui
        Dm = ti.Matrix.cols([u1, u2])  # 2x2
        Dm_inv = Dm.inverse()

        dx1 = pj - pi
        dx2 = pk - pi
        Ds = ti.Matrix.cols([dx1, dx2])  # 3x2

        F = Ds @ Dm_inv  # 3x2

        wu = F @ ti.Vector([1.0, 0.0])  # column 0
        wv = F @ ti.Vector([0.0, 1.0])  # column 1

        shear_scalar = wu.dot(wv)

        # Compute triangle area in UV (rest)
        a = 0.5 * ti.abs((uj - ui).cross(uk - ui))

        # Derivatives of F wrt x_i, x_j, x_k
        dphi_du = Dm_inv[0, :]
        dphi_dv = Dm_inv[1, :]

        # Each row: gradient of one coordinate in F
        dwu_dxi = -dphi_du[0] - dphi_du[1]
        dwv_dxi = -dphi_dv[0] - dphi_dv[1]

        dwu_dxj = dphi_du[0]
        dwv_dxj = dphi_dv[0]

        dwu_dxk = dphi_du[1]
        dwv_dxk = dphi_dv[1]

        # Total shear force is from gradient of wu ⋅ wv
        fi = -current_k_shear * a * (wv * dwu_dxi + wu * dwv_dxi)
        fj = -current_k_shear * a * (wv * dwu_dxj + wu * dwv_dxj)
        fk = -current_k_shear * a * (wv * dwu_dxk + wu * dwv_dxk)

        if pinned[i] == 0:
            cloth_force[i] += fi
        if pinned[j] == 0:
            cloth_force[j] += fj
        if pinned[k] == 0:
            cloth_force[k] += fk

@ti.kernel
def update_bending_spring(current_k_bend: ti.f32):
    for e in range(total_bending_edges):
        i, j, k, l = bending_edges[e]
        pk = cloth_verts[k]
        pl = cloth_verts[l]

        rest_len = (rest_angle[e])  # reuse as rest length (just name repurpose)
        cur_len = (pk - pl).norm()
        direction = (pk - pl).normalized() if cur_len > 1e-6 else ti.Vector([0.0, 0.0, 0.0])
        f = -current_k_bend * (cur_len - rest_len) * direction

        if pinned[k] == 0:
            cloth_force[k] += f
        if pinned[l] == 0:
            cloth_force[l] -= f

@ti.kernel
def update_seam_forces(current_w_seam: ti.f32, current_t_seam: ti.f32, current_E_seam: ti.f32):
    for e in range(seam_edges_ti.shape[0]):
        i, j = seam_edges_ti[e]
        pi = cloth_verts[i]
        pj = cloth_verts[j]
        rest_len = seam_rest_lengths[e]

        dir_vec = pj - pi
        cur_len = dir_vec.norm()

        if cur_len > 1e-6:
            dir_hat = dir_vec / cur_len
            dU_dl = current_w_seam * current_t_seam * current_E_seam / rest_len * (cur_len - rest_len)
            f = dU_dl * dir_hat

            if pinned[i] == 0:
                cloth_force[i] += f
            if pinned[j] == 0:
                cloth_force[j] -= f


# THEN in kernel:
@ti.kernel
def apply_frech_mass():
    for idx in range(ss_field.shape[0]):
        i = ss_field[idx][0]
        j = ss_field[idx][1]
        cloth_masses[i] = cloth_mass * 5
        cloth_masses[j] = cloth_mass * 5
        # cloth_colors[i] = ti.Vector([0.0, 1.0, 1.0])  # cyan
        # cloth_colors[j] = ti.Vector([0.0, 1.0, 1.0])  # cyan

# Video output setup
result_dir = './recordings/'
video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=60, automatic_build=False)

# Taichi UI setup
scene = ti.ui.Scene()
camera = ti.ui.Camera()
window = ti.ui.Window("SkirtSim: Skirt/Seam Simulation", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0.6, 0.6, 1.0))

# Set initial camera position
camera.position(0.0, 0.8, 5.0)
#camera.position(0.5, 0.8, -1.5) back view
camera.lookat(0.5, 0.2, 0.5)
camera.fov(30.0)

# Time tracking
start_t = 0.0
current_t = 0.0
color_sideseam = False

initial_mass()
# Main simulation loop
while True:
    gui = window.GUI
    gui.begin("Controls", 0.02, 0.02, 0.25, 0.25)
    color_sideseam = gui.checkbox("Highlight Side Seam",color_sideseam)
    if color_sideseam:
        sideseam_colorguide(1)
    else:
        sideseam_colorguide(3)

    gui.text("Fabric Type:")
    new_fabric_selection = CHOSEN_FABRIC
    if gui.button("Jersey"):
        new_fabric_selection = "jersey"
    if gui.button("Denim"):
        new_fabric_selection = "denim"

    if new_fabric_selection != CHOSEN_FABRIC:
        CHOSEN_FABRIC = new_fabric_selection
        update_fabric_parameters(CHOSEN_FABRIC)
        reset_simulation()
    gui.text(f"Current: {CHOSEN_FABRIC}")

    gui.text("Seam Type:")
    new_seam_selection = CHOSEN_SEAM_TYPE
    if gui.button("Regular Seam"):
        new_seam_selection = "regular"
        reset_simulation()
    if gui.button("French Seam"):
        new_seam_selection = "french"
        reset_simulation()
    # if gui.button("No Seam"):
    #     new_seam_selection = "none"
    #     reset_simulation()
    if new_seam_selection != CHOSEN_SEAM_TYPE:
        CHOSEN_SEAM_TYPE = new_seam_selection
    gui.text(f"Current: {CHOSEN_SEAM_TYPE}")
    if gui.button("Stop simulation"):
        break
    gui.end()


    # Rotate camera with arrow keys
    if window.is_pressed(ti.ui.LEFT):
        yaw -= 0.03  # You can adjust speed
    if window.is_pressed(ti.ui.RIGHT):
        yaw += 0.03

    # Zoom with UP and DOWN arrow keys
    if window.is_pressed(ti.ui.UP):
        radius -= 0.05  # Zoom in
    if window.is_pressed(ti.ui.DOWN):
        radius += 0.05  # Zoom out

    # Update camera position (orbit horizontally)
    x = target[0] + radius * math.sin(yaw)
    z = target[2] + radius * math.cos(yaw)
    y = 0.8  # fixed height

    camera.position(x, y, z)
    camera.lookat(*target)

    for _ in range(substeps):
        current_t += dt
        # timestep()
        reset()
        update_stretch(k_su, k_sv)
        update_stretch_damping(kd)
        update_shear(k_shear)
        update_bending_spring(k_bend)

        if CHOSEN_SEAM_TYPE == "regular":
            update_seam_forces(w_seam, t_seam, E_seam)
        elif CHOSEN_SEAM_TYPE == "french":
            #apply_french_seam_force()
            apply_frech_mass()
        # If CHOSEN_SEAM_TYPE == "none", no seam-specific force is applied.

        update_g()
        update_v()
        update_x()
        handle_collisions()

    # Set up the scene for rendering
    scene.set_camera(camera)
    # you can also control camera movement in a window
    # camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))

    # Draw the cloth disk
    #scene.mesh(cloth_verts, indices=cloth_tris, color=(1.0, 0.6, 0.6), two_sided=False)
    scene.mesh(cloth_verts, indices=cloth_tris, per_vertex_color=cloth_colors, two_sided=False)

    # Draw the collision object (e.g., cone)
    scene.mesh(obstacle.verts, indices=obstacle.tris, color=(0.8, 0.7, 0.6))

    canvas.scene(scene)

    if record:
        img = window.get_image_buffer_as_numpy()
        video_manager.write_frame(img)

    window.show()

if record:
    video_manager.make_video(gif=False, mp4=True)
