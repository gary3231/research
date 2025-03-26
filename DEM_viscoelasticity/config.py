import numpy as np
# ------------------------------ network settings ---------------------------------------------------
iteration = 20
D_in = 2
H = 30
D_out = 2  #2
lr = 0.001
# ------------------------------ material parameter -------------------------------------------------
model_energy = 'viscoelasticity'
E0 = 70*(10**3)  
E1 = 20*(10**3)
eta1 = 1*(10**3)
nu = 0.
# def dotC(e):
#     c = nu/(1+nu)/(1-nu)*np.trace(e)*np.identity(2) + 1/(1+nu)*e
#     return c
sigc = 100
epsr = 0.001

# R=0.5
# d=0.02
# param_c1 = 630
# param_c2 = -1.2
# param_c = 100
# ----------------------------- define structural parameters ---------------------------------------
Length = 0.1
Height = 0.2
Depth = 1.0
known_left_ux = 0
known_left_uy = 0
bc_left_penalty = 1.0

known_right_tx = 0
known_right_ty = 0
bc_right_penalty = 1.0
# ------------------------------ define domain and collocation points -------------------------------
Nx = 31 # 120  # 120
Ny = 61 # 30  # 60
x_min, y_min = (0.0, 0.0)
hx = Length / (Nx - 1)
hy = Height / (Ny - 1)
shape = [Nx, Ny]
dxdy = [hx, hy]

# ------------------------------ data testing -------------------------------------------------------
num_test_x = Nx
num_test_y = Ny