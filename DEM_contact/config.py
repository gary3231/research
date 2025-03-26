
# ------------------------------ network settings ---------------------------------------------------
iteration = 20
D_in = 2
H = 60
D_out = 6  #2
lr = 0.001
# ------------------------------ material parameter -------------------------------------------------
model_energy = 'NeoHookean2D'
E = 10  
nu = 0.3
B = E/3/(1-2*nu)
# R=0.5
# d=0.02
# param_c1 = 630
# param_c2 = -1.2
# param_c = 100
# ----------------------------- define structural parameters ---------------------------------------
Length = 1.0
Height = 1.0
Depth = 1.0
known_left_ux = 0
known_left_uy = 0
bc_left_penalty = 1.0

known_right_tx = 0
known_right_ty = 0
bc_right_penalty = 1.0
# ------------------------------ define domain and collocation points -------------------------------
Nx = 100 # 120  # 120
Ny = 100 # 30  # 60
x_min, y_min = (0.0, 0.0)
hx = Length / (Nx - 1)
hy = Height / (Ny - 1)
shape = [Nx, Ny]
dxdy = [hx, hy]

# ------------------------------ data testing -------------------------------------------------------
num_test_x = Nx
num_test_y = Ny