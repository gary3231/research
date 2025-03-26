import numpy as np
# ------------------------------ network settings ---------------------------------------------------
iteration = 50
lr = 0.1
D_in = 2
H = 30
D_out = 6
# ------------------------------ material parameter -------------------------------------------------
E = 300 
nu = 0.3
B = E/(1-2*nu)/3
# param_c1 = 630
# param_c2 = -1.2
# param_c = 10000
model_energy = 'NeoHookean2D'
# ----------------------------- define structural parameters ---------------------------------------
R = 1
ang_min = np.pi
ang_max = np.pi*2
# d = 0.02
# Length = 1.0
# Height = 1.0
# Depth = -1.0
origin = (0,3)
known_right_ux = 0
# known_left_uy = 0
# known_left_uz = 0
# bc_left_penalty = 5000.0
bc_left_penalty = 1.0 #5000

# known_right_tx = 0
known_above_ty = 0.5
# known_right_tz = 0
bc_right_penalty = 1.0
# ------------------------------ define domain and collocation points -------------------------------
Nr = 51  # 120  # 120
Na = 51  # 30  # 60
# Nz = 40  # 30  # 10
# x_min, y_min, z_min = (0.0, 0.0, 0.0)
(hx, hy) = (R / (Nr - 1), (ang_max-ang_min) / (Na - 1))
shape = [Nr, Na]
dxdydz = [hx, hy]
# ------------------------------ data testing -------------------------------------------------------
# num_test_x = Nx
# num_test_y = Ny
# num_test_z = Nz
# ------------------------------ filename output ----------------------------------------------------
filename_out = "./NeoHook3D_beam20x5_NeoHook_traction-1p25_20iter_100_25_5P_pen100000"

#-------------------------------------------wall-----------------------------------------------------#
wall_x = [0,R]
wall_y = [-R,-R]