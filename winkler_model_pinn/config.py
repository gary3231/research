import math

# ------------------------------ network settings ---------------------------------------------------
iteration = 20
D_in = 2
H = 30
D_out = 2  #2
lr = 0.1
# ------------------------------ material parameter -------------------------------------------------
model_energy = 'NeoHookean2D'
Es = 10**3  #substrate not SLS
Ec = 10**2
Eci_div_Ec = 10

E = 0
nu = 0

nuc = 0.3
nus = 0.3
# B = E/3/(1-2*nu)

W = 10
R = 9.525
a = math.pow(3*(W)*R*(1-nus**2)/(4*Es),1/3)
ph = 3*(W)/(2*math.pi*(a**2))
delta = (a**2)/R
l = (10**(-3))*a

# Ecb = Ec/ph
# lb = l/a

Re = Eci_div_Ec-1
De = 0.01#[0.01, 1, 5, 10]
tau = 0.01
vm = -De/tau*a

Ec_bar = Ec/ph
l_bar = l/delta

beta = tau/(1+Re)
# param_c1 = 630
# param_c2 = -1.2
# param_c = 100
# ----------------------------- define structural parameters ---------------------------------------
Length = 12*a
Height = 8*a
Depth = l
known_left_ux = 0
known_left_uy = 0
bc_left_penalty = 1.0

known_right_tx = 0
known_right_ty = 0
bc_right_penalty = 1.0
# ------------------------------ define domain and collocation points -------------------------------
Nx = 241 # 120  # 120
Ny = 121 # 30  # 60
x_min, y_min = (-8*a, -4*a)
# x_min, y_min = (-8, -4)
hx = Length / (Nx - 1)
hy = Height / (Ny - 1)
shape = [Nx, Ny]
dxdy = [hx, hy]

# ------------------------------ data testing -------------------------------------------------------
num_test_x = Nx
num_test_y = Ny