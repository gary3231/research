import math

# ------------------------------ network settings ---------------------------------------------------
iteration = 20
D_in = 2
H = 30
D_out = 2  #2
lr = 0.1
# ------------------------------ material parameter -------------------------------------------------
model_energy = 'NeoHookean2D'
Es = 117*10**3  #substrate not SLS
Ec = 10**3
Eci_div_Ec = 1000   

E = 0
nu = 0
# nuc = 0.3
# nus = 0.3
# B = E/3/(1-2*nu)
eta_i = 447*10**(-4)
W = 10
R = 9.525
a = 1.07*10**(-1)
ph = 4.18*10**2#3*(W)/(2*math.pi*(a**2))
pi = 196
delta = (a**2)/R
l = (10**(-3))*a

# Ecb = Ec/ph
# lb = l/a

Re = Eci_div_Ec-1
# De = 0.01#[0.01, 1, 5, 10]
# tau = 0.01
vm = -10 #[0, 10, 100, 10**3, 10**6]
Es_v = Ec/Re*(Re+1)


Ec_bar = Ec/ph
l_bar = l/delta

# beta = tau/(1+Re)
# param_c1 = 630
# param_c2 = -1.2
# param_c = 100
# ----------------------------- define structural parameters ---------------------------------------
Length = 6*a
Height = 6*a
Depth = l
known_left_ux = 0
known_left_uy = 0
bc_left_penalty = 1.0

known_right_tx = 0
known_right_ty = 0
bc_right_penalty = 1.0
# ------------------------------ define domain and collocation points -------------------------------
Nx = 241 # 120  # 120
Ny = 241 # 30  # 60
x_min, y_min = (-3*a, -3*a)
# x_min, y_min = (-8, -4)
hx = Length / (Nx - 1)
hy = Height / (Ny - 1)
shape = [Nx, Ny]
dxdy = [hx, hy]

# ------------------------------ data testing -------------------------------------------------------
num_test_x = Nx
num_test_y = Ny