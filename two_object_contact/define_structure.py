from dem_hyperelasticity.config import *
from dem_hyperelasticity.two_object_contact.config import *


def setup_domain():
    R_dom = R/Nr, R, Nr-1
    ang_dom = ang_min, ang_max, Na
    # z_dom = z_min, Depth, Nz
    # create points
    lin_R = np.linspace(R_dom[0], R_dom[1], R_dom[2])
    lin_ang = np.linspace(ang_dom[0], ang_dom[1], ang_dom[2])
    # lin_z = np.linspace(z_dom[0], z_dom[1], z_dom[2])
    dom = np.zeros(((Nr-1) * Na+1 , 2))
    c = 0
    dom[0] = np.array([0,0])
    for x in np.nditer(lin_R):
        tb = ang_dom[2] * c + 1
        te = tb + ang_dom[2]
        c += 1
        dom[tb:te, 0] = x*np.cos(lin_ang)+origin[0]
        dom[tb:te, 1] = x*np.sin(lin_ang)+origin[1]
    print(dom.shape)
    np.meshgrid(lin_R, lin_ang)
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    ax.scatter(dom[:, 0], dom[:, 1], s=0.005, facecolor='blue')
    ax.set_xlabel('X', fontsize=3)
    ax.set_ylabel('Y', fontsize=3)
    ax.tick_params(labelsize=4)
    # ------------------------------------ BOUNDARY ----------------------------------------
    # Left boundary condition (Dirichlet BC)

    bcl_u_pts_idx_1 = np.where(abs(dom[:, 1]-origin[1]) < 10**(-14))
    bcl_u_pts_1 = dom[bcl_u_pts_idx_1, :][0]
    bcl_u_1 = np.ones(np.shape(bcl_u_pts_1)) * [0, -0.1]

    bcr_t_pts_idx_1 =  np.where((abs((dom[:, 0]-origin[0])**2+(dom[:, 1]-origin[1])**2 - R**2) < 10**(-14)) & ((dom[:,1]-origin[1])>=R*np.sin((ang_max+ang_min)/2-np.pi/4)))
    bcr_t_pts_1 = dom[bcr_t_pts_idx_1, :][0]
    bcr_t_1 = np.ones(np.shape(bcr_t_pts_1)) * [0, 0]


    bcr_t_pts_idx_2 =  np.where((abs((dom[:, 0]-origin[0])**2+(dom[:, 1]-origin[1])**2 - R**2) < 10**(-14)) & ((dom[:,1]-origin[1])<R*np.sin((ang_max+ang_min)/2-np.pi/4)))
    bcr_t_pts_2 = dom[bcr_t_pts_idx_2, :][0]
    bcr_t_2 = np.ones(np.shape(bcr_t_pts_2)) * [0, 0]


    # F = 4/3*np.sqrt(R*d)*d*E/(1-nu**2)

    # bcr_t_pts_idx_6 = np.where((dom[:, 2] == Depth) & (((dom[:,0]-Length/2)**2+(dom[:,1]-Height/2)**2)<=R*d))
    # bcr_t_pts_6 = dom[bcr_t_pts_idx_6, :][0]
    # bcr_t_6 = np.ones(np.shape(bcr_t_pts_6)) * [known_left_ux, known_left_uy, F]
    # bcr_t_6[:,2]=(-d+1/(2*R)*((dom[bcr_t_pts_idx_6,0]-Length/2)**2+(dom[bcr_t_pts_idx_6,1]-Height/2)**2))

    inner_pts_idx = np.where((abs(dom[:,0]) > 10**(-14))&(abs(dom[:, 1]) > 10**(-14))&(abs(dom[:, 0]**2+dom[:, 1]**2 - R**2) > 10**(-14)))
    inner_pts = dom[inner_pts_idx, :][0]
    # ax.scatter(inner_pts[:, 0], inner_pts[:, 1], s=0.005, facecolor='blue')
    # ax.scatter(bcl_u_pts_1[:, 0], bcl_u_pts_1[:, 1], s=0.5, facecolor='green')
    # ax.scatter(bcr_t_pts_1[:, 0], bcr_t_pts_1[:, 1], s=0.5, facecolor='red')
    # ax.scatter(bcr_t_pts_2[:, 0], bcr_t_pts_2[:, 1], s=0.5, facecolor='black')
    # plt.show()
    # plt.show()

    boundary_neumann = {
        # condition on the right
        "neumann_1": {
            "coord": bcr_t_pts_1,
            "known_value": bcr_t_1,
            "penalty": bc_right_penalty,
            "idx":bcr_t_pts_idx_1
        },
        "neumann_2": {
            "coord": bcr_t_pts_2,
            "known_value": bcr_t_2,
            "penalty": bc_right_penalty,
            "idx":bcr_t_pts_idx_2
        },
        # adding more boundary condition here ...
    }
    boundary_dirichlet = {
        # condition on the left
        "dirichlet_1": {
            "coord": bcl_u_pts_1,
            "known_value": bcl_u_1,
            "penalty": bc_right_penalty
        },
        # # adding more boundary condition here ...
    }
    return dom, boundary_neumann, boundary_dirichlet

def set_wall():
    lin_wall_x = np.linspace(wall_x[0],wall_x[1],50)
    lin_wall_y = np.linspace(wall_y[0],wall_y[1],50)
    wall = np.vstack(lin_wall_x,lin_wall_y).T
    return wall

# -----------------------------------------------------------------------------------------------------
# prepare inputs for testing the model
# -----------------------------------------------------------------------------------------------------
def get_datatest():
    R_dom_test = 0, R, Nr-1
    ang_dom_test = ang_min, ang_max, Na
    # create points
    lin_R = np.linspace(R_dom_test[0], R_dom_test[1], R_dom_test[2])
    lin_ang = np.linspace(ang_dom_test[0], ang_dom_test[1], ang_dom_test[2])
    xGrid, yGrid = np.meshgrid(lin_R, lin_ang)
    data_test = np.concatenate(
        (np.array([xGrid.flatten()]).T, np.array([yGrid.flatten()]).T), axis=1)
    return lin_R, lin_ang, data_test

if __name__ == '__main__':
    setup_domain()