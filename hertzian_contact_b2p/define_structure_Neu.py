from dem_hyperelasticity.config import *
from dem_hyperelasticity.hertzian_contact_b2p.config import *


def setup_domain():
    x_dom = x_min, x_min+Length, Nx
    y_dom = y_min, y_min+Height, Ny
    # create points
    lin_x = np.linspace(x_dom[0], x_dom[1], x_dom[2])
    lin_y = np.linspace(y_dom[0], y_dom[1], y_dom[2])
    # lin_x = lin_x**2
    # lin_y = -lin_y**2
    dom = np.zeros((Nx * Ny, 2))
    # dom_idx = np.arange(Nx * Ny)
    c = 0
    for x in np.nditer(lin_x):
        tb = y_dom[2] * c
        te = tb + y_dom[2]
        c += 1
        dom[tb:te, 0] = x
        dom[tb:te, 1] = lin_y
    print(dom.shape)
    np.meshgrid(lin_x, lin_y)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.scatter(dom[:, 0], dom[:, 1], s=0.005, facecolor='blue')
    ax.set_xlabel('X', fontsize=3)
    ax.set_ylabel('Y', fontsize=3)
    ax.tick_params(labelsize=4)
# ------------------------------------ BOUNDARY ----------------------------------------
    # Left boundary condition (Dirichlet BC)
    bcl_u_pts_idx = np.where(dom[:,1] == y_min)
    # bcl_u_pts_idx_1 = np.where((dom[:,1] == 0) & (dom[:,0]**2<=R**2))
    # bcl_u_pts_idx_2 = np.where(dom[:,1] == y_min)

    bcl_u_pts = dom[bcl_u_pts_idx, :][0]  #get first position of dirichlet BC
    # bcl_u_pts_1 = dom[bcl_u_pts_idx_1,:][0]
    # bcl_u_pts_2 = dom[bcl_u_pts_idx_2,:][0]

    bcl_u = np.ones(np.shape(bcl_u_pts)) * [0, 0]
    # bcl_u_1 = np.ones(np.shape(bcl_u_pts_1))*[0,0]
    # bcl_u_2 = np.ones(np.shape(bcl_u_pts_2))*[0,0]

    # Right boundary condition (Neumann BC)
    bcr_t_pts_idx = np.where((dom[:, 0] == x_min))
    # bcr_t_pts_idx_1 = np.where(dom[:,1] == 0)
    bcr_t_pts_idx_1 = np.where((dom[:,1] == (y_min+Height)) & (dom[:,0]**2<=(R**2)))
    bcr_t_pts_idx_2 = np.where((dom[:, 1] == (y_min+Height)) & (dom[:,0]**2 > (R**2)))
    bcr_t_pts_idx_3 = np.where((dom[:, 0] == (x_min+Length)))
    # bcr_t_pts_idx_4 = np.where((dom[:, 1] == y_min))
    
    bcr_t_pts = dom[bcr_t_pts_idx, :][0]
    bcr_t_pts_1 = dom[bcr_t_pts_idx_1, :][0]    
    bcr_t_pts_2 = dom[bcr_t_pts_idx_2, :][0]
    bcr_t_pts_3 = dom[bcr_t_pts_idx_3, :][0]
    # bcr_t_pts_4 = dom[bcr_t_pts_idx_4, :][0]
    # bcr_t_pts_c = np.concatenate((bcr_t_pts,bcr_t_pts_2),0)
    bcr_t = np.ones(np.shape(bcr_t_pts)) * [0, 0]
    bcr_t_1 = np.ones(np.shape(bcr_t_pts_1)) * [0, 0]
    bcr_t_2 = np.ones(np.shape(bcr_t_pts_2)) * [0, 0]
    bcr_t_3 = np.ones(np.shape(bcr_t_pts_3)) * [0, 0]
    # bcr_t_4 = np.ones(np.shape(bcr_t_pts_4)) * [0, 0]
    # bcr_t = np.concatenate((bcr_t,bcr_t_2),0)

    ax.scatter(dom[:, 0], dom[:, 1], s=0.005, facecolor='blue')
    ax.scatter(bcl_u_pts[:, 0], bcl_u_pts[:, 1], s=0.5, facecolor='red')
    # ax.scatter(bcl_u_pts_1[:, 0], bcl_u_pts_1[:, 1], s=0.5, facecolor='red')
    # ax.scatter(bcl_u_pts_2[:, 0], bcl_u_pts_2[:, 1], s=0.5, facecolor='red')
    ax.scatter(bcr_t_pts[:, 0], bcr_t_pts[:, 1], s=0.5, facecolor='green')
    # ax.scatter(bcr_t_pts_1[:, 0], bcr_t_pts_1[:, 1], s=0.5, facecolor='black')
    ax.scatter(bcr_t_pts_2[:, 0], bcr_t_pts_2[:, 1], s=0.5, facecolor='green')
    ax.scatter(bcr_t_pts_3[:, 0], bcr_t_pts_3[:, 1], s=0.5, facecolor='green')
    # ax.scatter(bcr_t_pts_4[:, 0], bcr_t_pts_4[:, 1], s=0.5, facecolor='green')
    plt.show()
    # exit()
    boundary_neumann = {
        # condition on the right
        "neumann_1": {
            "coord": bcr_t_pts, #size is same as below
            "known_value": bcr_t,   #size is [N,2]
            "penalty": bc_right_penalty,  #in config file
            "idx":bcr_t_pts_idx
        },
        "neumann_2": {
            "coord": bcr_t_pts_1, #size is same as below
            "known_value": bcr_t_1,   #size is [N,2]
            "penalty": bc_right_penalty,  #in config file
            "idx":bcr_t_pts_idx_1
        },
        "neumann_3": {
            "coord": bcr_t_pts_2, #size is same as below
            "known_value": bcr_t_2,   #size is [N,2]
            "penalty": bc_right_penalty,  #in config file
            "idx":bcr_t_pts_idx_2
        },
        "neumann_4": {
            "coord": bcr_t_pts_3, #size is same as below
            "known_value": bcr_t_3,   #size is [N,2]
            "penalty": bc_right_penalty,  #in config file
            "idx":bcr_t_pts_idx_3
        }
        # adding more boundary condition here ...
    }
    boundary_dirichlet = {          #two dict
        # condition on the left
        "dirichlet_1": {
            "coord": bcl_u_pts,
            "known_value": bcl_u,
            "penalty": bc_left_penalty
        },
        # "dirichlet_2": {
        #     "coord": bcl_u_pts_1,
        #     "known_value": bcl_u_1,
        #     "penalty": bc_left_penalty
        # },
        # "dirichlet_2": {
        #     "coord": bcl_u_pts_2,
        #     "known_value": bcl_u_2,
        #     "penalty": bc_left_penalty
        # },
        # adding more boundary condition here ...
    }
    s=np.hstack((bcr_t_pts_idx_1,bcr_t_pts_idx_2))
    
    dom = np.delete(dom,s[0],0)
    # print(dom)
    return dom, boundary_neumann, boundary_dirichlet


# -----------------------------------------------------------------------------------------------------
# prepare inputs for testing the model
# -----------------------------------------------------------------------------------------------------
def get_datatest(Nx=num_test_x, Ny=num_test_y):
    x_dom_test = x_min, x_min+Length, Nx
    y_dom_test = y_min, y_min+Height, Ny
    # create points
    x_space = np.linspace(x_dom_test[0], x_dom_test[1], x_dom_test[2])
    y_space = np.linspace(y_dom_test[0], y_dom_test[1], y_dom_test[2])
    # x_space = x_space**2
    # y_space = -y_space**2
    xGrid, yGrid = np.meshgrid(x_space, y_space)
    data_test = np.concatenate(
        (np.array([xGrid.flatten()]).T, np.array([yGrid.flatten()]).T), axis=1)
    return x_space, y_space, data_test


# ------------------------------------
# Author: minh.nguyen@ikm.uni-hannover.de
# Initial date: 09.09.2019
# an additional functionality : get interior points without taking boundary points
# ------------------------------------
def setup_domain_v2(interData=False):
    Nx_temp, Ny_temp = 2000, 500
    x_dom = x_min, Length, Nx_temp
    y_dom = y_min, Height, Ny_temp
    # create points
    lin_x = np.linspace(x_dom[0], x_dom[1], x_dom[2])
    lin_y = np.linspace(y_dom[0], y_dom[1], y_dom[2])
    dom = np.zeros((Nx_temp * Ny_temp, 2))
    c = 0
    for x in np.nditer(lin_x):
        tb = y_dom[2] * c
        te = tb + y_dom[2]
        c += 1
        dom[tb:te, 0] = x
        dom[tb:te, 1] = lin_y
    print(dom.shape)
    np.meshgrid(lin_x, lin_y)
    fig = plt.figure(figsize=(30, 1))
    ax = fig.add_subplot(111)
    ax.scatter(dom[:, 0], dom[:, 1], s=0.005, facecolor='blue')
    ax.set_xlabel('X', fontsize=3)
    ax.set_ylabel('Y', fontsize=3)
    ax.tick_params(labelsize=4)
    # ------------------------------------ BOUNDARY ----------------------------------------
    # Left boundary condition (Dirichlet BC)
    bcl_u_pts_idx = np.where(dom[:, 0] == x_min)
    bcl_u_pts = dom[bcl_u_pts_idx, :][0]
    bcl_u = np.ones(np.shape(bcl_u_pts)) * [known_left_ux, known_left_uy]

    # Right boundary condition (Neumann BC)
    bcr_t_pts_idx = np.where(dom[:, 0] == Length)
    bcr_t_pts = dom[bcr_t_pts_idx, :][0]
    bcr_t = np.ones(np.shape(bcr_t_pts)) * [known_right_tx, known_right_ty]

    ax.scatter(dom[:, 0], dom[:, 1], s=0.005, facecolor='blue')
    ax.scatter(bcl_u_pts[:, 0], bcl_u_pts[:, 1], s=0.5, facecolor='red')
    ax.scatter(bcr_t_pts[:, 0], bcr_t_pts[:, 1], s=0.5, facecolor='green')
    plt.show()
    if interData == 1:
        x_dom = x_min, Length, Nx
        y_dom = y_min, Height, Ny
        # create points
        lin_x = np.linspace(x_dom[0], x_dom[1], x_dom[2])
        lin_y = np.linspace(y_dom[0], y_dom[1], y_dom[2])
        dom = np.zeros((Nx * Ny, 2))
        c = 0
        for x in np.nditer(lin_x):
            tb = y_dom[2] * c
            te = tb + y_dom[2]
            c += 1
            dom[tb:te, 0] = x
            dom[tb:te, 1] = lin_y
        id1 = np.where((dom[:, 1] > y_min))
        dom = dom[id1, :][0]
        id2 = np.where((dom[:, 1] < Height))
        dom = dom[id2, :][0]
        id3 = np.where((dom[:, 0] > x_min))
        dom = dom[id3, :][0]
        id4 = np.where((dom[:, 0] < Length))
        dom = dom[id4, :][0]
        fig = plt.figure(figsize=(30, 1))
        ax = fig.add_subplot(111)
        ax.scatter(dom[:, 0], dom[:, 1], s=0.005, facecolor='blue')
        plt.show()
    # exit()
    boundary_neumann = {
        # condition on the right
        "neumann_1": {
            "coord": bcr_t_pts,
            "known_value": bcr_t,
            "penalty": bc_right_penalty
        }
        # adding more boundary condition here ...
    }
    boundary_dirichlet = {
        # condition on the left
        "dirichlet_1": {
            "coord": bcl_u_pts,
            "known_value": bcl_u,
            "penalty": bc_left_penalty
        }
        # adding more boundary condition here ...
    }
    return dom, boundary_neumann, boundary_dirichlet

def get_datatest_v2(Nx=num_test_x, Ny=num_test_y, interData=False):
    x_dom_test = x_min, Length, Nx
    y_dom_test = y_min, Height, Ny
    # create points
    x_space = np.linspace(x_dom_test[0], x_dom_test[1], x_dom_test[2])
    y_space = np.linspace(y_dom_test[0], y_dom_test[1], y_dom_test[2])
    xGrid, yGrid = np.meshgrid(x_space, y_space)
    data_test = np.concatenate(
        (np.array([xGrid.flatten()]).T, np.array([yGrid.flatten()]).T), axis=1)
    if interData == 1:
        id1 = np.where((data_test[:, 1] > y_min))
        data_test = data_test[id1, :][0]
        id2 = np.where((data_test[:, 1] < Height))
        data_test = data_test[id2, :][0]
        id3 = np.where((data_test[:, 0] > x_min))
        data_test = data_test[id3, :][0]
        id4 = np.where((data_test[:, 0] < Length))
        data_test = data_test[id4, :][0]
        fig = plt.figure(figsize=(30, 1))
        ax = fig.add_subplot(111)
        ax.scatter(data_test[:, 0], data_test[:, 1], s=0.005, facecolor='blue')
        plt.show()
        return x_space[1:-1], y_space[1:-1], data_test
    fig = plt.figure(figsize=(30, 1))
    ax = fig.add_subplot(111)
    ax.scatter(data_test[:, 0], data_test[:, 1], s=0.005, facecolor='blue')
    plt.show()
    return x_space, y_space, data_test

if __name__ == '__main__':
    setup_domain()
