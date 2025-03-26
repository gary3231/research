"""
@author: sfmt4368 (Simon), texa5140 (Cosmin), minh.nguyen@ikm.uni-hannover.de

Implements the 2D Hyperelastic beam models (Neo-Hookean)
"""                                                                                                                                                                                                                                                                                                                          

import os,sys
sys.path.append("")
from dem_hyperelasticity.hertzian_contact_b2p import define_structure_Neu as des
from dem_hyperelasticity.MultiLayerNet import *
from dem_hyperelasticity.hertzian_contact_b2p import PINNmodel_neu as md
from dem_hyperelasticity import Utility as util
from dem_hyperelasticity.hertzian_contact_b2p import config as cf
from dem_hyperelasticity.IntegrationLoss import *
from dem_hyperelasticity.hertzian_contact_b2p.PINNmodel_neu import *
from dem_hyperelasticity.hertzian_contact_b2p import pin_alg as pin
import numpy as np
import time
import torch


mpl.rcParams['figure.dpi'] = 100
# fix random seeds
axes = {'labelsize' : 'large'}
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 17}
legend = {'fontsize': 'medium'}
lines = {'linewidth': 3,
         'markersize' : 7}
mpl.rc('font', **font)
mpl.rc('axes', **axes)
mpl.rc('legend', **legend)
mpl.rc('lines', **lines)


class DeepEnergyMethod:
    # Instance attributes
    def __init__(self, model, numIntType, energy, dim):
        # self.data = data
        self.model = MultiLayerNet(model[0], model[1], model[2])
        self.model = self.model.to(dev)
        #self.intLoss = IntegrationLoss(numIntType, dim)
        self.energy = energy
        # self.post = PostProcessing(energy, dim)
        self.dim = dim
        self.lossArray = []

    def train_model(self, data, neumannBC, dirichletBC, LHD, iteration, learning_rate):
        x = torch.from_numpy(data).float()
        x = x.to(dev)
        x.requires_grad_(True)
        # get tensor inputs and outputs for boundary conditions
        # -------------------------------------------------------------------------------
        #                             Dirichlet BC
        # -------------------------------------------------------------------------------
        dirBC_coordinates = {}  # declare a dictionary
        dirBC_values = {}  # declare a dictionary
        dirBC_penalty = {}
        for i, keyi in enumerate(dirichletBC):
            dirBC_coordinates[i] = torch.from_numpy(dirichletBC[keyi]['coord']).float().to(dev)
            dirBC_coordinates[i].requires_grad_(True)
            dirBC_values[i] = torch.from_numpy(dirichletBC[keyi]['known_value']).float().to(dev)
            dirBC_penalty[i] = torch.tensor(dirichletBC[keyi]['penalty']).float().to(dev)
        # -------------------------------------------------------------------------------
        #                           Neumann BC
        # -------------------------------------------------------------------------------
        neuBC_coordinates = {}  # declare a dictionary
        neuBC_values = {}  # declare a dictionary
        neuBC_penalty = {}
        idx = {}
        for i, keyi in enumerate(neumannBC):
            neuBC_coordinates[i] = torch.from_numpy(neumannBC[keyi]['coord']).float().to(dev)
            neuBC_coordinates[i].requires_grad_(True)
            neuBC_values[i] = torch.from_numpy(neumannBC[keyi]['known_value']).float().to(dev)
            neuBC_penalty[i] = torch.tensor(neumannBC[keyi]['penalty']).float().to(dev)
            idx[i] = neumannBC[keyi]['idx']
        # ----------------------------------------------------------------------------------
        # Minimizing loss function (energy and boundary conditions)
        # ----------------------------------------------------------------------------------
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=learning_rate, max_iter=50) #max_iter_change
        start_time = time.time()
        energy_loss_array = []
        boundary_loss_array = []
        val = neuBC_values
        loss_pre = 10**(-16)
        for t in range(iteration):
            # Zero gradients, perform a backward pass, and update the weights.
            def closure():
                it_time = time.time()
                # ----------------------------------------------------------------------------------
                # Internal Energy
                # ----------------------------------------------------------------------------------
                u_pred = self.getU(x)
                u_pred.double()
                # s_pred.double()
                strong,_ = self.energy.NeoHookean2D(u_pred, x,-1)
                gorven_goal = torch.zeros(*strong.size())
                strong = self.loss_squared_sum(strong,gorven_goal)
                
                # ce = self.energy.CE(u_pred, x)
                # ss = self.loss_squared_sum(ce,s_pred)
                # internal2 = self.intLoss.montecarlo2D(storedEnergy, LHD[0], LHD[1])
                # external2 = torch.zeros(len(neuBC_coordinates))
                bc_t_crit = torch.zeros((len(neuBC_coordinates),2))

                # in_energy = torch.zeros((len(x))-len(neuBC_coordinates)-len(dirBC_coordinates))
                # j=0
                # for i in x:
                #     if i[0] != 0 or i[0] !=4:
                #         n_u_pred = self.getU(i.unsqueeze(0))
                #         storedEnergy,_ = self.energy.getStoredEnergy(n_u_pred,i)
                #         gorven_goal = torch.zeros(*storedEnergy.size())
                #         in_energy[j] = self.loss_squared_sum(storedEnergy,gorven_goal)

                for i, vali in enumerate(neuBC_coordinates):
                    
                    neu_u_pred = self.getU(neuBC_coordinates[i])
                    
                    
                    # if i == 3:
                    #     X,c,r = pin.set_pinball(dom+u_pred.cpu().detach().numpy(),idx[i])
                    #     # wall = des.set_wall()
                    #     for j in range(X.shape[0]):
                    #         # g = (c[j,1]+neu_u_pred.cpu().detach().numpy()[j,1])-r[j]
                    #         g = (c[j,1])-r[j]
                    #         if g<0:
                    #             A = np.pi*r[j]*np.arccos((r[j]+g)/(r[j]))
                    #             p2 = cf.B/(np.pi*r[j]**2)*(A**2)
                    #             pf = -1*p2*g
                                
                    #             neuBC_values[i][j][1]=torch.add(val[i][j,1],torch.from_numpy(pf).float().to(dev))
                                # neuBC_values[i][j+1][1]=torch.add(val[i][j+1,1],torch.from_numpy(pf/2).float().to(dev))

                            # g=-7.84*10**(-5)
                            # A = np.pi*r[j]*np.arccos((r[j]+g)/(r[j]))
                            # p2 = cf.B/(np.pi*r[j]**2)*(A**2)
                            # pf = 100*-p2*g
                            # neuBC_values[i][j][1]=torch.add(val[i][j,1],torch.from_numpy(pf/2).float().to(dev))
                            # neuBC_values[i][j+1][1]=torch.add(val[i][j+1,1],torch.from_numpy(pf/2).float().to(dev))
                    _, neu = self.energy.NeoHookean2D(neu_u_pred, neuBC_coordinates[i],i)
                    if i!=1:
                        bc_t_crit[i] = self.loss_squared_sum_2(neu, neuBC_values[i])
                    else:
                        
                        g = cf.R-torch.sqrt(cf.R**2-(neuBC_coordinates[i][:,0])**2)-cf.d-(neuBC_coordinates[i][:,1]+neu_u_pred[:,1])
                        g = (1-torch.sign(g))*g/2
                        # g= neu_u_pred[:,1]
                        # mu = cf.E #/ (2 * (1 + cf.nu))
                        # p = neu[:,1]/cf.E#*mu
                        p = g*1000
                        
                        # p = md.EnergyModel.normalize(p)
                        # g = md.EnergyModel.normalize(g)
                        # lkkt methode
                        # lkkt_u = self.loss_squared_sum_3((1-torch.sign(g))*g/2, 0)
                        # lkkt_n = self.loss_squared_sum_3((1+torch.sign(p))*p/2, 0)
                        # lkkt_un = self.loss_squared_sum_3(p*g, 0)

                        lkkt_t = self.loss_squared_sum_3(neu[:,0], 0)

                        # fischer Burmester
                        # lkkt = self.loss_squared_sum_3(g-p-torch.sqrt(g**2+p**2), 0)
                        pen_loss = self.loss_squared_sum_3(neu[:,1], p)
                    # gorven_goal = torch.empty(*storedEnergy.size())
                    # storedEnergy = self.loss_squared_sum(storedEnergy,gorven_goal)
                    # fext = torch.bmm((neu_u_pred + neuBC_coordinates[i]).unsqueeze(1), neuBC_values[i].unsqueeze(2))
                    # external2[i] = self.intLoss.montecarlo1D(fext, LHD[1])
                bc_u_crit = torch.zeros((len(dirBC_coordinates),2))
                for i, vali in enumerate(dirBC_coordinates):
                    dir_u_pred = self.getU(dirBC_coordinates[i])
                    bc_u_crit[i] = self.loss_squared_sum_2(dir_u_pred, dirBC_values[i])
                
                dir_loss = bc_u_crit[0][0]
                dir_loss_1 = torch.sum(bc_u_crit[1])
                neu_loss_1 = bc_t_crit[0][1]
                neu_loss_2 = torch.sum(bc_t_crit[2])
                neu_loss_3= torch.sum(bc_t_crit[3])
                # neu_loss_4 = torch.sum(bc_t_crit[3])
                neu_loss = neu_loss_1+neu_loss_2+neu_loss_3
                # storedEnergy = 
                energy_loss = strong  # storedEnergy get problem
                # loss = strong+ss+100*dir_loss+0.1*neu_loss_1+0.1*neu_loss_2+0.1*neu_loss_3+10*lkkt_u+100*lkkt_n+10*lkkt_un+0.1*lkkt_t # u is relate to bc_t #normalization
                loss = strong+1000*dir_loss+1000*dir_loss_1+0.1*neu_loss_1+0.1*neu_loss_2+0.1*neu_loss_3+0.1*pen_loss+0.1*lkkt_t# 0.5 best
                # loss = loss*1000
                optimizer.zero_grad()
                loss.backward()
                print('Iter: %d Loss: %.6e strong_form: %.6e dir_bc: %.6e neu_bc: %.6e  Time: %.3e'
                      % (t + 1, loss.item(), strong.item(),strong.item(),strong.item(), time.time() - it_time))
                energy_loss_array.append(energy_loss.data)
                boundary_loss_array.append(dir_loss.data)
                self.lossArray.append(loss.data)
                return loss
            optimizer.step(closure)
            loss = closure()
            # if abs((loss-loss_pre)/loss_pre)<10**(-7):
                
            #     break
            # else:
                # loss_pre = loss
        elapsed = time.time() - start_time
        print('Training time: %.4f' % elapsed)

    def getU(self, x):
        u = self.model(x)
        Ux = u[:, 0]*x[:, 0]*(1+x[:,1])/cf.E
        Uy = u[:, 1]*(1+x[:,1])/cf.E
        # sigxx = u[:,2]*(1-x[:, 0])
        # sigyy = u[:,3]
        # sigxy = u[:,4]*(1-x[:,0])*x[:,0]

        Ux = Ux.reshape(Ux.shape[0], 1)
        Uy = Uy.reshape(Uy.shape[0], 1)
        u_pred = torch.cat((Ux, Uy), -1)

        # Sxx = sigxx.reshape(sigxx.shape[0], 1)
        # Syy = sigyy.reshape(sigyy.shape[0], 1)
        # Sxy = sigxy.reshape(sigxy.shape[0], 1)
        # s_pred = torch.cat((Sxx,Syy,Sxy), -1)
        return u_pred #,s_pred

    # --------------------------------------------------------------------------------
    # Evaluate model to obtain:
    # 1. U - Displacement
    # 2. E - Green Lagrange Strain
    # 3. S - 2nd Piola Kirchhoff Stress
    # 4. F - Deformation Gradient
    # Date implement: 20.06.2019
    # --------------------------------------------------------------------------------
    def evaluate_model(self, x, y, z):
        energy_type = self.energy.type
        mu = self.energy.mu
        lmbda = self.energy.lam
        dim = self.dim

        Nx = len(x)
        Ny = len(y)
        xGrid, yGrid = np.meshgrid(x, y)
        x1D = xGrid.flatten()
        y1D = yGrid.flatten()
        xy = np.concatenate((np.array([x1D]).T, np.array([y1D]).T), axis=-1)
        xy_tensor = torch.from_numpy(xy).float()
        xy_tensor = xy_tensor.to(dev)
        xy_tensor.requires_grad_(True)
        u_pred_torch = self.getU(xy_tensor)

        duxdxy = grad(u_pred_torch[:, 0].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                       create_graph=True, retain_graph=True)[0]
        duydxy = grad(u_pred_torch[:, 1].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                       create_graph=True, retain_graph=True)[0]
        
        epsxx = duxdxy[:,0].unsqueeze(1)
        epsxy = (0.5*duxdxy[:,1]+0.5*duydxy[:,0]).unsqueeze(1)
        epsyx = epsxy
        epsyy = duydxy[:,1].unsqueeze(1)

        treps = epsxx+epsyy
        sigmaxx = lmbda*treps+2*mu*epsxx
        sigmaxy = 2*mu*epsxy
        sigmayx = 2*mu*epsyx
        sigmayy = lmbda*treps+2*mu*epsyy
        if energy_type == 'neohookean' and dim == 2:
            pass
        else:
            print("This energy model will be implemented later !!!")
            exit()

        u_pred = u_pred_torch.detach().cpu().numpy()
        # s_pred = s_pred_torch.detach().cpu().numpy()

        # a = np.sqrt(cf.R*cf.d)
        # F = 4/3*cf.E*a*cf.d/(1-cf.nu**2)
        # P0=3*F/(2*np.pi*a**2)
        # print(max(u_pred[:, 1]))
        Px=sigmaxx.reshape(Ny, Nx,1).detach().cpu().numpy()
        Py=sigmayy.reshape(Ny, Nx,1).detach().cpu().numpy()
        Pz = np.zeros([Nx, Ny, 1])
        # Px = Px[:, 0].reshape(Ny, Nx,1)
        # Py = s_pred[:, 1].reshape(Ny, Nx,1)
        # Pxy = s_pred[:, 2].reshape(Ny, Nx,1)
        
        surUx = u_pred[:, 0].reshape(Ny, Nx,1)
        surUy = u_pred[:, 1].reshape(Ny, Nx,1)
        surUz = np.zeros([Nx, Ny, 1])

        epsilon = 1e-4
        # Eu =  (np.abs((u_pred[:, 1]-((-0.1)/cf.E*(1-cf.nu**2))*xy[:,1])/((-0.1)/(cf.E*(1-cf.nu**2))*xy[:,1]-epsilon))).reshape(Ny, Nx,1)
        e=1.5*10**4
        v=0.3
        Ebar, nubar, I = e/(1-v**2), v/(1-v), (12**3)/12
        f=-0.05*e
        H=12
        L=48
        Eu =  ((f/(6*Ebar*I))*(3*nubar*pow(xy[:,1],2)*(L-xy[:,0])+(4+5*nubar)*pow(H,2)*xy[:,0]/4 + (3*L-xy[:,0])*pow(xy[:,0],2))).reshape(Ny, Nx,1)
        Eyy = Pz
        Exy = Pz
        # Eu =  (np.abs((u_pred[:, 1]-((-0.1)/cf.E*(1-cf.nu**2))*xy[:,1])/(u_pred[:, 1]+(-0.1)/(cf.E*(1-cf.nu**2))*xy[:,1]))).reshape(Ny, Nx,1)
        # Eyy = (np.abs((s_pred[:, 1]+0.1)/(s_pred[:, 1]-0.1))).reshape(Ny, Nx,1)
        # Exy = (np.abs(s_pred[:, 2]/(s_pred[:, 2]))).reshape(Ny, Nx,1)

        # Eu =  np.arctan(np.abs((surUy-((-0.1)/cf.E*(1-cf.nu**2))*xy[:,1])/((-0.1)/(cf.E*(1-cf.nu**2))*xy[:,1]+epsilon))).reshape(Ny, Nx,1)
        # Eyy = np.arctan(np.abs((Py+0.1)/(-1+epsilon))).reshape(Ny, Nx,1)
        # Exy = np.arctan(np.abs(Pxy/(0+epsilon))).reshape(Ny, Nx,1)

        U = (np.float64(surUx), np.float64(surUy),np.float64(surUz))
        P = (np.float64(Px), (-np.float64(Py)),(np.float64(Pz)))  #NTC      
        Er = (np.float64(Eu), (np.float64(Eyy)),(np.float64(Exy)))
            # print('max F is %.5f: '%(F))
        # print('max P is %.5f: '%(max(abs(sigmayy))))
        
        return U,P,Er
    # --------------------------------------------------------------------------------
    # method: loss sum for the energy part
    # --------------------------------------------------------------------------------
    @staticmethod
    def loss_sum(tinput):
        return torch.sum(tinput) / tinput.data.nelement()

    # --------------------------------------------------------------------------------
    # purpose: loss square sum for the boundary part
    # --------------------------------------------------------------------------------
    @staticmethod
    def loss_squared_sum(tinput, target):
        row, column = tinput.shape
        loss = 0
        for j in range(column):
            loss += torch.sum((tinput[:, j] - target[:, j]) ** 2) / tinput[:, j].data.nelement()
        return loss
    @staticmethod
    def loss_squared_sum_3(tinput, target):
        row = tinput.shape
        # loss = 0
        
        loss = torch.sum((tinput - target) ** 2)/tinput.data.nelement()
        return loss

    def mean(tinput,target):
        row, column = tinput.shape
        loss = 0
        for j in range(column):
            loss += torch.sum((tinput[:, j] - target[:, j]) ** 2) 
        return loss/row

    def printLoss(self):
        self.loss
    
    @staticmethod
    def loss_squared_sum_2(tinput, target):
        row, column = tinput.shape
        loss = torch.zeros((column))
        for j in range(column):
            loss[j] = torch.sum((tinput[:, j] - target[:, j]) ** 2) / tinput[:, j].data.nelement()
        return loss


if __name__ == '__main__':
    # ----------------------------------------------------------------------
    #                   STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
    # ----------------------------------------------------------------------
    dom, boundary_neumann, boundary_dirichlet = des.setup_domain()
    x, y, datatest = des.get_datatest()
    # ----------------------------------------------------------------------
    #                   STEP 2: SETUP MODEL
    # ----------------------------------------------------------------------
    mat = md.EnergyModel('neohookean', 2, cf.E, cf.nu)
    dem = DeepEnergyMethod([cf.D_in, cf.H, cf.D_out], 'montecarlo', mat, 2)
    # ----------------------------------------------------------------------
    #                   STEP 3: TRAINING MODEL
    # ----------------------------------------------------------------------
    start_time = time.time()
    shape = [cf.Nx, cf.Ny]
    dxdy = [cf.hx, cf.hy]
    cf.iteration = 500 #550 exp

    # cf.filename_out = "dem_hyperelasticity/hertzian_contact_square/output/hert_20_itr_sb"
    cf.filename_out = "dem_hyperelasticity/hertzian_contact_b2p/output/2rd-te"
    dem.train_model(dom, boundary_neumann, boundary_dirichlet, [cf.Length, cf.Height, cf.Depth], cf.iteration, cf.lr)
    end_time = time.time() - start_time
    print("End time: %.5f" % end_time)
    z = np.array([0.])
    U, P,E = dem.evaluate_model(x, y, z)
    util.write_vtk_v2(cf.filename_out, x, y, z, U, P,E)
    surUx, surUy, surUz = U
    # L2norm =                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           util.getL2norm2D(surUx, surUy, cf.Nx, cf.Ny, cf.hx, cf.hy)
    # H10norm = util.getH10norm2D(F11, F12, F21, F22, cf.Nx, cf.Ny, cf.hx, cf.hy)
    # print("L2 norm = %.10f" % L2norm)
    # print("H10 norm = %.10f" % H10norm)
    print('Loss convergence')

    stop = int(input("input 1 to continue: "))
    ##################################################################################################################################
    start_time = time.time()
    shape = [cf.Nx, cf.Ny]
    dxdy = [cf.hx, cf.hy]
    cf.iteration = 290 #550 exp

    cf.filename_out = "dem_hyperelasticity/hertzian_contact_square/output/hert_1500itr"
    dem.train_model(dom, boundary_neumann, boundary_dirichlet, [cf.Length, cf.Height, cf.Depth], cf.iteration, cf.lr)
    end_time = time.time() - start_time
    print("End time: %.5f" % end_time)
    z = np.array([0.])
    U, P,E = dem.evaluate_model(x, y, z)
    util.write_vtk_v2(cf.filename_out, x, y, z, U, P,E)
    surUx, surUy, surUz = U
    # L2norm =                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           util.getL2norm2D(surUx, surUy, cf.Nx, cf.Ny, cf.hx, cf.hy)
    # H10norm = util.getH10norm2D(F11, F12, F21, F22, cf.Nx, cf.Ny, cf.hx, cf.hy)
    # print("L2 norm = %.10f" % L2norm)
    # print("H10 norm = %.10f" % H10norm)
    print('Loss convergence')

    
    stop = int(input("input 1 to continue: "))
    # ###################################################################################################################################
    # start_time = time.time()
    # shape = [cf.Nx, cf.Ny]
    # dxdy = [cf.hx, cf.hy]
    # cf.iteration = 500 #550 exp

    # cf.filename_out = "C:/Users/530/Desktop/code/dem_hyperelasticity/Beam2D/output/dem/100_0.1_0.1_500itr"
    # dem.train_model(dom, boundary_neumann, boundary_dirichlet, [cf.Length, cf.Height, cf.Depth], cf.iteration, cf.lr)
    # end_time = time.time() - start_time
    # print("End time: %.5f" % end_time)
    # z = np.array([0.])
    # U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, F11, F12, F21, F22 = dem.evaluate_model(x, y, z)
    # util.write_vtk_v2(cf.filename_out, x, y, z, U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises)
    # surUx, surUy, surUz = U
    # L2norm =                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           util.getL2norm2D(surUx, surUy, cf.Nx, cf.Ny, cf.hx, cf.hy)
    # H10norm = util.getH10norm2D(F11, F12, F21, F22, cf.Nx, cf.Ny, cf.hx, cf.hy)
    # print("L2 norm = %.10f" % L2norm)
    # print("H10 norm = %.10f" % H10norm)
    # print('Loss convergence')