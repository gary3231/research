"""
@author: sfmt4368 (Simon), texa5140 (Cosmin), minh.nguyen@ikm.uni-hannover.de

Implements the 2D Hyperelastic beam models (Neo-Hookean)
"""
import os,sys
sys.path.append("")
from dem_hyperelasticity.DEM_contact import define_structure as des
from dem_hyperelasticity.MultiLayerNet import *
from dem_hyperelasticity.DEM_contact import EnergyModel_mix as md
from dem_hyperelasticity import Utility as util
from dem_hyperelasticity.DEM_contact import config as cf
from dem_hyperelasticity.IntegrationLoss import *
from dem_hyperelasticity.DEM_contact.EnergyModel_mix import *
import numpy as np
import time
import torch

mpl.rcParams['figure.dpi'] = 350
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
        self.intLoss = IntegrationLoss(numIntType, dim)
        self.energy = energy
        self.dim = dim
        self.lossArray = []

    def train_model(self, shape, dxdydz, data, neumannBC, dirichletBC, iteration, learning_rate,opt):
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
            dirBC_values[i] = torch.from_numpy(dirichletBC[keyi]['known_value']).float().to(dev)
            dirBC_penalty[i] = torch.tensor(dirichletBC[keyi]['penalty']).float().to(dev)
        # -------------------------------------------------------------------------------
        #                           Neumann BC
        # -------------------------------------------------------------------------------
        neuBC_coordinates = {}  # declare a dictionary
        neuBC_values = {}  # declare a dictionary
        neuBC_penalty = {}
        for i, keyi in enumerate(neumannBC):
            neuBC_coordinates[i] = torch.from_numpy(neumannBC[keyi]['coord']).float().to(dev)
            neuBC_coordinates[i].requires_grad_(True)
            neuBC_values[i] = torch.from_numpy(neumannBC[keyi]['known_value']).float().to(dev)
            neuBC_penalty[i] = torch.tensor(neumannBC[keyi]['penalty']).float().to(dev)
        # ----------------------------------------------------------------------------------
        # Minimizing loss function (energy and boundary conditions)
        # ----------------------------------------------------------------------------------
        # optimizer = torch.optim.LBFGS(self.model.parameters(), lr=learning_rate, max_iter=20)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        if opt == 0:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.LBFGS(self.model.parameters(), lr=learning_rate)
        start_time = time.time()
        energy_loss_array = []
        boundary_loss_array = []
        loss_pre = 10**(-16)
        for t in range(iteration):
            # Zero gradients, perform a backward pass, and update the weights.
            def closure():
                it_time = time.time()
                # ----------------------------------------------------------------------------------
                # Internal Energy
                # ----------------------------------------------------------------------------------
                u_pred,s_pred = self.getU(x)
                u_pred.double()
                s_pred.double()
                storedEnergy = self.energy.getStoredEnergy(u_pred, x)
                internal2 = self.intLoss.lossInternalEnergy(storedEnergy, dx=dxdydz[0], dy=dxdydz[1], shape=shape)
                strong_u = self.energy.strong_u(u_pred,x)
                strong_s = self.energy.strong_s(s_pred,x)
                gorven_goal = torch.zeros(*strong_u.size())
                strong_u = torch.sum(self.loss_squared_sum_2(strong_u, gorven_goal))
                # gorven_goal = torch.zeros(*strong_u.size())
                strong_s = torch.sum(self.loss_squared_sum_2(strong_s, gorven_goal))
                # print(sf)
              
                ce = self.energy.CE(u_pred, x)
                ss = self.loss_squared_sum(ce,s_pred)
                bc_t_crit = torch.zeros((len(neuBC_coordinates),2))
                external2 = torch.zeros(len(neuBC_coordinates))
                for i, vali in enumerate(neuBC_coordinates):
                    
                    neu_u_pred,neu_s_pred = self.getU(neuBC_coordinates[i])
                    neu = self.energy.NBC(neu_s_pred, neuBC_coordinates[i],i)
                    if (i ==1) or (i==2):
                        fext = torch.bmm((neu_u_pred + neuBC_coordinates[i]).unsqueeze(1), neuBC_values[i].unsqueeze(2))
                        external2[i] = self.intLoss.lossExternalEnergy(fext, dx=dxdydz[1])
                        bc_t_crit[i] = self.loss_squared_sum_2(neu, neuBC_values[i])
                        
                    else:
                        fext = torch.bmm((neu_u_pred + neuBC_coordinates[i]).unsqueeze(1), neuBC_values[i].unsqueeze(2))
                        # external2[i] = self.intLoss.lossExternalEnergy(fext, dx=dxdydz[0])
                        bc_t_crit[i] = self.loss_squared_sum_2(neu, neuBC_values[i])
                        # g= neu_u_pred[:,1]#+neuBC_coordinates[i][:,1]
                        # p= -neu[:,1]
                        # pn = torch.zeros(neu_u_pred.shape)
                        # pn[:,1] = p
                        
                        # lkkt methode
                        # lkkt_u = self.loss_squared_sum_3((1-torch.sign(g))*g/2, 0)
                        # lkkt_n = self.loss_squared_sum_3((1+torch.sign(p))*p/2, 0)
                        # lkkt_un = self.loss_squared_sum_3(p*g, 0)

                        # lkkt_t = self.loss_squared_sum_3(-neu[:,0], 0)

                        # # fischer Burmester
                        # lkkt = self.loss_squared_sum_3(g-p-torch.sqrt(g**2+p**2), 0)
                    
                        # kkt_un = self.intLoss.lossExternalEnergy(kkt_un, dx=dxdydz[0])
                        # kkt_f = (1+torch.sign(kkt_f))*kkt_f/2
                        # if all(p>=0):
                        #     kkt_f = self.intLoss.lossExternalEnergy(fext, dx=dxdydz[1])
                            
                        # else:
                        #     kkt_f = torch.tensor(0)
                        # lkkt_t = self.loss_squared_sum_3(-neu[:,0], 0)
                        # lkkt_ut = self.loss_squared_sum_3(kkt_f*g, 0)
                    # else:
                    #     fext = torch.bmm((neu_u_pred + neuBC_coordinates[i]).unsqueeze(1), neuBC_values[i].unsqueeze(2))
                    #     external2[i] = self.intLoss.lossExternalEnergy(fext, dy=dxdydz[0],dx=dxdydz[1])

                bc_u_crit = torch.zeros((len(dirBC_coordinates),2))
                for i, vali in enumerate(dirBC_coordinates):
                    dir_u_pred,dir_s_pred = self.getU(dirBC_coordinates[i])
                    bc_u_crit[i] = self.loss_squared_sum_2(dir_u_pred, dirBC_values[i])
                energy_loss = internal2 - torch.sum(external2)
                boundary_loss =torch.sum(bc_u_crit[0])#bc_u_crit[0][0]
                neu_loss_1 = torch.sum(bc_t_crit[0])
                neu_loss_2 = torch.sum(bc_t_crit[1])
                neu_loss_3= torch.sum(bc_t_crit[2])
                neu_loss = neu_loss_1+neu_loss_2+neu_loss_3+torch.sum(bc_t_crit[3])
                loss = strong_u+strong_s+1*ss+1*(neu_loss_1+neu_loss_2+neu_loss_3+torch.sum(bc_t_crit[3]))+1*boundary_loss   #+0.1*lkkt_t+0.1*lkkt_n+100*lkkt_un
                optimizer.zero_grad()
                loss.backward()
                print('Iter: %d Loss: %.6e Energy: %.6e   lkkt_un: %.6e kkt_f: %.6eTime: %.3e'
                      % (t + 1, loss.item(), energy_loss.item(),ss.item(),neu_loss.item() ,time.time() - it_time))
                energy_loss_array.append(energy_loss.data)
                boundary_loss_array.append(boundary_loss.data)
                self.lossArray.append(loss.data)
                return loss
            optimizer.step(closure)
            loss = closure()
            # if abs((loss-loss_pre)/loss_pre)<10**(-20):
                
            #     break
            # else:
            #     loss_pre = loss
        elapsed = time.time() - start_time
        print('Training time: %.4f' % elapsed)

    def getU(self, x):
        u = self.model(x)
        Ux = u[:, 0]*x[:, 0]/cf.E
        Uy = u[:, 1]*x[:, 0]/cf.E
        sigxx = u[:,2]
        sigxy = u[:,3]*(1-x[:, 1])*(x[:, 1])
        sigyx = u[:,4]
        sigyy = u[:,5]*(1-x[:, 1])*(x[:, 1])

        Ux = Ux.reshape(Ux.shape[0], 1)
        Uy = Uy.reshape(Uy.shape[0], 1)
        u_pred = torch.cat((Ux, Uy), -1)

        Sxx = sigxx.reshape(sigxx.shape[0], 1)
        Sxy = sigyy.reshape(sigxy.shape[0], 1)
        Syx = sigxy.reshape(sigyx.shape[0], 1)
        Syy = sigxy.reshape(sigyy.shape[0], 1)
        s_pred = torch.cat((Sxx,Sxy,Syx,Syy), -1)
        return u_pred ,s_pred
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
        u_pred_torch, s_pred_torch = self.getU(xy_tensor)

        duxdxy = grad(u_pred_torch[:, 0].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxy = grad(u_pred_torch[:, 1].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        Fxx = duxdxy[:, 0].unsqueeze(1) + 1
        Fxy = duxdxy[:, 1].unsqueeze(1) + 0
        Fyx = duydxy[:, 0].unsqueeze(1) + 0
        Fyy = duydxy[:, 1].unsqueeze(1) + 1
        # F = torch.cat((Fxx,Fxy,Fyx,Fyy),1)
        # F.requires_grad_(True)
        detF = Fxx * Fyy - Fxy * Fyx
        invF11 = Fyy / detF
        invF22 = Fxx / detF
        invF12 = -Fxy / detF
        invF21 = -Fyx / detF
        sigmaxx = mu * Fxx + (lmbda * torch.log(detF) - mu) * invF11
        sigmaxy = mu * Fxy + (lmbda * torch.log(detF) - mu) * invF21
        sigmayx = mu * Fyx + (lmbda * torch.log(detF) - mu) * invF12
        sigmayy = mu * Fyy + (lmbda * torch.log(detF) - mu) * invF22
        if energy_type == 'neohookean' and dim == 2:
            pass
        else:
            print("This energy model will be implemented later !!!")
            exit()

        u_pred = u_pred_torch.detach().cpu().numpy()
        s_pred = s_pred_torch.detach().cpu().numpy()

        Pxx = s_pred[:, 0].reshape(Ny, Nx,1)
        Pxy = s_pred[:, 1].reshape(Ny, Nx,1)
        Pyx = s_pred[:, 2].reshape(Ny, Nx,1)
        Pyy = s_pred[:, 3].reshape(Ny, Nx,1)
        # Pz = np.zeros([Nx, Ny, 1])
        surUx = u_pred[:, 0].reshape(Ny, Nx,1)
        surUy = u_pred[:, 1].reshape(Ny, Nx,1)
        surUz = np.zeros([Nx, Ny, 1])

        Pxx_u=sigmaxx.reshape(Ny, Nx,1).detach().cpu().numpy()
        Pxy_u=sigmaxy.reshape(Ny, Nx,1).detach().cpu().numpy()
        Pyx_u=sigmayx.reshape(Ny, Nx,1).detach().cpu().numpy()
        Pyy_u=sigmayy.reshape(Ny, Nx,1).detach().cpu().numpy()

        epsilon = 1e-4
        # Eu =  np.abs(u_pred[:, 1]-((-0.1)/cf.E*(1-cf.nu**2))*xy[:,1]).reshape(Ny, Nx,1)
        # Eyy = np.abs(sigmayy.detach().cpu().numpy()+0.1).reshape(Ny, Nx,1)
        # Exy = np.abs(sigmaxy.detach().cpu().numpy()).reshape(Ny, Nx,1)
        # Eu =  (np.abs((u_pred[:, 1]-((-0.1)/cf.E*(1-cf.nu**2))*xy[:,1])/(u_pred[:, 1]+(-0.1)/(cf.E*(1-cf.nu**2))*xy[:,1]))).reshape(Ny, Nx,1)
        # Eyy = (np.abs((s_pred[:, 1]+0.1)/(s_pred[:, 1]-0.1))).reshape(Ny, Nx,1)
        # Exy = (np.abs(s_pred[:, 2]/(s_pred[:, 2]))).reshape(Ny, Nx,1)

        # Eu =  np.arctan(np.abs((surUy-((-0.1)/cf.E*(1-cf.nu**2))*xy[:,1])/((-0.1)/(cf.E*(1-cf.nu**2))*xy[:,1]+epsilon))).reshape(Ny, Nx,1)
        # Eyy = np.arctan(np.abs((Py+0.1)/(-1+epsilon))).reshape(Ny, Nx,1)
        # Exy = np.arctan(np.abs(Pxy/(0+epsilon))).reshape(Ny, Nx,1)

        U = (np.float64(surUx), np.float64(surUy),np.float64(surUz))
        P = (np.float64(Pxx), (np.float64(Pxy)),(np.float64(Pyx)),(np.float64(Pyy)))  #NTC      
        Er = (np.float64(Pxx_u), (np.float64(Pxy_u)),(np.float64(Pyx_u)),(np.float64(Pyy_u)))
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
    def loss_squared_sum_2(tinput, target):
        row, column = tinput.shape
        loss = torch.zeros((column))
        for j in range(column):
            loss[j] = torch.sum((tinput[:, j] - target[:, j]) ** 2) / tinput[:, j].data.nelement()
        return loss
    
    @staticmethod
    def loss_squared_sum_3(tinput, target):
        row = tinput.shape
        # loss = 0
        
        loss = torch.sum((tinput - target) ** 2)/tinput.data.nelement()
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
    dem = DeepEnergyMethod([cf.D_in, cf.H, cf.D_out], 'trapezoidal', mat, 2)
    # ----------------------------------------------------------------------
    #                   STEP 3: TRAINING MODEL
    # ----------------------------------------------------------------------
    start_time = time.time()
    shape = [cf.Nx, cf.Ny]
    dxdy = [cf.hx, cf.hy]
    cf.iteration = 10000
    cf.filename_out = "dem_hyperelasticity/DEM_contact/output/MDEM_ff"
    dem.train_model(shape, dxdy, dom, boundary_neumann, boundary_dirichlet,  cf.iteration, cf.lr,0)
    dem.train_model(shape, dxdy, dom, boundary_neumann, boundary_dirichlet,  100, 0.1,1)
    end_time = time.time() - start_time
    print("End time: %.5f" % end_time)
    z = np.array([0.])
    U, P,E = dem.evaluate_model(x, y, z)
    util.write_vtk_v2(cf.filename_out, x, y, z, U, P,E)
    # surUx, surUy, surUz = U
    # L2norm = util.getL2norm2D(surUx, surUy, cf.Nx, cf.Ny, cf.hx, cf.hy)
    # H10norm = util.getH10norm2D(F11, F12, F21, F22, cf.Nx, cf.Ny, cf.hx, cf.hy)
    # print("L2 norm = %.10f" % L2norm)
    # print("H10 norm = %.10f" % H10norm)
    # start_time = time.time()
    shape = [cf.Nx, cf.Ny]
    dxdy = [cf.hx, cf.hy]
    # cf.iteration = 400 #550 exp
    stop = int(input("input 1 to continue: "))
    cf.filename_out = "dem_hyperelasticity/DEM_contact/output/MDEM_ff+100itr"
    dem.train_model(shape, dxdy, dom, boundary_neumann, boundary_dirichlet,  100, 0.1,1)
    end_time = time.time() - start_time
    print("End time: %.5f" % end_time)
    z = np.array([0.])
    U, P,E = dem.evaluate_model(x, y, z)
    util.write_vtk_v2(cf.filename_out, x, y, z, U, P,E)
    surUx, surUy, surUz = U