"""
@author: sfmt4368 (Simon), texa5140 (Cosmin), minh.nguyen@ikm.uni-hannover.de

Implements the 2D Hyperelastic beam models (Neo-Hookean)
"""                                                                                                                                                                                                                                                                                                                          

import os,sys
sys.path.append("")
from dem_hyperelasticity.two_object_contact import define_structure as des
from dem_hyperelasticity.two_object_contact import define_structure_slab as des_s
from dem_hyperelasticity.MultiLayerNet import *
from dem_hyperelasticity.two_object_contact import PINNmodel as md
from dem_hyperelasticity.two_object_contact import PINNmodel_slab as md_s
from dem_hyperelasticity.two_object_contact  import Utility as util
from dem_hyperelasticity.two_object_contact import config as cf
from dem_hyperelasticity.two_object_contact import config_slab as cf_s
from dem_hyperelasticity.IntegrationLoss import *
# from dem_hyperelasticity.two_object_contact.PINNmodel import *
# from dem_hyperelasticity.two_object_contact.PINNmodel_slab import *
import numpy as np
import time
import torch
import sys
from dem_hyperelasticity.two_object_contact.search import *


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
    def __init__(self, model, model_s, numIntType, energy, energy_s, dim):
        # self.data = data
        self.model = MultiLayerNet(model[0], model[1], model[2])
        self.model = self.model.to(dev)

        self.model_s = MultiLayerNet(model_s[0], model_s[1], model_s[2])
        self.model_s = self.model_s.to(dev)

        #self.intLoss = IntegrationLoss(numIntType, dim)
        self.energy = energy
        self.energy_s = energy_s
        # self.post = PostProcessing(energy, dim)
        self.dim = dim
        self.lossArray = []

    def train_model(self, data, data_s, neumannBC, neumannBC_s, dirichletBC, dirichletBC_s, iteration, learning_rate):
        x = torch.from_numpy(data).float()
        x_s = torch.from_numpy(data_s).float()

        x = x.to(dev)
        x_s = x_s.to(dev)
        x.requires_grad_(True)
        x_s.requires_grad_(True)
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

        dirBC_coordinates_s = {}  # declare a dictionary
        dirBC_values_s = {}  # declare a dictionary
        for i, keyi in enumerate(dirichletBC_s):
            dirBC_coordinates_s[i] = torch.from_numpy(dirichletBC_s[keyi]['coord']).float().to(dev)
            dirBC_coordinates_s[i].requires_grad_(True)
            dirBC_values_s[i] = torch.from_numpy(dirichletBC_s[keyi]['known_value']).float().to(dev)
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

        neuBC_coordinates_s = {}  # declare a dictionary
        neuBC_values_s = {}  # declare a dictionary

        for i, keyi in enumerate(neumannBC_s):
            neuBC_coordinates_s[i] = torch.from_numpy(neumannBC_s[keyi]['coord']).float().to(dev)
            neuBC_coordinates_s[i].requires_grad_(True)
            neuBC_values_s[i] = torch.from_numpy(neumannBC_s[keyi]['known_value']).float().to(dev)

        c_to_slab = search(neuBC_coordinates[1], neuBC_coordinates_s[2])
        c_to_slab.requires_grad_(True)

        neuBC_coordinates_s[3] = c_to_slab
        uni = torch.unique(c_to_slab, dim=0)
        mask = ~torch.any(torch.all(neuBC_coordinates_s[2].unsqueeze(1) == uni, dim=2), dim=1)
        neuBC_coordinates_s[2] = neuBC_coordinates_s[2][mask]
        neuBC_values_s[2] = torch.ones(neuBC_coordinates_s[2].shape) * torch.tensor([0, 0])
        # ----------------------------------------------------------------------------------
        # Minimizing loss function (energy and boundary conditions)
        # ----------------------------------------------------------------------------------
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=0.5, max_iter=50)
        optimizer_s = torch.optim.LBFGS(self.model_s.parameters(), lr=0.1, max_iter=50) #max_iter_change
        start_time = time.time()
        energy_loss_array = []
        boundary_loss_array = []
        val = neuBC_values

        loss_pre = 10**(-20)

        for t in range(iteration):
            # Zero gradients, perform a backward pass, and update the weights.
            def closure():
                it_time = time.time()
                # ----------------------------------------------------------------------------------
                # Internal Energy
                # ----------------------------------------------------------------------------------
                u_pred, s_pred = self.getU(x)
                u_pred.double()
                s_pred.double()
                strong = self.energy.strong(s_pred, x)
                gorven_goal = torch.zeros(*strong.size())
                strong = self.loss_squared_sum(strong,gorven_goal)
                
                ce = self.energy.CE(u_pred, x)
                ss = self.loss_squared_sum(ce,s_pred)

                # internal2 = self.intLoss.montecarlo2D(storedEnergy, LHD[0], LHD[1])
                # external2 = torch.zeros(len(neuBC_coordinates))

                bc_t_crit = torch.zeros((len(neuBC_coordinates),2))
                for i, vali in enumerate(neuBC_coordinates):
                    neu_u_pred, neu_s_pred = self.getU(neuBC_coordinates[i])
                    neu = self.energy.NBC(neu_s_pred, neuBC_coordinates[i],i)
                    if i!=1:
                        bc_t_crit[i] = self.loss_squared_sum_2(neu, neuBC_values[i])
                    else:
                        neu_u_pred_s, _ = self.getU_slab(neuBC_coordinates_s[3])
                        neu_u_pred_s = neu_u_pred_s.detach()
                        x_c = (neuBC_coordinates_s[3]).detach()
                        # g = (neuBC_coordinates[i][:,1]+neu_u_pred[:,1])
                        l= torch.sqrt(((neuBC_coordinates[i][:,0]) - (x_c[:,0]))**2+((neuBC_coordinates[i][:,1]) - (x_c[:,1]))**2)
                        n = ((x_c) - (neuBC_coordinates[i]))/ l.unsqueeze(1)
                        # n = torch.cat((neuBC_coordinates[i][:, 0].unsqueeze(1), (neuBC_coordinates[i][:, 1]-cf.origin[1]).unsqueeze(1)), 1)
                        g = (n[:,0].unsqueeze(1)*((x_c+neu_u_pred_s) - (neu_u_pred+neuBC_coordinates[i]))[:,0].unsqueeze(1)+n[:,1].unsqueeze(1)*((x_c+neu_u_pred_s) - (neu_u_pred+neuBC_coordinates[i]))[:,1].unsqueeze(1))
                        
                        tt = torch.cat(((n[:,1]).unsqueeze(1), -n[:,0].unsqueeze(1)), 1)
                        trac = torch.cat((n[:,0].unsqueeze(1)*neu_s_pred[:,0].unsqueeze(1)+(n[:,1].unsqueeze(1))*neu_s_pred[:,1].unsqueeze(1), n[:,0].unsqueeze(1)*neu_s_pred[:,2].unsqueeze(1)+(n[:,1].unsqueeze(1))*neu_s_pred[:,3].unsqueeze(1)),1)
                        p = trac[:,0]*n[:,0]+trac[:,1]*n[:,1]
                        # p = neu[:,0]*n[:,0]+neu[:,1]*n[:,1]
                        # ptest =  neu*n
                        # ptest = ptest[:,0]+ptest[:,1]
                        p = p/cf.E
                        # g_m = g.cpu().detach().numpy().min()
                        # lkkt_u = self.loss_squared_sum_3((1-torch.sign(g))*g/2, 0)                        
                        pt = trac[:,0]*tt[:,0]+trac[:,1]*tt[:,1]

                        # fischer Burmester
                        lkkt = self.loss_squared_sum_3(g+p-torch.sqrt(g**2+p**2), 0) 
                        # lkkt = self.loss_squared_sum_3(g, 0) 
                        lkkt_t = self.loss_squared_sum_3(pt, 0)

                bc_u_crit = torch.zeros((len(dirBC_coordinates),2))
                for i, vali in enumerate(dirBC_coordinates):
                    dir_u_pred, _ = self.getU(dirBC_coordinates[i])
                    bc_u_crit[i] = self.loss_squared_sum_2(dir_u_pred, dirBC_values[i])
                
                dir_loss = torch.sum(bc_u_crit[0])
                neu_loss = torch.sum(bc_t_crit[0])
                # storedEnergy = 
                energy_loss = strong  # storedEnergy get problem
                loss = strong+ss+dir_loss+0.1*neu_loss+100*lkkt+lkkt_t
                # loss = ss+strong+100*dir_loss+0.1*neu_loss_1+0.1*neu_loss_2+0.1*neu_loss_3+100*lkkt+0.1*lkkt_t
                # loss = loss*3
                optimizer.zero_grad()
                optimizer_s.zero_grad()
                loss.backward(retain_graph=True)
                # loss.backward()
                print('Iter: %d Loss: %.6e strong_form: %.6e dir_bc: %.3e neu_bc: %.6e kkt: %.6e fs: %.6e Time: %.3e'
                      % (t + 1, loss.item(), strong.item(),dir_loss.item(),neu_loss.item(),lkkt.item(),lkkt_t.item(), time.time() - it_time))
                # energy_loss_array.append(energy_loss.data)
                # boundary_loss_array.append(dir_loss_1.data)
                self.lossArray.append(loss.data)
                return loss
            
            def closure_s():
                it_time = time.time()
                # ----------------------------------------------------------------------------------
                # Internal Energy
                # ----------------------------------------------------------------------------------

                u_pred_s, s_pred_s = self.getU_slab(x_s)
                u_pred_s.double()
                s_pred_s.double()
                strong_s = self.energy_s.strong(s_pred_s, x_s)
                gorven_goal_s = torch.zeros(*strong_s.size())
                strong_s = self.loss_squared_sum(strong_s,gorven_goal_s)

                ce_s = self.energy_s.CE(u_pred_s, x_s)
                ss_s = self.loss_squared_sum(ce_s,s_pred_s)
                # internal2 = self.intLoss.montecarlo2D(storedEnergy, LHD[0], LHD[1])
                # external2 = torch.zeros(len(neuBC_coordinates))
                
                bc_t_crit_s = torch.zeros((len(neuBC_coordinates_s),2))
                for i, vali in enumerate(neuBC_coordinates_s):
                    neu_u_pred_s, neu_s_pred_s = self.getU_slab(neuBC_coordinates_s[i])
                    neu_s = self.energy_s.NBC(neu_s_pred_s, neuBC_coordinates_s[i],i)
                    if i!=3:
                        bc_t_crit_s[i] = self.loss_squared_sum_2(neu_s, neuBC_values_s[i])
                    else:
                        neu_u_pred, neu_s_pred = self.getU(neuBC_coordinates[1])
                        neu_c = self.energy.NBC(neu_s_pred, neuBC_coordinates[1],1)
                        neu_c = neu_c.detach()
                        # lkkt_s = self.loss_squared_sum_2(neu_s, -neu_c)
                        # lkkt_s = torch.sum(lkkt_s)
                        neu_u_pred = neu_u_pred.detach()
                        x_sc = (neuBC_coordinates[1]).detach()
                        # g = (neuBC_coordinates[i][:,1]+neu_u_pred[:,1])
                        l_s= torch.sqrt(((neuBC_coordinates_s[i][:,0]) - (x_sc[:,0]))**2+((neuBC_coordinates_s[i][:,1]) - (x_sc[:,1]))**2)
                        n_s = ((x_sc) - (neuBC_coordinates_s[i]))/ l_s.unsqueeze(1)
                        # n_s = torch.cat((neuBC_coordinates_s[i][:, 1].unsqueeze(1)*0, neuBC_coordinates_s[i][:, 1].unsqueeze(1)), 1)/2
                        trac_s = -torch.cat((n_s[:,0].unsqueeze(1)*neu_s_pred_s[:,0].unsqueeze(1)+(n_s[:,1].unsqueeze(1))*neu_s_pred_s[:,1].unsqueeze(1), n_s[:,0].unsqueeze(1)*neu_s_pred_s[:,2].unsqueeze(1)+(n_s[:,1].unsqueeze(1))*neu_s_pred_s[:,3].unsqueeze(1)),1)
                        n_c = -n_s
                        # n_c = torch.cat((neuBC_coordinates[1][:, 0].unsqueeze(1), (neuBC_coordinates[1][:, 1]-cf.origin[1]).unsqueeze(1)), 1)
                        # trac_c = -torch.cat((n_c[:,0].unsqueeze(1)*neu_s_pred[:,0].unsqueeze(1)+(n_c[:,1].unsqueeze(1))*neu_s_pred[:,1].unsqueeze(1), n_c[:,0].unsqueeze(1)*neu_s_pred[:,2].unsqueeze(1)+(n_c[:,1].unsqueeze(1))*neu_s_pred[:,3].unsqueeze(1)),1)
                        # g_s = (n_s[:,0].unsqueeze(1)*((neu_u_pred+x_sc) - (neu_u_pred_s+neuBC_coordinates_s[i]))[:,0].unsqueeze(1)+n_s[:,1].unsqueeze(1)*((neu_u_pred+x_sc) - (neu_u_pred_s+neuBC_coordinates_s[i]))[:,1].unsqueeze(1))
                        
                        tt_s = torch.cat(((-n_s[:,1]).unsqueeze(1), n_s[:,0].unsqueeze(1)), 1)

                        # p_s = neu_s[:,0]*n_s[:,0]+neu_s[:,1]*n_s[:,1]
                        p_s = trac_s[:,0]*n_s[:,0]+trac_s[:,1]*n_s[:,1]
                        p_s = p_s/cf_s.E

                        p_c = neu_c[:,0]*n_s[:,0]+neu_c[:,1]*n_s[:,1]
                        p_c = (p_c).detach()

                        pt_s = trac_s[:,0]*tt_s[:,0]+trac_s[:,1]*tt_s[:,1]
                        lkkt_s = self.loss_squared_sum_2(neu_s, neu_c)
                        lkkt_s = torch.sum(lkkt_s)
                        # fischer Burmester
                        # lkkt_s = self.loss_squared_sum_3(g_s-p_s-torch.sqrt(g_s**2+p_s**2), 0)
                        lkkt_t_s = self.loss_squared_sum_3(pt_s, 0)
                
                bc_u_crit_s = torch.zeros((len(dirBC_coordinates_s),2))
                for i, vali in enumerate(dirBC_coordinates_s):
                    dir_u_pred_s, _ = self.getU_slab(dirBC_coordinates_s[i])
                    bc_u_crit_s[i] = self.loss_squared_sum_2(dir_u_pred_s, dirBC_values_s[i])
                
                dir_loss_s = torch.sum(bc_u_crit_s[0])
                neu_loss_s = torch.sum(bc_t_crit_s[0])+torch.sum(bc_t_crit_s[1])+torch.sum(bc_t_crit_s[2])
                # storedEnergy = 
                # loss = ss+strong+100*dir_loss+0.1*neu_loss_1+0.1*neu_loss_2+0.1*neu_loss_3+100*lkkt+0.1*lkkt_t
                loss_s = strong_s+ss_s+dir_loss_s+0.1*neu_loss_s+1*lkkt_s+1*lkkt_t_s
                # loss_s = loss_s*3
                optimizer_s.zero_grad()
                # optimizer.zero_grad()
                loss_s.backward(retain_graph=True)
                # loss_s.backward()
                print('        Loss: %.6e strong_form: %.6e dir_bc: %.3e neu_bc: %.6e lkkt: %.6e ss: %.6e Time: %.3e'
                      % ( loss_s.item(), strong_s.item(),dir_loss_s.item(),neu_loss_s.item(), lkkt_s.item(),ss_s.item(), time.time() - it_time))
                # energy_loss_array.append(energy_loss.data)
                # boundary_loss_array.append(dir_loss_1.data)
                self.lossArray.append(loss_s.data)
                return loss_s
            
            optimizer.step(closure)
            if t>250:
                optimizer_s.step(closure_s)

            loss = closure()
            loss_s = closure_s()
            # if abs((loss-loss_pre)/loss_pre)<10**(-7):
                
            #     break
            # else:
            #     loss_pre = loss
        elapsed = time.time() - start_time
        print('Training time: %.4f' % elapsed)

    def getU(self, x):
        u = self.model(x)
        Ux = u[:, 0]*(x[:, 1]-3)/cf.E
        Uy = u[:, 1]*(x[:, 1]-3)/cf.E-0.1
        sigxx = u[:,2]
        sigxy = u[:,3]
        sigyx = u[:,4]
        sigyy = u[:,5]

        Ux = Ux.reshape(Ux.shape[0], 1)
        Uy = Uy.reshape(Uy.shape[0], 1)
        u_pred = torch.cat((Ux, Uy), -1)

        Sxx = sigxx.reshape(sigxx.shape[0], 1)
        Sxy = sigyy.reshape(sigxy.shape[0], 1)
        Syx = sigxy.reshape(sigyx.shape[0], 1)
        Syy = sigxy.reshape(sigyy.shape[0], 1)
        s_pred = torch.cat((Sxx,Sxy,Syx,Syy), -1)

        return u_pred ,s_pred
    
    def getU_slab(self, x):
        u = self.model_s(x)
        Ux = u[:, 0]*(x[:, 1])/cf_s.E
        Uy = u[:, 1]*(x[:, 1])/cf_s.E
        sigxx = u[:,2]*(x[:, 0]-cf_s.x_min)*(x[:, 0]-(cf_s.x_min+cf_s.Length))
        sigxy = u[:,3]
        sigyx = u[:,4]*(x[:, 0]-cf_s.x_min)*(x[:, 0]-(cf_s.x_min+cf_s.Length))
        sigyy = u[:,5]

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
    def evaluate_model(self, x, y, z,dom):
        energy_type = self.energy.type
        mu = self.energy.mu
        lmbda = self.energy.lam
        dim = self.dim

        Nx = len(x)
        Ny = len(y)

        xy_tensor = torch.from_numpy(dom).float()
        xy_tensor = xy_tensor.to(dev)
        xy_tensor.requires_grad_(True)
        u_pred_torch, s_pred_torch = self.getU(xy_tensor)

        duxdxy = grad(u_pred_torch[:, 0].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                       create_graph=True, retain_graph=True)[0]
        duydxy = grad(u_pred_torch[:, 1].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                       create_graph=True, retain_graph=True)[0]
        
        F11 = duxdxy[:, 0].unsqueeze(1) + 1
        F12 = duxdxy[:, 1].unsqueeze(1) + 0
        F21 = duydxy[:, 0].unsqueeze(1) + 0
        F22 = duydxy[:, 1].unsqueeze(1) + 1
        detF = F11 * F22 - F12 * F21
        invF11 = F22 / detF
        invF22 = F11 / detF
        invF12 = -F12 / detF
        invF21 = -F21 / detF
        C11 = F11**2 + F21**2
        C12 = F11*F12 + F21*F22
        C21 = F12*F11 + F22*F21
        C22 = F12**2 + F22**2
        E11 = 0.5 * (C11 - 1)
        E12 = 0.5 * C12
        E21 = 0.5 * C21
        E22 = 0.5 * (C22 - 1)
        if energy_type == 'neohookean' and dim == 2:
            P11 = mu * F11 + (lmbda * torch.log(detF) - mu) * invF11
            P12 = mu * F12 + (lmbda * torch.log(detF) - mu) * invF21
            P21 = mu * F21 + (lmbda * torch.log(detF) - mu) * invF12
            P22 = mu * F22 + (lmbda * torch.log(detF) - mu) * invF22
        else:
            print("This energy model will be implemented later !!!")
            exit()
        

        u_pred = u_pred_torch.detach().cpu().numpy()
        s_pred = s_pred_torch.detach().cpu().numpy()
        F11_pred = F11.detach().cpu().numpy()
        F12_pred = F12.detach().cpu().numpy()
        F21_pred = F21.detach().cpu().numpy()
        F22_pred = F22.detach().cpu().numpy()
        surUx = u_pred[:, 0]
        surUy = u_pred[:, 1]
        surUz = np.zeros([Nx, Ny, 1])
        surE11 = np.zeros([Nx, Ny, 1])
        surE12 = np.zeros([Nx, Ny, 1])
        surE13 = np.zeros([Nx, Ny, 1])
        surE21 = np.zeros([Nx, Ny, 1])
        surE22 = np.zeros([Nx, Ny, 1])
        surE23 = np.zeros([Nx, Ny, 1])
        surE33 = np.zeros([Nx, Ny, 1])
        surS11 = s_pred[:, 0]
        surS12 = s_pred[:, 1]
        surS13 = np.zeros([Nx, Ny, 1])
        surS21 = s_pred[:, 2]
        surS22 = s_pred[:, 3]
        surS23 = np.zeros([Nx, Ny, 1])
        surS33 = np.zeros([Nx, Ny, 1])
        SVonMises = np.float64(np.sqrt(0.5 * ((surS11 - surS22) ** 2 + (surS22) ** 2 + (-surS11) ** 2 + 6 * (surS12 ** 2))))
        U = (np.float64(surUx), np.float64(surUy), np.float64(surUz))
        return U, np.float64(surS11), np.float64(surS12), np.float64(surS13), np.float64(surS22), np.float64(
            surS23), \
            np.float64(surS33), np.float64(surE11), np.float64(surE12), \
            np.float64(surE13), np.float64(surE22), np.float64(surE23), np.float64(surE33), np.float64(
            SVonMises), \
            np.float64(F11_pred), np.float64(F12_pred), np.float64(F21_pred), np.float64(F22_pred)
    
    def evaluate_model_s(self, x, y, z, dom):
        energy_type = self.energy_s.type
        mu = self.energy_s.mu
        lmbda = self.energy_s.lam
        dim = self.dim
        Nx = len(x)
        Ny = len(y)

        xy_tensor = torch.from_numpy(dom).float()
        xy_tensor = xy_tensor.to(dev)
        xy_tensor.requires_grad_(True)
        u_pred_torch, s_pred_torch = self.getU_slab(xy_tensor)

        duxdxy = grad(u_pred_torch[:, 0].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                       create_graph=True, retain_graph=True)[0]
        duydxy = grad(u_pred_torch[:, 1].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                       create_graph=True, retain_graph=True)[0]
        
        F11 = duxdxy[:, 0].unsqueeze(1) + 1
        F12 = duxdxy[:, 1].unsqueeze(1) + 0
        F21 = duydxy[:, 0].unsqueeze(1) + 0
        F22 = duydxy[:, 1].unsqueeze(1) + 1
        detF = F11 * F22 - F12 * F21
        invF11 = F22 / detF
        invF22 = F11 / detF
        invF12 = -F12 / detF
        invF21 = -F21 / detF
        C11 = F11**2 + F21**2
        C12 = F11*F12 + F21*F22
        C21 = F12*F11 + F22*F21
        C22 = F12**2 + F22**2
        E11 = 0.5 * (C11 - 1)
        E12 = 0.5 * C12
        E21 = 0.5 * C21
        E22 = 0.5 * (C22 - 1)
        if energy_type == 'neohookean' and dim == 2:
            P11 = mu * F11 + (lmbda * torch.log(detF) - mu) * invF11
            P12 = mu * F12 + (lmbda * torch.log(detF) - mu) * invF21
            P21 = mu * F21 + (lmbda * torch.log(detF) - mu) * invF12
            P22 = mu * F22 + (lmbda * torch.log(detF) - mu) * invF22
        else:
            print("This energy model will be implemented later !!!")
            exit()
        

        u_pred = u_pred_torch.detach().cpu().numpy()
        s_pred = s_pred_torch.detach().cpu().numpy()
        F11_pred = F11.detach().cpu().numpy()
        F12_pred = F12.detach().cpu().numpy()
        F21_pred = F21.detach().cpu().numpy()
        F22_pred = F22.detach().cpu().numpy()
        surUx = u_pred[:, 0]
        surUy = u_pred[:, 1]
        surUz = np.zeros([Nx, Ny, 1])
        surE11 = np.zeros([Nx, Ny, 1])
        surE12 = np.zeros([Nx, Ny, 1])
        surE13 = np.zeros([Nx, Ny, 1])
        surE21 = np.zeros([Nx, Ny, 1])
        surE22 = np.zeros([Nx, Ny, 1])
        surE23 = np.zeros([Nx, Ny, 1])
        surE33 = np.zeros([Nx, Ny, 1])
        surS11 = s_pred[:, 0]
        surS12 = s_pred[:, 1]
        surS13 = np.zeros([Nx, Ny, 1])
        surS21 = s_pred[:, 2]
        surS22 = s_pred[:, 3]
        surS23 = np.zeros([Nx, Ny, 1])
        surS33 = np.zeros([Nx, Ny, 1])
        SVonMises = np.float64(np.sqrt(0.5 * ((surS11 - surS22) ** 2 + (surS22) ** 2 + (-surS11) ** 2 + 6 * (surS12 ** 2))))
        U = (np.float64(surUx), np.float64(surUy), np.float64(surUz))
        return U, np.float64(surS11), np.float64(surS12), np.float64(surS13), np.float64(surS22), np.float64(
            surS23), \
            np.float64(surS33), np.float64(surE11), np.float64(surE12), \
            np.float64(surE13), np.float64(surE22), np.float64(surE23), np.float64(surE33), np.float64(
            SVonMises), \
            np.float64(F11_pred), np.float64(F12_pred), np.float64(F21_pred), np.float64(F22_pred)
        # print('max P is %.5f: '%(max(abs(sigmayy))))
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
    # x, y, datatest = des.get_datatest()
    x, y, _ = des.get_datatest()

    dom_s, boundary_neumann_s, boundary_dirichlet_s = des_s.setup_domain()
    x_s, y_s, datatest_s = des_s.get_datatest()
    # ----------------------------------------------------------------------
    #                   STEP 2: SETUP MODEL
    # ----------------------------------------------------------------------
    mat = md.EnergyModel('neohookean', 2, cf.E, cf.nu)
    mat_s = md_s.EnergyModel('neohookean', 2, cf_s.E, cf_s.nu)
    dem = DeepEnergyMethod([cf.D_in, cf.H, cf.D_out], [cf_s.D_in, cf_s.H, cf_s.D_out], 'montecarlo', mat, mat_s, 2)

    
    # dem_s = DeepEnergyMethod([cf_s.D_in, cf_s.H, cf_s.D_out], 'montecarlo', mat_s, 2)
    # ----------------------------------------------------------------------
    #                   STEP 3: TRAINING MODEL
    # ----------------------------------------------------------------------
    start_time = time.time()
    shape = [cf.Nr, cf.Na]
    dxdy = [cf.hx, cf.hy]
    cf.iteration = 1000 #550 exp

    cf.filename_out = "dem_hyperelasticity/two_object_contact/output/equal_force_c"
    cf_s.filename_out = "dem_hyperelasticity/two_object_contact/output/equal_force_s"
    # cf.filename_out = "dem_hyperelasticity/hertzian_contact_square/output/hert_20itr_fb"
    dem.train_model(dom, dom_s, boundary_neumann, boundary_neumann_s, boundary_dirichlet, boundary_dirichlet_s, cf.iteration, cf.lr)
    end_time = time.time() - start_time
    print("End time: %.5f" % end_time)
    z = np.array([0.])
    U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, F11, F12, F21, F22 = dem.evaluate_model(x, y, z, dom)
    U_s, S11_s, S12_s, S13_s, S22_s, S23_s, S33_s, E11_s, E12_s, E13_s, E22_s, E23_s, E33_s, SVonMises_s, F11_s, F12_s, F21_s, F22_s = dem.evaluate_model_s(x_s, y_s, z, dom_s)
    util.write_arr2DVTU(cf.filename_out, dom[:,0], dom[:,1], z, U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises)
    util.write_arr2DVTU(cf_s.filename_out, dom_s[:,0], dom_s[:,1], z, U_s, S11_s, S12_s, S13_s, S22_s, S23_s, S33_s, E11_s, E12_s, E13_s, E22_s, E23_s, E33_s, SVonMises_s)
    surUx, surUy, surUz = U
    # L2norm =                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           util.getL2norm2D(surUx, surUy, cf.Nx, cf.Ny, cf.hx, cf.hy)
    # H10norm = util.getH10norm2D(F11, F12, F21, F22, cf.Nx, cf.Ny, cf.hx, cf.hy)
    # print("L2 norm = %.10f" % L2norm)
    # print("H10 norm = %.10f" % H10norm)
    print('Loss convergence')
    ##################################################################################################################################
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