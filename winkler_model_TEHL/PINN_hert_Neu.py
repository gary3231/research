"""
@author: sfmt4368 (Simon), texa5140 (Cosmin), minh.nguyen@ikm.uni-hannover.de

Implements the 2D Hyperelastic beam models (Neo-Hookean)
"""                                                                                                                                                                                                                                                                                                                          

import os,sys
sys.path.append("")
from dem_hyperelasticity.winkler_model_TEHL import define_structure_vis as des_v
from dem_hyperelasticity.winkler_model_TEHL import define_structure_sub as des_s
from dem_hyperelasticity.MultiLayerNet import *
from dem_hyperelasticity.winkler_model_TEHL import PINNmodel_neu as md
from dem_hyperelasticity import Utility as util
from dem_hyperelasticity.winkler_model_TEHL import config as cf
from dem_hyperelasticity.IntegrationLoss import *
from dem_hyperelasticity.winkler_model_TEHL.PINNmodel_neu import *
# from dem_hyperelasticity.winkler_model_TEHL import pin_alg as pin
from dem_hyperelasticity.winkler_model_TEHL import substrate as sub
import numpy as np
import time
import torch
import math


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
        self.numIntType = numIntType
        self.model = MultiLayerNet(model[0], model[1], model[2])
        self.model.load_state_dict(torch.load("winkler_oil_v1_base_final.pt"))
        self.model.eval()
        # self.model = torch.load("winkler_uc_new.pt")
        # self.model_2 = torch.load("winkler_uc_new.pt")
        
        self.model = self.model.to(dev)
        self.intLoss = IntegrationLoss(numIntType, dim)
        self.energy = energy
        # self.post = PostProcessing(energy, dim)
        self.dim = dim
        self.lossArray = []
        self.substrate = sub.Substrate(dim)

    def train_model(self, shape, dxdydz, set_dom_vis, set_dom_sub, LHD, iteration, learning_rate):
        data, neumannBC, dirichletBC = set_dom_vis
        data_s, neumannBC_s, dirichletBC_s = set_dom_sub
        
        x = torch.from_numpy(data).float()
        x = x.to(dev)
        # xs = torch.from_numpy(data).float()
        # xs = xs.to(dev)
        x.requires_grad_(True)
        # x.retain_grad_(True)

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
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=learning_rate, max_iter=50)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas = (0.9,0.999)) #max_iter_change
        start_time = time.time()
        energy_loss_array = []
        boundary_loss_array = []
        val = neuBC_values
        loss_pre = 10**(16)
        # t_pre = -1
        for t in range(iteration):
            # Zero gradients, perform a backward pass, and update the weights.
            # _, p = self.getU(x)
            # # uz.double()
            # # p = p.detach()
            # p.double()
            # inf_vis = (p, x, self.numIntType)
            # self.substrate.train_model(shape, dxdydz, set_dom_sub,inf_vis,LHD,iteration,learning_rate)
            # us = self.substrate.get_sub(x)
            # us.double()
            def closure():
                it_time = time.time()
                # ----------------------------------------------------------------------------------
                # Internal Energy
                # ----------------------------------------------------------------------------------
                uz, pz = self.getU(x)
                uz.double()
                pz.double()
                # uz_2, pz_2 = self.getU_2(x)
                # uz_2.double().detach()
                # pz_2.double().detach()
                # strong = self.energy.strong(s_pred, x)
                # gorven_goal = torch.zeros(*strong.size())
                # strong = self.loss_squared_sum(strong,gorven_goal)
                
                s_u, s_p = self.energy.CE(uz, pz, x)
                # ce_goal = torch.zeros(*ce.size())
                ss = self.loss_squared_sum(s_u,s_p)
                load = self.intLoss.lossInternalEnergy(pz, dx=dxdydz[0], dy=dxdydz[1], shape=shape)
                b_loss = (load-(cf.W))**2
                if t >= 0 :
                    us = sub.deform_sub(pz,x,x,dxdydz,shape,'trapezoidal',self.dim)
                    u = us+uz#.detach()
                    # d =  3*10**(-4)*cf.a
                    d = cf.delta#math.pow((2*torch.pi/3), 2/3)*(cf.a**2)/cf.R
                    # if (cf.R**2-(x[:,0])**2-(x[:,1])**2)<0:
                    #     p = p+2
                    # g = cf.R-torch.sqrt(cf.R**2-(x[:,0])**2-(x[:,1])**2).unsqueeze(1)-d+u                    
                    # g = ((x[:,0])**2/(2*cf.R)+(x[:,1])**2/(2*cf.R)).unsqueeze(1)-d+u
                    g = ((x[:,0])**2/(2*cf.R)+(x[:,1])**2/(2*cf.R)).unsqueeze(1)-d+u
                    # g = (((((x[:,0])**2)/(2*cf.R)) + (((x[:,1])**2)/(2*cf.R)))/140.13267).unsqueeze(1) -d + uz
                    # g = (((((x[:,0])**2)/(1340)) + (((x[:,1])**2)/(1340)))).unsqueeze(1) -d + uz
                    # g = (((((x[:,0])**2)/(2*cf.R)) + (((x[:,1])**2)/(2*cf.R)))-((((x[:,0])**2)/(19.324612)) + (((x[:,1])**2)/(19.324612)))).unsqueeze(1) -d + uz
                    g = g/cf.delta#(cf.a**2/cf.R)
                    p = pz/(cf.ph)
                    lkkt_u = self.loss_squared_sum_3((1-torch.sign(g))*g/2, 0) 
                    lkkt_n = self.loss_squared_sum_3((1-torch.sign(p))*pz/2, 0)
                    lkkt_un = self.loss_squared_sum_3(p*g, 0)
                    
                    # eta = p*g
                    # eta = (1+torch.sign(eta))*eta/2
                    # lkkt = self.loss_squared_sum_3(g+p-torch.sqrt(g**2+p**2+2*eta), 0)
                    lkkt = self.loss_squared_sum_3(g+p-torch.sqrt(g**2+p**2), 0)

                # external2 = torch.zeros(len(neuBC_coordinates))
                bc_t_crit = torch.zeros(((len(neuBC_coordinates)-1),1))

                # in_energy = torch.zeros((len(x))-len(neuBC_coordinates)-len(dirBC_coordinates))
                # j=0
                # for i in x:
                #     if i[0] != 0 or i[0] !=4:
                #         n_u_pred = self.getU(i.unsqueeze(0))
                #         storedEnergy,_ = self.energy.getStoredEnergy(n_u_pred,i)
                #         gorven_goal = torch.zeros(*storedEnergy.size())
                #         in_energy[j] = self.loss_squared_sum(storedEnergy,gorven_goal)
                
                for i, vali in enumerate(neuBC_coordinates):
                    if i != 3:
                        neu_uz, neu_pz = self.getU(neuBC_coordinates[i])
                        bc_t_crit[i] = self.loss_squared_sum_2(neu_pz, neuBC_values[i])
                    elif t >= 0:
                        a = torch.zeros(*g[idx[i]].size())
                        g_loss = (self.loss_squared_sum(g[idx[i]], a))
                    # else:
                        
                    #     g = cf.R-torch.sqrt(cf.R**2-(neuBC_coordinates[i][:,0])**2)-cf.d-(neuBC_coordinates[i][:,1]+neu_u_pred[:,1])
                    #     g = g*cf.E
                    #     # g= neu_u_pred[:,1]
                    #     # mu = cf.E #/ (2 * (1 + cf.nu))
                    #     p = -neu[:,1]#*mu
                        
                    #     # p = md.EnergyModel.normalize(p)
                    #     # g = md.EnergyModel.normalize(g)
                    #     # lkkt methode
                    #     # lkkt_u = self.loss_squared_sum_3((1-torch.sign(g))*g/2, 0)
                    #     # lkkt_n = self.loss_squared_sum_3((1+torch.sign(p))*p/2, 0)
                    #     # lkkt_un = self.loss_squared_sum_3(p*g, 0)

                    #     lkkt_t = self.loss_squared_sum_3(-neu[:,0], 0)

                    #     # fischer Burmester
                    #     lkkt = self.loss_squared_sum_3(g-p-torch.sqrt(g**2+p**2), 0)
                bc_u_crit = torch.zeros((len(dirBC_coordinates),1))
                for i, vali in enumerate(dirBC_coordinates):
                    dir_uz, dir_pz = self.getU(dirBC_coordinates[i])
                    # if i ==1:
                    #     root = torch.ones(dir_uz.size())
                    #     root_u = self.loss_squared_sum(dir_uz,3*10**(-4)*root)
                    #     root_p = self.loss_squared_sum(dir_pz,1*root)
                    # else:    
                    bc_u_crit[i] = self.loss_squared_sum_2(dir_uz, dirBC_values[i])


                
                
                # u_shape = u.shape
                # mid = (cf.Nx//2)*cf.Ny+(cf.Ny//2)
                # if (cf.Nx*cf.Ny) % 2 ==0:
                    
                #     u_mid = (u[mid-1,0]+u[mid,0]+u[mid-1-cf.Ny,0]+u[mid-cf.Ny,0])/2
                # else:
                #     u_mid = u[mid,0]
                # d =  math.sqrt(cf.a**2/cf.R)#math.pow(cf.W, 2/3) #0.0277644729
                


                dir_loss = torch.sum(bc_u_crit[0])#+torch.sum(bc_u_crit[1])
                # neu_loss_1 = bc_t_crit[0][1]
                # neu_loss_2 = torch.sum(bc_t_crit[1]zzz)
                # neu_loss_3= torch.sum(bc_t_crit[2])
                # neu_loss_4 = torch.sum(bc_t_crit[3])
                neu_loss = torch.sum(bc_t_crit[0])+torch.sum(bc_t_crit[1])+torch.sum(bc_t_crit[2])#
                # storedEnergy = 
                energy_loss = ss  # storedEnergy get problem
                # loss = strong+ss+100*dir_loss+0.1*neu_loss_1+0.1*neu_loss_2+0.1*neu_loss_3+10*lkkt_u+100*lkkt_n+10*lkkt_un+0.1*lkkt_t # u is relate to bc_t #normalization
                # loss = 10*ss+10**7*neu_loss+1*10**4*b_loss#+0.1*lkkt#+10**2*root_u+10**2*root_p#+1*g_loss#+lkkt_n+10*lkkt_u+10*lkkt_un#+10**5*dir_loss
                # if t < 500:
                #     ws = 0.001#*np.sign(t//5)
                ws = t//100*0.2+0.001
                loss =ws*ss+12*b_loss+40*neu_loss+20*lkkt#+10**(-7)*g_loss#+1000*lkkt_un
                # loss = 0.000001*ss+12*b_loss+40*neu_loss+0*lkkt
                # loss = 1*self.loss_squared_sum_2(uz,uz_2)+1*self.loss_squared_sum_2(pz,pz_2)+0.1*lkkt
                # if t >= 20:
                # loss+=+ws*(t//20)*ss#+0*lkkt
                # if t >= 500:
                #     loss+=1*ss
                    # loss+=2*lkkt#+500*g_loss
                #     loss+=10**2*g_loss
                    # loss+=50*ss
                  

                # loss = loss*100
                optimizer.zero_grad()
                loss.backward()#retain_graph=True
                if  t < 0 :
                    print('Iter: %d Loss: %.6e ss: %.3e load: %.3e  dir_bc: %.3e neu_bc: %.3e Time: %.3e'  
                          % (t + 1, loss.item(), ss.item(), load,dir_loss.item(),neu_loss.item() ,time.time() - it_time))
                else:
                    print('Iter: %d Loss: %.6e ss: %.3e load: %.3e kktu: %.3e kkt: %.3e  dir_bc: %.3e neu_bc: %.3e g_loss: %.3e Time: %.3e'
                      % (t + 1, loss.item(), ss.item(), load,lkkt_u.item(),lkkt.item(),dir_loss.item(),neu_loss.item(), us.max(), time.time() - it_time))
                energy_loss_array.append(energy_loss.data)
                boundary_loss_array.append(dir_loss.data)
                self.lossArray.append(loss.data)
                return loss
            optimizer.step(closure)
            loss = closure()
            with open('loss_his_v1_kkt20.csv','a') as fd:
                fd.write(str(loss.item()) + '\n')
            if math.isnan(loss): 
                self.model.load_state_dict(torch.load("winkler_oil_v1_final.pt"))
                self.model.eval()
                break
            
            elif loss<loss_pre:
                time.sleep(0.02)
                torch.save(self.model.state_dict(),'winkler_oil_v1_final.pt')

                # time.sleep(0.02)
            #     break
            # else:
                loss_pre = loss
                t_pre = t
        elapsed = time.time() - start_time
        
        print('Training time: %.4f' % elapsed)

    def getU(self, x):
        u = self.model(x)

        Uz = u[:, 0]*(x[:,0]-cf.x_min)*(x[:,0]-(cf.x_min+cf.Length))*(x[:,1]-cf.y_min)*(x[:,1]-(cf.y_min+cf.Height))*cf.delta#/100000
        Pz = u[:,1]*(x[:,0]-cf.x_min)*(x[:,0]-(cf.x_min+cf.Length))*(x[:,1]-cf.y_min)*(x[:,1]-(cf.y_min+cf.Height))*cf.ph
        # Uz_norm = self.nomalize(Uz)
        # Pz_norm = self.nomalize(Pz)

        uz = Uz.reshape(Uz.shape[0], 1)
        pz = Pz.reshape(Pz.shape[0], 1)
        
        return uz ,pz

    def getU_2(self, x):
        u = self.model_2(x)

        Uz = u[:, 0]*cf.delta
        Pz = u[:,1]*cf.ph

        uz = Uz.reshape(Uz.shape[0], 1)
        pz = Pz.reshape(Pz.shape[0], 1)
        
        return uz ,pz 
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
        u_pred_torch, p_pred_torch = self.getU(xy_tensor)

        # duxdxy = grad(u_pred_torch[:, 0].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
        #                create_graph=True, retain_graph=True)[0]
        # duydxy = grad(u_pred_torch[:, 1].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
        #                create_graph=True, retain_graph=True)[0]
        
        # epsxx = duxdxy[:,0].unsqueeze(1)
        # epsxy = (0.5*duxdxy[:,1]+0.5*duydxy[:,0]).unsqueeze(1)
        # epsyx = epsxy
        # epsyy = duydxy[:,1].unsqueeze(1)

        # treps = epsxx+epsyy
        # sigmaxx = lmbda*treps+2*mu*epsxx
        # sigmaxy = 2*mu*epsxy
        # sigmayx = 2*mu*epsyx
        # sigmayy = lmbda*treps+2*mu*epsyy
        if energy_type == 'neohookean' and dim == 2:
            pass
        else:
            print("This energy model will be implemented later !!!")
            exit()

        u_pred = u_pred_torch.detach().cpu().numpy()
        p_pred = p_pred_torch.detach().cpu().numpy()

        # a = np.sqrt(cf.R*cf.d)
        # F = 4/3*cf.E*a*cf.d/(1-cf.nu**2)
        # P0=3*F/(2*np.pi*a**2)
        # print(max(u_pred[:, 1]))
        # Px=sigmaxx.reshape(Ny, Nx,1).detach().cpu().numpy()
        # Py=sigmayy.reshape(Ny, Nx,1).detach().cpu().numpy()
        Pz = p_pred.reshape(Ny, Nx,1)
        # Pz = np.zeros([Nx, Ny, 1])
        uz = u_pred.reshape(Ny, Nx,1)
        # surUz = np.zeros([Nx, Ny, 1])

        # surUy = (((-0.1)/cf.E*(1-cf.nu**2))*xy[:,1]).reshape(Ny, Nx,1)
        # Py = (np.ones(s_pred[:, 1].shape)*(-0.1)).reshape(Ny, Nx,1)
        # Pxy = np.zeros(s_pred[:, 2].shape).reshape(Ny, Nx,1)

        # epsilon = 1e-4
        # # Eu =  (np.abs((u_pred[:, 1]-((-0.1)/cf.E*(1-cf.nu**2))*xy[:,1])/((-0.1)/(cf.E*(1-cf.nu**2))*xy[:,1]-epsilon))).reshape(Ny, Nx,1)
        # e=1.5*10**4
        # v=0.3
        # Ebar, nubar, I = e/(1-v**2), v/(1-v), (12**3)/12
        # f=-0.05*e
        # H=12
        # L=48
        # Eu =  ((f/(6*Ebar*I))*(3*nubar*pow(xy[:,1],2)*(L-xy[:,0])+(4+5*nubar)*pow(H,2)*xy[:,0]/4 + (3*L-xy[:,0])*pow(xy[:,0],2))).reshape(Ny, Nx,1)
        # Eyy = (np.abs((s_pred[:, 1]+0.1)/(-1-epsilon))).reshape(Ny, Nx,1)
        # Exy = (np.abs(s_pred[:, 2]/(0-epsilon))).reshape(Ny, Nx,1)
        # Eu =  (np.abs((u_pred[:, 1]-((-0.1)/cf.E*(1-cf.nu**2))*xy[:,1])/(u_pred[:, 1]+(-0.1)/(cf.E*(1-cf.nu**2))*xy[:,1]))).reshape(Ny, Nx,1)
        # Eyy = (np.abs((s_pred[:, 1]+0.1)/(s_pred[:, 1]-0.1))).reshape(Ny, Nx,1)
        # Exy = (np.abs(s_pred[:, 2]/(s_pred[:, 2]))).reshape(Ny, Nx,1)

        # Eu =  np.arctan(np.abs((surUy-((-0.1)/cf.E*(1-cf.nu**2))*xy[:,1])/((-0.1)/(cf.E*(1-cf.nu**2))*xy[:,1]+epsilon))).reshape(Ny, Nx,1)
        # Eyy = np.arctan(np.abs((Py+0.1)/(-1+epsilon))).reshape(Ny, Nx,1)
        # Exy = np.arctan(np.abs(Pxy/(0+epsilon))).reshape(Ny, Nx,1)

        # U = (np.float64(surUx), np.float64(surUy),np.float64(surUz))
        # P = (np.float64(Px), (np.float64(Py)),(np.float64(Pxy)))  #NTC      
        # Er = (np.float64(Eu), (np.float64(Eyy)),(np.float64(Exy)))
            # print('max F is %.5f: '%(F))
        # print('max P is %.5f: '%(max(abs(sigmayy))))
        
        return uz, Pz
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
    @staticmethod
    def nomalize(input:torch.Tensor) ->torch.Tensor:
        """normalize nn output""" 
        min = torch.min(input)
        max = torch.max(input)
        output = (input-min)/(max-min)
        return output

if __name__ == '__main__':
    # ----------------------------------------------------------------------
    #                   STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
    # ----------------------------------------------------------------------
    set_dom_vis = des_v.setup_domain()
    x, y, datatest = des_v.get_datatest()
    set_dom_sub = des_s.setup_domain()
    x_s, y_s, datatest_s = des_s.get_datatest()
    # ----------------------------------------------------------------------
    #                   STEP 2: SETUP MODEL
    # ----------------------------------------------------------------------
    mat = md.EnergyModel('neohookean', 2, cf.E, cf.nu, cf.vm, cf.Ec, cf.l)
    dem = DeepEnergyMethod([cf.D_in, cf.H, cf.D_out], 'trapezoidal', mat, 2)
    # ----------------------------------------------------------------------
    #                   STEP 3: TRAINING MODEL
    # ----------------------------------------------------------------------
    start_time = time.time()
    shape = [cf.Nx, cf.Ny]
    dxdy = [cf.hx, cf.hy]
    cf.iteration = 1000 #550 exp

    # cf.filename_out = "dem_hyperelasticity/hertzian_contact_square/output/hert_20_itr_sb"
    cf.filename_out = "dem_hyperelasticity/hertzian_contact_square/output/tt"
    dem.train_model(shape, dxdy, set_dom_vis, set_dom_sub, [cf.Length, cf.Height, cf.Depth], cf.iteration, cf.lr)
    end_time = time.time() - start_time
    print("End time: %.5f" % end_time)
    z = np.array([0.])
    U, P= dem.evaluate_model(x, y, z)
    # util.write_vtk_v2(cf.filename_out, x, y, z, U, P,E)
    # surUx, surUy, surUz = U
    # L2norm =                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           util.getL2norm2D(surUx, surUy, cf.Nx, cf.Ny, cf.hx, cf.hy)
    # H10norm = util.getH10norm2D(F11, F12, F21, F22, cf.Nx, cf.Ny, cf.hx, cf.hy)
    # print("L2 norm = %.10f" % L2norm)
    # print("H10 norm = %.10f" % H10norm)
    # np.save("winkler_uz_oil_v1.npy", U)
    # np.save("winkler_pz_oil_v1.npy", P)
    print('Loss convergence')

    # stop = int(input("input 1 to continue: "))
    ##################################################################################################################################
    # start_time = time.time()
    # shape = [cf.Nx, cf.Ny]
    # dxdy = [cf.hx, cf.hy]
    # cf.iteration = 400 #550 exp

    # cf.filename_out = "dem_hyperelasticity/hertzian_contact_square/output/hert_1500itr"
    # dem.train_model(dom, boundary_neumann, boundary_dirichlet, [cf.Length, cf.Height, cf.Depth], cf.iteration, cf.lr)
    # end_time = time.time() - start_time
    # print("End time: %.5f" % end_time)
    # z = np.array([0.])
    # U, P,E = dem.evaluate_model(x, y, z)
    # util.write_vtk_v2(cf.filename_out, x, y, z, U, P,E)
    # surUx, surUy, surUz = U
    # # L2norm =                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           util.getL2norm2D(surUx, surUy, cf.Nx, cf.Ny, cf.hx, cf.hy)
    # # H10norm = util.getH10norm2D(F11, F12, F21, F22, cf.Nx, cf.Ny, cf.hx, cf.hy)
    # # print("L2 norm = %.10f" % L2norm)
    # # print("H10 norm = %.10f" % H10norm)
    # print('Loss convergence')

    
    # stop = int(input("input 1 to continue: "))
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