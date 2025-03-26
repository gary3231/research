import dem_hyperelasticity.winkler_model_TEHL.Integration as Integration
import dem_hyperelasticity.winkler_model_TEHL.config as cf 
from dem_hyperelasticity.winkler_model_TEHL.Integration import *
from dem_hyperelasticity.winkler_model_TEHL import define_structure_sub as des_s
from dem_hyperelasticity.winkler_model_TEHL.MultiLayerNet_sub import *
import torch
import numpy as np
import pandas as pd

def load(p, xv, xs):
    dived = ((-xv[:,0]+xs[0])**2+(-xv[:,1]+xs[1])**2).unsqueeze(1)  #torch.sqrt
    # dived_d = (dived)**(0.5)
    div = torch.sqrt(dived)
    # div = torch.nan_to_num(div, nan = (10**(-2)))
    # return dived
    l = torch.div(cf.ph,div)
    return l

def deform_sub(p, xv, xs, dxdydz, shape, IntType, dim):

    integrate = Integration.IntegrationLoss(IntType, dim)

    us = torch.zeros((xs.shape[0],1))
    x_load = torch.where((xs[:,0]**2+xs[:,1]**2) < (cf.a**2))[0]
    loads = p[x_load,:]*torch.pi**2/2/torch.sqrt(1-((xs[x_load,0].unsqueeze(1))**2+(xs[x_load,1].unsqueeze(1))**2)/cf.a**2)*cf.a#torch.sqrt(cf.a**2-((xs[i,0])**2+(xs[i,1])**2))
        # loads = p[i,:]*(cf.a*2)*torch.pi
    us[x_load,:] =  loads/torch.pi/cf.Es#*2/(torch.pi**2)*cf.Ec/cf.Es
    # for i in range(xs.shape[0]):
    #     #assump the dimession of both object is same
    #     # jump = (xv == xs[i,:]).nonzero(as_tuple=True)[0][0]
    #     # xpr = torch.cat((xv[:jump,:],xv[jump+1:,:]),0)
    #     # l = load(p, xv, (xs[i,:])-4*10**(-2))
    #     # loads = integrate.lossInternalEnergy(l, dx = dxdydz[0], dy = dxdydz[1], shape = shape)
    #     if ((xs[i,0])**2+(xs[i,1])**2)<cf.a**2:
    #         loads = p[i,:]*torch.pi**2/2/torch.sqrt(1-((xs[i,0])**2+(xs[i,1])**2)/cf.a**2)*cf.a#torch.sqrt(cf.a**2-((xs[i,0])**2+(xs[i,1])**2))
    #     else:
    #         loads = 0
    #     # loads = p[i,:]*(cf.a*2)*torch.pi
    #     us[i,:] =  loads*(1-cf.nus**2)/torch.pi/cf.Es#*2/(torch.pi**2)*cf.Ec/cf.Es

    return us

class Substrate:

    def __init__(self, dim):
        self.model = MultiLayerNet(cf.D_in, cf.H, cf.D_out)
        self.model = self.model.to(dev)
        self.dim = dim
        self.lossArray = []

    def train_model(self, shape, dxdydz, set_dom_sub, inf_vis, LHD, iteration, learning_rate):
        p, xv, IntType = inf_vis
        p = p#.detach()
        
        data, neumannBC, dirichletBC = set_dom_sub
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
        # for i, keyi in enumerate(dirichletBC):
        #     dirBC_coordinates[i] = torch.from_numpy(dirichletBC[keyi]['coord']).float().to(dev)
        #     dirBC_coordinates[i].requires_grad_(True)
        #     dirBC_values[i] = torch.from_numpy(dirichletBC[keyi]['known_value']).float().to(dev)
        #     dirBC_penalty[i] = torch.tensor(dirichletBC[keyi]['penalty']).float().to(dev)
        # -------------------------------------------------------------------------------
        
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=learning_rate, max_iter=50) #max_iter_change
        energy_loss_array = []
        boundary_loss_array = []
        loss_pre = 10**(-16)
        for t in range(iteration):
            # Zero gradients, perform a backward pass, and update the weights.
            def closure():
                uz = self.get_sub(x)
                
                # if True in check:
                #     print(0)

                uz.double()
                us = deform_sub(p, xv, x, dxdydz, shape, IntType, self.dim)
                # bc_u_crit = torch.zeros((len(dirBC_coordinates),1))
                # for i, vali in enumerate(dirBC_coordinates):
                #     dir_uz= self.get_sub(dirBC_coordinates[i])
                #     bc_u_crit[i] = self.loss_squared_sum_2(dir_uz, dirBC_values[i])

                # dir_loss = torch.sum(bc_u_crit[0])+torch.sum(bc_u_crit[1])+torch.sum(bc_u_crit[2])+torch.sum(bc_u_crit[3])
                # check = torch.isin(xv[:,1], x[:,1])
                # check_n = us.detach().cpu().numpy()
                # df = pd.DataFrame(check_n) #convert to a dataframe
                # df.to_csv("testfile.csv",index=False) #save to file
                
                # print(cf.check_point)
                us = us#.detach()
                loss1 = self.loss_squared_sum(uz, us)#+1000*dir_loss

                optimizer.zero_grad()
                loss1.backward(retain_graph=True)
                
                return loss1
            
            optimizer.step(closure)
            loss1 = closure()
            print('Iter: %d Loss: %.6e '
                      % (t + 1, loss1))
            if abs((loss1-loss_pre)/loss_pre)<10**(-4):
                
                break
            else:
                loss_pre = loss1


    def get_sub(self, x):
        u = self.model(x)
        u = u*cf.delta

        # Uz = u
        # Pz = u[:,1]*cf.ph

        uz = u.reshape(u.shape[0], 1)
        # pz = Pz.reshape(Pz.shape[0], 1)
        
        return uz 
    
    # def evaluate_model(self, x, y):

    #     Nx = len(x)
    #     Ny = len(y)
    #     xGrid, yGrid = np.meshgrid(x, y)
    #     x1D = xGrid.flatten()
    #     y1D = yGrid.flatten()
    #     xy = np.concatenate((np.array([x1D]).T, np.array([y1D]).T), axis=-1)
    #     xy_tensor = torch.from_numpy(xy).float()
    #     xy_tensor = xy_tensor.to(dev)
    #     xy_tensor.requires_grad_(True)
    #     u_pred_torch = self.getU(xy_tensor)

    #     u_pred = u_pred_torch.detach().cpu().numpy()
    #     uz = u_pred.reshape(Ny, Nx,1)
    #     return uz
    
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
