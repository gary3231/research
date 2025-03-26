from dem_hyperelasticity.config import *
from typing import List

class EnergyModel:
    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, energy, dim, time_step = None , E0=None, E1=None,eta1 = None, nu=None):
        self.type = energy
        self.dim = dim
        if self.type == 'viscoelasticity':
            self.step = time_step
            self.nu = nu
            self.E0 = E0
            self.E1 = E1
            self.eta1 = eta1

    def getStoredEnergy(self, u, epsv, x,  epsv_old) -> List[torch.Tensor]:
        if self.type == 'viscoelasticity':
            if self.dim == 2:
                return self.viscoelasticity2D(u,epsv, x , epsv_old)
            # if self.dim == 3:
            #     return self.viscoelasticity3D(u, x,  epsv_old)
    # ---------------------------------------------------------------------------------------
    # Purpose: calculate Neo-Hookean potential energy in 3D
    # ---------------------------------------------------------------------------------------
    def viscoelasticity3D(self, u, x,i):
        duxdxyz = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxyz = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duzdxyz = grad(u[:, 2].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        epsxx = duxdxyz[:,0].unsqueeze(1)
        epsxy = (0.5*duxdxyz[:,1]+0.5*duydxyz[:,0]).unsqueeze(1)
        epsxz = (0.5*duxdxyz[:,2]+0.5*duzdxyz[:,0]).unsqueeze(1)
        epsyx = epsxy
        epsyy = duydxyz[:,1].unsqueeze(1)
        epsyz = (0.5*duydxyz[:,2]+0.5*duzdxyz[:,1]).unsqueeze(1)
        epszx = epsxz
        epszy = epsyz
        epszz = duzdxyz[:,2].unsqueeze(1)
        treps = epsxx+epsyy+epszz
        sigmaxx = self.lam*treps+2*self.mu*epsxx
        sigmaxy = 2*self.mu*epsxy
        sigmaxz = 2*self.mu*epsxz
        sigmayx = 2*self.mu*epsyx
        sigmayy = self.lam*treps+2*self.mu*epsyy
        sigmayz = 2*self.mu*epsyz
        sigmazx = 2*self.mu*epszx
        sigmazy = 2*self.mu*epszy
        sigmazz = self.lam*treps+2*self.mu*epszz
        dsxxdxyz = grad(sigmaxx, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        dsxydxyz = grad(sigmaxy, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        dsxzdxyz = grad(sigmaxz, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        dsyxdxyz = grad(sigmayx, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        dsyydxyz = grad(sigmayy, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        dsyzdxyz = grad(sigmayz, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        dszxdxyz = grad(sigmazx, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        dszydxyz = grad(sigmazy, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        dszzdxyz = grad(sigmazz, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        divsxx = dsxxdxyz[:,0].unsqueeze(1)
        divsxy = dsxydxyz[:,1].unsqueeze(1)
        divsxz = dsxzdxyz[:,2].unsqueeze(1)
        divsyx = dsyxdxyz[:,0].unsqueeze(1)
        divsyy = dsyydxyz[:,1].unsqueeze(1)
        divsyz = dsyzdxyz[:,2].unsqueeze(1)
        divszx = dszxdxyz[:,0].unsqueeze(1)
        divszy = dszydxyz[:,1].unsqueeze(1)
        divszz = dszzdxyz[:,2].unsqueeze(1)
        divsigma = torch.cat((divsxx+divsxy+divsxz,divsyx+divsyy+divsyz,divszx+divszy+divszz),1)
        
        if i == 0:
            return divsigma , torch.cat((sigmaxx,sigmayx,sigmazx),1)
        elif i == 1:
            return divsigma , torch.cat((sigmaxy,sigmayy,sigmazy),1)
        elif i == 2:
            return divsigma, torch.cat((sigmaxz,sigmayz,sigmazz),1)
        else:
            return divsigma,0
    # ---------------------------------------------------------------------------------------
    # Purpose: calculate Neo-Hookean potential energy in 2D
    # ---------------------------------------------------------------------------------------
    def strainenergy(self, E, epsxx, epsxy, epsyx, epsyy):

        treps = epsxx+epsyy
        # c = self.nu/(1+self.nu)/(1-self.nu)*treps*np.identity(2) + 1/(1+self.nu)*e
        sigmaxx = E*(self.nu/(1+self.nu)/(1-self.nu)*treps+ 1/(1+self.nu)*epsxx)
        sigmaxy = E*(1/(1+self.nu)*epsxy)
        sigmayx = E*(1/(1+self.nu)*epsyx)
        sigmayy = E*(self.nu/(1+self.nu)/(1-self.nu)*treps+ 1/(1+self.nu)*epsyy)
        # trC = (F[:,0].unsqueeze(1))**2+(F[:,1].unsqueeze(1))**2+(F[:,2].unsqueeze(1))**2+(F[:,3].unsqueeze(1))**2
        strainEnergy = 0.5 * (sigmaxx*epsxx+sigmayy*epsyy+sigmaxy*epsxy+sigmayx*epsyx)
        return strainEnergy
    
    def dotC(self,co ,epsxx, epsxy, epsyx, epsyy):

        treps = epsxx+epsyy
        # c = self.nu/(1+self.nu)/(1-self.nu)*treps*np.identity(2) + 1/(1+self.nu)*e
        xx = co*(self.nu/(1+self.nu)/(1-self.nu)*treps+ 1/(1+self.nu)*epsxx)
        xy = co*(1/(1+self.nu)*epsxy)
        yx = co*(1/(1+self.nu)*epsyx)
        yy = co*(self.nu/(1+self.nu)/(1-self.nu)*treps+ 1/(1+self.nu)*epsyy)

        return xx, xy, yx, yy



    def dissipation_potentail(self, eta, epsvxx_dif, epsvxy_dif, epsvyx_dif, epsvyy_dif):
        
        epsvxx_dot = epsvxx_dif/self.step
        epsvxy_dot = epsvxy_dif/self.step
        epsvyx_dot = epsvyx_dif/self.step
        epsvyy_dot = epsvyy_dif/self.step

        energy = 0.5*eta*(epsvxx_dot*epsvxx_dot + epsvxy_dot*epsvxy_dot + epsvyx_dot*epsvyx_dot + epsvyy_dot*epsvyy_dot)

        return energy

    def viscoelasticity2D(self, u, epsv, x, epsv_old):
        duxdxy = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxy = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]

        epsxx = duxdxy[:,0].unsqueeze(1)
        epsxy = (0.5*duxdxy[:,1]+0.5*duydxy[:,0]).unsqueeze(1)
        epsyx = epsxy
        epsyy = duydxy[:,1].unsqueeze(1)

        epsvxx_old = epsv_old[:,0].unsqueeze(1)
        epsvxy_old = epsv_old[:,1].unsqueeze(1)
        epsvyx_old = epsv_old[:,2].unsqueeze(1)
        epsvyy_old = epsv_old[:,3].unsqueeze(1)

        # identy_2 = torch.eye(2)
        # identy_4 = torch.eye(4)
        # c = self.nu/(1-self.nu**2)*torch.kron(identy_2,identy_2)+1*(1+self.nu)*identy_4

        # eps = torch.cat((epsxx,epsxy,epsyx,epsyy),1)
        # epsv = torch.inverse(identy_4+self.step*self.E1/self.eta1*c)*(epsv_old+self.step*self.E1/self.eta1*eps)

        # mu = self.E1/2/(1+self.nu)
        # wk = 1000
        # wj = 8000/3
        # k1 = self.E1/2/(1-self.nu)


        epsvxx = epsv[:,0].unsqueeze(1)
        epsvxy = epsv[:,1].unsqueeze(1)
        epsvyx = epsv[:,2].unsqueeze(1)
        epsvyy = epsv[:,3].unsqueeze(1)

        elasic_energy = self.strainenergy(self.E0 ,epsxx, epsxy, epsyx, epsyy)
        visco_energy = self.strainenergy(self.E1, epsxx-epsvxx, epsxy-epsvxy, epsyx-epsvyx, epsyy-epsvyy)
        dissiaption = self.dissipation_potentail(self.eta1, epsvxx-epsvxx_old, epsvxy-epsvxy_old, epsvyx-epsvyx_old, epsvyy-epsvyy_old)

        strain_energy = elasic_energy+ visco_energy

        return strain_energy, dissiaption  #, torch.cat((epsvxx,epsvxy,epsvyx,epsvyy))

    
    def strong(self,s,x):
        dsxxdxyz = grad(s[:,0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        dsyydxyz = grad(s[:,1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        dsxydxyz = grad(s[:,2].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0] 

        divsxx = dsxxdxyz[:,0].unsqueeze(1)
        divsxy = dsxydxyz[:,1].unsqueeze(1)
        divsyx = dsxydxyz[:,0].unsqueeze(1)
        divsyy = dsyydxyz[:,1].unsqueeze(1)

        divsigma = torch.cat((divsxx+divsxy,divsyx+divsyy),1)

        return divsigma
    
    def NBC(self, s, x, i):
        sxx = s[:,0].unsqueeze(1)
        syy = s[:,1].unsqueeze(1)
        sxy = s[:,2].unsqueeze(1)
        if i == 0:
            return  torch.cat((-sxx,-sxy,),1)
        if i == 1:
            return  torch.cat((sxy,syy),1)
        if i == 2:
            # l = self.normalize(x)
            return  torch.cat((sxx,sxy,),1)
        if i == 3:
            # l = self.normalize(x)
            return  torch.cat((-sxy,-syy),1)