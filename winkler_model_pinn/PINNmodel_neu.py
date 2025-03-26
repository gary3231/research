from dem_hyperelasticity.config import *
import dem_hyperelasticity.winkler_model_pinn.config as cf


class EnergyModel:
    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, energy, dim, E=None, nu=None, vm = None, beta = None, Ec = None, l = None, tau = None):
        self.type = energy
        self.dim = dim
        if self.type == 'neohookean':
            self.mu = E / (2 * (1 + nu))
            self.lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
            self.vm = vm
            self.beta = beta
            self.Ec = Ec
            self.l = l
            self.tau = tau
            

    def getStoredEnergy(self, u, x,i):
        if self.type == 'neohookean':
            if self.dim == 2:
                return self.NeoHookean2D(u, x,i)
            if self.dim == 3:
                return self.NeoHookean3D(u, x,i)

    # ---------------------------------------------------------------------------------------
    # Purpose: calculate Neo-Hookean potential energy in 3D
    # ---------------------------------------------------------------------------------------
    def NeoHookean3D(self, u, x,i):
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
    def NeoHookean2D(self, u, x,i):
        duxdxy = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxy = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        
        epsxx = duxdxy[:,0].unsqueeze(1)
        epsxy = (0.5*duxdxy[:,1]+0.5*duydxy[:,0]).unsqueeze(1)
        # epsxz = 0
        epsyx = epsxy
        epsyy = duydxy[:,1].unsqueeze(1)
        # epsyz = 0
        # epszx = 0
        # epszy = 0
        # epszz = 0

        treps = epsxx+epsyy

        sigmaxx = self.lam*treps+2*self.mu*epsxx
        sigmaxy = 2*self.mu*epsxy
        # sigmaxz = 0
        sigmayx = 2*self.mu*epsyx
        sigmayy = self.lam*treps+2*self.mu*epsyy
        # sigmayz = 0
        # sigmazx = 0
        # sigmazy = 0
        # sigmazz = self.lam*treps+2*self.mu*epszz

        dsxxdxyz = grad(sigmaxx, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        dsxydxyz = grad(sigmaxy, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0] 
        dsyxdxyz = grad(sigmayx, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        dsyydxyz = grad(sigmayy, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        # dszzdxyz = grad(sigmazz, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
       
        divsxx = dsxxdxyz[:,0].unsqueeze(1)
        divsxy = dsxydxyz[:,1].unsqueeze(1)
        divsyx = dsyxdxyz[:,0].unsqueeze(1)
        divsyy = dsyydxyz[:,1].unsqueeze(1)

        divsigma = torch.cat((divsxx+divsxy,divsyx+divsyy),1)
        
        if i == 0:
            return divsigma , torch.cat((sigmaxx,sigmayx,),1)
        if i == 1:
            return divsigma , torch.cat((sigmaxy,sigmayy),1)
        if i == 2:
            # l = self.normalize(x)
            return divsigma , torch.cat((x[:,0]*sigmaxx+x[:,1]*sigmaxy, x[:,0]*sigmayx+x[:,1]*sigmayy),1).reshape(-1,2)
        if i == 3:
            # l = self.normalize(x)
            return divsigma , torch.cat((x[:,0]*sigmaxx+x[:,1]*sigmaxy, x[:,0]*sigmayx+x[:,1]*sigmayy),1).reshape(-1,2)
        else:
            return divsigma,0
    
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
    
    def CE(self,uz,pz,x): #constitutive equation
        r"""x is x dimension not whole domain"""
        duzdxy = grad(uz, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        dpzdxy = grad(pz, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        
        duzdx = duzdxy[:,0].unsqueeze(1)
        dpzdx = dpzdxy[:,0].unsqueeze(1)

        s_pz = pz+self.vm*self.beta*dpzdx
        s_uz = self.Ec/self.l*(uz+self.vm*self.tau*duzdx)
        # s_pz = pz+cf.De/(1+cf.Re)*dpzdx
        # s_uz = cf.Ec_bar/cf.l_bar*(uz+cf.De*duzdx)

        return s_pz,s_uz
    
    def NBC(self, s, x, i):
        sxx = s[:,0].unsqueeze(1)
        syy = s[:,1].unsqueeze(1)
        sxy = s[:,2].unsqueeze(1)
        if i == 0:
            return  torch.cat((sxx,sxy,),1)
        if i == 1:
            return  torch.cat((sxy,syy),1)
        if i == 2:
            # l = self.normalize(x)
            return  torch.cat((x[:,0].unsqueeze(1)*sxx+x[:,1].unsqueeze(1)*sxy, x[:,0].unsqueeze(1)*sxy+x[:,1].unsqueeze(1)*syy),1)
        if i == 3:
            # l = self.normalize(x)
            return  torch.cat((x[:,0].unsqueeze(1)*sxx+x[:,1].unsqueeze(1)*sxy, x[:,0].unsqueeze(1)*sxy+x[:,1].unsqueeze(1)*syy),1)