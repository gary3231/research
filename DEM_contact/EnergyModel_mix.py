from dem_hyperelasticity.config import *


class EnergyModel:
    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, energy, dim, E=None, nu=None, param_c1=None, param_c2=None, param_c=None):
        self.type = energy
        self.dim = dim
        if self.type == 'neohookean':
            self.mu = E / (2 * (1 + nu))
            self.lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        if self.type == 'mooneyrivlin':
            self.param_c1 = param_c1
            self.param_c2 = param_c2
            self.param_c = param_c
            self.param_d = 2 * (self.param_c1 + 2 * self.param_c2)

    def getStoredEnergy(self, u, x):
        if self.type == 'neohookean':
            if self.dim == 2:
                return self.NeoHookean2D(u, x)
            if self.dim == 3:
                return self.NeoHookean3D(u, x)
        if self.type == 'mooneyrivlin':
            if self.dim == 2:
                return self.MooneyRivlin2D(u, x)
            if self.dim == 3:
                return self.MooneyRivlin3D(u, x)

    def MooneyRivlin3D(self, u, x):
        duxdxyz = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxyz = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duzdxyz = grad(u[:, 2].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        Fxx = duxdxyz[:, 0].unsqueeze(1) + 1
        Fxy = duxdxyz[:, 1].unsqueeze(1) + 0
        Fxz = duxdxyz[:, 2].unsqueeze(1) + 0
        Fyx = duydxyz[:, 0].unsqueeze(1) + 0
        Fyy = duydxyz[:, 1].unsqueeze(1) + 1
        Fyz = duydxyz[:, 2].unsqueeze(1) + 0
        Fzx = duzdxyz[:, 0].unsqueeze(1) + 0
        Fzy = duzdxyz[:, 1].unsqueeze(1) + 0
        Fzz = duzdxyz[:, 2].unsqueeze(1) + 1
        detF = Fxx * (Fyy * Fzz - Fyz * Fzy) - Fxy * (Fyx * Fzz - Fyz * Fzx) + Fxz * (Fyx * Fzy - Fyy * Fzx)
        C11 = Fxx ** 2 + Fyx ** 2 + Fzx ** 2
        C12 = Fxx * Fxy + Fyx * Fyy + Fzx * Fzy
        C13 = Fxx * Fxz + Fyx * Fyz + Fzx * Fzz
        C21 = Fxy * Fxx + Fyy * Fyx + Fzy * Fzx
        C22 = Fxy ** 2 + Fyy ** 2 + Fzy ** 2
        C23 = Fxy * Fxz + Fyy * Fyz + Fzy * Fzz
        C31 = Fxz * Fxx + Fyz * Fyx + Fzz * Fzx
        C32 = Fxz * Fxy + Fyz * Fyy + Fzz * Fzy
        C33 = Fxz ** 2 + Fyz ** 2 + Fzz ** 2
        trC = C11 + C22 + C33
        trC2 = C11*C11 + C12*C21 + C13*C31 + C21*C12 + C22*C22 + C23*C32 + C31*C13 + C32*C23 + C33*C33
        I1 = trC
        I2 = 0.5 * (trC*trC - trC2)
        J = detF
        strainEnergy = self.param_c * (J - 1) ** 2 - self.param_d * torch.log(J) + self.param_c1 * (
                I1 - 3) + self.param_c2 * (I2 - 3)
        return strainEnergy


    def MooneyRivlin2D(self, u, x):
        duxdxy = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxy = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        Fxx = duxdxy[:, 0].unsqueeze(1) + 1
        Fxy = duxdxy[:, 1].unsqueeze(1) + 0
        Fyx = duydxy[:, 0].unsqueeze(1) + 0
        Fyy = duydxy[:, 1].unsqueeze(1) + 1
        detF = Fxx * Fyy - Fxy * Fyx
        C11 = Fxx * Fxx + Fyx * Fyx
        C12 = Fxx * Fxy + Fyx * Fyy
        C21 = Fxy * Fxx + Fyy * Fyx
        C22 = Fxy * Fxy + Fyy * Fyy
        J = detF
        traceC = C11 + C22
        I1 = traceC
        trace_C2 = C11 * C11 + C12 * C21 + C21 * C12 + C22 * C22
        I2 = 0.5 * (traceC ** 2 - trace_C2)
        strainEnergy = self.param_c * (J - 1) ** 2 - self.param_d * torch.log(J) + self.param_c1 * (I1 - 2) + self.param_c2 * (I2 - 1)
        return strainEnergy

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
    def NeoHookean2D(self,u , x):
        duxdxy = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxy = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
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
        P11 = self.mu * Fxx + (self.lam * torch.log(detF) - self.mu) * invF11
        P12 = self.mu * Fxy + (self.lam * torch.log(detF) - self.mu) * invF21
        P21 = self.mu * Fyx + (self.lam * torch.log(detF) - self.mu) * invF12
        P22 = self.mu * Fyy + (self.lam * torch.log(detF) - self.mu) * invF22
        # detF = F[:,0].unsqueeze(1)*F[:,3].unsqueeze(1)-F[:1].unsqueeze(1)*F[:2].unsqueeze(1)
        trC = Fxx ** 2 + Fxy ** 2 + Fyx ** 2 + Fyy ** 2
        # trC = (F[:,0].unsqueeze(1))**2+(F[:,1].unsqueeze(1))**2+(F[:,2].unsqueeze(1))**2+(F[:,3].unsqueeze(1))**2
        strainEnergy = 0.5 * self.lam * (torch.log(detF) * torch.log(detF)) - self.mu * torch.log(detF) + 0.5 * self.mu * (trC - 2)
        
        return strainEnergy
    
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0: 
          return v
        return v / norm
    
    def strong_s(self,s,x):
        dsxxdxyz = grad(s[:,0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        dsxydxyz = grad(s[:,1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        dsyxdxyz = grad(s[:,2].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        dsyydxyz = grad(s[:,3].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]  

        divsxx = dsxxdxyz[:,0].unsqueeze(1)
        divsxy = dsxydxyz[:,1].unsqueeze(1)
        divsyx = dsyxdxyz[:,0].unsqueeze(1)
        divsyy = dsyydxyz[:,1].unsqueeze(1)

        divsigma = torch.cat((divsxx+divsxy,divsyx+divsyy),1)

        return divsigma
    
    def strong_u(self,u,x):
        duxdxy = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxy = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
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
        P11 = self.mu * Fxx + (self.lam * torch.log(detF) - self.mu) * invF11
        P12 = self.mu * Fxy + (self.lam * torch.log(detF) - self.mu) * invF21
        P21 = self.mu * Fyx + (self.lam * torch.log(detF) - self.mu) * invF12
        P22 = self.mu * Fyy + (self.lam * torch.log(detF) - self.mu) * invF22
        # detF = F[:,0].unsqueeze(1)*F[:,3].unsqueeze(1)-F[:1].unsqueeze(1)*F[:2].unsqueeze(1)
        trC = Fxx ** 2 + Fxy ** 2 + Fyx ** 2 + Fyy ** 2
        # trC = (F[:,0].unsqueeze(1))**2+(F[:,1].unsqueeze(1))**2+(F[:,2].unsqueeze(1))**2+(F[:,3].unsqueeze(1))**2
        strainEnergy = 0.5 * self.lam * (torch.log(detF) * torch.log(detF)) - self.mu * torch.log(detF) + 0.5 * self.mu * (trC - 2)
        # v=0.3
        # E=1000
        # # density = 
        # g=9.81
        # body_force = 0#(x[-1,0]-x[0,0]*x[-1,1]-x[0,1])*density*g
        
        dP11dxy = grad(P11,x,torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        dP12dxy = grad(P12,x,torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        dP21dxy = grad(P21,x,torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        dP22dxy = grad(P22,x,torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        div_P11 = dP11dxy[:,0].unsqueeze(1)#+dP11dxy[:,1].unsqueeze(1)
        div_P12 = dP12dxy[:,1].unsqueeze(1)#+dP12dxy[:,1].unsqueeze(1)
        div_P21 = dP21dxy[:,0].unsqueeze(1)#+dP21dxy[:,1].unsqueeze(1)
        div_P22 = dP22dxy[:,1].unsqueeze(1)#+dP22dxy[:,1].unsqueeze(1)
        div_P = torch.cat((div_P11+div_P12,div_P21+div_P22),1)

        return div_P
    
    def CE(self,u,x): #constitutive equation
        duxdxy = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxy = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
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
        P11 = self.mu * Fxx + (self.lam * torch.log(detF) - self.mu) * invF11
        P12 = self.mu * Fxy + (self.lam * torch.log(detF) - self.mu) * invF21
        P21 = self.mu * Fyx + (self.lam * torch.log(detF) - self.mu) * invF12
        P22 = self.mu * Fyy + (self.lam * torch.log(detF) - self.mu) * invF22


        return torch.cat((P11,P12,P21,P22),1)
    
    def NBC(self, s, x, i):
        sxx = s[:,0].unsqueeze(1)
        sxy = s[:,1].unsqueeze(1)
        syx = s[:,2].unsqueeze(1)
        syy = s[:,3].unsqueeze(1)
        if i == 0:
            return  torch.cat((sxy,syy),1)
        if i == 1:
            return  torch.cat((sxx,syx),1)
        if i == 2:
            # l = self.normalize(x)
            return  torch.cat((sxx,syx),1)
        if i == 3:
            # l = self.normalize(x)
            return  torch.cat((-sxy,-syy),1)