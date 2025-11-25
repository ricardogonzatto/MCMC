from cobaya.likelihood import Likelihood
import numpy as np
import pyccl as ccl

class my_likelihood(Likelihood):
    params = {"Oc":{}, "Ob":{}, "w0":{}, "S8":{}}
    
    def initialize(self):
        sims = np.load('/share/storage2/fvs/test_cls/cls_lens_zbin0_all_fiducial.npy')
        dndz= np.load('/share/storage2/fvs/input_dndzs/n_z_sourcegals_forfalcon_z_0.npy')
        
        self.cov = np.cov(sims.T)
        self.icov = np.linalg.pinv(self.cov)
        self.fiducial = np.mean(sims, axis=0)
        
        self.zz= dndz[:,0]
        self.nz = dndz[:,1]
        self.ell = np.arange(1, 2202)
        
        
    def get_cl_theory(self, **params): 
    

        Oc = params["Oc"]
        Ob = params["Ob"]
        w0 = params["w0"]
        
        Om = Ob + Oc
        sigma_8 = params["S8"]/((Om / 0.3)**0.5)
        
        cosmo = ccl.Cosmology(Omega_c=Oc,
                        Omega_b=Ob,
                        h=0.673,
                        n_s=0.9649,
                        sigma8=sigma_8,
                        w0=w0,
                        T_CMB=2.7255,
                        matter_power_spectrum= 'halofit',
                        baryonic_effects=None)

        lens = ccl.WeakLensingTracer(cosmo, dndz=(self.zz,self.nz))
        cls = cosmo.angular_cl(lens, lens, self.ell)
    
        return cls
        
    def logp(self, **params):
        """
        Gaussian likelihood.
        """
        
        t = self.get_cl_theory(**params)
        
        r = (t - self.fiducial)
        chi2 = np.dot(r,  np.dot(self.icov, r))

        return -0.5 * chi2
