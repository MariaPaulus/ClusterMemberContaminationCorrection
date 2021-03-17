from __future__ import division
from math import sqrt
import numpy as np
import sys
sys.path.append('/home/grandis/source/')
import ClusterCosmologyModules as ccm
from colossus.cosmology import cosmology
from colossus import halo
from colossus.halo import concentration
from numpy.lib import scimath as sm
from astropy import units as u

class ClustContCorr(object):
    
    def __init__(self, cosmo1, constants, mapping, list_lambda, list_z, list_n_s, list_n_f, edges_f, list_num, m500):
        
        self.constants = constants
        self.mapping = mapping
        self.list_lambda = list_lambda
        self.list_z = list_z
        self.list_num = list_num
        self.num_clusters = len(list_lambda)
        self.m500 = m500
        
        # initialise here anything that can compute E(z),
        self.cosmology = cosmo1
        
        assert(len(list_z)==len(list_lambda))
        
        self.list_n_s = list_n_s
        self.list_n_f = list_n_f
        self.edges_f = edges_f
        self.bin_size = edges_f[1]-edges_f[0]

        return
    
    def setup(self):
        
        return
    
    
    def computeLikelihood(self,ctx):
        
        p1 = ctx.getParams()
        
        params = self.constants.copy()
        
        for key,value in self.mapping.items():
            params[key]=p1[value]
        
        lnL = 0
        for i in xrange(self.num_clusters):
            #print i
            
            if (self.list_num[i]==0).any():
                lnL += 0
            else:
                lnL += self.single_clusterlike(i,params)

        
        return lnL

    
    
    def single_clusterlike(self,i,params):

        
        n_s_ind = self.list_n_s[i]

        list_num_ind = self.list_num[i]

        
        e = self.model(i,params)

        
        if np.any(e <= 0): 
            lnL = -np.inf
            print lnL

        else:
            # num is the no of background objects for each cluster
            lnL = self.bin_size * np.sum(list_num_ind[:, None] * (n_s_ind * np.log(e) - e) )
            print lnL
            
        if np.isnan(lnL):
            print 'cluster %i has nan likelihood'%i
        if not np.isfinite(lnL):
            print 'cluster %i has infinite likelihood'%i

        
        return lnL 
    
   
    
    
    def cont_distr(self,i,params):
        

        Afcl_arr  = np.array([params['A_fcl_1'],params['A_fcl_2'],params['A_fcl_3'],params['A_fcl_4'],params['A_fcl_5'],params['A_fcl_6'],params['A_fcl_7'],params['A_fcl_8'],params['A_fcl_9'],params['A_fcl_10']])#,params['A_fcl_11']])#,params['A_fcl_12'],params['A_fcl_13'],params['A_fcl_14'],params['A_fcl_15']])

        #zbins = np.array([0.15,0.20,0.25,0.30,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.3])
        #zbins = np.array([0.15,0.20,0.25,0.30,0.4,0.5,0.6,0.7,0.8,0.9,1.3])
        #zbins = np.array([0.1,0.15,0.20,0.25,0.30,0.4,0.5,0.6,0.7,1.3])
        #zbins = np.array([0.15,0.20,0.25,0.30,0.4,0.5,0.6,0.7,0.9])
        
        idx = np.digitize(self.list_z[i], zbins) - 1

        A_fcl = Afcl_arr[idx]
        
        Amp = A_fcl * (self.list_lambda[i]/params['lambda0'])**params['B_fcl']
        Amp *= self.projNFW(i, params)
        f_cl = Amp/(Amp+1)
        
        mu = self.list_z[i] + params['mu0'] * ((1.+self.list_z[i])/(1.+params['z0']))**params['C_mu']

        try:
            sigma0 = params['sigma0']
        except KeyError:
            sigma0 = np.exp(params['lnsigma0'])
            
        sigma = sigma0 * ((1+self.list_z[i])/(1+params['z0']))**params['C_sigma']
        
        gaussian=1./np.sqrt(2.*np.pi*sigma**2)* np.exp(-0.5*(self.edges_f-mu)**2/sigma**2) 

        
        integrad_in_bin = 0.5*(gaussian[1:]+gaussian[:-1])

        
        assert np.sum(integrad_in_bin) != 0
        
        return f_cl, integrad_in_bin
        
    def model(self,i,params):
        

        f_cl, integrad_in_bin = self.cont_distr(i,params)
        
        n_f_ind = self.list_n_f[i]

        if np.any(f_cl) <= 0:
            e = -1
        elif np.any(f_cl) > 1:
            e = -1
        else:
            e = (1-f_cl)[:,None] * n_f_ind[None,:] + f_cl[:,None] * integrad_in_bin[None,:]  # b[:,None]*
        
        return e
    
    def arcsec(self, z):
        val1 = 1j / z
        val2 = sm.sqrt(1 - 1./z**2)
        val = 1j * np.log(val2 + val1)
        return .5 * np.pi + val
    
    # Sigma_NFW[Radius][Mass]
    def get_Sigma(self, x):
        val1 = 1. / (x**2 - 1)
        val2 = (self.arcsec(x) / (sm.sqrt(x**2 - 1))**3).real
        return (val1-val2)
    
    def duffy_concentration(self, i, m200):
        """
        Compute the concentration of the Duffy et al. (2008)
        mass concentration relation for 200 rho_crit.
        Parameters:
        ===========
        m200: float, array_like or astropy.Quantity
            Mass of halo at 200rho_crit, [Msun] if float
        z: float or array_like
            Halo redshift
        cosmo: astropy.cosmology
        Returns:
        ========
        conc: float or array
            Halo concentration
        Notes:
        ======
        Halo masses must be given in physical units with factors of h
        divided out.
        """

        m200 = np.asanyarray(m200)
        z = self.list_z[i]
        m200 = u.Quantity(m200* self.cosmology.h, u.solMass) #* self.cosmology.h ???
        a = 5.71
        b = -0.084
        c = -0.47
        m_pivot = u.Quantity(2e12, u.solMass)
        conc = a * (m200 / m_pivot)**b * (1 + z)**c
        return conc.value
    
   
    
    
    def projNFW(self, i, params):
        
        z_cl = self.list_z[i]

        
        c500c = np.array(concentration.concentration(self.m500[i]*self.cosmology.h, '500c', z_cl, model = 'duffy08',range_warning = True))
        
        M200c,r200c,c200c = np.array(halo.mass_defs.changeMassDefinition(self.m500[i]*self.cosmology.h, c500c , z_cl, '500c', '200c'))
        

        rs = r200c/(params['c']*self.cosmology.h*1000.)

        
        rbin_edges = np.logspace(np.log10(params['rmin']),np.log10(params['rmax']),params['rbins']+1,endpoint=True)
        rmean = 2./3. *(rbin_edges[1:]**3-rbin_edges[:-1]**3)/ (rbin_edges[1:]**2-rbin_edges[:-1]**2)

        x = rmean[0:8] / rs #* np.pi/180. # , rs and rmean should both be in Mpc!

        x0=params['x0']
        
        result = self.get_Sigma(x)#/self.get_Sigma(x0)

        
        return result/self.get_Sigma(x0)
