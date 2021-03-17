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
    
    def __init__(self, cosmo1, constants, mapping, list_lambda, list_z, list_n_s, list_n_f, edges_f, list_num, m500): #randoms_var_smoothed):
        
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
        
        #self.var_sys = var_sys
        #self.var_tot = var_tot
        #self.relerrsys = relerrsys
        #self.smoothed_var_sys = smoothed_var_sys

        return
    
    def setup(self):
        
        
        
        return
    
    
    def computeLikelihood(self,ctx):
        
        p1 = ctx.getParams()
        #print p1
        
        params = self.constants.copy()
        #print params
        
        for key,value in self.mapping.items():
            params[key]=p1[value]
            #print params[key]
            #print params
            #print p1[value]
        
        lnL = 0
        for i in xrange(self.num_clusters):
            #print i
            
            if (self.list_num[i]==0).any():
                lnL += 0
            else:
                #lnL += self.single_clusterlike(i,params)
                lnL += self.single_clusterlike(i,params)
        
        #lnprior = -0.5 * (params['sigma0'] - 0.08)**2/0.04**2
        #lnprior += -0.5 * (params['mu0'] - 0.2)**2/0.05**2
        
        return lnL #+ lnprior
    
    
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
            #lnL += 200000
            
        if np.isnan(lnL):
            print 'cluster %i has nan likelihood'%i
        if not np.isfinite(lnL):
            print 'cluster %i has infinite likelihood'%i

        
        return lnL 
    
    
    def single_clusterlike_new(self,i,params):
        
        #bin_size = edges_f[1]-edges_f[0]
        
        n_s_ind = self.list_n_s[i]
        list_num_ind = self.list_num[i]
        var_sys_ind = self.var_sys[i]
        
        e = self.model(i,params)#[:,:70]
        #print e.shape
        
        if np.any(e <= 0): 
            lnL = -np.inf
            print lnL
            
        else:
            lnL = -0.5 * self.bin_size**2 * np.sum(((list_num_ind[:, None] * (e-n_s_ind))**2)/(list_num_ind[:, None]+ var_sys_ind[None,:]*list_num_ind[:, None]**2 * self.bin_size**2))
            #lnL = -0.5 * self.bin_size**2 *np.sum(((list_num_ind[:, None] * e-n_s_ind)**2)/(self.var_tot[None,:]* list_num_ind[:, None]**2)) 
            #lnL += 200000
            
        if np.isnan(lnL):
            print 'cluster %i has nan likelihood'%i
        if not np.isfinite(lnL):
            print 'cluster %i has infinite likelihood'%i

        
        return lnL 
    
    
    def cont_distr(self,i,params):
        

        Afcl_arr  = np.array([params['A_fcl_1'],params['A_fcl_2'],params['A_fcl_3'],params['A_fcl_4'],params['A_fcl_5'],params['A_fcl_6'],params['A_fcl_7'],params['A_fcl_8'],params['A_fcl_9'],params['A_fcl_10']])#,params['A_fcl_11']])#,params['A_fcl_12'],params['A_fcl_13'],params['A_fcl_14'],params['A_fcl_15']])

        #zbins = np.array([0.15,0.20,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.9])
        
        #zbins = np.array([0.15,0.20,0.25,0.30,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.3])
        #zbins = np.array([0.15,0.20,0.25,0.30,0.4,0.5,0.6,0.7,0.8,0.9,1.3])
        
        zbins = np.array([0.15,0.20,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.3])
        #zbins = np.array([0.1,0.15,0.20,0.25,0.30,0.4,0.5,0.6,0.7,1.3])
        #zbins = np.array([0.15,0.20,0.25,0.30,0.4,0.5,0.6,0.7,0.9])
        
        idx = np.digitize(self.list_z[i], zbins) - 1
        #print idx
        A_fcl = Afcl_arr[idx]
        
        Amp = A_fcl * (self.list_lambda[i]/params['lambda0'])**params['B_fcl']
        Amp *= self.projNFW(i, params)
        #print Amp
        #print f_cl.shape
        f_cl = Amp/(Amp+1)
        
        mu = self.list_z[i] + params['mu0'] * ((1.+self.list_z[i])/(1.+params['z0']))**params['C_mu']
        #mu = params['mu0'] * ((1.+self.list_z[i])/(1.+params['z0']))**params['C_mu']
        #mu = self.list_z[i] * params['C_mu'] +  params['mu0']
        try:
            sigma0 = params['sigma0']
        except KeyError:
            sigma0 = np.exp(params['lnsigma0'])
            
        sigma = sigma0 * ((1+self.list_z[i])/(1+params['z0']))**params['C_sigma']
        
        gaussian=1./np.sqrt(2.*np.pi*sigma**2)* np.exp(-0.5*(self.edges_f-mu)**2/sigma**2) 
        #print gaussian.shape
        
        integrad_in_bin = 0.5*(gaussian[1:]+gaussian[:-1]) #*(edges[1:]-edges[:-1])
        #print np.min(np.sum(integrad_in_bin))
        #selec =  np.where(np.sum(integrad_in_bin) == 0)
        #print integrad_in_bin[selec]
        #plt.plot(integrad_in_bin[selec])
        
        assert np.sum(integrad_in_bin) != 0
        #print np.where(np.sum(integrad_in_bin) != 0)
        #integrad_in_bin /= np.sum(integrad_in_bin*self.bin_size)  # make sure it sums to one
        #print np.sum(integrad_in_bin)
        
        return f_cl, integrad_in_bin
        
    def model(self,i,params):
        
        #b = np.array([params['b1'],params['b2'],params['b3'],params['b4'],params['b5'],params['b6'],params['b7'],params['b8']])

        f_cl, integrad_in_bin = self.cont_distr(i,params)
        
        n_f_ind = self.list_n_f[i]
        #n_f_ind = self.list_n_s[i,12,:]
        #print n_f_all.shape
        
        #if np.isnan(integrad_in_bin).any():
        #    print i
        
        if np.any(f_cl) <= 0:
            e = -1
        elif np.any(f_cl) > 1:
            e = -1
        else:
            e = (1-f_cl)[:,None] * n_f_ind[None,:] + f_cl[:,None] * integrad_in_bin[None,:]  # b[:,None]*
        
        #if np.any(f_cl) <= 0:
        #    e = -1
        #elif np.any(f_cl) > 1:
        #    e = -1
        
        #if np.any(e <= 0):
        #    idx = e <= 0
        #    e[idx] = 1e-6
            
        #print e.shape
        
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
    
    
    
    #def f(x):

    #    return x**3*(np.log(1.+1./x)-1./(1.+x))

    #def get_c500_from_c200(c200, amin=0.01, amax=100):

    #    g = lambda x: f(x) - 500/200*f(1./c200)

    #    return 1./op.brentq(g, amin, amax)
    
    
    #def get_M500_from_M200(self, i, M200, z):
    
    #    c200 = sr.concentration_Duffy08(M200, z)
    
    #    c500 = get_c500_from_c200(c200)
    
    #    return M200*500/200*(c500/c200)**3
    

    #def get_M200_from_M500(self, i,  M500, z):
    
    #    g = lambda x: M500 - get_M500_from_M200(x, z)
    
    #    return op.brentq(g, 1e10, 1e18)
    
    
    
    def projNFW(self, i, params):
        
        z_cl = self.list_z[i]
        
        #M200m = 10**params['M0']*(self.list_lambda[i]/30.)**params['Flam']*((1+z_cl)/1.5)**params['Gz']  # m refers to the mean matter density of the Universe
        #c200m = np.array(concentration.concentration(M200m*self.cosmology.h, '200m', z_cl, model = 'duffy08',range_warning = True))
        #M200c,r200c,c200c = np.array(halo.mass_defs.changeMassDefinition(M200m*self.cosmology.h, c200m , z_cl, '200m', '200c')) 
        # c refers to the critical density of the Universe
        #print c200c
        
        c500c = np.array(concentration.concentration(self.m500[i]*self.cosmology.h, '500c', z_cl, model = 'duffy08',range_warning = True))
        
        M200c,r200c,c200c = np.array(halo.mass_defs.changeMassDefinition(self.m500[i]*self.cosmology.h, c500c , z_cl, '500c', '200c'))
        
        #print r200c
        
        
        #conc = self.duffy_concentration(i, M200c)
        #print conc

        rs = r200c/(params['c']*self.cosmology.h*1000.)
        #rs = r200c/(c200c*self.cosmology.h*1000.)
        #rs = r200c/(conc*self.cosmology.h*1000.)
        #print type(rs)
        #print rs
        
        #Delta_c = 200./3. * params['c']**3. / (np.log(1.+params['c']) - params['c']/(1.+params['c']))
        #Delta_c = 200./3. * c200c**3. / (np.log(1.+c200c) - c200c/(1.+c200c))
    
        # how does your cosmology computer (self.cosmology) give you E(z)
        rho_crit_z = ccm.rho_crit * self.cosmology.Ez(self.list_z[i])**2
        
        #frac = 2 * rs * rho_crit_z * Delta_c
        
        rbin_edges = np.logspace(np.log10(params['rmin']),np.log10(params['rmax']),params['rbins']+1,endpoint=True)
        rmean = 2./3. *(rbin_edges[1:]**3-rbin_edges[:-1]**3)/ (rbin_edges[1:]**2-rbin_edges[:-1]**2)
        #r = np.insert(rmean,0,0)
        #print r
        #print type(rmean)
        #print rs
        #print rmean.shape
        # x = radial bin centers / rs
        x = rmean[0:8] / rs #* np.pi/180. # , rs and rmean should both be in Mpc!
        #x = rmean / rs
        #print x.shape
        #s = np.linspace(0.01,2.,99)
        #s = np.array([0.99,1.01])
        x0=params['x0']
        
        result = self.get_Sigma(x)#/self.get_Sigma(x0)
        #print result.shape
        #print result
        #print result/np.sum(result)
        #print self.get_Sigma(x0).shape
        #print self.get_Sigma(x)/self.get_Sigma(x0)
        
        return result/self.get_Sigma(x0)