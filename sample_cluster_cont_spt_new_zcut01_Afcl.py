import numpy as np
from cosmoHammer import LikelihoodComputationChain, CosmoHammerSampler, MpiCosmoHammerSampler
from cosmoHammer.util import FlatPositionGenerator
from ExtSampleFileUtil import ExtSampleFileUtil
import sys
sys.path.append('/home/grandis/source/')
from RestartFromChain import RestartFromChain
from astropy.io import fits
#from clust_cont_like_gauss_new import ClustContCorr
from clust_cont_like_Afcl import ClustContCorr
from ClustContCorrPrior import ClustContCorrPrior
from colossus.cosmology import cosmology
from colossus import halo
from colossus.halo import concentration

cosmo_params = {'flat': True, 'H0': 70., 'Om0': 0.3, 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95}
cosmo1 = cosmology.setCosmology('myCosmo', cosmo_params)


params = np.array([#[0.1, -0.2, 1.2,  0.04],
                   #[0.105, 0.001, 1.2,  0.004],
                   [0.26, 0.001, 1.2,  0.004],
                   [0.36, 0.001, 1.2,  0.004],
                   #[0.105, 0.001, 1.2,  0.004],
                   [0.51, 0.001, 1.2,  0.004],
                   #[0.11, 0.001, 1.2,  0.004],
                   [0.18, 0.001, 1.2,  0.004],
                   [0.19, 0.001, 1.2,  0.004],
                   [0.34, 0.001, 1.2,  0.004],
                   [0.21, 0.001, 1.2,  0.004],
                   [0.51, 0.001, 1.2,  0.004],
                   [0.56, -0.3, 3.2,  0.004],
                   #[0.10, 0.001, 1.2,  0.004],
                   #[0.10, 0.001, 1.2,  0.004],
                   #[0.55, 0.001, 3.2,  0.004],
                   [0.74, -0.001, 7.2,  0.004],
                   [0.7, -0.05, 1.2, 0.04],   
                   [0.15, 0.01, 0.2, 0.003],
                   [0.11, 0.02, 0.17, 0.003],
                   [3.8, 0.05, 7.0, 0.1]])

  
# define the likelihood computation chain using the choosen bounds
chain = LikelihoodComputationChain(params[:, 1], params[:, 2])


#a = 0   #154
#b = 289 #264 

#lamb0 = np.load('cluster_cont_like_data/lamb_spt_xi4_z015_z09_fcont01_mass.npy')[a:b]
#z0 = np.load('cluster_cont_like_data/z_spt_xi4_z015_z09_fcont01_mass.npy')[a:b]
#list_n_s10 = np.load('cluster_cont_like_data/list_n_s1_spt_xi4_z015_z09_fcont01_mass.npy')[a:b,:,:]
#edges_f = np.load('cluster_cont_like_data/edges_f_spt_xi4_z015_z09_fcont01_mass.npy')
#list_num0 = np.load('cluster_cont_like_data/list_num_spt_xi4_z015_z09_fcont01_mass.npy')[a:b,:]
#m5000 = np.load('cluster_cont_like_data/mass500_spt_xi4_z015_z09_fcont01_mass.npy')[a:b]



#lamb0 = np.load('cluster_cont_like_data/lamb_spt_xi4_z025_z09_fcont01_mass_01_30zbins.npy')[a:b]
#z0 = np.load('cluster_cont_like_data/z_spt_xi4_z025_z09_fcont01_mass_01_30zbins.npy')[a:b]
#list_n_s10 = np.load('cluster_cont_like_data/list_n_s1_spt_xi4_z025_z09_fcont01_mass_01_30zbins.npy')[a:b,:,:]
#list_n_f10 = np.load('cluster_cont_like_data/list_n_f1_spt_xi4_z025_z09_fcont01_mass.npy')[a:b,:]
#list_n_f10 = np.load('cluster_cont_like_data/list_n_f1_200_mock_both_spt.npy')#[a:b,:]
#edges_f = np.load('cluster_cont_like_data/edges_f_spt_xi4_z025_z09_fcont01_mass_01_30zbins.npy')
#list_num0 = np.load('cluster_cont_like_data/list_num_spt_xi4_z025_z09_fcont01_mass_01_30zbins.npy')[a:b,:]
#m5000 = np.load('cluster_cont_like_data/mass500_spt_xi4_z025_z09_fcont01_mass_01_30zbins.npy')[a:b]

#lamb0 = np.load('cluster_cont_like_data/lamb_spt_xi4_z015_z09_fcont01_mass_01_30zbins.npy')[a:b]
#z0 = np.load('cluster_cont_like_data/z_spt_xi4_z015_z09_fcont01_mass_01_30zbins.npy')[a:b]
#list_n_s10 = np.load('cluster_cont_like_data/list_n_s1_spt_xi4_z015_z09_fcont01_mass_01_30zbins.npy')[a:b,:,:]
#edges_f = np.load('cluster_cont_like_data/edges_f_spt_xi4_z015_z09_fcont01_mass_01_30zbins.npy')
#list_num0 = np.load('cluster_cont_like_data/list_num_spt_xi4_z015_z09_fcont01_mass_01_30zbins.npy')[a:b,:]
#m5000 = np.load('cluster_cont_like_data/mass500_spt_xi4_z015_z09_fcont01_mass_01_30zbins.npy')[a:b]

#lamb0 = np.load('cluster_cont_like_data/lamb_spt_xi4_z015_z09_fcont01_mass_01_30zbins_szcentres.npy')[a:b]
#z0 = np.load('cluster_cont_like_data/z_spt_xi4_z015_z09_fcont01_mass_01_30zbins_szcentres.npy')[a:b]
#list_n_s10 = np.load('cluster_cont_like_data/list_n_s1_spt_xi4_z015_z09_fcont01_mass_01_30zbins_szcentres.npy')[a:b,:,:]
#edges_f = np.load('cluster_cont_like_data/edges_f_spt_xi4_z015_z09_fcont01_mass_01_30zbins_szcentres.npy')
#list_num0 = np.load('cluster_cont_like_data/list_num_spt_xi4_z015_z09_fcont01_mass_01_30zbins_szcentres.npy')[a:b,:]
#m5000 = np.load('cluster_cont_like_data/mass500_spt_xi4_z015_z09_fcont01_mass_01_30zbins_szcentres.npy')[a:b]


#field0 = (list_n_s10[:,7,:] * list_num0[:,7,None] + list_n_s10[:,8,:] * list_num0[:,8,None])/(list_num0[:,7,None] + list_num0[:,8,None]) 

a = 0   
b = 327  

lamb0 = np.load('cluster_cont_like_data/lamb_spt_xi4_z015_z13_fcont01_mass_01_30zbins.npy')[a:b]
z0 = np.load('cluster_cont_like_data/z_spt_xi4_z015_z13_fcont01_mass_01_30zbins.npy')[a:b]
list_n_s10 = np.load('cluster_cont_like_data/list_n_s1_spt_xi4_z015_z13_fcont01_mass_01_30zbins.npy')[a:b,:,:]
edges_f = np.load('cluster_cont_like_data/edges_f_spt_xi4_z015_z13_fcont01_mass_01_30zbins.npy')
list_num0 = np.load('cluster_cont_like_data/list_num_spt_xi4_z015_z13_fcont01_mass_01_30zbins.npy')[a:b,:]
m5000 = np.load('cluster_cont_like_data/mass500_spt_xi4_z015_z13_fcont01_mass_01_30zbins.npy')[a:b]

field0 = list_n_s10[:,8,:]
#field0 = (list_n_s10[:,7,:] * list_num0[:,7,None] + list_n_s10[:,8,:] * list_num0[:,8,None])/(list_num0[:,7,None] + list_num0[:,8,None]) 


def get_mask(array):
    mask = ~np.isfinite(array.sum(axis=1).sum(axis=1))
    return mask

mask = get_mask(list_n_s10)
mask_inv = np.invert(mask)

lamb = lamb0[mask_inv]
z = z0[mask_inv]
list_n_s1 = list_n_s10[mask_inv]
field = field0[mask_inv]
list_num = list_num0[mask_inv]
m500 = m5000[mask_inv]


#lamb = np.delete(lamb,[0,2,3,6,10,19,21,22,23,27,32,36,64,99,111,113,126,132,133,139,148,152,180,182,188,191,197])
#z = np.delete(z,[0,2,3,6,10,19,21,22,23,27,32,36,64,99,111,113,126,132,133,139,148,152,180,182,188,191,197])
#list_n_s1 = np.delete(list_n_s1,[0,2,3,6,10,19,21,22,23,27,32,36,64,99,111,113,126,132,133,139,148,152,180,182,188,191,197], axis=0)
#list_n_f1 = np.delete(list_n_f1,[0,2,3,6,10,19,21,22,23,27,32,36,64,99,111,113,126,132,133,139,148,152,180,182,188,191,197], axis=0)
#field_200 = np.delete(field_200,[0,2,3,6,10,19,21,22,23,27,32,36,64,99,111,113,126,132,133,139,148,152,180,182,188,191,197], axis=0)
#field = np.delete(field,[0,2,3,6,10,19,21,22,23,27,32,36,64,99,111,113,126,132,133,139,148,152,180,182,188,191,197], axis=0)
#list_num = np.delete(list_num,[0,2,3,6,10,19,21,22,23,27,32,36,64,99,111,113,126,132,133,139,148,152,180,182,188,191,197], axis=0)
#m500 = np.delete(m500,[0,2,3,6,10,19,21,22,23,27,32,36,64,99,111,113,126,132,133,139,148,152,180,182,188,191,197], axis=0)
#smoothed_var_new = np.delete(smoothed_var_new,[0,2,3,6,10,19,21,22,23,27,32,36,64,99,111,113,126,132,133,139,148,152,180,182,188,191,197], axis=0)
#smoothed_cosmic_var = np.delete(smoothed_cosmic_var,[0,2,3,6,10,19,21,22,23,27,32,36,64,99,111,113,126,132,133,139,148,152,180,182,188,191,197], axis=0)

field[field==0] = 10e-6





mapping = {'A_fcl_1':0,
           'A_fcl_2':1,
           'A_fcl_3':2,
           'A_fcl_4':3,
           'A_fcl_5':4,
           'A_fcl_6':5,
           'A_fcl_7':6,
           'A_fcl_8':7,
           'A_fcl_9':8,
           'A_fcl_10':9,
           #'A_fcl_11':10,
           #'A_fcl_12':11,
           #'A_fcl_13':12,
           #'A_fcl_14':13,
           #'A_fcl_15':14,
           'B_fcl': 10,
           'mu0': 11,
           'sigma0': 12,
           'c': 13}



x0 = float(sys.argv[1])
print x0

constants = {'lambda0': 72.,   #76.,     
             'z0': 0.56,  #0.53,
             'C_mu': 0,
             'C_sigma': 0,
             'M0': 14.371,
             'Flam': 1.12,
             'Gz': 0.18,
             'rmin': 0.25,
             'rmax': 10.4,
             'rbins': 9,
             'x0': x0}

#c = ClustContCorr(cosmo1, constants, mapping, lamb, z, list_n_s1, list_n_f1, edges_f, list_num, m500, smoothed_var_sys, smoothed_cosmic_var)#, randoms_var_smoothed)
c = ClustContCorr(cosmo1, constants, mapping, lamb, z, list_n_s1[:,0:8,:], field, edges_f, list_num[:,0:8], m500)
#c = ClustContCorr(cosmo1, constants, mapping, lamb1, z1, list_n_s11, list_n_f11, edges_f, list_num1, m5001, smoothed_var_sys1, smoothed_cosmic_var1)

chain.addLikelihoodModule(c)

#priors = ClustContCorrPrior(mapping, keys=['B_fcl','mu0', 'sigma0','c'], means=np.array([0.34, 0.12, 0.113, 1.54]), sigma=np.array([0.04, 0.003, 0.003, 0.1]))
priors = ClustContCorrPrior(mapping, keys=['B_fcl','mu0', 'sigma0','c'], means=np.array([0.35, 0.11, 0.10, 1.7]), sigma=np.array([0.04, 0.003, 0.003, 0.11]))



#chain.addLikelihoodModule(priors)

# setup the chain
chain.setup()

# setting up the storage Util
fileprefix = 'chains/spt_poisson_zcut01_binAfcl_30zbins_015_13_old_Amp_v2_shifted1_norm%.3f'%x0  # prefix of all output files, can also be path
ini_pos = FlatPositionGenerator()  # RestartFromChain('erosita_test2.out')

# initializing the sampler, choose MpiCosmoHammerSampler(same input) for parallel computation
sampler = MpiCosmoHammerSampler(params=params,
                                likelihoodComputationChain=chain,
                                filePrefix=fileprefix,
                                walkersRatio=12,
                                burninIterations=0,
                                sampleIterations=1000,
                                #storageUtil=storage,
                                initPositionGenerator=ini_pos)

# start sampling
sampler.startSampling()