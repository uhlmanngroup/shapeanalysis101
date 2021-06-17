"""
@author: Virginie UHLMANN, Anna SONG

Please acknowledge by citing:
Song, A., Uhlmann, V., Fageot, J., & Unser, M. (2020). Dictionary learning for two-dimensional Kendall shapes. SIAM Journal on Imaging Sciences, 13(1), 141-175.
"""

import numpy as np
from scipy.integrate import quad
from scipy.linalg import sqrtm
import sys
sys.path.append('./spline_models')
from spline_curve_model import B3

def innerProductL2(u,v):
    integral = quad(lambda t : u(t)*v(t),0,1)
    return integral[0]

def B3_Phi(M):
    Phi=np.zeros((M,M))
    def aux(k):
        return lambda t : B3().value(M*(t-1) - k) + B3().value(M*t - k) + B3().value(M*(t+1) - k)
    for k in range(M):
        for l in range(M):
            u = aux(k)
            v = aux(l)
            Phi[k,l] = innerProductL2(u,v)
    return Phi

class B3_shape_space:
    def __init__(self, M, closed):
        self.closed=closed
        if self.closed:
            self.M=M
        else:
            self.M=M+4

        self.Phi=B3_Phi(self.M)

    def hermitianProduct(self,z,w):
        '''Hermitian product of z and w in (C^N,Phi)'''
        return z.conj().T @ self.Phi @ w

    def configurationNorm(self,z):
        '''Norm of z in C^N w.r.t the Hermitian product Phi.'''
        squaredNorm=z.conj().T @ self.Phi @ z
        return np.sqrt(squaredNorm).real

    def configurationMean(self,z):
        '''Computes the temporal mean of the configuration z in C^N.'''
        m = np.mean(z)
        return m

    def preshape(self,z):
        '''Centers and normalizes the configuration z in C^N.'''
        m = self.configurationMean(z)
        meanSuppressed = z-m
        s = self.configurationNorm(meanSuppressed)
        return meanSuppressed/s

    def preshapeDataset(self,dataset):
        '''Centers and normalizes a dataset.'''
        K = len(dataset)
        preshaped = np.zeros_like(dataset)
        for k in range(K) :
            preshaped[k] = self.preshape(dataset[k])
        return preshaped

    def meanFrechet(self,dataset): 
        '''Input: dataset of preshapes.
        Output: Fréchet mean (w.r.t. the distance d_F) of the dataset of shapes.
        It is mathematically a shape, numerically handled as a preshape.'''
        SQ = dataset.T @ dataset.conj() @ self.Phi
        D,V = np.linalg.eig(SQ)
        ds = np.real(D)
        inds = np.argsort(ds)[::-1]
        m = V[:,inds[0]]
        m = self.preshape(m) 
        return m

    def theta(self,z,w): 
        '''Computes the optimal angle theta(z,w) = arg(z* Phi w).'''
        return np.angle(self.hermitianProduct(z,w))

    def align(self,z,w):
        '''Optimally rotate the shape z along w.'''
        ta = self.theta(z,w)
        rotated = np.exp(1j*ta)*z
        return rotated

    def alignDataset(self,dataset):
        '''Optimally rotate the shapes along their Fréchet mean with respect to d_F. '''
        K = len(dataset)
        rotated = np.zeros_like(dataset)
        shapeMean = self.meanFrechet(dataset)
        for k in range(K) :
            rotated[k] = self.align(dataset[k],shapeMean)
        return rotated

    def geodesicDistance(self,z,w): 
        '''Geodesic distance between [z] and [w].'''
        aux = np.abs(self.hermitianProduct(z,w))
        if aux > 1.0 : # catches numerical errors
            aux = 1.0
        return np.arccos(aux)

    def fullDistance(self,z,w):
        '''Full Procrustes distance between [z] and [w]'''
        return np.sqrt(1.0 - np.abs(hermitianProduct(z,w))**2)

    def partialDistance(self,z,w): 
        '''Partial Procrustes distance between [z] and [w]'''
        return np.sqrt(2.0 - 2.0*np.abs(hermitianProduct(z,w)))
    
    def geodesicPath(self,z,w,numSteps = 5): 
        '''Returns elements regularly spaced along the geodesic curve joining z to w (preshapes).'''
        ro = self.geodesicDistance(z,w)
        steps = np.arange(numSteps+1)/numSteps

        ta = self.theta(z,w)
        path = 1/np.sin(ro)*(np.sin((1-steps[:,None])*ro)*np.exp(1j*ta)*z + np.sin(steps[:,None]*ro)*w)
        return path

    def exponentialMap(self,z,v): 
        '''Considering z preshape, v in C^n referring to a tangent vector, computes the exponential of v at z, that corresponds to a preshape.'''
        t = self.configurationNorm(v)
        if t < 1e-16 : # catches numerical errors
            return z
        return np.cos(t)*z + v*np.sin(t)/t

    def log(self,z,w) : 
        '''Computes a preshape pertaining to the shape (equivalence class) log_[z] ([w])
        where log is relative the shape space Sigma.'''
        ta = self.theta(z,w)
        wt=np.exp(-1j*ta)*w
        ro = self.geodesicDistance(z,wt)
        return ro/np.sin(ro)*(wt - np.cos(ro)*z)

class B3_PCA:
    def __init__(self, M, closed, frechetMean):
        self.closed=closed
        if self.closed:
            self.M=M
        else:
            self.M=M+4

        self.frechetMean=frechetMean

        Phi = B3_Phi(self.M)
        self.Psi = np.concatenate((np.hstack((Phi,0*Phi)),np.hstack((0*Phi,Phi))))

        self.pcPhi=None
        self.diag=None
        self.complexPc=None

    def fit(self, data):
        ''' Classical linear PCA

        Parameters:
            - data in C^{(K,n)} is a complex array containing the horizontally stacked dataset [z_1,...,z_K]^T

        Returns: 
            - Eigenvalues
            - Eigenvectors (eigenmodes)
            
        '''

        K=len(data)

        mean = np.mean(data,0) # not necessarily zero
        V=(data - mean)
        V=data

        sqrtPsi = sqrtm(self.Psi)

        Vr = np.zeros((K,2*self.M))
        Vr[:,:self.M] = V.real
        Vr[:,self.M:] = V.imag

        Y = Vr @ sqrtPsi
        Diag,Vmodes = np.linalg.eig(Y.T @ Y)
        Diag = Diag.real
        Vmodes = Vmodes.real

        sortindr = np.argsort(Diag)
        sortindr = sortindr[::-1] # decreasing order of eigenvalues

        self.diag = Diag[sortindr]
        Vmodes = Vmodes[:,sortindr]
        
        Wmodes = np.linalg.inv(sqrtPsi) @ Vmodes # eigenmodes in the tangent plane at m
        W=np.zeros((self.M,2*self.M), dtype=complex)
        W.real=Wmodes[:self.M,:]
        W.imag=Wmodes[self.M:,:]
        self.pcPhi=Vmodes
        self.complexPC=W

    def project(self, v):
        sqrtPsi = sqrtm(self.Psi)

        vr = np.zeros((2*self.M))
        vr[:self.M] = v.real
        vr[self.M:] = v.imag

        pcWeight=np.zeros((2*self.M))
        for k in range(2*self.M):
            y = vr @ sqrtPsi
            pcWeight[k] = y @ self.pcPhi[:,k]

        return pcWeight

    def reconstruct(self, v, count=0):
        pcWeight=self.project(v)
        if count<1:
            count=len(pcWeight)

        reconstruction=np.zeros((self.M), dtype=complex)
        for k in range(count):
            reconstruction+=(pcWeight[k]*self.complexPC[:,k])

        return reconstruction