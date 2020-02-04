import os, sys
import numpy as np
from sklearn.neighbors import KernelDensity


def confinterval(xs, weights=None, conflevel=0.6826, cumulative=False):
    """
    Hand-made weighted quantile/percentile computation tool, since there is no numpy equivalent.

    :param xs: list, distribution
    :param weights: list, weights of the distribution
    :param conflevel: float, quantile (between 0 and 1) of the conf interval
    :param cumulative: bool. 
          * If set to True, return the max value of xs corresponding to the conflevel. For example, if conflevel=0.68, return the value such that xs<=val at 68% confidence.
          * If set to False, return the values of xs corresponding to conflevel/2.0 around the 50th percentile (median). For example, if conflevel = 0.68, return the 16th, 50 and 84th percentiles of xs.
          
    :return: 3 or 1 float(s), see cumulative description.
    
    __warning:: This function is not optimized, so it can be really slow if you apply it to very large data sets.
                However, for the samples we are dealing with here (around 50k), that's enough (~seconds)
    
    __note:: This function has been compared to numpy.percentiles for uniform weights 
                and give similar results (at 0.03%). For non-uniform weights, it is able to reproduce the
                results from the Planck collaboration with the same level of precision
    
    """

    hist, bin_edges = np.histogram(xs, weights=weights, bins=2000)

    if not cumulative:
        frac = 0
        for ind, val in enumerate(hist):
            frac += val
            if frac/sum(hist) > 0.5:
                meanval = (bin_edges[ind]+bin_edges[ind-1])/2.0
                break

        lowerhalf = 0
        upperhalf = 0
        for ind, val in enumerate(bin_edges[:-1]):
            if val <= meanval:
                lowerhalf += hist[ind]
            elif val >= meanval:
                upperhalf += hist[ind]


        limfrac = (1.0 - conflevel) / 2.0
        lowerfrac, upperfrac = 0, 0
        for ind, val in enumerate(hist):
            lowerfrac += val
            if lowerfrac/sum(hist) >= limfrac:
                bottomlimit = (bin_edges[ind]+bin_edges[ind-1])/2.0
                break

        for ind, val in enumerate(hist):
            upperfrac += val
            if upperfrac/sum(hist) >= 1.0 - limfrac:
                toplimit = (bin_edges[ind]+bin_edges[ind+1])/2.0
                break

        return meanval, (meanval-bottomlimit), -1.0*(meanval-toplimit)

    else:
        frac = 0
        for ind, val in enumerate(hist):
            frac += val
            if frac/sum(hist) >= conflevel:
                return (bin_edges[ind]+bin_edges[ind+1])/2.0
               

class Estimate():
    
    def __init__(self, name, mean=None, std=None, vals=None, weights=None):
        """
        Class to deal with an H0 estimate. 
        It can hold either a Gaussian approximation (using mean and std) or a list of weighted values (using vals and weights).
        
        :param name: string, name given to your estimate.
        :param 

        """
        
        if vals is None:
            assert (mean is not None and std is not None) 
                    
        self.name = name
        self.mean = mean
        self.std = std
        self.vals = vals
        self.weights = weights

        self.pcs = None
        
        if vals is not None:
            self.loglikelihood = self.kdelikelihood_hist()

    def kdelikelihood_hist(self, kde_kernel="gaussian", bandwidth=0.1, nbins_hist=200): 
        
        if self.weights is not None:
            weights = self.weights
        else:
            weights = np.ones(len(self.vals))
            
        hist = np.histogram(self.vals, bins=nbins_hist, weights=weights)
        vals = hist[0]
        bins = [(h + hist[1][i+1])/2.0 for i, h in enumerate(hist[1][:-1])]

        # ignore potential zero weights, sklearn does not like them
        kde_bins = [(b,) for v, b in zip(vals, bins) if v>0]
        kde_weights = [v for v in vals if v>0]
        
        kde = KernelDensity(kernel=kde_kernel, bandwidth=bandwidth).fit(kde_bins, sample_weight=kde_weights)        
        return kde.score_samples         
        
   
    def compute_percentiles(self):
        if self.vals is None:
            self.pcs = (self.mean-self.std, self.mean, self.mean+self.std)
        elif self.weights is None:
            self.pcs = np.percentile(a=self.vals, q=[16, 50, 84])
        else:
            ci = confinterval(xs=self.vals, weights=self.weights, conflevel=0.68)
            self.pcs = (ci[0]-ci[1], ci[0], ci[0]+ci[2])
      
    def getpcs(self):
        if self.pcs is None:
            self.compute_percentiles()
        return self.pcs
    
    
    def evaluate(self, H0s):
        if self.vals is None:
            evals = np.exp(-(H0s - self.mean)**2 /(2*self.std **2))
        else:
            evals = np.exp(self.loglikelihood(H0s.reshape(-1, 1)))
        return evals / np.sum(evals)

    
    
            
def combine_estimates(estimates, H0s):
    weightslist = [est.evaluate(H0s) for est in estimates] 
    combweights =  np.prod(weightslist, axis=0)
    combname = " x ".join([est.name for est in estimates])
    return Estimate(name=combname, vals=H0s, weights=combweights)
    

def compute_tension(est1, est2):
    if est1.getpcs()[1] >= est2.getpcs()[1]:
        high = est1
        low = est2
    else:
        high = est2
        low = est1
        
    meddiff = (high.getpcs()[1] - low.getpcs()[1])
    lowerr = low.getpcs()[2] - low.getpcs()[1]
    higherr = high.getpcs()[1] - high.getpcs()[0]
    
    return meddiff / np.sqrt(lowerr**2 + higherr**2)    