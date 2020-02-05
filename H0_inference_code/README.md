# H<sub>0</sub> inference code. 

This notebook should help you to reproduce our inference of the cosmological parameters in different cosmologies as presented in <a href="https://arxiv.org/abs/1907.04869">Wong et al. 2019</a>. You should first download the distance posteriors from the H0LiCOW analysis available [here](../h0licow_distance_chains). 

The cosmologies currently implemented are using the following priors : 
* FLCMD : flat-&Lambda;CDM cosmology, H<sub>0</sub> uniform in [0:150], &Omega;<sub>m</sub> in [0.05:0.5]
* ULCDM : flat-&Lambda;CDM cosmology, H<sub>0</sub> uniform in [0:150], &Omega;<sub>m</sub> fixed to 0.3.
* oLCDM : open-&Lambda;CDM cosmology with H<sub>0</sub> uniform in [0:150], &Omega;<sub>m</sub>  in [0.05:0.5], &Omega;<sub>k</sub>  in [-0.5:0.5].
* FwCDM : flat-wCDM cosmology, with H<sub>0</sub> uniform in [0:150], &Omega;<sub>m</sub> in [0.05:0.5], w in [-2.5:0.5].
* w0waCDM : flat w0waCDM cosmology, with H<sub>0</sub> uniform in [0:150], &Omega;<sub>m</sub>  in [0.05:0.5], w<sub>0</sub> in [-2.5:0.5], w<sub>a</sub> in [-2:2]

With a little bit of patience, you should be able to reproduce Fig. 2 of <a href="https://arxiv.org/abs/1907.04869">Wong et al. 2019</a>. : 
![H0_FLCDM.png](https://raw.githubusercontent.com/shsuyu/H0LiCOW-public/master/H0_inference_code/H0_FLCDM.png)


If you make use of this python notebook for H0 inference, please reference <a href="https://zenodo.org/record/3633035#.XjrsIhd7k0o">this Zenodo link</a> (bibtex entry <a href="https://zenodo.org/record/3633035/export/hx#.XjrsRhd7k0o">here</a>). 

### Requirements 
This notebook makes use of the following python package : 
 * `Lenstronomy` 
 * `numpy`
 * `pandas`
 * `sklearn`
 * `matplotlib`
 * `emcee`
 * `pickle`
 
 These packages can be installed with a simple : `pip install `
 
 
#### Contributors
Vivien Bonvin (vivien.bonvin@protonmail.ch)  
Martin Millon (martin.millon@epfl.ch)