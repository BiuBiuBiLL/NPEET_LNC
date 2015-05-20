NPEET_LNC
=====

Mutual information Estimation with Local Non-uniformity Correction (a branch of <a href="https://github.com/gregversteeg/NPEET/">NPEET</a> Non-parametric Entropy Estimation Toolbox)

This package contains Python code implementing mutual information estimation functions for continuous variables. This estimator gives a correction term to the traditional kNN estimator and can estimate mutual information more accurately than Kraskov estimator for strongly dependent variables with limited samples.

To use this package, it requires <a href="http://www.scipy.org">scipy</a> 0.12 or greater.


 
Example installation and usage:

git clone https://github.com/BiuBiuBiLL/NPEET_LNC.git

```python
>>> from lnc import MI
>>> import numpy as np
>>> x = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0]
>>> y = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0]
>>> MI.mi_LNC([x,y],k=5,base=np.exp(1),alpha=0.25) #Apply LNC estimator
Output: 25.29758574548632
>>> MI.mi_Kraskov([x,y],k=5,base=np.exp(1)) #Same data applied to Kraskov estimator
Output: 0.62745310245310382
```


To run the test example in ```test.py```, simply type the following in terminal

```shell
$python test.py
Output:
Testing 2D linear relationship Y=X+Uniform_Noise
noise level=1e-07, Nsamples = 500
True MI(x:y) 16.0472260191
Kraskov MI(x:y) 5.79082342999
LNC MI(x:y) 15.9355050641

Testing 2D quadratic relationship Y=X^2+Uniform_Noise
noise level=1e-07, Nsamples = 1000
True MI(x:y) 15.8206637545
Kraskov MI(x:y) 6.48347086055
LNC MI(x:y) 11.4586276609

Testing 3D linear relationship Y=X+Uniform_Noise, Z=X+Uniform_Noise
noise level=1e-07, Nsamples = 500
True MI(x:y:z) 32.2836569188
Kraskov MI(x:y:z) 11.58164686
LNC MI(x:y:z) 32.1846129957

Testing 3D quadratic relationship Y=X^2+Uniform_Noise, Z=X^2+Uniform_Noise
noise level=1e-07, Nsamples = 500
True MI(x:y:z) 31.5020968975
Kraskov MI(x:y:z) 11.57764686
LNC MI(x:y:z) 25.6686276941
```

One need to specify the thresholding parameter alpha when using LNC estimator. This parameter is related to the nearest-neighbor parameter ```k``` and  dimensionality ```d```, see ```alpha.xlsx``` for the detailed alpha value to use.

Also see the references on the implemented estimators.

				A Kraskov, H Stögbauer, P Grassberger. 
				http://pre.aps.org/abstract/PRE/v69/i6/e066138
				Estimating Mutual Information
				PRE, 2004.

				Shuyang Gao, Greg Ver Steeg and Aram Galstyan 
				http://arxiv.org/abs/1411.2003
				Efficient Estimation of Mutual Information for Strongly Dependent Variables
				AISTATS, 2015.


				
