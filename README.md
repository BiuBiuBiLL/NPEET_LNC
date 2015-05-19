NPEET_LNC
=====

Mutual information Estimation with Local Non-uniform Correction (a branch of <a href="https://github.com/gregversteeg/NPEET/">NPEET</a> Non-parametric Entropy Estimation Toolbox)

This package contains Python code implementing mutual information estimation functions for continuous variables. This estimator gives a correction term to the traditional kNN estimator and can estimator mutual information accurately for strongly dependent variables with limited samples.

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
		
See the references on implemented estimators.

				A Kraskov, H Stögbauer, P Grassberger. 
				http://pre.aps.org/abstract/PRE/v69/i6/e066138
				Estimating Mutual Information
				PRE, 2004.

				Shuyang Gao, Greg Ver Steeg and Aram Galstyan 
				http://arxiv.org/abs/1411.2003
				Efficient Estimation of Mutual Information for Strongly Dependent Variables
				AISTATS, 2015.


				
