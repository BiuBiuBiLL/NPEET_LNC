#Python 2.7
#Written by Shuyang Gao (BiLL), email: gaos@usc.edu


from scipy import stats
import numpy as np
import scipy.spatial as ss
from scipy.special import digamma,gamma
import numpy.random as nr
import random
import matplotlib.pyplot as plt
import re
from scipy.stats.stats import pearsonr
import numpy.linalg as la
from numpy.linalg import eig, inv, norm, det
from scipy import stats
from math import log,pi,hypot,fabs,sqrt
class MI:
	
	@staticmethod
	def zip2(*args):
		#zip2(x,y) takes the lists of vectors and makes it a list of vectors in a joint space
		#E.g. zip2([[1],[2],[3]],[[4],[5],[6]]) = [[1,4],[2,5],[3,6]]
		return [sum(sublist,[]) for sublist in zip(*args)]
	
	@staticmethod
	def avgdigamma(points,dvec):
	 	#This part finds number of neighbors in some radius in the marginal space
		#returns expectation value of <psi(nx)>
		N = len(points)
		tree = ss.cKDTree(points)
		avg = 0.
		for i in range(N):
	  		dist = dvec[i]
	  		#subtlety, we don't include the boundary point, 
	  		#but we are implicitly adding 1 to kraskov def bc center point is included
	  		num_points = len(tree.query_ball_point(points[i],dist-1e-15,p=float('inf'))) 
	  		avg += digamma(num_points)/N
		return avg
	
	@staticmethod 
	def mi_Kraskov(X,k=5,base=np.exp(1),intens=1e-10):
		'''The mutual information estimator by Kraskov et al.
		   ith row of X represents ith dimension of the data, e.g. X = [[1.0,3.0,3.0],[0.1,1.2,5.4]], if X has two dimensions and we have three samples
		   	
		'''
		#adding small noise to X, e.g., x<-X+noise
		x = [];
		for i in range(len(X)):
			tem = [];
			for j in range(len(X[i])):
				tem.append([X[i][j] + intens*nr.rand(1)[0]]);
			x.append(tem);
	
		points = [];
		for j in range(len(x[0])):
			tem = [];
			for i in range(len(x)):	
				tem.append(x[i][j][0]);
			points.append(tem);
		tree = ss.cKDTree(points);
		dvec = [];
		for i in range(len(x)):
			dvec.append([])	
		for point in points:
			#Find k-nearest neighbors in joint space, p=inf means max norm
			knn = tree.query(point,k+1,p=float('inf'));
			points_knn = [];
			for i in range(len(x)):
				dvec[i].append(float('-inf'));
				points_knn.append([]);
			for j in range(k+1):
				for i in range(len(x)):
					points_knn[i].append(points[knn[1][j]][i]);

			#Find distances to k-nearest neighbors in each marginal space
			for i in range(k+1):
				for j in range(len(x)):
					if dvec[j][-1] < fabs(points_knn[j][i]-points_knn[j][0]):
						dvec[j][-1] =  fabs(points_knn[j][i]-points_knn[j][0]);

		ret = 0.
		for i in range(len(x)):
			ret -= MI.avgdigamma(x[i],dvec[i]);
		ret += digamma(k) - (float(len(x))-1.)/float(k) + (float(len(x))-1.) * digamma(len(x[0]));
		return ret;	

	@staticmethod 
	def mi_LNC(X,k=5,base=np.exp(1),alpha=0.25,intens = 1e-10):
		'''The mutual information estimator by PCA-based local non-uniform correction(LNC)
		   ith row of X represents ith dimension of the data, e.g. X = [[1.0,3.0,3.0],[0.1,1.2,5.4]], if X has two dimensions and we have three samples
		   alpha is a threshold parameter related to k and d(dimensionality), please refer to our paper for details about this parameter
		'''
		#N is the number of samples
		N = len(X[0]);
		
		#First Step: calculate the mutual information using the Kraskov mutual information estimator
		#adding small noise to X, e.g., x<-X+noise
		x = [];
		for i in range(len(X)):
			tem = [];
			for j in range(len(X[i])):
				tem.append([X[i][j] + intens*nr.rand(1)[0]]);
			x.append(tem);
	
		points = [];
		for j in range(len(x[0])):
			tem = [];
			for i in range(len(x)):	
				tem.append(x[i][j][0]);
			points.append(tem);
		tree = ss.cKDTree(points);
		dvec = [];
		for i in range(len(x)):
			dvec.append([])	
		for point in points:
			#Find k-nearest neighbors in joint space, p=inf means max norm
			knn = tree.query(point,k+1,p=float('inf'));
			points_knn = [];
			for i in range(len(x)):
				dvec[i].append(float('-inf'));
				points_knn.append([]);
			for j in range(k+1):
				for i in range(len(x)):
					points_knn[i].append(points[knn[1][j]][i]);
			
			#Find distances to k-nearest neighbors in each marginal space
			for i in range(k+1):
				for j in range(len(x)):
					if dvec[j][-1] < fabs(points_knn[j][i]-points_knn[j][0]):
						dvec[j][-1] =  fabs(points_knn[j][i]-points_knn[j][0]);

		ret = 0.
		for i in range(len(x)):
			ret -= MI.avgdigamma(x[i],dvec[i]);
		ret += digamma(k) - (float(len(x))-1.)/float(k) + (float(len(x))-1.) * digamma(len(x[0]));

		#Second Step: Add the correction term (Local Non-Uniform Correction)
		e = 0.
		tot = -1;
  		for point in points:
			tot += 1;
			#Find k-nearest neighbors in joint space, p=inf means max norm
			knn = tree.query(point,k+1,p=float('inf'));
			knn_points = [];
			for i in range(k+1):
				tem = [];
				for j in range(len(point)):
					tem.append(points[knn[1][i]][j]);
				knn_points.append(tem);
				
		
			#Substract mean	of k-nearest neighbor points
			for i in range(len(point)):
				avg = knn_points[0][i];
				for j in range(k+1):
					knn_points[j][i] -= avg;
		
			#Calculate covariance matrix of k-nearest neighbor points, obtain eigen vectors
			covr = [];
			for i in range(len(point)):
				tem = 0;
				covr.append([]);
				for j in range(len(point)):
					covr[i].append(0);
			for i in range(len(point)):
				for j in range(len(point)):
					avg = 0.
					for ii in range(1,k+1):
						avg += knn_points[ii][i] * knn_points[ii][j] / float(k);
					covr[i][j] = avg;
			w, v = la.eig(covr);
			

			#Calculate PCA-bounding box using eigen vectors
			V_rect = 0;
			cur = [];
			for i in range(len(point)):
				maxV = 0.
				for j in range(0,k+1):
					tem = 0.;
					for jj in range(len(point)):
						tem += v[jj,i] * knn_points[j][jj];
					if fabs(tem) > maxV:
						maxV = fabs(tem);						
				cur.append(maxV);
				V_rect = V_rect + log(cur[i]);
		
			#Calculate the volume of original box
			log_knn_dist = 0.;
			for i in range(len(dvec)):
				log_knn_dist += log(dvec[i][tot]);

			#Perform local non-uniformity checking
			if V_rect >= log_knn_dist + log(alpha):
				V_rect = log_knn_dist;

			#Update correction term
			if (log_knn_dist - V_rect) > 0:
				e += (log_knn_dist - V_rect)/N;

		return (ret + e)/log(base);
	
	@staticmethod
	def entropy(x,k=3,base=np.exp(1),intens=1e-10):
	  """ The classic K-L k-nearest neighbor continuous entropy estimator
	      x should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
	      if x is a one-dimensional scalar and we have four samples
	  """
	  assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	  d = len(x[0])
	  N = len(x)
	  x = [list(p + intens*nr.rand(len(x[0]))) for p in x]
	  tree = ss.cKDTree(x)
	  nn = [tree.query(point,k+1,p=float('inf'))[0][k] for point in x]
	  const = digamma(N)-digamma(k) + d*log(2)
	  return (const + d*np.mean(map(log,nn)))/log(base)

