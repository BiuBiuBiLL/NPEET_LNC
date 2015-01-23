#Mutual Information Estimation by Kraskov
#Mutual Inofmration Estimation by modifying Kraskov

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
	def corrected(V_e, V_u, kk):
		if V_e >= V_u:
			return V_u;
		upper = (V_u**2)*(V_e**kk)*((kk-3)*V_e-(kk-2)*V_u)+(V_e**3)*(V_u**kk);
		lower = V_u * (V_e**kk)*((kk-2)*V_e-kk*V_u+V_u) + (V_e**2)*(V_u**kk);
		ret = float(kk-1)/float(kk-3) * float(upper) / float(lower);
		if ret <= 0:
			return V_e;
		return ret;


	@staticmethod
	def Kraskov_mi1(x,y,k=3,base=np.exp(1)):
		  """ Mutual information of x and y
		      x,y should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
		      if x is a one-dimensional scalar and we have four samples
		  """
		  assert len(x)==len(y), "Lists should have same length"
		  assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
		  intens = 0.0 #small noise to break degeneracy, see doc.
		  x = [list(p + intens*nr.rand(len(x[0]))) for p in x]
		  y = [list(p + intens*nr.rand(len(y[0]))) for p in y]
		  points = MI.zip2(x,y)
		  #Find nearest neighbors in joint space, p=inf means max-norm
		  tree = ss.cKDTree(points)
		  dvec = [tree.query(point,k+1,p=float('inf'))[0][k] for point in points]
		  a,b,c,d = MI.avgdigamma(x,dvec), MI.avgdigamma(y,dvec), digamma(k), digamma(len(x)) 
		  return (-a-b+c+d)/log(base)

	@staticmethod
	def Kraskov_mi2(x,y,k=3,base=np.exp(1),intens = 1e-8):
		""" Mutual information of x and y
		    x,y should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
		    if x is a one-dimensional scalar and we have four samples
		"""
		assert len(x)==len(y), "Lists should have same length"
		assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
		x = [list(p + intens*nr.rand(len(x[0]))) for p in x]
		y = [list(p + intens*nr.rand(len(y[0]))) for p in y]
		points = MI.zip2(x,y)
		#Find nearest neighbors in joint space, p=inf means max-norm
		tree = ss.cKDTree(points)
		dvec_x = [];
		dvec_y = [];
		for point in points:
			knn = tree.query(point,k+1,p=float('inf'));
			dvec_x.append(float('-inf'));
			dvec_y.append(float('-inf'));
			points_x = [];
			points_y = [];
			for j in range(k+1):
				tem = [];
				for i in range(len(x[0])):
					tem.append(points[knn[1][j]][i]);		
				points_x.append(tem);
				tem = [];
				for i in range(len(x[0]), len(y[0])+len(x[0])):
					tem.append(points[knn[1][j]][i]);
				points_y.append(tem);
			for i in range(k+1):
				if dvec_x[-1] < norm(np.array(points_x[i])-np.array(points_x[0]),float('inf')):
					dvec_x[-1] = norm(np.array(points_x[i])-np.array(points_x[0]),float('inf'));
				if dvec_y[-1] < norm(np.array(points_y[i])-np.array(points_y[0]),float('inf')):
					dvec_y[-1] = norm(np.array(points_y[i])-np.array(points_y[0]),float('inf'));

		a,b,c,d = MI.avgdigamma(x,dvec_x), MI.avgdigamma(y,dvec_y), digamma(k), digamma(len(x)) 
		return (-a-b+c+d-1./float(k))/log(base)



	
	@staticmethod
	def centered_pca_mi2(x,y,k=5,base=np.exp(1), BIC_theta=0.5, intens = 1e-8):
		""" Mutual information of x and y
		    x,y should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
		    if x is a one-dimensional scalar and we have four samples
		"""
		assert len(x)==len(y), "Lists should have same length"
		assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
		#intens = 0.0 #small noise to break degeneracy, see doc.
		x = [list(p + intens*nr.rand(len(x[0]))) for p in x]
		y = [list(p + intens*nr.rand(len(y[0]))) for p in y]
		points = MI.zip2(x,y)
		N = len(points);
		#Find nearest neighbors in joint space, p=inf means max-norm
		tree = ss.cKDTree(points)
		dvec_x = [];
		dvec_y = [];
		
		for point in points:
			knn = tree.query(point,k+1,p=float('inf'));
			dvec_x.append(float('-inf'));
			dvec_y.append(float('-inf'));
			points_x = [];
			points_y = [];
			for j in range(k+1):
				tem = [];
				for i in range(len(x[0])):
					tem.append(points[knn[1][j]][i]);		
				points_x.append(tem);
				tem = [];
				for i in range(len(x[0]), len(y[0])+len(x[0])):
					tem.append(points[knn[1][j]][i]);
				points_y.append(tem);
			for i in range(k+1):
				if dvec_x[-1] < norm(np.array(points_x[i])-np.array(points_x[0]),float('inf')):
					dvec_x[-1] = norm(np.array(points_x[i])-np.array(points_x[0]),float('inf'));
				if dvec_y[-1] < norm(np.array(points_y[i])-np.array(points_y[0]),float('inf')):
					dvec_y[-1] = norm(np.array(points_y[i])-np.array(points_y[0]),float('inf'));

		a,b,c,d = MI.avgdigamma(x,dvec_x), MI.avgdigamma(y,dvec_y), digamma(k), digamma(len(x)) 	
		#Correction Term 
		e = 0.
		tot = -1;
		cur_tot = random.randint(0,N-1);
		plot_x = [];
		plot_y = [];
		xxx = [];
		yyy = [];
  		for point in points:
			tot += 1;
			knn = tree.query(point,k+1,p=float('inf'));
			knn_points = [];
			knn_dist = [dvec_x[tot], dvec_y[tot]];
			xx = [];
			yy = [];	
			for i in range(k+1):
				tem = [];
				for j in range(len(point)):
					tem.append(points[knn[1][i]][j]);
				xx.append(tem[0]);
				yy.append(tem[1]);	
				knn_points.append(tem);
				
		
			#substract mean	
			for i in range(len(point)):
				avg = knn_points[0][i];
				for j in range(k+1):
					knn_points[j][i] -= avg;

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
						avg += knn_points[ii][i] * knn_points[ii][j] / float(k-1);
					covr[i][j] = avg;

			w, v = la.eig(covr);	
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

			log_BIC_theta = log(BIC_theta);
			if V_rect >= (log(knn_dist[0]) + log(knn_dist[1])) + log_BIC_theta:
				V_rect = log(knn_dist[0]) + log(knn_dist[1]);

			if (log(knn_dist[0]) + log(knn_dist[1]) - V_rect) > 0:
				e += (log(knn_dist[0]) + log(knn_dist[1]) - V_rect)/N;
		return (-a-b+c+d+e-1./float(k))/log(base);
	@staticmethod
	#unstable, to be continued..
	def non_centered_pca_mi2(x,y,k=5,base=np.exp(1),BIC_theta=0.25, intens = 1e-10):
		""" Mutual information of x and y
		    x,y should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
		    if x is a one-dimensional scalar and we have four samples
		"""
		assert len(x)==len(y), "Lists should have same length"
		assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
	#	intens = 1e-10 #small noise to break degeneracy, see doc.
		x = [list(p + intens*nr.rand(len(x[0]))) for p in x]
		y = [list(p + intens*nr.rand(len(y[0]))) for p in y]
		points = MI.zip2(x,y)
		N = len(points);
		#Find nearest neighbors in joint space, p=inf means max-norm
		tree = ss.cKDTree(points)
		dvec_x = [];
		dvec_y = [];
		for point in points:
			knn = tree.query(point,k+1,p=float('inf'));
			dvec_x.append(float('-inf'));
			dvec_y.append(float('-inf'));
			points_x = [];
			points_y = [];
			for j in range(k+1):
				tem = [];
				for i in range(len(x[0])):
					tem.append(points[knn[1][j]][i]);		
				points_x.append(tem);
				tem = [];
				for i in range(len(x[0]), len(y[0])+len(x[0])):
					tem.append(points[knn[1][j]][i]);
				points_y.append(tem);
			for i in range(k+1):
				if dvec_x[-1] < norm(np.array(points_x[i])-np.array(points_x[0]),float('inf')):
					dvec_x[-1] = norm(np.array(points_x[i])-np.array(points_x[0]),float('inf'));
				if dvec_y[-1] < norm(np.array(points_y[i])-np.array(points_y[0]),float('inf')):
					dvec_y[-1] = norm(np.array(points_y[i])-np.array(points_y[0]),float('inf'));

		a,b,c,d = MI.avgdigamma(x,dvec_x), MI.avgdigamma(y,dvec_y), digamma(k), digamma(len(x)) 	
		#Correction Term 
		e = 0.
		tot = -1;
		tot1 = 0;
  		for point in points:
			tot += 1;
			knn = tree.query(point,k+1,p=float('inf'));
			knn_points = [];
			for i in range(k+1):
				tem = [];
				for j in range(len(point)):
					tem.append(points[knn[1][i]][j]);
				knn_points.append(tem);
			covariance = np.cov(np.array(knn_points).T);
			w, v = la.eig(covariance);
			#substract mean
			for i in range(len(point)):
				avg = 0.;
				for j in range(k+1):
					avg += float(knn_points[j][i]) / float(k+1);
				for j in range(k+1):
					knn_points[j][i] -= avg;
			log_V_rect = 0;
			cur = [];
			knn_dist = [dvec_x[tot], dvec_y[tot]];
			
			for i in range(len(point)):
				maxV = 0.
				for j in range(0,k+1):
					tem = 0.;
					for jj in range(len(point)):
						tem += v[jj,i] * knn_points[j][jj];
					if fabs(tem) > maxV:
						maxV = fabs(tem);
						

				cur.append(maxV);
		
			cur.sort();
			knn_dist.sort();
			m = 0;
		#	if log(cur[0]) + log(cur[1]) > log(knn_dist[0]) + log(knn_dist[1]) + log(BIC_theta):
		#		continue;
				#cur[0] = cur[0];
			for i in range(len(point)):
			#	print cur[i] / knn_dist[i];
				if log(cur[i]) > log(knn_dist[i]) + log(BIC_theta):	
					#if cur[i] < knn_dist[i]:
					 cur[i] = knn_dist[i];
				else:
					cur[i] = MI.corrected(cur[i] * 2, knn_dist[i] * 2, k+1) /2.; 
						
			if (log(knn_dist[0]) + log(knn_dist[1]) - log(cur[0]) - log(cur[1])) > 0:
				e += (log(knn_dist[0]) + log(knn_dist[1]) - log(cur[0]) - log(cur[1]))/N;
		return (-a-b+c+d+e-1./float(k))/log(base);

	@staticmethod
	def entropy(x,k=3,base=np.exp(1),intens=0.0):
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


		
	
		 
	


