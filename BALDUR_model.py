# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 17:50:11 2023

@author: Albert
"""

import numpy as np
from scipy import linalg
import copy
from scipy.stats import norm
from sklearn.preprocessing import label_binarize
import math
import sys
from sklearn.metrics import hamming_loss
from sklearn.metrics import roc_curve, auc, mean_absolute_error, balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn import metrics
#import torch
#from torch import nn, optim
#import pyro.contrib.gp as gp
import pickle
import os
import time

class LR_ARD(object):    
    
    def __init__(self):
        pass

    def fit(self, z, y, z_tst = None, y_tst = None, states = None, hyper = None, prune = 0, prune_feat = 0, maxit = 500, 
            pruning_crit = 1e-6, pruning_feat_crit = 1e-6, tol = 1e-6, Kc = 4, direction = 0, fold = 0, seed = 0, Z_debug = [np.nan,np.nan], W_debug = [np.nan, np.nan], A_debug = [np.nan, np.nan], Wy_debug = np.nan, fix_tau = 0, fix_alpha = 0, fix_gamma = 0):

        self.X = z  #(NxK)
        self.X_tst = z_tst  #(NxK_tst)
        self.Xv = z.copy()
        self.t = y  #(NxD)
        self.t_tst = y_tst  #(NxD_tst)
        self.s = states
        self.direction = direction
        self.fold = fold
        self.seed = seed

        self.m = len(self.X)

        self.d = []
        for m1 in range(self.m):
            self.d.append(self.X[m1].shape[1])
        
        self.Q = self.t.shape[1] #num dimensiones output
        self.N = self.X[0].shape[0] # num datos
        self.Nv = self.Xv[0].shape[0]
        self.Kc = Kc
        self.N_tst = self.X_tst[0].shape[0]
        self.accs = []
        self.contador = 0
        ##########
        self.W_debug = W_debug
        self.Z_debug = Z_debug
        self.A_debug = A_debug
        ##########

        self.index_K = np.arange(self.Kc)
        self.index_feats = []
        for m2 in range(self.m):
            self.index_feats.append(np.arange(self.d[m2]))

        ##################
        self.accs = []
        self.bal_accs = []
        self.precisions = []
        self.recalls = []
        self.f1s = []
        self.aucs = []
        ###################

        self.L = []
        self.mse = []
        self.mse_tst = []        
        self.R2 = []
        self.R2_tst = []
        #self.AUC = []
        #self.AUC_tst = []
        self.K_vec = []
        self.labels_pred = []
        #self.input_idx = np.ones(self.K, bool)
        if hyper == None:
            self.hyper = HyperParameters(self.m,self.Kc)
        else:
            self.hyper = hyper
        self.q_dist = Qdistribution(self.N, self.Nv, self.d, self.Q,self.Kc, self.m, self.s,self.hyper, self.W_debug, self.A_debug, self.Z_debug)

        self.fit_vb(prune, prune_feat, maxit, pruning_crit, pruning_feat_crit, tol)
                    
            
    def fit_vb(self, prune, prune_feat, maxit=10, pruning_crit = 1e-1, pruning_feat_crit = 1e-3,tol = 1e-6):
        q = self.q_dist
        for i in range(maxit):
            self.update()

            soft, pred = self.predict(self.X_tst)
            
            self.labels_pred.append(pred)
            print(i, flush = True)
            print('Accuracy: ', accuracy_score(self.labels_pred[-1], self.t_tst), flush = True)
            print('Balanced Accuracy: ', balanced_accuracy_score(self.labels_pred[-1], self.t_tst), flush = True)
            self.accs.append(accuracy_score(self.labels_pred[-1], self.t_tst))

            #Calculamos todas las métricas binarias
            acc = accuracy_score(self.labels_pred[-1], self.t_tst)
            bal_acc = balanced_accuracy_score(self.labels_pred[-1], self.t_tst)
            precision = metrics.precision_score(self.labels_pred[-1], self.t_tst)
            recall = metrics.recall_score(self.labels_pred[-1], self.t_tst)
            f1 = metrics.f1_score(self.labels_pred[-1], self.t_tst)
            fpr, tpr, thresholds = metrics.roc_curve(self.t_tst, soft)
            auc = metrics.auc(fpr, tpr)
            #Las guardamos temporalmente segun el fold y la seed
            self.accs.append(acc)
            self.bal_accs.append(bal_acc)
            self.precisions.append(precision)
            self.recalls.append(recall)
            self.f1s.append(f1)
            self.aucs.append(auc)
            

            if prune and i>5:
                self.pruning(pruning_crit)
            print('K: ', q.Kc, flush = True)
            if prune_feat and i>5:
                self.pruning_feat(pruning_feat_crit)
            feats = []
            for m in range(self.m):
                feats.append(self.d[m])
            print('Feats: ', feats, flush = True)
            self.contador = i


  
    def pruning_feat(self, pruning_feat_crit):
        q = self.q_dist
        
        for m in np.arange(self.m):
            if self.s[m] == 0:
                maximo_W = np.max(abs(q.W[m]['mean'].T))
                fact_sel = np.where(np.any(abs(q.W[m]['mean'].T)>pruning_feat_crit*maximo_W, axis=1))[0].tolist()
                q.W[m]['mean'] = q.W[m]['mean'][:,fact_sel]
                for k in range(q.Kc):
                    q.W_cov[m]['mean'][k] = q.W_cov[m]['mean'][k][fact_sel,:][:,fact_sel]
            elif self.s[m] == 1:
                maximo_A = np.max(abs(self.Xv[m].T @ q.A[m]['mean']))
                fact_sel = np.where(np.any(abs(self.Xv[m].T @ q.A[m]['mean'])>pruning_feat_crit*maximo_A, axis=1))[0].tolist()
                self.Xv[m] = self.Xv[m][:,fact_sel]
            q.gamma[m]['b'] = q.gamma[m]['b'][fact_sel]
            self.d[m] = len(fact_sel)
            self.X[m] = self.X[m][:,fact_sel]
            self.X_tst[m] = self.X_tst[m][:,fact_sel]

            self.index_feats[m] = self.index_feats[m][fact_sel]
            #fact_sel.clear()
    

    def pruning(self, pruning_crit):
        """Pruning of the latent variables.
            
        Checks the values of the projection matrices W and keeps the latent 
        variables if there is no relevant value for any feature. Updates the 
        dimensions of all the model variables and the value of Kc.
        
        """
        
        q = self.q_dist
        fact_sel = []
        for m in np.arange(self.m):
            if self.s[m] == 0:
                maximo_W = np.max(abs(q.W[m]['mean'].T))
                fact_sel = fact_sel + np.where(np.any(abs(q.W[m]['mean'].T)>pruning_crit*maximo_W, axis=0))[0].tolist()
            elif self.s[m] == 1:
                maximo_A = np.max(abs(self.Xv[m].T @ q.A[m]['mean']))
                fact_sel = fact_sel + np.where(np.any(abs(self.Xv[m].T @ q.A[m]['mean'])>pruning_crit*maximo_A, axis=0))[0].tolist()
        maximo_Wy = np.max(abs(q.Wy['mean']))
        fact_sel = fact_sel + np.where(np.any(abs(q.Wy['mean'])>pruning_crit*maximo_Wy, axis=0))[0].tolist()
        fact_sel = np.unique(fact_sel).astype(int)
        # Pruning Z
        q.Z['mean'] = q.Z['mean'][:,fact_sel]
        q.Z['cov'] = q.Z['cov'][fact_sel,:][:,fact_sel]
        q.Z['prodT'] = q.Z['prodT'][fact_sel,:][:,fact_sel]            
         # Pruning W and alpha
        for m in np.arange(self.m):
            if self.s[m] == 0:
                q.W[m]['mean'] = q.W[m]['mean'][fact_sel,:]
                q.W_cov[m]['mean'] = [q.W_cov[m]['mean'][f] for f in fact_sel]
            if self.s[m] == 1:
                q.A[m]['mean'] = q.A[m]['mean'][:,fact_sel]
                q.A_cov[m]['mean'] = [q.A_cov[m]['mean'][f] for f in fact_sel]
            q.alpha[m]['b'] = q.alpha[m]['b'][fact_sel]
        q.Wy['mean'] = q.Wy['mean'][:,fact_sel]
        q.Wy['cov'] = q.Wy['cov'][fact_sel,:][:,fact_sel]
        q.Wy['prodT'] = q.Wy['prodT'][fact_sel,:][:,fact_sel]
        q.psi['a'] = q.psi['a'][fact_sel]
        q.psi['b'] = q.psi['b'][fact_sel]
        self.hyper.psi_a = self.hyper.psi_a[fact_sel]
        self.hyper.psi_b = self.hyper.psi_b[fact_sel]
        self.index_K = self.index_K[fact_sel]
        
        q.Kc = len(fact_sel)    
        


    def calcAUC(self, Y_pred, Y_tst):
        n_classes = Y_pred.shape[1]
        
        # Compute ROC curve and ROC area for each class    
        fpr = dict()
        tpr = dict()
        roc_auc = np.zeros((n_classes,1))
        for i in np.arange(n_classes):
            fpr[i], tpr[i], _ = roc_curve(Y_tst[:,i], Y_pred[:,i]/n_classes)
            roc_auc[i] = auc(fpr[i], tpr[i])
            
        return roc_auc.flatten()
    
    def predict(self,x_tst):
        q = self.q_dist
        
        z_tst = np.zeros((np.shape(x_tst[0])[0],q.Kc))
        for m in range(self.m):
            if self.s[m] == 0:
                z_tst += x_tst[m] @ q.W[m]['mean'].T
            elif self.s[m] == 1:
                z_tst += x_tst[m] @ self.Xv[m].T @ q.A[m]['mean']
        med = z_tst @ q.Wy['mean'].T
        sig = q.tau_y_mean()*np.eye(self.N_tst) + z_tst @ q.Wy['cov'] @ z_tst.T
        sig = np.diagonal(sig)
        sig = sig[:, np.newaxis]
        soft_pred = self.sigmoid(med/np.sqrt(1 + (np.pi/8)*sig))
        
        hard_pred = np.where(soft_pred >= 0.5, 1, 0)
        
        return soft_pred, hard_pred
    
    def predict_tr(self,X):
        q = self.q_dist
        med = q.Z['mean'] @ q.Wy['mean'].T
        sig = q.tau_y_mean()*np.eye(self.N) + q.Z['mean'] @ q.Wy['cov'] @ q.Z['mean'].T
        sig = np.diagonal(sig)
        sig = sig[:, np.newaxis]

        soft_pred = self.sigmoid(med/np.sqrt(1 + (np.pi/8)*sig))
        
        hard_pred = np.where(soft_pred >= 0.5, 1, 0)
        
        return soft_pred, hard_pred

    
    def return_model_variables(self):
        q = self.q_dist


        weights = []
        for m in range(self.m):
            if self.s[m] == 0:
                weights.append(q.W[m]['mean'])
            else:
                weights.append(self.Xv[m].T @ q.A[m]['mean'])
        return weights, q.Z['mean']
        

    
    
    def update(self):
        """Update the variables of the model.
        
        This function updates all the variables of the model and stores the 
        lower bound as well as the Hamming Loss or MSE if specified.
        
        Parameters
        ----------
        __verbose: bool, (default 0). 
            Whether or not to print all the lower bound updates.
        __Y_tst: dict, (default [None]).
            If specified, it is used as the output view to calculate the 
            Hamming Loss. This dictionary can be built using the function 
            "struct_data".
        __X_tst: dict, (default [None]).
            If specified, it is used as the input view to calculate the 
            Hamming Loss. This dictionary can be built using the function 
            "struct_data".
            
        """

        q = self.q_dist
        
        for m in np.arange(self.m): 
            if not np.any(np.isnan(self.W_debug[0])):
                print('W not updated')
            else:
                if self.s[m] == 0:
                    print('Updating W', flush = True)
                    self.update_W(m)
                elif self.s[m] == 1:
                    print('Updating A', flush = True)
                    self.update_A(m)
                print('Updating gamma', flush = True)
                self.update_gamma(m)
                print('Updating alpha', flush = True)
                self.update_alpha(m)
            # self.update_W(m)
            # self.update_gamma(m)
            # self.update_alpha(m)
        print('Updating tau')
        self.update_tau()
        print('Updating Wy')
        self.update_Wy()
        print('Updating tauy')
        self.update_tau_y()
        print('Updating psi')
        self.update_psi()
        if not np.any(np.isnan(self.Z_debug)):
            print('Z not updated')
        else:
            print('Updating Z')
            self.update_Z()
        print('Updating xi')
        self.update_xi()
        print('Updating Y')
        self.update_y()
    
    
    
    def myInverse(self,X):
        """Computation of the inverse of a matrix.
        
        This function calculates the inverse of a matrix in an efficient way 
        using the Cholesky decomposition.
        
        Parameters
        ----------
        __A: bool, (default 0). 
            Whether or not to print all the lower bound updates.
            
        """
        
        try:
           L = linalg.pinv(np.linalg.cholesky(X), rcond=1e-10) #np.linalg.cholesky(A)
           return np.dot(L.T,L) #linalg.pinv(L)*linalg.pinv(L.T)
            # return np.linalg.pinv(X)
        except:
            return np.nan
        
    def myInverse_w(self,X):
        """Computation of the inverse of a matrix.
        
        This function calculates the inverse of a matrix in an efficient way 
        using the Cholesky decomposition.
        
        Parameters
        ----------
        __A: bool, (default 0). 
            Whether or not to print all the lower bound updates.
            
        """
        
        try:

            return np.linalg.pinv(X)
        except:
            return np.nan
        
    def sigmoid(self,x):
        """Computation of the sigmoid function.
        
        Parameters
        ----------
        __x: bool, (default 0). 
            Whether or not to print all the lower bound updates.
            
        """
        
        return np.exp(-np.log(1 + np.exp(-x)))
#        return 1. / (1 + np.exp(-x))
  
    def lambda_func(self,x):
        """Computation of the lambda function.
        
        This function calculates the lambda function defined in the paper.
        
        Parameters
        ----------
        __x: bool, (default 0). 
            Whether or not to print all the lower bound updates.
            
        """
        return np.exp(np.log(self.sigmoid(x) - 0.5) - np.log(2*x))
    
    def Hsk(self,m,s,k):
        q = self.q_dist
        if s == 0:
            return self.X[m] @ q.W[m]['mean'][np.newaxis,k,:].T
        
        elif s == 1:
            return self.X[m] @ self.Xv[m].T @ q.A[m]['mean'][:,k,np.newaxis]
        else:
            print('State not recognized')
            

    def Hs(self,m,s):
        q = self.q_dist
        if s == 0:
            return self.X[m] @ q.W[m]['mean'].T
        elif s == 1:
            return self.X[m] @ self.Xv[m].T @ q.A[m]['mean']
        else:
            print('State not recognized')
    
    
########################
    
    def update_Z(self):
        q = self.q_dist

        z_cov = q.tau_mean()*np.eye(q.Kc) + q.tau_y_mean()*q.Wy['prodT']
        z_cov_inv = self.myInverse(z_cov)    
        if not np.any(np.isnan(z_cov_inv)):
            q.Z['cov'] = z_cov_inv
            z_m = np.zeros((self.N, q.Kc))
            for m1 in range(self.m):
                z_m += self.Hs(m1, self.s[m1])
            z_mean = (q.tau_mean()*z_m + q.tau_y_mean()*q.Y['mean'] @ q.Wy['mean']) @ q.Z['cov']
            q.Z['mean'] = z_mean
            q.Z['prodT'] = q.Z['mean'].T @ q.Z['mean'] + q.Z['cov']
        else:
            print ('Cov Z is not invertible, not updated')
        
    
    def update_Wy(self):
        q = self.q_dist
        w_cov = np.diag(q.psi_mean()) + q.Z['prodT'] * q.tau_y_mean()
        w_cov_inv = self.myInverse(w_cov)
        if not np.any(np.isnan(w_cov_inv)):
            q.Wy['cov'] = w_cov_inv
            q.Wy['mean'] = q.tau_y_mean() * q.Y['mean'].T @ q.Z['mean'] @ q.Wy['cov']
            q.Wy['prodT'] = q.Wy['mean'].T @ q.Wy['mean'] + q.Wy['cov']
        else:
            print ('Cov Wy is not invertible, not updated')
            
        
        
    def update_tau_y(self):
        q = self.q_dist
        q.tauy['a'] = (self.N*self.Q)/2 + self.hyper.tau_y_a
        
#        ys = 0
#        for n in range(self.N):
#            ys += self.y[np.newaxis,n,:]@self.y[np.newaxis,n,:].T
        ys = np.trace(q.Y['prodT'])
        q.tauy['b'] = self.hyper.tau_y_b + (1/2)*ys - np.trace(q.Y['mean'] @ q.Wy['mean'] @ q.Z['mean'].T) + (1/2)*np.trace(q.Wy['prodT'] @ q.Z['prodT'])
        
    def update_psi(self):
        q = self.q_dist
        q.psi['a'] = (self.Q/2 + self.hyper.psi_a)/self.Q
        q.psi['b'] = (self.hyper.psi_b + 0.5*np.diag(q.Wy['prodT']))/self.Q    

    
    def calculate_w_cov(self, fix, vary):
        return fix*vary
    
    def calculate_w_mean(self, fix, vary):
        # print('Shape fix: ', fix)
        # print('Shape vary: ', vary)
        return fix @ vary
    
    def take_value_list(self, lista, value):
        return lista[value]
    
    def calculate_a_cov(self, fix, vary):
        return vary*fix
    
    def calculate_a_mean(self, tau, vary, fix):
        return tau*vary@fix
    
    def take_value_a(self, mat, value):
        return mat[:,value, np.newaxis]
    
    def extract_diag(self, mat1, mat2):
        return ((mat1)*(mat2).T).sum(-1)
    
    def prod_cov_esc(self, esc, cov):
        return esc*cov
    
    def diag_sum_plus(self,mat1,mat2):
        return np.sum(((mat1)*(mat2).T).sum(-1))
    
    def update_W(self,m):
        q = self.q_dist
        #Computation of covariance
#        for k in range(q.Kc):
#            w_cov = np.diagflat(q.gamma_mean(m)) * q.alpha_mean(m)[k] + q.tau_mean()*self.X[m].T @ self.X[m]
#            w_cov_inv = self.myInverse(w_cov)
#            if not np.any(np.isnan(w_cov_inv)):
#                q.W_cov[m]['mean'][k] = w_cov_inv
#            else: 
#                print('Cov W is not invertible, not updated')
        ################
        fixed = q.tau_mean()*self.X[m].T @ self.X[m]
        w_covs = list(map(self.calculate_w_cov,q.Kc*[np.diagflat(q.gamma_mean(m))], list(q.alpha_mean(m))))
        w_covs = w_covs + fixed
        w_covs_inv = list(map(self.myInverse_w, w_covs))
        #######################
        indices_nonan = [index for index, value in enumerate(w_covs_inv) if not isinstance(value, float) or not (value != value)]
        for pos in indices_nonan:
            q.W_cov[m]['mean'][pos] = w_covs_inv[pos]

        #######################
        #q.W_cov[m]['mean'] = w_covs_inv
        ###############
        #Computation of the mean
        w_m = np.zeros((q.Kc,self.N))
        for mo in range(self.m):
            w_m += self.Hs(mo,self.s[mo]).T
        w_m = w_m - self.Hs(m,self.s[m]).T
        w_mean_list = list(map(self.calculate_w_mean, q.Kc*[q.tau_mean()*(q.Z['mean'].T - w_m) @ self.X[m]], q.W_cov[m]['mean']))
        w_mean_def = list(map(self.take_value_list,w_mean_list, np.arange(0,q.Kc)))
        w_mean_def = np.array(w_mean_def)
        q.W[m]['mean'] = w_mean_def
        #############
        #Computation of the mean
#        w_mean = np.zeros((q.Kc, self.d[m]))
#        for k in range(q.Kc):
#            print(k)
#            w_m = np.zeros((1,self.N))
#            for mo in range(self.m):
#                w_m += self.Hsk(mo,self.s[mo],k).T
#            w_m = w_m - self.Hsk(m,self.s[m],k).T
#            w_mean[k,:] = q.tau_mean()*(q.Z['mean'][:,k, np.newaxis].T - w_m) @ self.X[m] @ q.W_cov[m]['mean'][k]
#        q.W[m]['mean'] = w_mean
        
    def update_A(self,m):
        q = self.q_dist
        #Covariance
#        for k in range(q.Kc):
#            a_cov = q.alpha_mean(m)[k] * (self.Xv[m] @ (q.gamma_mean(m)[:,np.newaxis]*self.Xv[m].T)) + q.tau_mean()*self.Xv[m] @ self.X[m].T @ self.X[m] @ self.Xv[m].T
#            a_cov_inv = self.myInverse(a_cov)
#            if not np.any(np.isnan(a_cov_inv)):
#                q.A_cov[m]['mean'][k] = a_cov_inv
#            else: 
#                print('Cov A is not invertible, not updated')
                
        ##################
        print('New way to calculate the covariance')
        fixed = q.tau_mean()*self.Xv[m] @ self.X[m].T @ self.X[m] @ self.Xv[m].T
        a_covs = list(map(self.calculate_a_cov, list(q.alpha_mean(m)),q.Kc*[self.Xv[m] @ (q.gamma_mean(m)[:,np.newaxis]*self.Xv[m].T)]))
        a_covs = a_covs + fixed
        a_covs_inv = list(map(self.myInverse, a_covs))
        #####################33
        indices_nonan = [index for index, value in enumerate(a_covs_inv) if not isinstance(value, float) or not (value != value)]
        for pos in indices_nonan:
            q.A_cov[m]['mean'][pos] = a_covs_inv[pos]

        #####################

        #q.A_cov[m]['mean'] = a_covs_inv
        ##############
        print('New way to calculate the mean')
        a_m = np.zeros((self.Nv,q.Kc))
        for mo in range(self.m):
            a_m += self.Hs(mo,self.s[mo])
        a_m = a_m - self.Hs(m,self.s[m])
        a_mean_list = list(map(self.calculate_a_mean, q.Kc*[q.tau_mean()], q.A_cov[m]['mean'], q.Kc*[self.Xv[m] @ self.X[m].T @ (q.Z['mean'] - a_m)]))
        a_mean_def = list(map(self.take_value_a,a_mean_list, np.arange(0,q.Kc)))
        a_mean_def = np.array(a_mean_def)
        a_mean_def = a_mean_def.T[0]
        q.A[m]['mean'] = a_mean_def
        ##################
#        a_mean = np.zeros((self.Nv, q.Kc))
#        for k in range(q.Kc):
#            a_m = np.zeros((self.Nv,1))
#            for mo in range(self.m):
#                a_m += self.Hsk(mo,self.s[mo],k)
#            a_m = a_m - self.Hsk(m,self.s[m],k)
#            a_mean[:,k,np.newaxis] = q.tau_mean()*q.A_cov[m]['mean'][k] @ self.Xv[m] @ self.X[m].T @ (q.Z['mean'][:,k,np.newaxis] - a_m)
#        q.A[m]['mean'] = a_mean
        
        
    def diag_sum(self,mat):
        return np.sum(np.diag(mat))
                    
    def update_alpha(self,m):
        
        q = self.q_dist
        
        if self.s[m] == 0:
            q.alpha[m]['a'] = self.d[m]/2 + self.hyper.alpha_a[m]
            
            ###############
#            alpha_b = np.zeros((q.Kc,))
#            for k in range(q.Kc):
#                alpha_b[k] = np.sum(q.gamma_mean(m)*q.W[m]['mean'][np.newaxis,k,:]**2) + np.sum(q.gamma_mean(m)*np.diag(q.W_cov[m]['mean'][k]))
            ###############
            alpha_b = np.diag(q.gamma_mean(m)*q.W[m]['mean']@q.W[m]['mean'].T) + list(map(self.diag_sum,q.gamma_mean(m)*q.W_cov[m]['mean']))
            q.alpha[m]['b'] = self.hyper.alpha_b[m] + (1/2)*alpha_b
        elif self.s[m] == 1:
            q.alpha[m]['a'] = (self.d[m]/2 + self.hyper.alpha_a[m])/self.d[m]
            
            ####################
            alpha_b = np.zeros((q.Kc,))

            for k in range(q.Kc):
                alpha_b[k] = ((q.gamma_mean(m)[:,np.newaxis]*self.Xv[m].T @ q.A[m]['mean'][:,k,np.newaxis])*(q.A[m]['mean'][:,k,np.newaxis].T @ self.Xv[m]).T).sum() + ((q.gamma_mean(m)[:,np.newaxis]*self.Xv[m].T @ q.A_cov[m]['mean'][k])*(self.Xv[m]).T).sum()

            q.alpha[m]['b'] = (self.hyper.alpha_b[m] + (1/2)*alpha_b)/self.d[m]
        
        
    def update_gamma(self,m):
        q = self.q_dist
        
        
        if self.s[m] == 0:
            q.gamma[m]['a'] = q.Kc/2 + self.hyper.gamma_a[m]
            
            gamma_b = np.zeros((self.d[m],))
            for d in range(self.d[m]):
                prov = 0
                for k in range(q.Kc):
                    prov += q.alpha_mean(m)[k]*(q.W[m]['mean'][k,d]*q.W[m]['mean'][k,d] + q.W_cov[m]['mean'][k][d,d])
                gamma_b[d] = prov

            q.gamma[m]['b'] = self.hyper.gamma_b[m] + (1/2)*gamma_b
        elif self.s[m] == 1:
            q.gamma[m]['a'] = (q.Kc/2 + self.hyper.gamma_a[m])/q.Kc
        
            
            term_1 = ((q.alpha_mean(m)*(self.Xv[m].T @ q.A[m]['mean'])**2).sum(1))
            term_2 = np.zeros((self.d[m],))
            for k in range(q.Kc):
                term_2 +=  ((q.alpha_mean(m)[k]*self.Xv[m].T @ q.A_cov[m]['mean'][k])*(self.Xv[m]).T).sum(-1)
            gamma_b = term_1 + term_2

            q.gamma[m]['b'] = (self.hyper.gamma_b[m] + (1/2)*gamma_b)/q.Kc
    
    
    def update_tau(self):
        q = self.q_dist
        q.tau['a'] = (self.Kc*self.N)/2 + self.hyper.tau_a
        #Term b
        #Vamos término a término actualizando
        term1 = np.trace(q.Z['mean'] @ q.Z['mean'].T) + self.N*np.trace(q.Z['cov'])
        
        term2_prov = 0
        for m in range(self.m):
            term2_prov += self.Hs(m, self.s[m]).T
        term2 = np.trace(q.Z['mean'] @ term2_prov)

        term3_prov = np.zeros((self.N, self.N))
        for m1 in range(self.m):
            for m2 in range(self.m):
                if m1 != m2:
                    term3_prov += self.Hs(m1,self.s[m1]) @ self.Hs(m2,self.s[m2]).T
                elif m1 == m2:
                    if self.s[m1] == 0:
                        wtw = np.zeros((self.d[m1], self.d[m1]))
                        for k in range(q.Kc):
                            wtw += q.W[m1]['mean'][np.newaxis,k,:].T @ q.W[m1]['mean'][np.newaxis,k,:] + q.W_cov[m1]['mean'][k]
                        term3_prov += self.X[m1] @ wtw @ self.X[m1].T
                    elif self.s[m1] == 1:
                        ata = np.zeros((self.Nv, self.Nv))
                        for k in range(q.Kc):
                            ata += q.A[m1]['mean'][:,k,np.newaxis] @ q.A[m1]['mean'][:,k,np.newaxis].T + q.A_cov[m1]['mean'][k]
                        term3_prov += self.X[m1] @ self.Xv[m1].T @ ata @ self.Xv[m1] @ self.X[m1].T
        term3 = np.trace(term3_prov)

        q.tau['b'] = self.hyper.tau_b + 0.5*term1 - term2 + 0.5*term3
    
    def update_y(self):
        q = self.q_dist
        
        q.Y['cov'] = self.myInverse(q.tau_y_mean()*np.eye(self.N) + 2*np.diagflat(q.xi['mean']))
        q.Y['mean'] = q.Y['cov'] @ (self.t - np.full((self.N,1),1/2) + q.tau_y_mean()*q.Z['mean']@q.Wy['mean'].T)
        q.Y['prodT'] = q.Y['mean'] @ q.Y['mean'].T + q.Y['cov']
    
    def update_xi(self):
        q = self.q_dist
        q.xi['mean'] = np.sqrt((q.Y['mean']**2) + np.reshape(np.diag(q.Y['cov']),(self.N,1)))
    
    #################
                
    

       
    def HGamma(self, a, b):
        """Compute the entropy of a Gamma distribution.

        Parameters
        ----------
        __a: float. 
            The parameter a of a Gamma distribution.
        __b: float. 
            The parameter b of a Gamma distribution.

        """
        
        return -np.log(b + sys.float_info.epsilon)
    
    def HGauss(self, mn, cov, entr):
        """Compute the entropy of a Gamma distribution.
        
        Uses slogdet function to avoid numeric problems. If there is any 
        infinity, doesn't update the entropy.
        
        Parameters
        ----------
        __mean: float. 
            The parameter mean of a Gamma distribution.
        __covariance: float. 
            The parameter covariance of a Gamma distribution.
        __entropy: float.
            The entropy of the previous update. 

        """
        
        H = 0.5*mn.shape[0]*np.linalg.slogdet(cov)[1]
        return self.checkInfinity(H, entr)
    
    def checkInfinity(self, H, entr):
        """Checks if three is any infinity in th entropy.
        
        Goes through the input matrix H and checks if there is any infinity.
        If there is it is not updated, if there isn't it is.
        
        Parameters
        ----------
        __entropy: float.
            The entropy of the previous update. 

        """
        
        if abs(H) == np.inf:
            return entr
        else:
            return H

    def update_bound(self):
        q = self.q_dist

        q.Z['LH'] = (self.N/2)*self.HGauss(q.Z['mean'], q.Z['cov'], q.Z['LH'])
        q.Wy['LH'] = 0.5*self.HGauss(q.Wy['mean'], q.Wy['cov'], q.Wy['LH'])
        q.Y['LH'] = (self.N/2)*self.HGauss(q.Y['mean'], q.Y['cov'], q.Y['LH'])

        outputs = -(2 + self.N/2 -self.hyper.tau_a)*np.log(q.tau['b']) + (-2 + self.hyper.psi_a[0])*np.sum(np.log(q.psi['b'])) - (2 + self.N/2 -self.hyper.tau_y_a)*np.log(q.tauy['b']) + q.Z['LH']  + q.Wy['LH'] + q.Y['LH']

        #Lets compute the input elbo
        inputs = 0
        for m in range(self.m):
            inputs += (-2 + self.hyper.gamma_a[m] +self.Kc/2)*np.sum(np.log(q.gamma[m]['b'])) + (-2 + self.hyper.alpha_a[m] +self.d[m]/2)*np.sum(np.log(q.alpha[m]['b'])) \
            + self.hyper.gamma_b[m]*np.sum(q.gamma[m]['a']/q.gamma[m]['b']) + self.hyper.alpha_b[m]*np.sum(q.alpha[m]['a']/q.alpha[m]['b'])
            if self.s[m] == 0:
                primal = 0
                for d in range(self.d[m]):
                    for k in range(q.Kc):
                        primal += ((q.gamma[m]['a']*q.alpha[m]['a'])/(q.gamma[m]['b'][d]*q.alpha[m]['b'][k]))*(q.W[m]['mean'][k,d]**2 + q.W_cov[m]['mean'][k][d,d])
                inputs += primal
            elif self.s[m] == 1:
                dual = 0
                for d in range(self.d[m]):
                    for k in range(q.Kc):
                        dual += ((q.gamma[m]['a']*q.alpha[m]['a'])/(q.gamma[m]['b'][d]*q.alpha[m]['b'][k]))*np.trace(self.Xv[m] @ self.Xv[m].T @ (q.A[m]['mean'][:,k] @ q.A[m]['mean'][:,k].T + q.A_cov[m]['mean'][k]))
                inputs += dual
        return outputs + inputs

        
class HyperParameters(object):
    """ Hyperparameter initialisation.
    
    Parameters
    ----------
    __m : int.
        number of views in the model.
    
    """
    def __init__(self, m_total, K):
        # self.alpha_a = []
        # self.alpha_b = []
        # self.gamma_a = []
        # self.gamma_b = []
        # for m in np.arange(m_total): 
        #     self.alpha_a.append(2)
        #     self.alpha_b.append(1)
            
        #     self.gamma_a.append(2)
        #     self.gamma_b.append(1)
        # self.psi_a = 2*np.ones((K,))
        # self.psi_b = 1*np.ones((K,))
        # self.tau_y_a = 2
        # self.tau_y_b = 1
        # self.tau_a = 2
        # self.tau_b = 1

        self.alpha_a = []
        self.alpha_b = []
        self.gamma_a = []
        self.gamma_b = []
        for m in np.arange(m_total): 
            self.alpha_a.append(1e-12)
            self.alpha_b.append(1e-14)
            
            self.gamma_a.append(1e-12)
            self.gamma_b.append(1e-14)
        self.psi_a = 1e-12*np.ones((K,))
        self.psi_b = 1e-14*np.ones((K,))
        self.tau_y_a = 1e-14
        self.tau_y_b = 1e-14
        self.tau_a = 1e-14
        self.tau_b = 1e-14
            
class Qdistribution(object):
    """ Hyperparameter initialisation.
    
    Parameters
    ----------
    __m : int.
        number of views in the model.
    
    """
    def __init__(self, n_max, nv,  d, q, Kc, m, s, hyper, W_debug, A_debug, Z_debug):
        self.n_max = n_max
        self.d = d
        self.q = q
        self.Kc = Kc
        self.Nv = nv
        self.m = m
        self.states = s
        self.W_debug = W_debug
        self.A_debug = A_debug
        self.Z_debug = Z_debug
        # Initialize some parameters that are constant
        #self.alpha = self.qGamma(hyper.alpha_a,hyper.alpha_b,self.m,self.m*[self.Kc]) 
        self.alpha = self.qGamma(hyper.alpha_a,hyper.alpha_b,self.m,self.m*[self.Kc]) 
        self.psi = self.qGamma_uni(hyper.psi_a,hyper.psi_b,self.Kc)
        self.gamma = self.qGamma(hyper.gamma_a,hyper.gamma_b,self.m,self.d) 
        self.tau = self.qGamma_uni(hyper.tau_a,hyper.tau_b,1) 
        self.tauy = self.qGamma_uni(hyper.tau_y_a,hyper.tau_y_b,1)
        
        self.init_rnd()

    def init_rnd(self):
        """ Hyperparameter initialisation.
    
        Parameters
        ----------
        __m : int.
            number of views in the model.
            
        """
        
        
        self.W = [None]*self.m
        self.W_cov = [None]*self.m

        self.A = [None]*self.m
        self.A_cov = [None]*self.m
        
        self.Z = {
                "mean":     None,
                "cov":      None,
                "prodT":    None,
                "LH":       0,
                "Elogp":    0,
            }
        
        self.Wy = {
                "mean":     None,
                "cov":      None,
                "prodT":    None,
                "LH":       0,
                "Elogp":    0,
            }
        
        for m in np.arange(self.m):
            info = {
                "mean":     None,
                "cov":      None,
                "prodT":    None,
                "LH":       0,
                "Elogp":    0,
            }
            self.W[m] = info
        for m in np.arange(self.m):
            info_Wcov = {
                "mean":     [None]*self.Kc,
                "cov":      None,
                "prodT":    None,
                "LH":       0,
                "Elogp":    0,
            }
            self.W_cov[m] = info_Wcov
            
        for m in np.arange(self.m):
            info = {
                "mean":     None,
                "cov":      None,
                "prodT":    None,
                "LH":       0,
                "Elogp":    0,
            }
            self.A[m] = info
        for m in np.arange(self.m):
            info_Acov = {
                "mean":     [None]*self.Kc,
                "cov":      None,
                "prodT":    None,
                "LH":       0,
                "Elogp":    0,
            }
            self.A_cov[m] = info_Acov
        self.Y = {
                "mean":     None,
                "cov":      None,
                "prodT":    None,
                "LH":       0,
                "Elogp":    0,
            }
        
        self.xi = {
                "mean":     None,
            }
        
        # Initialization of the latent space matrix Z
        if not np.any(np.isnan(self.Z_debug)):
            self.Z["mean"] = self.Z_debug
            self.Z['cov'] = np.zeros((self.Kc,self.Kc)) 
            self.Z["prodT"] = self.Z["mean"].T @ self.Z["mean"]
        else:
            self.Z['mean'] = np.random.normal(0.0, 1.0, self.n_max * self.Kc).reshape(self.n_max, self.Kc)
            self.Z['cov'] = np.eye(self.Kc) 
            self.Z['prodT'] = np.dot(self.Z['mean'].T, self.Z['mean']) + self.n_max*self.Z['cov']
        
        self.Wy['mean'] = np.random.normal(0.0, 1.0, self.Kc * self.q).reshape(self.q, self.Kc)
        self.Wy['cov'] = np.eye(self.Kc) 
        self.Wy['prodT'] = self.Wy['mean'].T @ self.Wy['mean'] + self.Wy['cov']
        
        self.Y['mean'] = self.Z['mean'] @ self.Wy['mean'].T
        self.Y['cov'] = self.tau_y_mean()*np.eye(self.n_max)
        self.Y['prodT'] = self.Y['mean'] @ self.Y['mean'].T + self.Y['cov']
        
        self.xi['mean'] = np.random.normal(0.0, 1.0, self.n_max).reshape(self.n_max,1)
        
        
        # for m in np.arange(self.m):   
        #     self.W[m]['mean'] = np.random.normal(0.0, 1.0, self.Kc * self.d[m]).reshape(self.Kc, self.d[m])
        #     for k in np.arange(self.Kc):
        #         self.W_cov[m]['mean'][k] = np.eye(self.d[m])

        for m in np.arange(self.m):   
            if not np.any(np.isnan(self.W_debug[0])):
                self.W[m]['mean'] = self.W_debug[m]
                for k in np.arange(self.Kc):
                    self.W_cov[m]['mean'][k] = np.zeros((self.d[m],self.d[m]))
            else:
                if self.states[m] == 0:
                    self.W[m]['mean'] = np.random.normal(0.0, 1.0, self.Kc * self.d[m]).reshape(self.Kc, self.d[m])
                    for k in np.arange(self.Kc):
                        self.W_cov[m]['mean'][k] = np.eye(self.d[m])
                else:
                    self.W[m]['mean'] = 0
                    for k in np.arange(self.Kc):
                        self.W_cov[m]['mean'][k] = 0
        
        for m in np.arange(self.m):   
            if not np.any(np.isnan(self.A_debug[0])):
                self.A[m]['mean'] = self.A_debug[m]
                for k in np.arange(self.Kc):
                    self.A_cov[m]['mean'][k] = np.zeros((self.Nv,self.Nv))
            else:
                if self.states[m] == 1:
                    self.A[m]['mean'] = np.random.normal(0.0, 1.0, self.Kc * self.Nv).reshape(self.Nv, self.Kc)
                    for k in np.arange(self.Kc):
                        self.A_cov[m]['mean'][k] = np.eye(self.Nv)
                else:
                    self.A[m]['mean'] = 0
                    for k in np.arange(self.Kc):
                        self.A_cov[m]['mean'][k] = 0
        
    def qGamma(self,a,b,m_i,r,mask=None,sp=[None]):
        """ Initialisation of variables with Gamma distribution..
    
        Parameters
        ----------
        __a : array (shape = [m_in, 1]).
            Initialistaion of the parameter a.        
        __b : array (shape = [m_in, 1]).
            Initialistaion of the parameter b.
        __m_i: int.
            Number of views. 
        __r: array (shape = [m_in, 1]).
            dimension of the parameter b for each view.
            
        """
        param = [None]*m_i
        for m in np.arange(m_i):
            if (None in sp) or sp[m] == 1:
                info = {                
                    "a":         a[m],
                    "LH":         None,
                    "ElogpWalp":  None,
                }
                if mask is None or mask[m] is None:
                    info["b"] = (b[m]*np.ones((r[m],))).flatten()
                else:
                    info["b"] = (b[m]*np.ones((len(np.unique(mask[m])),1))).flatten()
                param[m] = info
        return param
    
    def qGamma_uni(self,a,b,K):
        """ Initialisation of variables with Gamma distribution..
    
        Parameters
        ----------
        __a : array (shape = [1, 1]).
            Initialistaion of the parameter a.        
        __b : array (shape = [K, 1]).
            Initialistaion of the parameter b.
        __m_i: int.
            Number of views. 
        __K: array (shape = [K, 1]).
            dimension of the parameter b for each view.
            
        """
        
        param = {                
                "a":         a,
                "b":         b,
                "LH":         None,
                "ElogpWalp":  None,
            }
        return param
    
    
    def alpha_mean(self,m):
        """ Mean of alpha.
        It returns the mean value of the variable alpha for the specified view.
    
        Parameters
        ----------
        __m : int.
            View that wants to be used.
            
        """
        return self.alpha[m]["a"] / self.alpha[m]["b"]
    
    def tau_mean(self):
        """ Mean of tau.
        It returns the mean value of the variable tau for the specified view.
    
        Parameters
        ----------
        __m : int.
            View that wants to be used.
            
        """
        
        return self.tau["a"] / self.tau["b"]
    def tau_y_mean(self):
        
        
        
        return self.tauy["a"] / self.tauy["b"]

    def gamma_mean(self,m):
        """ Mean of gamma.
        It returns the mean value of the variable gamma for the specified view.
    
        Parameters
        ----------
        __m : int.
            View that wants to be used.
            
        """
        
        return self.gamma[m]["a"] / self.gamma[m]["b"]
    
    def psi_mean(self):
        """ Mean of gamma.
        It returns the mean value of the variable gamma for the specified view.
    
        Parameters
        ----------
        __m : int.
            View that wants to be used.
            
        """
        
        return self.psi["a"] / self.psi["b"]
