import functools
import os
import sys
import pprint
import typing
import random
import joblib
import json
import pickle
import warnings
import dcor
import math
from math import ceil, log, sqrt
import networkx as nx
import numpy as np
import pandas as pd
import pingouin as pg
from tqdm.auto import tqdm
from datetime import datetime
import scipy.stats as stats
import scipy.special as special
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression, LassoCV
from sklearn.metrics import adjusted_mutual_info_score
from collections import Counter, defaultdict
from econml.dml import CausalForestDML, LinearDML
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.PermutationBased.GRaSP import grasp
import semopy
from semopy import Model
from semopy.inspector import inspect


import crunch
#crunch = crunch.load_notebook()


#"""DAG的工具函数"""
def graph_nodes_representation(graph, nodelist):
    """
    Create an alternative representation of a graph which is hashable
    and equivalent graphs have the same hash.

    Python cannot PROPERLY use nx.Graph/DiGraph as key for
    dictionaries, because two equivalent graphs with just different
    order of the nodes would result in different keys. This is
    undesirable here.

    So here we transform the graph into an equivalent form that is
    based on a specific nodelist and that is hashable. In this way,
    two equivalent graphs, once transformed, will result in identical
    keys.

    So we use the following trick: extract the adjacency matrix
    (with nodes in a fixed order) and then make a hashable thing out
    of it, through tuple(array.flatten()):
    """

    # This get the adjacency matrix with nodes in a given order, as
    # numpy array (which is not hashable):
    adjacency_matrix = nx.adjacency_matrix(graph, nodelist=nodelist).todense()

    # This transforms the numpy array into a hashable object:
    hashable = tuple(adjacency_matrix.flatten())

    return hashable

def create_graph_label():
    """
    Create a dictionary from graphs to labels, in two formats.
    """
    graph_label = {
        nx.DiGraph([("X", "Y"), ("v", "X"), ("v", "Y")]): "Confounder",
        nx.DiGraph([("X", "Y"), ("X", "v"), ("Y", "v")]): "Collider",
        nx.DiGraph([("X", "Y"), ("X", "v"), ("v", "Y")]): "Mediator",
        nx.DiGraph([("X", "Y"), ("v", "X")]):             "Cause of X",
        nx.DiGraph([("X", "Y"), ("v", "Y")]):             "Cause of Y",
        nx.DiGraph([("X", "Y"), ("X", "v")]):             "Consequence of X",
        nx.DiGraph([("X", "Y"), ("Y", "v")]):             "Consequence of Y",
        nx.DiGraph({"X": ["Y"], "v": []}):                "Independent",
    }

    nodelist = ["v", "X", "Y"]

    # This is an equivalent alternative to graph_label but in a form
    # for which two equivalent graphs have the same key:
    adjacency_label = {
        graph_nodes_representation(graph, nodelist): label
        for graph, label in graph_label.items()
    }

    return graph_label, adjacency_label

def get_labels(adjacency_matrix, adjacency_label):
    """
    Transform an adjacency_matrix (as pd.DataFrame) into a dictionary of variable:label
    """

    result = {}
    for variable in adjacency_matrix.columns.drop(["X", "Y"]):
        submatrix = adjacency_matrix.loc[[variable, "X", "Y"], [variable, "X", "Y"]]  # this is not hashable
        key = tuple(submatrix.values.flatten())  # this is hashable and a compatible with adjacency_label

        result[variable] = adjacency_label[key]

    return result


# # """数据增强-旧"""
# def apply_mapping(df, mapping):
#     df_new = df.copy()
#     # 创建临时映射以避免冲突
#     temp_mapping = {k: f'_temp_{k}' for k in mapping.keys()}
#     df_new.rename(columns=temp_mapping, inplace=True)
#     if df_new.shape[0] == df_new.shape[1]:  # 如果是方阵，如标签矩阵
#         df_new.rename(index=temp_mapping, inplace=True)
#     # 应用最终映射
#     final_mapping = {f'_temp_{k}': v for k, v in mapping.items()}
#     df_new.rename(columns=final_mapping, inplace=True)
#     if df_new.shape[0] == df_new.shape[1]:
#         df_new.rename(index=final_mapping, inplace=True)
#     return df_new

# def check_duplicate_columns(df):
#     """检查是否存在重复的列名"""
#     return df.columns.duplicated().any()

# def augment_data(X_train, y_train, augment_factor=1.5):
#     """
#     扩增数据集。

#     参数:
#     - X_train: dict, 原始特征矩阵，键为样本ID，值为DataFrame
#     - y_train: dict, 原始标签矩阵，键为样本ID，值为DataFrame
#     - augment_factor: float, 数据扩增的倍数，例如2.5表示每个样本生成2到3个增强样本
#     """
#     new_X_train = X_train.copy()
#     new_y_train = y_train.copy()

#     for sample_id in X_train.keys():
#         X = X_train[sample_id]
#         y = y_train[sample_id]
#         variables = list(X.columns)
#         dim = len(variables)
#         # 提取因果关系对
#         edges = []
#         for u in y.index:
#             for v in y.columns:
#                 if y.loc[u, v] == 1:
#                     edges.append((u, v))
#         # 根据维度决定使用哪些边
#         if dim >= 4:
#             edges_no_XY = [(u, v) for (u, v) in edges if u not in ['X', 'Y'] and v not in ['X', 'Y']]
#             edges_to_use_base = edges_no_XY
#         else:
#             edges_to_use_base = edges

#         # 计算每个样本需要生成的增强样本数量
#         integer_part = math.floor(augment_factor)
#         fractional_part = augment_factor - integer_part

#         for augment_num in range(integer_part):
#             # 生成整数部分的增强样本
#             if dim >= 4:
#                 edges_to_use = edges_no_XY.copy()
#             else:
#                 edges_to_use = edges.copy()

#             if not edges_to_use:
#                 # 没有边可用，直接复制原始数据
#                 new_sample_id = f'{augment_num}{sample_id}'
#                 new_X_train[new_sample_id] = X.copy()
#                 new_y_train[new_sample_id] = y.copy()
#                 continue

#             attempts = 0
#             success = False
#             while attempts < 3 and not success:
#                 if not edges_to_use:
#                     break  # 没有合适的边，跳出循环
#                 u, v = random.choice(edges_to_use)
#                 mapping = {'X': u, 'Y': v, u: 'X', v: 'Y'}
#                 # 应用映射到特征矩阵和标签矩阵
#                 X_new = apply_mapping(X, mapping)
#                 y_new = apply_mapping(y, mapping)
#                 # 检查特征矩阵是否有重复列
#                 if check_duplicate_columns(X_new):
#                     attempts += 1
#                     edges_to_use.remove((u, v))  # 移除当前选择，避免重复尝试
#                     continue  # 重试
#                 else:
#                     # 没有重复列，存储新的数据
#                     new_sample_id = f'{augment_num}{sample_id}'
#                     new_X_train[new_sample_id] = X_new
#                     new_y_train[new_sample_id] = y_new
#                     success = True
#             if not success:
#                 # 没有找到合适的映射，复制原始数据
#                 new_sample_id = f'{augment_num}{sample_id}'
#                 new_X_train[new_sample_id] = X.copy()
#                 new_y_train[new_sample_id] = y.copy()

#         # 处理小数部分
#         if fractional_part > 0:
#             if random.random() < fractional_part:
#                 augment_num = integer_part  # 例如，2.5 -> 2
#                 if dim >= 4:
#                     edges_to_use = edges_no_XY.copy()
#                 else:
#                     edges_to_use = edges.copy()

#                 if not edges_to_use:
#                     # 没有边可用，直接复制原始数据
#                     new_sample_id = f'{augment_num}{sample_id}'
#                     new_X_train[new_sample_id] = X.copy()
#                     new_y_train[new_sample_id] = y.copy()
#                 else:
#                     attempts = 0
#                     success = False
#                     while attempts < 3 and not success:
#                         if not edges_to_use:
#                             break  # 没有合适的边，跳出循环
#                         u, v = random.choice(edges_to_use)
#                         mapping = {'X': u, 'Y': v, u: 'X', v: 'Y'}
#                         # 应用映射到特征矩阵和标签矩阵
#                         X_new = apply_mapping(X, mapping)
#                         y_new = apply_mapping(y, mapping)
#                         # 检查特征矩阵是否有重复列
#                         if check_duplicate_columns(X_new):
#                             attempts += 1
#                             edges_to_use.remove((u, v))  # 移除当前选择，避免重复尝试
#                             continue  # 重试
#                         else:
#                             # 没有重复列，存储新的数据
#                             new_sample_id = f'{augment_num}{sample_id}'
#                             new_X_train[new_sample_id] = X_new
#                             new_y_train[new_sample_id] = y_new
#                             success = True
#                     if not success:
#                         # 没有找到合适的映射，复制原始数据
#                         new_sample_id = f'{augment_num}{sample_id}'
#                         new_X_train[new_sample_id] = X.copy()
#                         new_y_train[new_sample_id] = y.copy()

#     return new_X_train, new_y_train


#"""数据增强-新"""
def apply_mapping(df, mapping, rename_index=False):
    df_new = df.copy()
    # 创建临时映射以避免冲突
    temp_mapping = {k: f'_temp_{k}' for k in mapping.keys()}
    df_new.rename(columns=temp_mapping, inplace=True)
    if rename_index:
        df_new.rename(index=temp_mapping, inplace=True)
    # 应用最终映射
    final_mapping = {temp_mapping[k]: mapping[k] for k in mapping.keys()}
    df_new.rename(columns=final_mapping, inplace=True)
    if rename_index:
        df_new.rename(index=final_mapping, inplace=True)
    return df_new

def check_duplicate_columns(df):
    """检查是否存在重复的列名"""
    return df.columns.duplicated().any()

def augment_data(X_train, y_train, augment_factor=1.5):
    """
    扩增数据集。

    参数:
    - X_train: dict, 原始特征矩阵，键为样本ID，值为DataFrame
    - y_train: dict, 原始标签矩阵，键为样本ID，值为DataFrame
    - augment_factor: float, 数据扩增的倍数，例如2.5表示每个样本生成2到3个增强样本
    """
    new_X_train = X_train.copy()
    new_y_train = y_train.copy()
    failure_counts = {dim: 0 for dim in range(3, 11)}  # 初始化失败计数器

    for sample_id in X_train.keys():
        X = X_train[sample_id]
        y = y_train[sample_id]
        variables = list(map(str, X.columns))
        dim = len(variables)

        # 将变量名都转换为字符串类型
        X.columns = X.columns.astype(str)
        if set(X.index) == set(X.columns):
            X.index = X.index.astype(str)
        y.columns = y.columns.astype(str)
        y.index = y.index.astype(str)

        # 第一步：将 'X' 和 'Y' 重命名为未使用的数字
        used_variables = set(X.columns) | set(y.index) | set(y.columns)
        all_numbers = set(map(str, range(10)))  # '0' 到 '9' 的字符串集合
        unused_numbers = list(all_numbers - used_variables)

        mapping_XY = {}
        if 'X' in used_variables:
            if unused_numbers:
                new_X_name = unused_numbers.pop()
                mapping_XY['X'] = new_X_name
            else:
                # 没有未使用的数字，无法重命名 'X'
                failure_counts[dim] += 1
                continue  # 跳过此样本
        if 'Y' in used_variables:
            if unused_numbers:
                new_Y_name = unused_numbers.pop()
                mapping_XY['Y'] = new_Y_name
            else:
                # 没有未使用的数字，无法重命名 'Y'
                failure_counts[dim] += 1
                continue  # 跳过此样本
        if mapping_XY:
            X = X.rename(columns=mapping_XY)
            if set(X.index) == set(X.columns):
                X = X.rename(index=mapping_XY)
            y = y.rename(index=mapping_XY, columns=mapping_XY)
            # 更新变量名
            variables = list(X.columns)

        # 重新提取因果关系对
        edges = []
        for u in y.index:
            for v in y.columns:
                if y.loc[u, v] == 1:
                    edges.append((u, v))

        # 计算每个样本需要生成的增强样本数量
        integer_part = math.floor(augment_factor)
        fractional_part = augment_factor - integer_part
        edges_to_use = edges.copy()

        edges_to_use.remove((mapping_XY['X'], mapping_XY['Y']))
        for augment_num in range(integer_part):

            # 移除之前映射的 (X, Y)

            if not edges_to_use:
                # 没有边可用，计数失败次数
                failure_counts[dim] += 1
                continue

            attempts = 0
            success = False
            while attempts < 3 and not success:
                if not edges_to_use:
                    break  # 没有合适的边，跳出循环
                u, v = random.choice(edges_to_use)
                edges_to_use.remove((u, v))
                mapping = {u: 'X', v: 'Y'}
                # 应用映射到特征矩阵和标签矩阵
                X_new = apply_mapping(X, mapping)
                y_new = apply_mapping(y, mapping, rename_index=True)
                # 检查特征矩阵是否有重复列
                if check_duplicate_columns(X_new):
                    attempts += 1
                    continue  # 重试
                else:
                    # 没有重复列，存储新的数据
                    new_sample_id = f'{augment_num}{sample_id}'
                    new_X_train[new_sample_id] = X_new
                    new_y_train[new_sample_id] = y_new
                    success = True
            if not success:
                # 计数失败次数
                failure_counts[dim] += 1

        # 处理小数部分
        if fractional_part > 0:
            if random.random() < fractional_part:
                augment_num = integer_part
                if not edges_to_use:
                    failure_counts[dim] += 1
                else:
                    attempts = 0
                    success = False
                    while attempts < 3 and not success:
                        if not edges_to_use:
                            break  # 没有合适的边，跳出循环
                        u, v = random.choice(edges_to_use)
                        edges_to_use.remove((u, v))
                        mapping = {u: 'X', v: 'Y'}
                        # 应用映射到特征矩阵和标签矩阵
                        X_new = apply_mapping(X, mapping)
                        y_new = apply_mapping(y, mapping, rename_index=True)
                        # 检查特征矩阵是否有重复列
                        if check_duplicate_columns(X_new):
                            attempts += 1
                            continue  # 重试
                        else:
                            # 没有重复列，存储新的数据
                            new_sample_id = f'{augment_num}{sample_id}'
                            new_X_train[new_sample_id] = X_new
                            new_y_train[new_sample_id] = y_new
                            success = True
                    if not success:
                        # 计数失败次数
                        failure_counts[dim] += 1

    return new_X_train, new_y_train, failure_counts


#X_train, y_train, X_test = crunch.load_data()


#"""Copula Entropy的工具函数"""
###  Cite: https://github.com/majianthu/pycopent
from scipy.special import digamma
from scipy.stats import rankdata as rank
from scipy.spatial.distance import cdist
from math import gamma, log, pi
from numpy import array, abs, max, hstack, vstack, ones, zeros, cov, matrix, where
from numpy.random import uniform, normal as rnorm
from numpy.linalg import det
from multiprocessing.pool import Pool,ThreadPool
import sys

##### constructing empirical copula density [1]
def construct_empirical_copula(x):
	(N,d) = x.shape
	xc = zeros([N,d])
	for i in range(0,d):
		xc[:,i] = rank(x[:,i]) / N

	return xc
##### Estimating entropy with kNN method [2]
def entknn(x, k = 3, dtype = 'chebychev'):
	(N,d) = x.shape

	g1 = digamma(N) - digamma(k)

	if dtype == 'euclidean':
		cd = pi**(d/2) / 2**d / gamma(1+d/2)
	else:	# (chebychev) maximum distance
		cd = 1;

	logd = 0
	dists = cdist(x, x, dtype)
	dists.sort()
	for i in range(0,N):
		logd = logd + log( 2 * dists[i,k] ) * d / N

	return (g1 + log(cd) + logd)
##### 2-step Nonparametric estimation of copula entropy [1]
def copent(x, k = 3, dtype = 'chebychev', log0 = False):
	xarray = array(x)

	if log0:
		(N,d) = xarray.shape
		max1 = max(abs(xarray), axis = 0)
		for i in range(0,d):
			if max1[i] == 0:
				xarray[:,i] = rnorm(0,1,N)
			else:
				xarray[:,i] = xarray[:,i] + rnorm(0,1,N) * max1[i] * 0.000005

	xc = construct_empirical_copula(xarray)

	try:
		return -entknn(xc, k, dtype)
	except ValueError: # log0 error
		return copent(x, k, dtype, log0 = True)

##### conditional independence test [3]
##### to test independence of (x,y) conditioned on z
def ci(x, y, z, k = 3, dtype = 'chebychev'):
	xyz = vstack((x,y,z)).T
	yz = vstack((y,z)).T
	xz = vstack((x,z)).T
	return copent(xyz,k,dtype) - copent(yz,k,dtype) - copent(xz,k,dtype)
##### estimating transfer entropy from y to x with lag [3]
def transent(x, y, lag = 1, k = 3, dtype = 'chebychev'):
	xlen = len(x)
	ylen = len(y)
	if (xlen > ylen):
		l = ylen
	else:
		l = xlen
	if (l < (lag + k + 1)):
		return 0
	x1 = x[0:(l-lag)]
	x2 = x[lag:l]
	y = y[0:(l-lag)]
	return ci(x2,y,x1,k,dtype)
##### multivariate normality test [4]
def mvnt(x, k = 3, dtype = 'chebychev'):
	return -0.5 * log(det(cov(x.T))) - copent(x,k,dtype)
##### two-sample test [5]
def tst(s0,s1,n=12, k = 3, dtype = 'chebychev'):
	(N0,d0) = s0.shape
	(N1,d1) = s1.shape
	x = vstack((s0,s1))
	stat1 = 0
	for i in range(0,n):
		y1 = vstack((ones([N0,1]),ones([N1,1])*2)) + uniform(0, 0.0000001,[N0+N1,1])
		y0 = ones([N0+N1,1]) + uniform(0,0.0000001,[N0+N1,1])
		stat1 = stat1 + copent(hstack((x,y1)),k,dtype) - copent(hstack((x,y0)),k,dtype)
	return stat1/n
##### single change point detection [6]
def init(X,N,K,DTYPE):
	global x,n,k,dtype
	x = X
	n = N
	k = K
	dtype = DTYPE

def tsti(i):
	s0 = x[0:(i+1),:]
	s1 = x[(i+2):,:]
	return tst(s0,s1,n,k,dtype)

def cpd(x, thd = 0.13, n = 30, k = 3, dtype = 'chebychev'):
	x = matrix(x)
	len1 = x.shape[0]
	if len1 == 1:
		len1 = x.shape[1]
		x = x.T
	pos = -1
	maxstat = 0
	if sys.platform.startswith("win"): # "win"
		pool = ThreadPool(initializer = init, initargs=(x,n,k,dtype))
	else: # "linux" or "darwin"
		pool = Pool(initializer = init, initargs=(x,n,k,dtype))
	stat1 = [0] + pool.map(tsti,range(len1-2)) + [0]
	pool.close()
	if(max(stat1) > thd):
		maxstat = max(stat1)
		pos = where(stat1 == maxstat)[0][0]+1
	return pos, maxstat, stat1
##### multiple change point detection [6]
def mcpd(x, maxp = 5, thd = 0.13, minseglen = 10, n = 30, k = 3, dtype = 'chebychev'):
	x = matrix(x)
	len1 = x.shape[0]
	if len1 == 1:
		len1 = x.shape[1]
		x = x.T
	maxstat = []
	pos = []
	bisegs = matrix([0,len1-1])
	for i in range(0,maxp):
		if i >= bisegs.shape[0]:
			break
		rpos, rmaxstat, _ = cpd(x[bisegs[i,0]:bisegs[i,1],:],thd,n,k,dtype)
		if rpos > -1 :
			rpos = rpos + bisegs[i,0]
			maxstat.append(rmaxstat)
			pos.append(rpos)
			if (rpos - bisegs[i,0]) > minseglen :
				bisegs = vstack((bisegs,[bisegs[i,0],rpos-1]))
			if (bisegs[i,1] - rpos +1) > minseglen :
				bisegs = vstack((bisegs,[rpos,bisegs[i,1]]))
	return pos,maxstat


#"""PPS"""
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, f1_score
from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_object_dtype,
    is_categorical_dtype,
    is_string_dtype,
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
)


def _calculate_model_cv_score_(
    df, target, feature, task, cross_validation, random_seed, **kwargs
):
    "Calculates the mean model score based on cross-validation"
    # Sources about the used methods:
    # https://scikit-learn.org/stable/modules/tree.html
    # https://scikit-learn.org/stable/modules/cross_validation.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
    metric = task["metric_key"]
    model = task["model"]
    # shuffle the rows - this is important for cross-validation
    # because the cross-validation just takes the first n lines
    # if there is a strong pattern in the rows eg 0,0,0,0,1,1,1,1
    # then this will lead to problems because the first cv sees mostly 0 and the later 1
    # this approach might be wrong for timeseries because it might leak information
    df = df.sample(frac=1, random_state=random_seed, replace=False)

    # preprocess target
    if task["type"] == "classification":
        label_encoder = preprocessing.LabelEncoder()
        df[target] = label_encoder.fit_transform(df[target])
        target_series = df[target]
    else:
        target_series = df[target]

    # preprocess feature
    if _dtype_represents_categories(df[feature]):
        one_hot_encoder = preprocessing.OneHotEncoder()
        array = df[feature].__array__()
        sparse_matrix = one_hot_encoder.fit_transform(array.reshape(-1, 1))
        feature_input = sparse_matrix
    else:
        # reshaping needed because there is only 1 feature
        array = df[feature].values
        if not isinstance(array, np.ndarray):  # e.g Int64 IntegerArray
            array = array.to_numpy()
        feature_input = array.reshape(-1, 1)

    # Cross-validation is stratifiedKFold for classification, KFold for regression
    # CV on one core (n_job=1; default) has shown to be fastest
    scores = cross_val_score(
        model, feature_input, target_series.to_numpy(), cv=cross_validation, scoring=metric
    )

    return scores.mean()


def _normalized_mae_score(model_mae, naive_mae):
    "Normalizes the model MAE score, given the baseline score"
    # # Value range of MAE is [0, infinity), 0 is best
    # 10, 5 ==> 0 because worse than naive
    # 10, 20 ==> 0.5
    # 5, 20 ==> 0.75 = 1 - (mae/base_mae)
    if model_mae > naive_mae:
        return 0
    else:
        return 1 - (model_mae / naive_mae)


def _mae_normalizer(df, y, model_score, **kwargs):
    "In case of MAE, calculates the baseline score for y and derives the PPS."
    df["naive"] = df[y].median()
    baseline_score = mean_absolute_error(df[y].to_numpy(), df["naive"].to_numpy())  # true, pred

    ppscore = _normalized_mae_score(abs(model_score), baseline_score)
    return ppscore, baseline_score


def _normalized_f1_score(model_f1, baseline_f1):
    "Normalizes the model F1 score, given the baseline score"
    # # F1 ranges from 0 to 1
    # # 1 is best
    # 0.5, 0.7 ==> 0 because model is worse than naive baseline
    # 0.75, 0.5 ==> 0.5
    #
    if model_f1 < baseline_f1:
        return 0
    else:
        scale_range = 1.0 - baseline_f1  # eg 0.3
        f1_diff = model_f1 - baseline_f1  # eg 0.1
        return f1_diff / scale_range  # 0.1/0.3 = 0.33


def _f1_normalizer(df, y, model_score, random_seed):
    "In case of F1, calculates the baseline score for y and derives the PPS."
    label_encoder = preprocessing.LabelEncoder()
    df["truth"] = label_encoder.fit_transform(df[y])
    df["most_common_value"] = df["truth"].value_counts().index[0]
    random = df["truth"].sample(frac=1, random_state=random_seed)

    baseline_score = max(
        f1_score(df["truth"], df["most_common_value"], average="weighted"),
        f1_score(df["truth"], random, average="weighted"),
    )

    ppscore = _normalized_f1_score(model_score, baseline_score)
    return ppscore, baseline_score


#VALID_CALCULATIONS = {
#    "regression": {
#        "type": "regression",
#        "is_valid_score": True,
#        "model_score": -1,
#        "baseline_score": -1,
#        "ppscore": -1,
#        "metric_name": "mean absolute error",
#        "metric_key": "neg_mean_absolute_error",
#        "model": tree.DecisionTreeRegressor(),
#        "score_normalizer": _mae_normalizer,
#    },
#    "classification": {
#        "type": "classification",
#        "is_valid_score": True,
#        "model_score": -1,
#        "baseline_score": -1,
#        "ppscore": -1,
#        "metric_name": "weighted F1",
#        "metric_key": "f1_weighted",
#        "model": tree.DecisionTreeClassifier(),
#        "score_normalizer": _f1_normalizer,
#    },
#    "predict_itself": {
#        "type": "predict_itself",
#        "is_valid_score": True,
#        "model_score": 1,
#        "baseline_score": 0,
#        "ppscore": 1,
#        "metric_name": None,
#        "metric_key": None,
#        "model": None,
#        "score_normalizer": None,
#    },
#    "target_is_constant": {
#        "type": "target_is_constant",
#        "is_valid_score": True,
#        "model_score": 1,
#        "baseline_score": 1,
#        "ppscore": 0,
#        "metric_name": None,
#        "metric_key": None,
#        "model": None,
#        "score_normalizer": None,
#    },
#    "target_is_id": {
#        "type": "target_is_id",
#        "is_valid_score": True,
#        "model_score": 0,
#        "baseline_score": 0,
#        "ppscore": 0,
#        "metric_name": None,
#        "metric_key": None,
#        "model": None,
#        "score_normalizer": None,
#    },
#    "feature_is_id": {
#        "type": "feature_is_id",
#        "is_valid_score": True,
#        "model_score": 0,
#        "baseline_score": 0,
#        "ppscore": 0,
#        "metric_name": None,
#        "metric_key": None,
#        "model": None,
#        "score_normalizer": None,
#    },
#}

#INVALID_CALCULATIONS = [
#    "target_is_datetime",
#    "target_data_type_not_supported",
#    "empty_dataframe_after_dropping_na",
#    "unknown_error",
#]


def _dtype_represents_categories(series) -> bool:
    "Determines if the dtype of the series represents categorical values"
    return (
        is_bool_dtype(series)
        or is_object_dtype(series)
        or is_string_dtype(series)
        or is_categorical_dtype(series)
    )


def _determine_case_and_prepare_df(df, x, y, sample=5_000, random_seed=123):
    "Returns str with the name of the determined case based on the columns x and y"
    if x == y:
        return df, "predict_itself"

    df = df[[x, y]]
    # IDEA: log.warning when values have been dropped
    df = df.dropna()

    if len(df) == 0:
        return df, "empty_dataframe_after_dropping_na"
        # IDEA: show warning
        # raise Exception(
        #     "After dropping missing values, there are no valid rows left"
        # )

    df = _maybe_sample(df, sample, random_seed=random_seed)

    if _feature_is_id(df, x):
        return df, "feature_is_id"

    category_count = df[y].value_counts().count()
    if category_count == 1:
        # it is helpful to separate this case in order to save unnecessary calculation time
        return df, "target_is_constant"
    if _dtype_represents_categories(df[y]) and (category_count == len(df[y])):
        # it is important to separate this case in order to save unnecessary calculation time
        return df, "target_is_id"

    if _dtype_represents_categories(df[y]):
        return df, "classification"
    if is_numeric_dtype(df[y]):
        # this check needs to be after is_bool_dtype (which is part of _dtype_represents_categories) because bool is considered numeric by pandas
        return df, "regression"

    if is_datetime64_any_dtype(df[y]) or is_timedelta64_dtype(df[y]):
        # IDEA: show warning
        # raise TypeError(
        #     f"The target column {y} has the dtype {df[y].dtype} which is not supported. A possible solution might be to convert {y} to a string column"
        # )
        return df, "target_is_datetime"

    # IDEA: show warning
    # raise Exception(
    #     f"Could not infer a valid task based on the target {y}. The dtype {df[y].dtype} is not yet supported"
    # )  # pragma: no cover
    return df, "target_data_type_not_supported"


def _feature_is_id(df, x):
    "Returns Boolean if the feature column x is an ID"
    if not _dtype_represents_categories(df[x]):
        return False

    category_count = df[x].value_counts().count()
    return category_count == len(df[x])


def _maybe_sample(df, sample, random_seed=None):
    """
    Maybe samples the rows of the given df to have at most `sample` rows
    If sample is `None` or falsy, there will be no sampling.
    If the df has fewer rows than the sample, there will be no sampling.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe that might be sampled
    sample : int or `None`
        Number of rows to be sampled
    random_seed : int or `None`
        Random seed that is forwarded to pandas.DataFrame.sample as `random_state`

    Returns
    -------
    pandas.DataFrame
        DataFrame after potential sampling
    """
    if sample and len(df) > sample:
        # this is a problem if x or y have more than sample=5000 categories
        # TODO: dont sample when the problem occurs and show warning
        df = df.sample(sample, random_state=random_seed, replace=False)
    return df


def _is_column_in_df(column, df):
    try:
        return column in df.columns
    except:
        return False


def _score(
    df, x, y, task, sample, cross_validation, random_seed, invalid_score, catch_errors
):
    df, case_type = _determine_case_and_prepare_df(
        df, x, y, sample=sample, random_seed=random_seed
    )
    task = _get_task(case_type, invalid_score)

    if case_type in ["classification", "regression"]:
        model_score = _calculate_model_cv_score_(
            df,
            target=y,
            feature=x,
            task=task,
            cross_validation=cross_validation,
            random_seed=random_seed,
        )
        # IDEA: the baseline_scores do sometimes change significantly, e.g. for F1 and thus change the PPS
        # we might want to calculate the baseline_score 10 times and use the mean in order to have less variance
        ppscore, baseline_score = task["score_normalizer"](
            df, y, model_score, random_seed=random_seed
        )
    else:
        model_score = task["model_score"]
        baseline_score = task["baseline_score"]
        ppscore = task["ppscore"]

    return {
        "x": x,
        "y": y,
        "ppscore": ppscore,
        "case": case_type,
        "is_valid_score": task["is_valid_score"],
        "metric": task["metric_name"],
        "baseline_score": baseline_score,
        "model_score": abs(model_score),  # sklearn returns negative mae
        "model": task["model"],
    }


def pps_score(
    df,
    x,
    y,
    task="NOT_SUPPORTED_ANYMORE",
    sample=5_000,
    cross_validation=4,
    random_seed=123,
    invalid_score=0,
    catch_errors=True,
):
    """
    Calculate the Predictive Power Score (PPS) for "x predicts y"
    The score always ranges from 0 to 1 and is data-type agnostic.

    A score of 0 means that the column x cannot predict the column y better than a naive baseline model.
    A score of 1 means that the column x can perfectly predict the column y given the model.
    A score between 0 and 1 states the ratio of how much potential predictive power the model achieved compared to the baseline model.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe that contains the columns x and y
    x : str
        Name of the column x which acts as the feature
    y : str
        Name of the column y which acts as the target
    sample : int or `None`
        Number of rows for sampling. The sampling decreases the calculation time of the PPS.
        If `None` there will be no sampling.
    cross_validation : int
        Number of iterations during cross-validation. This has the following implications:
        For example, if the number is 4, then it is possible to detect patterns when there are at least 4 times the same observation. If the limit is increased, the required minimum observations also increase. This is important, because this is the limit when sklearn will throw an error and the PPS cannot be calculated
    random_seed : int or `None`
        Random seed for the parts of the calculation that require random numbers, e.g. shuffling or sampling.
        If the value is set, the results will be reproducible. If the value is `None` a new random number is drawn at the start of each calculation.
    invalid_score : any
        The score that is returned when a calculation is invalid, e.g. because the data type was not supported.
    catch_errors : bool
        If `True` all errors will be catched and reported as `unknown_error` which ensures convenience. If `False` errors will be raised. This is helpful for inspecting and debugging errors.

    Returns
    -------
    Dict
        A dict that contains multiple fields about the resulting PPS.
        The dict enables introspection into the calculations that have been performed under the hood
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"The 'df' argument should be a pandas.DataFrame but you passed a {type(df)}\nPlease convert your input to a pandas.DataFrame"
        )
    if not _is_column_in_df(x, df):
        raise ValueError(
            f"The 'x' argument should be the name of a dataframe column but the variable that you passed is not a column in the given dataframe.\nPlease review the column name or your dataframe"
        )
    if len(df[[x]].columns) >= 2:
        raise AssertionError(
            f"The dataframe has {len(df[[x]].columns)} columns with the same column name {x}\nPlease adjust the dataframe and make sure that only 1 column has the name {x}"
        )
    if not _is_column_in_df(y, df):
        raise ValueError(
            f"The 'y' argument should be the name of a dataframe column but the variable that you passed is not a column in the given dataframe.\nPlease review the column name or your dataframe"
        )
    if len(df[[y]].columns) >= 2:
        raise AssertionError(
            f"The dataframe has {len(df[[y]].columns)} columns with the same column name {y}\nPlease adjust the dataframe and make sure that only 1 column has the name {y}"
        )

    if random_seed is None:
        from random import random

        random_seed = int(random() * 1000)

    try:
        return _score(
            df,
            x,
            y,
            task,
            sample,
            cross_validation,
            random_seed,
            invalid_score,
            catch_errors,
        )
    except Exception as exception:
        if catch_errors:
            case_type = "unknown_error"
            task = _get_task(case_type, invalid_score)
            return {
                "x": x,
                "y": y,
                "ppscore": task["ppscore"],
                "case": case_type,
                "is_valid_score": task["is_valid_score"],
                "metric": task["metric_name"],
                "baseline_score": task["baseline_score"],
                "model_score": task["model_score"],  # sklearn returns negative mae
                "model": task["model"],
            }
        else:
            raise exception


def _get_task(case_type, invalid_score):
    VALID_CALCULATIONS = {
    "regression": {
        "type": "regression",
        "is_valid_score": True,
        "model_score": -1,
        "baseline_score": -1,
        "ppscore": -1,
        "metric_name": "mean absolute error",
        "metric_key": "neg_mean_absolute_error",
        "model": tree.DecisionTreeRegressor(),
        "score_normalizer": _mae_normalizer,
    },
    "classification": {
        "type": "classification",
        "is_valid_score": True,
        "model_score": -1,
        "baseline_score": -1,
        "ppscore": -1,
        "metric_name": "weighted F1",
        "metric_key": "f1_weighted",
        "model": tree.DecisionTreeClassifier(),
        "score_normalizer": _f1_normalizer,
    },
    "predict_itself": {
        "type": "predict_itself",
        "is_valid_score": True,
        "model_score": 1,
        "baseline_score": 0,
        "ppscore": 1,
        "metric_name": None,
        "metric_key": None,
        "model": None,
        "score_normalizer": None,
    },
    "target_is_constant": {
        "type": "target_is_constant",
        "is_valid_score": True,
        "model_score": 1,
        "baseline_score": 1,
        "ppscore": 0,
        "metric_name": None,
        "metric_key": None,
        "model": None,
        "score_normalizer": None,
    },
    "target_is_id": {
        "type": "target_is_id",
        "is_valid_score": True,
        "model_score": 0,
        "baseline_score": 0,
        "ppscore": 0,
        "metric_name": None,
        "metric_key": None,
        "model": None,
        "score_normalizer": None,
    },
    "feature_is_id": {
        "type": "feature_is_id",
        "is_valid_score": True,
        "model_score": 0,
        "baseline_score": 0,
        "ppscore": 0,
        "metric_name": None,
        "metric_key": None,
        "model": None,
        "score_normalizer": None,
        },
    }
    INVALID_CALCULATIONS = [
    "target_is_datetime",
    "target_data_type_not_supported",
    "empty_dataframe_after_dropping_na",
    "unknown_error",
    ]
    if case_type in VALID_CALCULATIONS.keys():
        return VALID_CALCULATIONS[case_type]
    elif case_type in INVALID_CALCULATIONS:
        return {
            "type": case_type,
            "is_valid_score": False,
            "model_score": invalid_score,
            "baseline_score": invalid_score,
            "ppscore": invalid_score,
            "metric_name": None,
            "metric_key": None,
            "model": None,
            "score_normalizer": None,
        }
    raise Exception(f"case_type {case_type} is not supported")


def _format_list_of_dicts(scores, output, sorted):
    """
    Format list of score dicts `scores`
    - maybe sort by ppscore
    - maybe return pandas.Dataframe
    - output can be one of ["df", "list"]
    """
    if sorted:
        scores.sort(key=lambda item: item["ppscore"], reverse=True)

    if output == "df":
        df_columns = [
            "x",
            "y",
            "ppscore",
            "case",
            "is_valid_score",
            "metric",
            "baseline_score",
            "model_score",
            "model",
        ]
        data = {column: [score[column] for score in scores] for column in df_columns}
        scores = pd.DataFrame.from_dict(data)

    return scores


def pps_predictors(df, y, output="df", sorted=True, **kwargs):
    """
    Calculate the Predictive Power Score (PPS) of all the features in the dataframe
    against a target column

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe that contains the data
    y : str
        Name of the column y which acts as the target
    output: str - potential values: "df", "list"
        Control the type of the output. Either return a pandas.DataFrame (df) or a list with the score dicts
    sorted: bool
        Whether or not to sort the output dataframe/list by the ppscore
    kwargs:
        Other key-word arguments that shall be forwarded to the pps.score method,
        e.g. `sample, `cross_validation, `random_seed, `invalid_score`, `catch_errors`

    Returns
    -------
    pandas.DataFrame or list of Dict
        Either returns a tidy dataframe or a list of all the PPS dicts. This can be influenced
        by the output argument
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"The 'df' argument should be a pandas.DataFrame but you passed a {type(df)}\nPlease convert your input to a pandas.DataFrame"
        )
    if not _is_column_in_df(y, df):
        raise ValueError(
            f"The 'y' argument should be the name of a dataframe column but the variable that you passed is not a column in the given dataframe.\nPlease review the column name or your dataframe"
        )
    if len(df[[y]].columns) >= 2:
        raise AssertionError(
            f"The dataframe has {len(df[[y]].columns)} columns with the same column name {y}\nPlease adjust the dataframe and make sure that only 1 column has the name {y}"
        )
    if not output in ["df", "list"]:
        raise ValueError(
            f"""The 'output' argument should be one of ["df", "list"] but you passed: {output}\nPlease adjust your input to one of the valid values"""
        )
    if not sorted in [True, False]:
        raise ValueError(
            f"""The 'sorted' argument should be one of [True, False] but you passed: {sorted}\nPlease adjust your input to one of the valid values"""
        )

    scores = [pps_score(df, column, y, **kwargs) for column in df if column != y]

    return _format_list_of_dicts(scores=scores, output=output, sorted=sorted)


def pps_matrix(df, output="df", sorted=False, **kwargs):
    """
    Calculate the Predictive Power Score (PPS) matrix for all columns in the dataframe

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe that contains the data
    output: str - potential values: "df", "list"
        Control the type of the output. Either return a pandas.DataFrame (df) or a list with the score dicts
    sorted: bool
        Whether or not to sort the output dataframe/list by the ppscore
    kwargs:
        Other key-word arguments that shall be forwarded to the pps.score method,
        e.g. `sample, `cross_validation, `random_seed, `invalid_score`, `catch_errors`

    Returns
    -------
    pandas.DataFrame or list of Dict
        Either returns a tidy dataframe or a list of all the PPS dicts. This can be influenced
        by the output argument
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"The 'df' argument should be a pandas.DataFrame but you passed a {type(df)}\nPlease convert your input to a pandas.DataFrame"
        )
    if not output in ["df", "list"]:
        raise ValueError(
            f"""The 'output' argument should be one of ["df", "list"] but you passed: {output}\nPlease adjust your input to one of the valid values"""
        )
    if not sorted in [True, False]:
        raise ValueError(
            f"""The 'sorted' argument should be one of [True, False] but you passed: {sorted}\nPlease adjust your input to one of the valid values"""
        )

    scores = [pps_score(df, x, y, **kwargs) for x in df for y in df]

    return _format_list_of_dicts(scores=scores, output=output, sorted=sorted)


#"""工具函数"""
###### 回归的工具函数
def Squared_term(dataset, variables):
    for var in variables:
        dataset[f'{var}_squared_term'] = dataset[var] ** 2
    return dataset

def Interaction_term(dataset, variables):
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            dataset[f'{variables[i]}_{variables[j]}'] = dataset[variables[i]] * dataset[variables[j]]
    return dataset

def Cos_Sin_term(dataset, variables):
    for var in variables:
        dataset[f'{var}_cos_term'] = np.cos(dataset[var])
        dataset[f'{var}_sin_term'] = np.sin(dataset[var])
    return dataset

def Piecewise_term(dataset, variables):
    for var in variables:
        dataset[f'{var}_piecewise_term'] = np.maximum(dataset[var] - np.median(dataset[var]), 0)
    return dataset

###### 因果发现算法的工具函数
def handle_multicollinearity(data, epsilon=1e-7, corr_threshold=0.99):
    """
    检查数据中的多重共线性，并在需要时向存在多重共线性的列添加随机扰动。

    参数:
    - data (np.ndarray): 输入的数据矩阵，形状为 (样本数, 特征数)。
    - epsilon (float): 添加的随机扰动的尺度，默认值为1e-10。
    - corr_threshold (float): 判断高相关性的阈值，默认值为0.95。

    返回:
    - data (np.ndarray): 处理后的数据矩阵。
    """
    # 计算相关系数矩阵
    corr_matrix = np.corrcoef(data, rowvar=False)
    n_cols = corr_matrix.shape[0]

    # 使用集合存储所有涉及多重共线性的列索引，避免重复
    high_corr_indices = set()

    # 遍历相关系数矩阵的上三角部分，寻找高相关的列对
    for i in range(n_cols):
        for j in range(i+1, n_cols):
            if np.abs(corr_matrix[i, j]) > corr_threshold:
                high_corr_indices.add(i)
                high_corr_indices.add(j)

    if high_corr_indices:
        sorted_indices = sorted(high_corr_indices)
        # print(f"检测到多重共线性，涉及的列索引: {sorted_indices}。正在添加随机扰动...")

        # 生成与高相关性列对应的随机扰动
        noise = np.random.normal(0, epsilon, (data.shape[0], len(sorted_indices)))

        # 将扰动添加到相应的列
        data[:, sorted_indices] += noise

    return data

def convert_bidirectional_to_dag(bidirectional_adj: pd.DataFrame) -> pd.DataFrame:
    """
    将双向编码的邻接矩阵转换为单向编码的邻接矩阵（DAG表示）。

    在双向编码的邻接矩阵中：
    - [j, i] = 1 且 [i, j] = -1 表示 i → j
    - [j, i] = 1 且 [i, j] = 1 表示 i ↔ j（互为因果）
    - 其他情况 [i, j] = 0 表示无边

    转换后，单向编码的邻接矩阵将：
    - [i, j] = 1 表示 i → j
    - [i, j] = 1 且 [j, i] = 1 表示 i ↔ j
    - [i, j] = 0 表示无边

    参数:
    - bidirectional_adj (pd.DataFrame): 双向编码的邻接矩阵，使用 1 和 -1 表示有向边。

    返回:
    - pd.DataFrame: 单向编码的邻接矩阵（DAG表示）。
    """
    # 确保输入是一个方阵
    if bidirectional_adj.shape[0] != bidirectional_adj.shape[1]:
        raise ValueError("输入的邻接矩阵必须是方阵（行数等于列数）。")

    # 确保行列索引一致
    if not bidirectional_adj.index.equals(bidirectional_adj.columns):
        raise ValueError("邻接矩阵的行索引和列索引必须相同。")

    # 初始化一个全零的邻接矩阵
    single_direction_adj = pd.DataFrame(0, index=bidirectional_adj.index, columns=bidirectional_adj.columns)

    # 遍历每一对变量，确定有向边
    for source in bidirectional_adj.columns:
        for target in bidirectional_adj.index:
            if source == target:
                # 根据DAG定义，通常不允许自环，因此设置为0
                single_direction_adj.at[source, target] = 0
                continue

            # 检查双向编码的邻接矩阵来确定有向边
            if bidirectional_adj.at[target, source] == 1:
                if bidirectional_adj.at[source, target] == -1:
                    # 表示 source → target
                    single_direction_adj.at[source, target] = 1
                elif bidirectional_adj.at[source, target] == 1:
                    # 表示 source ↔ target（互为因果）
                    single_direction_adj.at[source, target] = 1
                    single_direction_adj.at[target, source] = 1
                # 如果 [source, target] == 0 或其他情况，不设置边
    return single_direction_adj

def convert_dag_with_strength_to_dag(adj: pd.DataFrame) -> pd.DataFrame:
    """
    将表示因果影响强度的邻接矩阵数据框转换为二元邻接矩阵数据框。
    无论存在正向影响还是负向影响，都将其转换为1；不存在影响则为0。

    参数:
    - adj (pd.DataFrame): 原始的因果影响强度邻接矩阵，元素可以是正数、负数或0。

    返回:
    - pd.DataFrame: 二元邻接矩阵，元素为0或1。
    """
    # 确保输入是一个方阵
    if adj.shape[0] != adj.shape[1]:
        raise ValueError("输入的邻接矩阵必须是方阵（行数等于列数）。")

    # 确保行列索引一致
    if not adj.index.equals(adj.columns):
        raise ValueError("邻接矩阵的行索引和列索引必须相同。")

    # 将所有非零元素转换为1，零元素保持为0
    return adj.ne(0).astype(int)


#"""离散化统计指标的工具函数"""
def discretize_sequence(x, ffactor=10):
    """
    将连续序列离散化。

    参数:
    x (array-like): 输入的连续变量，范围在 [-1, 1] 之间。
    ffactor (int): 离散化因子，用于缩放和离散化。

    返回:
    np.ndarray: 离散化后的序列（整数类型）。
    """
    # 确保输入在 [-1, 1] 范围内
    x = np.clip(x, -1, 1)
    # 缩放并四舍五入
    x = np.round(x * ffactor).astype(int)
    return x

def compute_discrete_probability(x):
    """
    计算离散概率分布。

    参数:
    x (iterable): 输入的离散化后的序列，可以是单变量或联合变量。

    返回:
    Counter: 元素及其计数。
    """
    return Counter(x)

def discrete_entropy(x, bias_factor=0.7):
    """
    计算离散熵。

    参数:
    x (iterable): 输入的离散化后的序列，可以是单变量或联合变量。
    bias_factor (float): 偏差因子，用于修正有限样本的熵估计。

    返回:
    float: 计算得到的熵值。
    """
    c = compute_discrete_probability(x)
    pk = np.array(list(c.values()), dtype=float)
    pk_sum = pk.sum()
    if pk_sum == 0:
        return 0.0
    pk /= pk_sum
    # 避免 log(0) 问题，添加一个很小的常数
    vec = pk * np.log(pk + 1e-12)
    S = -np.sum(vec)
    # 添加偏差项
    bias = bias_factor * (len(pk) - 1) / (2.0 * len(x))
    return S + bias

def discrete_joint_entropy(x, y):
    """
    计算两个离散序列的联合熵 H(X, Y)。

    参数:
    x, y (array-like): 输入的离散化后的序列。

    返回:
    float: 联合熵 H(X, Y)。
    """
    joint = list(zip(x, y))  # 将 x 和 y 配对
    return discrete_entropy(joint)

def normalized_error_probability(x, y):
    """
    计算归一化的错误概率。

    该函数通过构建联合概率矩阵，计算分类错误的概率，并将其归一化。

    参数:
    x, y (array-like): 输入的离散化后的序列。

    返回:
    float: 归一化的错误概率。
    """
    cx = Counter(x)
    cy = Counter(y)

    sorted_cx = sorted(cx.keys())
    sorted_cy = sorted(cy.keys())

    # 统计联合频数
    pxy = defaultdict(int)
    for a, b in zip(x, y):
        pxy[(a, b)] += 1

    total = sum(pxy.values())
    if total == 0:
        return 0.0  # 或者根据需求返回其他值

    # 构建联合概率矩阵
    pxy_matrix = np.array([
        [pxy.get((a, b), 0) for b in sorted_cy]
        for a in sorted_cx
    ], dtype=float)

    # 归一化为概率
    pxy_matrix /= total

    # 计算每行的最大概率
    max_per_row = pxy_matrix.max(axis=1)
    perr = 1 - np.sum(max_per_row)

    # 计算每列的概率和的最大值
    sum_per_column = pxy_matrix.sum(axis=0)
    max_perr = 1 - np.max(sum_per_column)

    # 归一化错误概率
    pnorm = perr / max_perr if max_perr > 0 else perr
    return pnorm

def discrete_divergence(cx, cy):
    """
    计算两个离散分布之间的KL散度（Kullback-Leibler Divergence）。

    KL散度衡量了分布 cx 相对于分布 cy 的差异，是信息论中的一个重要概念。

    参数:
    cx (Counter): 第一个离散分布的元素计数。
    cy (Counter): 第二个离散分布的元素计数。

    返回:
    float: KL散度 D_KL(cx || cy)。
    """
    # 创建 cy 的副本，避免修改原始对象
    cy = cy.copy()

    # 为了避免 cy 中某些元素的概率为零，将它们的计数设为 1
    for a in cx:
        if cy[a] == 0:
            cy[a] = 1

    # 计算概率
    nx = float(sum(cx.values()))
    ny = float(sum(cy.values()))

    kl_div = 0.0
    for a, v in cx.items():
        px = v / nx
        py = cy[a] / ny
        kl_div += px * np.log(px / py)
    return kl_div

def discrete_conditional_entropy(x, y):
    """
    计算两个离散序列的条件熵 H(X|Y)。

    条件熵衡量了在已知 Y 的情况下，X 的不确定性。

    参数:
    x, y (array-like): 输入的离散化后的序列。

    返回:
    float: 条件熵 H(X|Y)。
    """
    joint_entropy = discrete_joint_entropy(x, y)
    entropy_y = discrete_entropy(y)
    return joint_entropy - entropy_y

def adjusted_mutual_information_score(x, y):
    """
    计算两个离散序列的调整互信息（Adjusted Mutual Information, AMI）。

    AMI 是互信息的一种调整版本，考虑了随机期望的互信息，通常用于聚类评估。

    参数:
    x, y (array-like): 输入的离散化后的序列。

    返回:
    float: 调整后的互信息。
    """
    return adjusted_mutual_info_score(x, y)

def discrete_mutual_information(x, y):
    """
    计算两个离散序列的互信息 I(X; Y)。

    互信息衡量了两个变量之间共享的信息量，是信息论中的一个基本概念。

    参数:
    x, y (array-like): 输入的离散化后的序列。

    返回:
    float: 互信息 I(X; Y)。
    """
    entropy_x = discrete_entropy(x)
    entropy_y = discrete_entropy(y)
    joint_entropy = discrete_joint_entropy(x, y)
    mutual_info = entropy_x + entropy_y - joint_entropy
    # 避免由于数值误差导致的负值
    mutual_info = max(mutual_info, 0)
    return mutual_info

# -------------------
# Helper Functions
# -------------------

def normalize_discrete(x):
    """
    对离散化后的序列进行标准化处理。

    参数:
    x (array-like): 离散化后的x序列。

    返回:
    np.ndarray: 标准化后的x序列。
    """
    if len(set(x)) < 2:
        return np.zeros_like(x, dtype=float)
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std > 0:
        return (x - x_mean) / x_std
    else:
        return x - x_mean

def to_numerical(x_discrete, y_continuous):
    """
    将类别型的离散x转换为数值型，通过将每个唯一的x值替换为对应y的平均值。

    参数:
    x_discrete (array-like): 离散化后的x数组。
    y_continuous (array-like): 与x对应的连续y数组。

    返回:
    np.ndarray: 数值型的x数组，每个x值被替换为对应的y平均值。
    """
    dx = defaultdict(lambda: [0.0, 0])
    for a, b in zip(x_discrete, y_continuous):
        dx[a][0] += b
        dx[a][1] += 1
    for a in dx:
        dx[a][0] /= dx[a][1] if dx[a][1] > 0 else 1e-12
    x_numerical = np.array([dx[a][0] for a in x_discrete], dtype=float)
    x_numerical = (x_numerical - np.mean(x_numerical)) / np.std(x_numerical) if np.std(x_numerical) > 0 else x_numerical
    return x_numerical

def count_unique(x):
    """
    计算数组中唯一元素的数量。

    参数:
    x (array-like): 输入数组。

    返回:
    int: 唯一元素的数量。
    """
    return len(set(x))

# -------------------
# Feature Engineering Functions
# -------------------

def normalized_entropy_baseline(x):
    """
    计算给定归一化x的标准化熵基线。

    参数:
    x (array-like): 离散且归一化的x序列。

    返回:
    float: 标准化熵基线值。
    """
    if len(set(x)) < 2:
        return 0.0
    xs = np.sort(x)
    delta = xs[1:] - xs[:-1]
    delta = delta[delta != 0]
    if len(delta) == 0:
        return 0.0
    hx = np.mean(np.log(delta))
    hx += special.psi(len(delta))
    hx -= special.psi(1)
    return hx

def normalized_entropy(x, m=2):
    """
    计算标准化熵。

    参数:
    x (array-like): 离散且归一化的x序列。
    m (int): delta计算的参数。

    返回:
    float: 标准化熵值。
    """
    cx = Counter(x)
    if len(cx) < 2:
        return 0.0
    xk = np.array(list(cx.keys()), dtype=float)
    xk.sort()
    if len(xk) < 2:
        return 0.0
    delta = (xk[1:] - xk[:-1]) / m
    counter = np.array([cx[i] for i in xk], dtype=float)
    hx = np.sum(counter[1:] * np.log(delta / counter[1:])) / len(x)
    hx += (special.psi(len(delta)) - np.log(len(delta)))
    hx += np.log(len(x))
    hx -= (special.psi(m) - np.log(m))
    return hx

def igci(x, y):
    """
    计算IGCI（信息几何因果推断）度量。

    参数:
    x (array-like): 离散且归一化的x序列。
    y (array-like): 离散且归一化的y序列。

    返回:
    float: IGCI度量值。
    """
    # 检查是否有足够的唯一值
    if len(set(x)) < 2:
        return 0.0

    # 判断是否有重复的x值
    if len(x) != len(set(x)):
        dx = defaultdict(lambda: [0.0, 0])
        for a, b in zip(x, y):
            dx[a][0] += b
            dx[a][1] += 1
        for a in dx:
            dx[a][0] /= dx[a][1] if dx[a][1] > 0 else 1e-12
        # 构建联合序列
        xy = np.array([[a, dx[a][0]] for a in dx.keys()], dtype=float)
        # 获取每个x的计数
        counter = np.array([dx[a][1] for a in xy[:, 0]], dtype=float)
    else:
        # 如果x没有重复，直接排序
        xy = np.array(sorted(zip(x, y)), dtype=float)
        counter = np.ones(len(x))

    # 计算相邻差值
    delta = xy[1:] - xy[:-1]
    # 选择y差值不为0的样本
    selec = delta[:, 1] != 0
    delta = delta[selec]
    counter = np.minimum(counter[1:], counter[:-1])[selec]

    if len(delta) == 0:
        return 0.0

    # 添加一个极小值epsilon，避免log(0)
    epsilon = 1e-12
    ratio = (delta[:, 0] + epsilon) / np.abs(delta[:, 1])
    ratio = np.where(ratio > 0, ratio, epsilon)

    # 计算 hxy，避免返回 NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        hxy = np.sum(counter * np.log(ratio)) / len(x)

    # 检查 hxy 是否为有效数值
    if np.isnan(hxy):
        return 0.0

    return hxy

def uniform_divergence(x, m=2):
    """
    计算统一散度。

    参数:
    x (array-like): 离散且归一化的x序列。
    m (int): delta计算的参数。

    返回:
    float: 统一散度值。
    """
    cx = Counter(x)
    xk = np.array(list(cx.keys()), dtype=float)
    xk.sort()
    delta = np.zeros(len(xk))
    if len(xk) > 1:
        delta[0] = xk[1] - xk[0]
        if len(xk) > m:
            delta[1:-1] = (xk[m:] - xk[:-m]) / m
        else:
            delta[1:-1] = (xk[-1] - xk[0]) / (len(xk) - 1)
        delta[-1] = xk[-1] - xk[-2]
    else:
        delta = np.array([np.sqrt(12)], dtype=float)  # 假设均匀分布在[-1,1]

    counter = np.array([cx[i] for i in xk], dtype=float)
    delta_sum = np.sum(delta)
    if delta_sum > 0:
        delta = delta / delta_sum
    else:
        delta = delta
    if len(xk) > 1:
        hx = np.sum(counter * np.log(counter / delta)) / len(x)
    else:
        hx = 0.0
    hx -= np.log(len(x))
    hx += (special.psi(m) - np.log(m))
    return hx

def normalized_skewness(x):
    """
    计算x的标准化偏度。

    参数:
    x (array-like): 离散且归一化的x序列。

    返回:
    float: 标准化偏度值。
    """
    return stats.skew(x)

def normalized_kurtosis(x):
    """
    计算x的标准化峰度。

    参数:
    x (array-like): 离散且归一化的x序列。

    返回:
    float: 标准化峰度值。
    """
    return stats.kurtosis(x)

def normalized_moment(x, y, n, m):
    """
    计算x和y的标准化联合矩。

    参数:
    x (array-like): 离散且归一化的x序列。
    y (array-like): 离散且归一化的y序列。
    n (int): x的幂次。
    m (int): y的幂次。

    返回:
    float: 标准化的联合矩值。
    """
    return np.mean((x ** n) * (y ** m))

def moment21(x, y):
    """
    计算标准化联合矩 I(X^2 * Y)。

    参数:
    x (array-like): 离散且归一化的x序列。
    y (array-like): 离散且归一化的y序列。

    返回:
    float: 联合矩 I(X^2 * Y)。
    """
    return normalized_moment(x, y, 2, 1)

def moment22(x, y):
    """
    计算标准化联合矩 I(X^2 * Y^2)。

    参数:
    x (array-like): 离散且归一化的x序列。
    y (array-like): 离散且归一化的y序列。

    返回:
    float: 联合矩 I(X^2 * Y^2)。
    """
    return normalized_moment(x, y, 2, 2)

def moment31(x, y):
    """
    计算标准化联合矩 I(X^3 * Y)。

    参数:
    x (array-like): 离散且归一化的x序列。
    y (array-like): 离散且归一化的y序列。

    返回:
    float: 联合矩 I(X^3 * Y)。
    """
    return normalized_moment(x, y, 3, 1)

def fit_pairwise(x, y):
    """
    拟合多项式到x和y，并基于系数计算一个复杂的度量值。

    参数:
    x (array-like): 离散且归一化的x序列（数值型）。
    y (array-like): 离散且归一化的y序列（数值型）。

    返回:
    float: 拟合度量值。
    """
    if count_unique(x) <= 2 or count_unique(y) <= 2:
        return 0.0
    x_std = x if np.std(x) == 1 else (x - np.mean(x)) / np.std(x) if np.std(x) > 0 else x
    y_std = y if np.std(y) == 1 else (y - np.mean(y)) / np.std(y) if np.std(y) > 0 else y
    try:
        xy1 = np.polyfit(x_std, y_std, 1)
        xy2 = np.polyfit(x_std, y_std, 2)
        return abs(2 * xy2[0]) + abs(xy2[1] - xy1[0])
    except np.RankWarning:
        return 0.0
    except Exception:
        return 0.0

def fit_error(x, y, m=2):
    """
    计算x和y之间的拟合误差。

    参数:
    x (array-like): 离散且归一化的x序列。
    y (array-like): 离散且归一化的y序列。
    m (int): 拟合时使用的多项式的阶数。

    返回:
    float: 拟合误差。
    """
    if count_unique(x) <= m or count_unique(y) <= m:
        poly_degree = min(count_unique(x), count_unique(y)) - 1
    else:
        poly_degree = m

    if poly_degree < 1:
        return 0.0

    try:
        poly = np.polyfit(x, y, poly_degree)
        y_pred = np.polyval(poly, x)
        return np.std(y - y_pred)
    except np.RankWarning:
        return 0.0
    except Exception:
        return 0.0

def fit_noise_entropy(x, y, minc=10):
    """
    计算拟合噪声熵。

    参数:
    x (array-like): 离散且归一化的x序列。
    y (array-like): 离散且归一化的y序列。
    minc (int): 计算熵的最小计数阈值。

    返回:
    float: 拟合噪声熵。
    """
    cx = Counter(x)
    entyx = []
    for a in cx:
        if cx[a] > minc:
            y_subset = y[x == a]
            entyx.append(discrete_entropy(y_subset))
    if len(entyx) == 0:
        return 0.0
    n = count_unique(y)
    return np.std(entyx) / np.log(n) if n > 0 else 0.0

def fit_noise_skewness(x, y, minc=8):
    """
    计算拟合噪声偏度的标准差。

    参数:
    x (array-like): 离散且归一化的x序列。
    y (array-like): 离散且归一化的y序列。
    minc (int): 计算偏度的最小计数阈值。

    返回:
    float: 拟合噪声偏度的标准差。
    """
    cx = Counter(x)
    skewyx = []
    for a in cx:
        if cx[a] >= minc:
            y_subset = y[x == a]
            skewyx.append(normalized_skewness(y_subset))
    if len(skewyx) == 0:
        return 0.0
    return np.std(skewyx)

def fit_noise_kurtosis(x, y, minc=8):
    """
    计算拟合噪声峰度的标准差。

    参数:
    x (array-like): 离散且归一化的x序列。
    y (array-like): 离散且归一化的y序列。
    minc (int): 计算峰度的最小计数阈值。

    返回:
    float: 拟合噪声峰度的标准差。
    """
    cx = Counter(x)
    kurtyx = []
    for a in cx:
        if cx[a] >= minc:
            y_subset = y[x == a]
            kurtyx.append(normalized_kurtosis(y_subset))
    if len(kurtyx) == 0:
        return 0.0
    return np.std(kurtyx)

def conditional_distribution_similarity(x, y, minc=12):
    """
    计算条件分布相似性。

    参数:
    x (array-like): 离散且归一化的x序列。
    y (array-like): 离散且归一化的y序列。
    minc (int): 计算条件分布的最小计数阈值。

    返回:
    float: 条件分布相似性度量。
    """
    cx = Counter(x)
    cy = Counter(y)
    yrange = sorted(cy.keys())
    ny = len(yrange)

    py = np.array([cy[i] for i in yrange], dtype=float)
    py = py / py.sum() if py.sum() > 0 else py

    pyx = []
    for a in cx:
        if cx[a] > minc:
            yx = y[x == a]
            cyx = Counter(yx)
            pyxa = np.array([cyx.get(i, 0.0) for i in yrange], dtype=float)
            if pyxa.sum() == 0:
                continue
            pyxa = pyxa / pyxa.sum()
            pyx.append(py * pyxa)  # 修正这里，将 pyx * pyxa 改为 py * pyxa

    if len(pyx) == 0:
        return 0.0

    pyx = np.array(pyx)
    pyx = pyx - pyx.mean(axis=0)
    return np.std(pyx)


#"""Cloud的工具函数"""
def log2(n):
    return log(n or 1, 2)

def C_MN(n: int, K: int):
    """Computes the normalizing term of NML distribution recursively. O(n+K)

    For more detail, please refer to eq (19) (Theorem1) in
    "NML Computation Algorithms for Tree-Structured Multinomial Bayesian Networks"
    https://pubmed.ncbi.nlm.nih.gov/18382603/

    and algorithm 2 in
    "Computing the Multinomial Stochastic Complexity in Sub-Linear Time"
    http://pgm08.cs.aau.dk/Papers/31_Paper.pdf


    Args
    ----------
        n (int): sample size of a dataset
        K (int): K-value multinomal distribution

    Returns
    ----------
        float: (Approximated) Multinomal Normalizing Sum

    """

    total = 1
    b = 1
    d = 10 # 10 digit precision

    #bound = int(ceil(2 + sqrt( -2 * n * np.log(2 * 10**(-d) - 100 ** (-d)))))
    bound = int(ceil(2 + sqrt(2 * n * d * log(10))))  # using equation (38)

    for k in range(1, bound + 1):
        b = (n - k + 1) / n * b
        total += b

    log_old_sum = log2(1.0)
    log_total = log2(total)
    log_n = log2(n)
    for j in range(3, K + 1):
        log_x = log_n + log_old_sum - log_total - log2(j - 2)
        x = 2 ** log_x
        log_one_plus_x = log2(1 + x)
        log_new_sum = log_total + log_one_plus_x
        log_old_sum = log_total
        log_total = log_new_sum

    if K == 1:
        log_total = log2(1.0)

    return log_total

def parametric_complexity(X, Y, model_type: str, X_ndistinct_vals=None, Y_ndistinct_vals=None):
    """Computes the Parametric Complexity of Multinomals.

    Args
    ----------
        model_type (str): ["to", "gets", "indep", "confounder"]
        X (sequence): sequence of discrete outcomes
        Y (sequence): sequence of discrete outcomes
        X_ndistinct_vals (int): number of distinct values of the multinomial r.v X.
        Y_ndistinct_vals (int): number of distinct values of the multinomial r.v Y.

    Returns
    ----------
        float: Parametric Complexity of Multinomals

    """

    assert len(X)==len(Y)
    n = len(X)
    X_ndistinct_vals = X_ndistinct_vals or len(set(X))
    Y_ndistinct_vals = Y_ndistinct_vals or len(set(Y))


    if model_type == "confounder":
        return  C_MN(n=n, K=X_ndistinct_vals * Y_ndistinct_vals)

    else:
        return  C_MN(n=n, K=X_ndistinct_vals) + C_MN(n=n, K=Y_ndistinct_vals)

# ref: https://github.molgen.mpg.de/EDA/cisc/blob/master/formatter.py
def stratify(X, Y):
    """Stratifies Y based on unique values of X.
    Args:
        X (sequence): sequence of discrete outcomes
        Y (sequence): sequence of discrete outcomes
    Returns:
        (dict): list of Y-values for a X-value
    """
    Y_grps = defaultdict(list)
    for i, x in enumerate(X):
        Y_grps[x].append(Y[i])
    return Y_grps

def map_to_majority(X, Y):
    """Creates a function that maps x to most frequent y.
    Args:
        X (sequence): sequence of discrete outcomes
        Y (sequence): sequence of discrete outcomes
    Returns:
        (dict): map from Y-values to frequently co-occuring X-values
    """
    f = dict()
    Y_grps = stratify(X, Y)
    for x, Ys in Y_grps.items():
        frequent_y, _ = Counter(Ys).most_common(1)[0]
        f[x] = frequent_y
    return f

def update_regression(C, E, f, max_niterations=1000):
    """Update discrete regression with C as a cause variable and Y as a effect variable
    so that it maximize likelihood
    Args
    -------
        C (sequence): sequence of discrete outcomes
        E (sequence): sequence of discrete outcomes
        f (dict): map from C to Y

    """
    supp_C = list(set(C))
    supp_E = list(set(E))
    mod_E = len(supp_E)
    n = len(C)

    # N_E's log likelihood
    # optimize f to minimize N_E's log likelihood
    cur_likelihood = 0
    res = [(e - f[c]) % mod_E for c, e in zip(C, E)]
    for freq in Counter(res).values():
        cur_likelihood += freq * (log2(n) - log2(freq))

    j = 0
    minimized = True
    while j < max_niterations and minimized:
        minimized = False

        for c_to_map in supp_C:
            best_likelihood = sys.float_info.max
            best_e = None

            for cand_e in supp_E:
                if cand_e == f[c_to_map]:
                    continue

                f_ = f.copy()
                f_[c_to_map] = cand_e

                """
                if len(set(f_.values())) == 1:
                    continue
                """

                neglikelihood = 0
                res = [(e - f_[c]) % mod_E for c, e in zip(C, E)]
                for freq in Counter(res).values():
                    neglikelihood += freq * (log2(n) - log2(freq))

                if neglikelihood < best_likelihood:
                    best_likelihood = neglikelihood
                    best_e = cand_e

            if best_likelihood < cur_likelihood:
                cur_likelihood = best_likelihood
                f[c_to_map] = best_e
                minimized = True
        j += 1

    return f

def cause_effect_negloglikelihood(C, E, func):
    """Compute negative log likelihood for cause & effect pair.
    Model type : C→E

    Args
    -------
        C (sequence): sequence of discrete outcomes (Cause)
        E (sequence): sequence of discrete outcomes (Effect)
        func (dict): map from C-value to E-value

    Returns
    -------
        (float): maximum log likelihood
    """
    mod_C = len(set(C))
    mod_E = len(set(E))
    supp_C = list(set(C))
    supp_E = list(set(E))

    C_freqs = Counter(C)
    n = len(C)

    pair_cnt = defaultdict(lambda: defaultdict(int))
    for c, e in zip(C, E):
        pair_cnt[c][e] += 1

    loglikelihood = 0

    for freq in C_freqs.values():
        loglikelihood += freq * (log2(n) - log2(freq))

    for e_E in supp_E:
        freq = 0
        for e in supp_E:
            for c in supp_C:
                if (func[c] + e_E) % mod_E == e:
                    freq += pair_cnt[c][e]
        loglikelihood += freq * (log2(n) - log2(freq))

    return loglikelihood

def neg_log_likelihood(X, Y, model_type):
    """Compute negative maximum log-likelihood of the model given observations z^n.

    Args
    ------
        X (sequence): sequence of discrete outcomes
        Y (sequence): sequence of discrete outcomes
        model_type (str): one of ["to", "gets", "indep", "confounder"]
        f (dict): map from Y-values to frequently co-occuring X-values
    Returns
    -----------
        (float): (negative) maximum log likelihood
    """

    n = len(X)
    loglikelihood = 0

    if model_type == "to":
        f = map_to_majority(X, Y)
        f = update_regression(X, Y, f)
        loglikelihood = cause_effect_negloglikelihood(X, Y, f)

    elif model_type == "gets":
        g = map_to_majority(Y, X)
        g = update_regression(Y, X, g)
        loglikelihood = cause_effect_negloglikelihood(Y, X, g)

    elif model_type == "indep":
        X_freqs = Counter(X)
        Y_freqs = Counter(Y)
        for freq in X_freqs.values():
            loglikelihood += freq * (log2(n) - log2(freq))
        for freq in Y_freqs.values():
            loglikelihood += freq * (log2(n) - log2(freq))

    elif model_type == "confounder":
        pair_cnt = defaultdict(lambda: defaultdict(int))
        for x, y in zip(X, Y):
            pair_cnt[x][y] += 1

        for x in list(set(X)):
            for y in list(set(Y)):
                loglikelihood += pair_cnt[x][y] * (log2(n) - log2(pair_cnt[x][y]))

    return loglikelihood

def sc(X, Y, model_type: str, X_ndistinct_vals=None, Y_ndistinct_vals=None):
    """Computes the stochastic complexity of z^n(two discrete sequences).

    Args
    ------
        X (sequence): sequence of discrete outcomes
        Y (sequence): sequence of discrete outcomes
        model_type (str): ["to", "gets", "indep", "confounder"]
        X_ndistinct_vals (int): number of distinct values of the multinomial r.v X.
        Y_ndistinct_vals (int): number of distinct values of the multinomial r.v Y.

    Returns
    ----------
        float: Stochastic Complexity of a given dataset
    """
    assert len(X)==len(Y)
    X_ndistinct_vals = X_ndistinct_vals or len(set(X))
    Y_ndistinct_vals = Y_ndistinct_vals or len(set(Y))

    data_cost =  neg_log_likelihood(X, Y, model_type)
    model_cost = parametric_complexity(X, Y, model_type, X_ndistinct_vals, Y_ndistinct_vals)

    stochastic_complexity = data_cost + model_cost

    # add function code length
    if model_type == "to":
        stochastic_complexity += log2(Y_ndistinct_vals**(X_ndistinct_vals - 1) - 1)
    elif model_type == "gets":
        stochastic_complexity += log2(X_ndistinct_vals**(Y_ndistinct_vals - 1) - 1)

    return stochastic_complexity

def Cloud_print(score, llabel="X", rlabel="Y"):
    score.sort(key=lambda x: x[0])
    pred = score[0][1]
    if pred == "to":
        arrow = "⇒"
    elif pred == "gets":
        arrow = "⇐"
    elif pred == "indep":
        arrow = "⫫"
    elif pred == "confounder":
        arrow = "⇐  C ⇒"
    conf = abs(score[0][0] - score[1][0])
    out_str = "Cloud Inference Result:: %s %s %s\t Δ=%.2f" % \
                          (llabel, arrow, rlabel, conf)
    print(out_str)

def Cloud(X, Y, n_candidates=4, is_print=False, X_ndistinct_vals=None, Y_ndistinct_vals=None):
    """main function in our study.
    Cloud (Code Length-based method for Unobserved factor in Discrete data)

    Args
    ----------
        X (sequence): sequence of discrete outcomes
        Y (sequence): sequence of discrete outcomes
        n_candidates (int): the number of model candidates
        is_print (bool): whether or not to print inference result
        X_ndistinct_vals (int): number of distinct values of the multinomial r.v X.
        Y_ndistinct_vals (int): number of distinct values of the multinomial r.v Y.

    Returns
    ----------
        (List) : each element is tuple that contains code length L(z^n, M) (float) and causal model' label (str)

    """
    if n_candidates == 4:
        MODEL_CANDIDATES = ["to", "gets", "indep", "confounder"]
    elif n_candidates == 2:
        MODEL_CANDIDATES = ["to", "gets"]
    elif n_candidates == 1:
        MODEL_CANDIDATES = ["confounder"]
    else:
        MODEL_CANDIDATES = ["to", "gets", "indep"]

    # prepare data
    le_X = LabelEncoder()
    X = le_X.fit_transform(X)
    le_Y = LabelEncoder()
    Y = le_Y.fit_transform(Y)

    results = []

    for model_type in MODEL_CANDIDATES:
        stochastic_complexity = sc(X, Y, model_type, X_ndistinct_vals, Y_ndistinct_vals)
        results.append((stochastic_complexity, model_type))

    if is_print:
        Cloud_print(results)

    return results

def Cloud_output(score):
    score.sort(key=lambda x: x[0])
    pred = score[0][1]
    conf = abs(score[0][0] - score[1][0])
    return pred, conf


#"""特征工程"""
###########################################################################
################################相关系数特征################################
###########################################################################
#"""皮尔逊相关系数"""
def pearson_correlation(dataset):
    """
    Given a dataset, we compute the correlation-based features for each
    varibale, which are the correlation between that variable with X and Y,
    as well as summary statistics (max, min, mean, std) of all pairs
    of correlations.
    """

    variables = dataset.columns.drop(["X", "Y"])

    df = []
    for variable in variables:
        tmp = dataset.corr().drop([variable], axis="columns").loc[variable].abs()

        df.append({
            "variable": variable,
            "corr(v,X)": dataset[[variable, "X"]].corr().loc[variable, "X"],
            "corr(v,Y)": dataset[[variable, "Y"]].corr().loc[variable, "Y"],
            "max(corr(v, others))": tmp.max(),
            "min(corr(v, others))": tmp.min(),
            "mean(corr(v, others))": tmp.mean(),
            "std(corr(v, others))": tmp.std(),
            "25%(corr(v, others))": tmp.quantile(0.25),
            "75%(corr(v, others))": tmp.quantile(0.75),
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    df["corr(X,Y)"] = dataset[["X", "Y"]].corr().loc["X", "Y"]

    # pearsonr is NaN when the variance is 0, so we fill with 0
    df.fillna(0, inplace=True)

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

#"""滚动皮尔逊相关系数绝对值"""
def rolling_abs_pearson_correlation(dataset):
    variables = dataset.columns.drop(["X", "Y"])

    df = []
    for variable in variables:
        tmp = dataset[[variable, "X", "Y"]].copy()
        tmp = tmp.sort_values(by=variable, ascending=True).reset_index(drop=True)
        # 计算窗口内v-"X"的皮尔逊相关系数绝对值
        abs_rolling_corr_x = tmp[variable].rolling(window=300).corr(tmp['X']).abs()
        # 计算窗口内v-"Y"的皮尔逊相关系数绝对值
        abs_rolling_corr_y = tmp[variable].rolling(window=300).corr(tmp['Y']).abs()

        df.append({
            "variable": variable,
            "max(abs_rolling_corr(v, X))": abs_rolling_corr_x.max(),
            "min(abs_rolling_corr(v, X))": abs_rolling_corr_x.min(),
            "mean(abs_rolling_corr(v, X))": abs_rolling_corr_x.mean(),
            # "std(abs_rolling_corr(v, X))": abs_rolling_corr_x.std(),  # 掉分
            # "25%(abs_rolling_corr(v, X))": abs_rolling_corr_x.quantile(0.25),
            # "75%(abs_rolling_corr(v, X))": abs_rolling_corr_x.quantile(0.75),
            "max(abs_rolling_corr(v, Y))": abs_rolling_corr_y.max(),
            "min(abs_rolling_corr(v, Y))": abs_rolling_corr_y.min(),
            "mean(abs_rolling_corr(v, Y))": abs_rolling_corr_y.mean(),
            # "std(abs_rolling_corr(v, Y))": abs_rolling_corr_y.std(),
            # "25%(abs_rolling_corr(v, Y))": abs_rolling_corr_y.quantile(0.25),
            # "75%(abs_rolling_corr(v, Y))": abs_rolling_corr_y.quantile(0.75)
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

#"""斯皮尔曼相关系数"""
def spearman_correlation(dataset):
    """
    Given a dataset, we compute the Spearman rank correlation-based features for each
    variable, which are the Spearman correlation between that variable with X and Y,
    as well as summary statistics (max, min, mean, std) of all pairs of Spearman correlations.
    """
    variables = dataset.columns.drop(["X", "Y"])

    df = []
    for variable in variables:
        tmp = dataset.corr(method='spearman').drop([variable], axis="columns").loc[variable].abs()

        df.append({
            "variable": variable,
            "spearman_corr(v,X)": dataset[[variable, "X"]].corr(method='spearman').loc[variable, "X"],
            "spearman_corr(v,Y)": dataset[[variable, "Y"]].corr(method='spearman').loc[variable, "Y"],
            "max(spearman_corr(v, others))": tmp.max(),
            "min(spearman_corr(v, others))": tmp.min(),
            "mean(spearman_corr(v, others))": tmp.mean(),
            "std(spearman_corr(v, others))": tmp.std(),
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    df["spearman_corr(X,Y)"] = dataset[["X", "Y"]].corr(method='spearman').loc["X", "Y"]

    # Spearman correlation is NaN when there are ties in rank, so we fill with 0
    df.fillna(0, inplace=True)

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

#"""肯德尔相关系数"""
def kendall_correlation(dataset):
    """
    Given a dataset, we compute the Kendall's tau correlation-based features for each
    variable, which are the Kendall's tau correlation between that variable with X and Y,
    as well as summary statistics (max, min, mean, std) of all pairs of Kendall's tau correlations.
    """
    variables = dataset.columns.drop(["X", "Y"])

    df = []
    for variable in variables:
        tmp = dataset.corr(method='kendall').drop([variable], axis="columns").loc[variable].abs()

        df.append({
            "variable": variable,
            "kendall_corr(v,X)": dataset[[variable, "X"]].corr(method='kendall').loc[variable, "X"],
            "kendall_corr(v,Y)": dataset[[variable, "Y"]].corr(method='kendall').loc[variable, "Y"],
            "max(kendall_corr(v, others))": tmp.max(),
            "min(kendall_corr(v, others))": tmp.min(),
            "mean(kendall_corr(v, others))": tmp.mean(),
            "std(kendall_corr(v, others))": tmp.std(),
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    df["kendall_corr(X,Y)"] = dataset[["X", "Y"]].corr(method='kendall').loc["X", "Y"]

    # Kendall's tau correlation can be NaN in some cases, so we fill with 0
    df.fillna(0, inplace=True)

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

#"""互信息"""
def mutual_information(dataset):
    """
    Given a dataset, we compute the mutual-information-based features
    for each variable, which are the MI between that variable
    and X and Y, as well as summary statistics (max, min, mean, std) of
    all pairs of MI.
    """

    variables = dataset.columns.drop(["X", "Y"])

    df = []
    for variable in variables:
        tmp = mutual_info_regression(dataset.drop(columns=[variable]), dataset[variable])
        tmp = pd.Series(tmp)  # Convert tmp to a Pandas Series

        df.append({
            "variable": variable,
            "MI(v,X)": mutual_info_regression(dataset[[variable]], dataset["X"], discrete_features=False)[0],
            "MI(v,Y)": mutual_info_regression(dataset[[variable]], dataset["Y"], discrete_features=False)[0],
            "max(MI(v, others))": tmp.max(),
            "min(MI(v, others))": tmp.min(),
            "mean(MI(v, others))": tmp.mean(),
            "std(MI(v, others))": tmp.std(),
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    df["MI(X,Y)"] = mutual_info_regression(dataset[["X"]], dataset["Y"], discrete_features=False)[0]

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

#"""条件互信息"""
def conditional_mutual_information(dataset):
    """
    Calculate conditional mutual information for each variable with X and Y.
    """
    variables = dataset.columns.drop(["X", "Y"])

    df = []
    for variable in variables:
        # Calculate conditional MI(v, X | Y)
        mi_vx_given_y = mutual_info_regression(dataset[[variable, "Y"]], dataset["X"], discrete_features=False)[0] - \
                        mutual_info_regression(dataset[["Y"]], dataset["X"], discrete_features=False)[0]

        # Calculate conditional MI(v, Y | X)
        mi_vy_given_x = mutual_info_regression(dataset[[variable, "X"]], dataset["Y"], discrete_features=False)[0] - \
                        mutual_info_regression(dataset[["X"]], dataset["Y"], discrete_features=False)[0]

        # Calculate conditional MI(X, Y | v)
        mi_xy_given_v = mutual_info_regression(dataset[["X", variable]], dataset["Y"], discrete_features=False)[0] - \
                        mutual_info_regression(dataset[[variable]], dataset["Y"], discrete_features=False)[0]

        df.append({
            "variable": variable,
            "conditional_MI(v,X|Y)": mi_vx_given_y,
            "conditional_MI(v,Y|X)": mi_vy_given_x,
            "conditional_MI(X,Y|v)": mi_xy_given_v,
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

#"""距离相关系数"""
def distance_correlation(dataset):
    """
    Given a dataset, we compute the distance correlation-based features for each
    variable, which are the distance correlation between that variable with X and Y,
    as well as summary statistics (max, min, mean, std) of all pairs of distance correlations.
    """
    variables = dataset.columns.drop(["X", "Y"])

    df = []
    for variable in variables:
        tmp = []
        # Compute distance correlation between 'variable' and all other variables (excluding itself)
        other_variables = dataset.columns.drop([variable])
        for other_var in other_variables:
            corr = dcor.distance_correlation(dataset[variable], dataset[other_var])
            tmp.append(corr)
        tmp = pd.Series(tmp)  # Convert tmp to a Pandas Series

        distance_correlation_v_X = dcor.distance_correlation(dataset[variable], dataset["X"])
        distance_correlation_v_Y = dcor.distance_correlation(dataset[variable], dataset["Y"])
        distance_correlation_X_Y = dcor.distance_correlation(dataset["X"], dataset["Y"])
        distance_correlation_v_X_square = distance_correlation_v_X ** 2
        distance_correlation_v_Y_square = distance_correlation_v_Y ** 2
        distance_correlation_X_Y_square = distance_correlation_X_Y ** 2

        df.append({
            "variable": variable,
            "dcor(v,X)": distance_correlation_v_X,
            "dcor(v,Y)": distance_correlation_v_Y,
            "dcor(v,X)^2": distance_correlation_v_X_square,
            "dcor(v,Y)^2": distance_correlation_v_Y_square,
            "max(dcor(v, others))": tmp.max(),
            "min(dcor(v, others))": tmp.min(),
            "mean(dcor(v, others))": tmp.mean(),
            "std(dcor(v, others))": tmp.std(),
            "25%(dcor(v, others))": tmp.quantile(0.25),
            "75%(dcor(v, others))": tmp.quantile(0.75),
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    df["dcor(X,Y)"] = distance_correlation_X_Y
    df["dcor(X,Y)^2"] = distance_correlation_X_Y_square

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

#"""距离相关系数-尝试不同的欧几里得距离指数"""
def distance_correlation_different_exponent(dataset):
    """
    Given a dataset, we compute the distance correlation-based features for each
    variable, which are the distance correlation between that variable with X and Y,
    with different exponent, as well as summary statistics (max, min, mean, std) of all pairs of distance correlations.
    """
    variables = dataset.columns.drop(["X", "Y"])

    df = []
    for variable in variables:
        tmp = []
        # Compute distance correlation between 'variable' and all other variables (excluding itself)
        other_variables = dataset.columns.drop([variable])
        for other_var in other_variables:
            corr = dcor.distance_correlation(dataset[variable], dataset[other_var])
            tmp.append(corr)
        tmp = pd.Series(tmp)  # Convert tmp to a Pandas Series

        distance_correlation_v_X = dcor.distance_correlation(dataset[variable], dataset["X"], exponent=0.5)
        distance_correlation_v_Y = dcor.distance_correlation(dataset[variable], dataset["Y"], exponent=0.5)
        distance_correlation_X_Y = dcor.distance_correlation(dataset["X"], dataset["Y"], exponent=0.5)
        distance_correlation_v_X_square = distance_correlation_v_X ** 2
        distance_correlation_v_Y_square = distance_correlation_v_Y ** 2
        distance_correlation_X_Y_square = distance_correlation_X_Y ** 2

        df.append({
            "variable": variable,
            "dcor_0.5exp(v,X)": distance_correlation_v_X,
            "dcor_0.5exp(v,Y)": distance_correlation_v_Y,
            "dcor_0.5exp(v,X)^2": distance_correlation_v_X_square,
            "dcor_0.5exp(v,Y)^2": distance_correlation_v_Y_square,
            "max(dcor_0.5exp(v, others))": tmp.max(),
            "min(dcor_0.5exp(v, others))": tmp.min(),
            "mean(dcor_0.5exp(v, others))": tmp.mean(),
            "std(dcor_0.5exp(v, others))": tmp.std(),
            "25%(dcor_0.5exp(v, others))": tmp.quantile(0.25),
            "75%(dcor_0.5exp(v, others))": tmp.quantile(0.75),
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    df["dcor_0.5exp(X,Y)"] = distance_correlation_X_Y
    df["dcor_0.5exp(X,Y)^2"] = distance_correlation_X_Y_square

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

#"""能量距离"""
def energy_distance_features(dataset):
    """
    Given a dataset, we compute the energy distance-based features for each
    variable, which are the energy distance between that variable with X and Y,
    as well as summary statistics (max, min, mean, std) of all pairs of energy distances.
    """
    variables = dataset.columns.drop(["X", "Y"])

    df = []
    for variable in variables:
        # tmp = []
        # # Compute energy distance between 'variable' and all other variables (excluding itself)
        # other_variables = dataset.columns.drop([variable])
        # for other_var in other_variables:
        #     energy_dist = dcor.energy_distance(dataset[variable], dataset[other_var])
        #     tmp.append(energy_dist)
        # tmp = pd.Series(tmp)  # Convert tmp to a Pandas Series

        energy_distance_v_X = dcor.energy_distance(dataset[variable], dataset["X"])
        energy_distance_v_Y = dcor.energy_distance(dataset[variable], dataset["Y"])
        energy_distance_X_Y = dcor.energy_distance(dataset["X"], dataset["Y"])

        df.append({
            "variable": variable,
            "energy_dist(v,X)": energy_distance_v_X,
            "energy_dist(v,Y)": energy_distance_v_Y,
            # "max(energy_dist(v, others))": tmp.max(),
            # "min(energy_dist(v, others))": tmp.min(),
            # "mean(energy_dist(v, others))": tmp.mean(),
            # "std(energy_dist(v, others))": tmp.std(),
            # "25%(energy_dist(v, others))": tmp.quantile(0.25),
            # "75%(energy_dist(v, others))": tmp.quantile(0.75),
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    df["energy_dist(X,Y)"] = energy_distance_X_Y

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

#"""偏相关系数"""
def partial_correlation(dataset):
    warnings.filterwarnings("ignore")
    """
    Compute partial correlation coefficients for each variable with X and Y,
    controlling for the other variable, as well as the partial correlation
    between X and Y controlling for each variable.
    """
    variables = dataset.columns.drop(["X", "Y"])

    df = []
    for variable in variables:
        # Compute partial correlations
        pcorr_vX_Y = pg.partial_corr(data=dataset, x=variable, y='X', covar='Y')['r'].iloc[0]
        pcorr_vY_X = pg.partial_corr(data=dataset, x=variable, y='Y', covar='X')['r'].iloc[0]
        pcorr_XY_v = pg.partial_corr(data=dataset, x='X', y='Y', covar=variable)['r'].iloc[0]

        df.append({
            "variable": variable,
            "partial_corr(v,X|Y)": pcorr_vX_Y,
            "partial_corr(v,Y|X)": pcorr_vY_X,
            "partial_corr(X,Y|v)": pcorr_XY_v,
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

#"""Copula Entropy"""
def copula_entropy(dataset):
    warnings.filterwarnings("ignore")
    """
    Given a dataset, we compute the Copula entropy-based features for each
    variable, which are the Copula entropy between that variable with X and Y,
    as well as summary statistics of all pairs of Copula entropies.
    """
    variables = dataset.columns.drop(["X", "Y"])

    df = []
    for variable in variables:
        # Compute copula entropy between v and X
        ce_v_X = copent(dataset[[variable, "X"]].values)

        # Compute copula entropy between v and Y
        ce_v_Y = copent(dataset[[variable, "Y"]].values)

        # # Compute transfer entropy from v to X and X to v
        # te_v_X = transent(dataset[variable].values, dataset["X"].values)
        # te_X_v = transent(dataset["X"].values, dataset[variable].values)

        # # Compute transfer entropy from v to Y and Y to v
        # te_v_Y = transent(dataset[variable].values, dataset["Y"].values)
        # te_Y_v = transent(dataset["Y"].values, dataset[variable].values)


        df.append({
            "variable": variable,
            "copula_entropy(v,X)": ce_v_X,
            "copula_entropy(v,Y)": ce_v_Y,
            # "transfer_entropy(v->X)": te_v_X,
            # "transfer_entropy(X->v)": te_X_v,
            # "transfer_entropy(v->Y)": te_v_Y,
            # "transfer_entropy(Y->v)": te_Y_v,
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    # Compute copula entropy between X and Y
    df["copula_entropy(X,Y)"] = copent(dataset[["X", "Y"]].values)

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

#"""Predictive Power Score"""
def PPS_feature(dataset):
    warnings.filterwarnings("ignore")
    variables = dataset.columns.drop(["X", "Y"]).tolist()

    matrix_df = pps_matrix(dataset)
    pivot_df = pd.pivot_table(matrix_df, index='x', columns='y', values='ppscore')

    df = []
    for variable in variables:
        df.append({
            "variable": variable,
            "PPS(v,X)": pivot_df.loc[variable, 'X'],
            "PPS(X,v)": pivot_df.loc['X', variable],
            "PPS(v,Y)": pivot_df.loc[variable, 'Y'],
            "PPS(Y,v)": pivot_df.loc['Y', variable],
            "PPS(X,Y)": pivot_df.loc['X', 'Y'],
            "max(PPS(v,others))": pivot_df.loc[variable, variables].max(),
            "mean(PPS(v,others))": pivot_df.loc[variable, variables].mean(),
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

###########################################################################
################################其他类型特征################################
###########################################################################
#"""维度信息"""
def add_dimension_feature(dataset):
    """
    Add a dimension feature to the dataset.
    """
    variables = dataset.columns.drop(["X", "Y"])
    dimension = len(variables)
    square_dimension = dimension * dimension
    df = pd.DataFrame({
        "variable": variables,
        "dimension": dimension,
        "square_dimension": square_dimension
    })
    df["dataset"] = dataset.name

    return df

#"""网格化信息"""
def grid_describe(df, var_col, dir_col, grid=5, xlim=[-1.0, 1.0], ylim=[-1.0, 1.0]):
    """
    将指定的空间根据 xlim 和 ylim 划分为 grid x grid 的小窗格，并统计每个窗格内的数据点数量。
    确保所有划分出的区域都被记录，即使某些区域内没有数据点，计数为0。

    参数:
    - df: pandas.DataFrame 包含 var_col 和 dir_col 两列
    - var_col: 用于 X 轴的列名（例如 'X'）
    - dir_col: 用于 Y 轴的列名（例如 'Y'）
    - grid: 网格的数量，默认为 5
    - xlim: X 轴的范围，默认为 [-1.0, 1.0]
    - ylim: Y 轴的范围，默认为 [-1.0, 1.0]

    返回:
    - 一个字典，键为 (x_bin, y_bin) 的元组，值为对应窗格内的数据点数量
    """
    # 定义网格边界
    x_bins = np.linspace(xlim[0], xlim[1], grid + 1)
    y_bins = np.linspace(ylim[0], ylim[1], grid + 1)

    # 使用 numpy.histogram2d 计算2D直方图
    counts, _, _ = np.histogram2d(df[var_col], df[dir_col], bins=[x_bins, y_bins])

    # 将计数结果转换为字典，键为 (x_bin, y_bin)，值为计数
    grid_counts = {}
    for x in range(grid):
        for y in range(grid):
            grid_counts[(x, y)] = int(counts[x, y])  # 转换为整数

    return grid_counts

def grid_feature(dataset):
    variables = dataset.columns.drop(["X", "Y"]).tolist()

    grid = 5
    xlim = [-1.0, 1.0]
    ylim = [-1.0, 1.0]

    df = []
    for variable in variables:
        temp = dataset[[variable, "X"]].sort_values(by=variable, ascending=True)
        grid_features_v_to_X = grid_describe(temp, variable, 'X', grid, xlim, ylim)
        temp = dataset[[variable, "Y"]].sort_values(by=variable, ascending=True)
        grid_features_v_to_Y = grid_describe(temp, variable, 'Y', grid, xlim, ylim)

        df.append({
            "variable": variable,
            "grid(v,X)(0,0)": grid_features_v_to_X[(0, 0)],
            "grid(v,X)(0,1)": grid_features_v_to_X[(0, 1)],
            "grid(v,X)(0,2)": grid_features_v_to_X[(0, 2)],
            "grid(v,X)(0,3)": grid_features_v_to_X[(0, 3)],
            "grid(v,X)(0,4)": grid_features_v_to_X[(0, 4)],
            "grid(v,X)(1,0)": grid_features_v_to_X[(1, 0)],
            "grid(v,X)(1,1)": grid_features_v_to_X[(1, 1)],
            "grid(v,X)(1,2)": grid_features_v_to_X[(1, 2)],
            "grid(v,X)(1,3)": grid_features_v_to_X[(1, 3)],
            "grid(v,X)(1,4)": grid_features_v_to_X[(1, 4)],
            "grid(v,X)(2,0)": grid_features_v_to_X[(2, 0)],
            "grid(v,X)(2,1)": grid_features_v_to_X[(2, 1)],
            "grid(v,X)(2,2)": grid_features_v_to_X[(2, 2)],
            "grid(v,X)(2,3)": grid_features_v_to_X[(2, 3)],
            "grid(v,X)(2,4)": grid_features_v_to_X[(2, 4)],
            "grid(v,X)(3,0)": grid_features_v_to_X[(3, 0)],
            "grid(v,X)(3,1)": grid_features_v_to_X[(3, 1)],
            "grid(v,X)(3,2)": grid_features_v_to_X[(3, 2)],
            "grid(v,X)(3,3)": grid_features_v_to_X[(3, 3)],
            "grid(v,X)(3,4)": grid_features_v_to_X[(3, 4)],
            "grid(v,X)(4,0)": grid_features_v_to_X[(4, 0)],
            "grid(v,X)(4,1)": grid_features_v_to_X[(4, 1)],
            "grid(v,X)(4,2)": grid_features_v_to_X[(4, 2)],
            "grid(v,X)(4,3)": grid_features_v_to_X[(4, 3)],
            "grid(v,X)(4,4)": grid_features_v_to_X[(4, 4)],
            "grid(v,Y)(0,0)": grid_features_v_to_Y[(0, 0)],
            "grid(v,Y)(0,1)": grid_features_v_to_Y[(0, 1)],
            "grid(v,Y)(0,2)": grid_features_v_to_Y[(0, 2)],
            "grid(v,Y)(0,3)": grid_features_v_to_Y[(0, 3)],
            "grid(v,Y)(0,4)": grid_features_v_to_Y[(0, 4)],
            "grid(v,Y)(1,0)": grid_features_v_to_Y[(1, 0)],
            "grid(v,Y)(1,1)": grid_features_v_to_Y[(1, 1)],
            "grid(v,Y)(1,2)": grid_features_v_to_Y[(1, 2)],
            "grid(v,Y)(1,3)": grid_features_v_to_Y[(1, 3)],
            "grid(v,Y)(1,4)": grid_features_v_to_Y[(1, 4)],
            "grid(v,Y)(2,0)": grid_features_v_to_Y[(2, 0)],
            "grid(v,Y)(2,1)": grid_features_v_to_Y[(2, 1)],
            "grid(v,Y)(2,2)": grid_features_v_to_Y[(2, 2)],
            "grid(v,Y)(2,3)": grid_features_v_to_Y[(2, 3)],
            "grid(v,Y)(2,4)": grid_features_v_to_Y[(2, 4)],
            "grid(v,Y)(3,0)": grid_features_v_to_Y[(3, 0)],
            "grid(v,Y)(3,1)": grid_features_v_to_Y[(3, 1)],
            "grid(v,Y)(3,2)": grid_features_v_to_Y[(3, 2)],
            "grid(v,Y)(3,3)": grid_features_v_to_Y[(3, 3)],
            "grid(v,Y)(3,4)": grid_features_v_to_Y[(3, 4)],
            "grid(v,Y)(4,0)": grid_features_v_to_Y[(4, 0)],
            "grid(v,Y)(4,1)": grid_features_v_to_Y[(4, 1)],
            "grid(v,Y)(4,2)": grid_features_v_to_Y[(4, 2)],
            "grid(v,Y)(4,3)": grid_features_v_to_Y[(4, 3)],
            "grid(v,Y)(4,4)": grid_features_v_to_Y[(4, 4)],
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

#"""离散化统计指标"""
def discrete_statistic_features(dataset):
    warnings.filterwarnings("ignore")
    variables = dataset.columns.drop(["X", "Y"]).tolist()

    ffactor = 10
    x = dataset["X"].values
    y = dataset["Y"].values
    x_discrete = discretize_sequence(x, ffactor=ffactor)
    y_discrete = discretize_sequence(y, ffactor=ffactor)
    x_normalized = normalize_discrete(x_discrete)
    y_normalized = normalize_discrete(y_discrete)

    df = []
    for variable in variables:
        v = dataset[variable].values
        v_discrete = discretize_sequence(v, ffactor=ffactor)
        v_normalized = normalize_discrete(v_discrete)

        # 计算联合熵 H(v,x) 和 H(v,y)
        H_vx = discrete_joint_entropy(v_discrete, x_discrete)
        H_vy = discrete_joint_entropy(v_discrete, y_discrete)
        # 计算条件熵 H(v|x) 和 H(v|y)
        H_v_given_x = discrete_conditional_entropy(v_discrete, x_discrete)
        H_v_given_y = discrete_conditional_entropy(v_discrete, y_discrete)
        # 计算互信息 I(v,x) 和 I(v,y)
        I_vx = discrete_mutual_information(v_discrete, x_discrete)
        I_vy = discrete_mutual_information(v_discrete, y_discrete)
        # 计算调整互信息 AMI(v,x) 和 AMI(v,y)
        AMI_vx = adjusted_mutual_information_score(v_discrete, x_discrete)
        AMI_vy = adjusted_mutual_information_score(v_discrete, y_discrete)
        # 计算归一化的错误概率 NPE(v,x) 和 NPE(v,y)
        error_prob_vx = normalized_error_probability(v_discrete, x_discrete)
        error_prob_vy = normalized_error_probability(v_discrete, y_discrete)

        # 计算归一化熵基线 H_baseline 和归一化熵 H_entropy 和均匀散度 uniform_div
        # H_baseline = normalized_entropy_baseline(v_normalized)
        # H_entropy = normalized_entropy(v_normalized)
        # uniform_div = uniform_divergence(v_normalized)
        # 计算IGCI (v,x) 和 IGCI(v,y)
        igci_vx = igci(v_normalized, x_normalized)
        igci_vy = igci(v_normalized, y_normalized)
        # 计算IGCI (x,v) 和 IGCI(y,v)
        igci_xv = igci(x_normalized, v_normalized)
        igci_yv = igci(y_normalized, v_normalized)
        # 计算矩特征 MM(v^2 * x) 和 MM(v^2 * y)
        moment_21_vx = moment21(v_normalized, x_normalized)
        moment_21_vy = moment21(v_normalized, y_normalized)
        # 计算矩特征 MM(v^2 * x^2) 和 MM(v^2 * y^2)
        moment_22_vx = moment22(v_normalized, x_normalized)
        moment_22_vy = moment22(v_normalized, y_normalized)
        # 计算矩特征 MM(v^3 * x) 和 MM(v^3 * y)
        moment_31_vx = moment31(v_normalized, x_normalized)
        moment_31_vy = moment31(v_normalized, y_normalized)
        # 计算拟合度量值 fit 和拟合误差 fit_error 和拟合噪声熵 fit_noise_entropy 和拟合噪声偏度标准差 fit_noise_skewness 和拟合噪声峰度标准差 fit_noise_kurtosis
        fit_val_vx = fit_pairwise(v_normalized, x_normalized)
        fit_err_vx = fit_error(v_normalized, x_normalized, m=2)
        fit_noise_ent_vx = fit_noise_entropy(v_normalized, x_normalized, minc=8)
        fit_noise_skew_vx = fit_noise_skewness(v_normalized, x_normalized, minc=8)
        fit_noise_kurt_vx = fit_noise_kurtosis(v_normalized, x_normalized, minc=8)
        fit_val_vy = fit_pairwise(v_normalized, y_normalized)
        fit_err_vy = fit_error(v_normalized, y_normalized, m=2)
        fit_noise_ent_vy = fit_noise_entropy(v_normalized, y_normalized, minc=8)
        fit_noise_skew_vy = fit_noise_skewness(v_normalized, y_normalized, minc=8)
        fit_noise_kurt_vy = fit_noise_kurtosis(v_normalized, y_normalized, minc=8)
        # 计算条件分布相似度 cond_dist_sim
        cond_dist_sim_vx = conditional_distribution_similarity(v_normalized, x_normalized, minc=8)
        cond_dist_sim_vy = conditional_distribution_similarity(v_normalized, y_normalized, minc=8)

        df.append({
            "variable": variable,
            "JH(v,x)": H_vx,
            "JH(v,y)": H_vy,
            "CH(v|x)": H_v_given_x,
            "CH(v|y)": H_v_given_y,
            "I(v,x)": I_vx,
            "I(v,y)": I_vy,
            "AMI(v,x)": AMI_vx,
            "AMI(v,y)": AMI_vy,
            "NPE(v,x)": error_prob_vx,
            "NPE(v,y)": error_prob_vy,
            # "H_baseline": H_baseline,
            # "H_entropy": H_entropy,
            # "uniform_div": uniform_div,
            "IGCI(v,x)": igci_vx,
            "IGCI(v,y)": igci_vy,
            "IGCI(x,v)": igci_xv,
            "IGCI(y,v)": igci_yv,
            "MM(v^2 * x)": moment_21_vx,
            "MM(v^2 * y)": moment_21_vy,
            "MM(v^2 * x^2)": moment_22_vx,
            "MM(v^2 * y^2)": moment_22_vy,
            "MM(v^3 * x)": moment_31_vx,
            "MM(v^3 * y)": moment_31_vy,
            "fit(v,x)": fit_val_vx,
            "fit(v,y)": fit_val_vy,
            "fit_error(v,x)": fit_err_vx,
            "fit_error(v,y)": fit_err_vy,
            "fit_noise_entropy(v,x)": fit_noise_ent_vx,
            "fit_noise_entropy(v,y)": fit_noise_ent_vy,
            "fit_noise_skewness(v,x)": fit_noise_skew_vx,
            "fit_noise_skewness(v,y)": fit_noise_skew_vy,
            "fit_noise_kurtosis(v,x)": fit_noise_kurt_vx,
            "fit_noise_kurtosis(v,y)": fit_noise_kurt_vy,
            "cond_dist_sim(v,x)": cond_dist_sim_vx,
            "cond_dist_sim(v,y)": cond_dist_sim_vy,
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

###########################################################################
################################回归系数特征################################
###########################################################################
#"""线性回归"""
def linear_regression_feature(dataset):
    warnings.filterwarnings("ignore")
    variables = dataset.columns.drop(["X", "Y"]).tolist()
    scaler = StandardScaler()

    # model1: Fit X, v, v^2, v_i*v_j, v_i*X, cos(v), sin(v) ~ Y
    model1_features = ["X"] + variables
    d1 = Squared_term(dataset[model1_features], model1_features)
    d1 = Interaction_term(d1, model1_features)
    d1 = Cos_Sin_term(d1, variables)
    model1_features = d1.columns.tolist()
    d1_scaled = scaler.fit_transform(d1)
    model1 = LinearRegression().fit(d1_scaled, dataset[["Y"]])
    model1_coefs = model1.coef_[0].tolist()
    model1_dict = {name: coef for name, coef in zip(model1_features, model1_coefs)}

    # model2: Fit v, v^2, v_i*v_j, cos(v), sin(v) ~ X
    model2_features = variables
    d2 = Squared_term(dataset[model2_features], model2_features)
    d2 = Interaction_term(d2, model2_features)
    d2 = Cos_Sin_term(d2, variables)
    model2_features = d2.columns.tolist()
    d2_scaled = scaler.fit_transform(d2)
    model2 = LinearRegression().fit(d2_scaled, dataset[["X"]])
    model2_coefs = model2.coef_[0].tolist()
    model2_dict = {name: coef for name, coef in zip(model2_features, model2_coefs)}

    df = []
    for i, variable in enumerate(variables):
        # model3: Fit other v, X, Y ~ v
        model3_features = ["X", "Y"] + dataset.columns.drop(["X", "Y", variable]).tolist()
        d3 = Squared_term(dataset[model3_features], model3_features)
        d3 = Interaction_term(d3, model3_features)
        d3 = Cos_Sin_term(d3, model3_features)
        model3_features = d3.columns.tolist()
        d3_scaled = scaler.fit_transform(d3)
        model3 = LinearRegression().fit(d3_scaled, dataset[[variable]])
        model3_coefs = model3.coef_[0].tolist()
        model3_dict = {name: coef for name, coef in zip(model3_features, model3_coefs)}

        df.append({
            "variable": variable,
            "v~Y_coefficient": model1_dict[variable],     # <--- model1
            "v_squared~Y_coefficient": model1_dict[f"{variable}_squared_term"],
            "v*X~Y_coefficient": model1_dict[f"X_{variable}"],
            "v_cos~Y_coefficient": model1_dict[f"{variable}_cos_term"],
            "v_sin~Y_coefficient": model1_dict[f"{variable}_sin_term"],
            "v~X_coefficient": model2_dict[variable],     # <--- model2
            "v_squared~X_coefficient": model2_dict[f"{variable}_squared_term"],
            "v_cos~X_coefficient": model2_dict[f"{variable}_cos_term"],
            "v_sin~X_coefficient": model2_dict[f"{variable}_sin_term"],
            "X~v_coefficient": model3_dict["X"],          # <--- model3
            "X_squared~v_coefficient": model3_dict["X_squared_term"],
            "X_cos~v_coefficient": model3_dict["X_cos_term"],
            "X_sin~v_coefficient": model3_dict["X_sin_term"],
            "Y~v_coefficient": model3_dict["Y"],
            "Y_squared~v_coefficient": model3_dict["Y_squared_term"],
            "Y_cos~v_coefficient": model3_dict["Y_cos_term"],
            "Y_sin~v_coefficient": model3_dict["Y_sin_term"],
            "X*Y~v_coefficient": model3_dict["X_Y"]
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    df["X~Y_coefficient"] = model1_dict["X"]

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

#"""分段线性回归"""
def piecewise_linear_regression_estimate(dataset, X_col, y_col):
    warnings.filterwarnings("ignore")
    X = dataset[X_col].values.reshape(-1, 1)
    y = dataset[y_col].values

    break_point = np.median(X)
    X_piecewise = np.column_stack((X, np.maximum(X - break_point, 0)))
    model = LinearRegression().fit(X_piecewise, y)
    return model.coef_

def piecewise_linear_regression_feature(dataset):
    warnings.filterwarnings("ignore")
    """
    Given a dataset, we compute piecewise linear regression features for each
    variable with X and Y, using a single breakpoint at the median.
    We also include interaction terms between v and X for predicting Y.
    """
    variables = dataset.columns.drop(["X", "Y"])

    df = []
    for variable in variables:
        # For v ~ X
        v2X_coef = piecewise_linear_regression_estimate(dataset, variable, "X")
        # For v ~ Y
        v2Y_coef = piecewise_linear_regression_estimate(dataset, variable, "Y")
        # For X ~ v
        X2v_coef = piecewise_linear_regression_estimate(dataset, "X", variable)
        # For Y ~ v
        Y2v_coef = piecewise_linear_regression_estimate(dataset, "Y", variable)


        df.append({
            "variable": variable,
            "v~X_piecewise_coef1": v2X_coef[0],
            "v~X_piecewise_coef2": v2X_coef[1],
            "v~Y_piecewise_coef1": v2Y_coef[0],
            "v~Y_piecewise_coef2": v2Y_coef[1],
            "X~v_piecewise_coef1": X2v_coef[0],
            "X~v_piecewise_coef2": X2v_coef[1],
            "Y~v_piecewise_coef1": Y2v_coef[0],
            "Y~v_piecewise_coef2": Y2v_coef[1],
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

def piecewise_quadratic_regression_feature(dataset):
    warnings.filterwarnings("ignore")
    """
    Compute piecewise quadratic regression features for each variable with X and Y,
    using a single breakpoint at the median.
    """
    variables = dataset.columns.drop(["X", "Y"]).tolist()

    df = []
    for variable in variables:
        # Prepare data
        v = dataset[variable].values.reshape(-1, 1)
        X = dataset["X"].values.reshape(-1, 1)
        Y = dataset["Y"].values.reshape(-1, 1)

        # Create piecewise quadratic features
        v_breakpoint = np.median(v)
        X_breakpoint = np.median(X)
        Y_breakpoint = np.median(Y)

        v_piecewise = np.column_stack((v, v**2, np.maximum(v - v_breakpoint, 0), np.maximum(v - v_breakpoint, 0)**2))
        X_piecewise = np.column_stack((X, X**2, np.maximum(X - X_breakpoint, 0), np.maximum(X - X_breakpoint, 0)**2))
        Y_piecewise = np.column_stack((Y, Y**2, np.maximum(Y - Y_breakpoint, 0), np.maximum(Y - Y_breakpoint, 0)**2))

        # Fit models
        model_v_X = LinearRegression().fit(v_piecewise, X)
        model_v_Y = LinearRegression().fit(v_piecewise, Y)
        model_X_v = LinearRegression().fit(X_piecewise, v)
        model_Y_v = LinearRegression().fit(Y_piecewise, v)

        # Store coefficients
        df.append({
            "variable": variable,
            "v~X_piecewise_coef1": model_v_X.coef_[0][0],
            "v~X_piecewise_coef2": model_v_X.coef_[0][1],
            "v~X_piecewise_coef3": model_v_X.coef_[0][2],
            "v~X_piecewise_coef4": model_v_X.coef_[0][3],
            "v~Y_piecewise_coef1": model_v_Y.coef_[0][0],
            "v~Y_piecewise_coef2": model_v_Y.coef_[0][1],
            "v~Y_piecewise_coef3": model_v_Y.coef_[0][2],
            "v~Y_piecewise_coef4": model_v_Y.coef_[0][3],
            "X~v_piecewise_coef1": model_X_v.coef_[0][0],
            "X~v_piecewise_coef2": model_X_v.coef_[0][1],
            "X~v_piecewise_coef3": model_X_v.coef_[0][2],
            "X~v_piecewise_coef4": model_X_v.coef_[0][3],
            "Y~v_piecewise_coef1": model_Y_v.coef_[0][0],
            "Y~v_piecewise_coef2": model_Y_v.coef_[0][1],
            "Y~v_piecewise_coef3": model_Y_v.coef_[0][2],
            "Y~v_piecewise_coef4": model_Y_v.coef_[0][3],
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    # Reorder columns
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

def piecewise_quadratic_regression_feature_4_improved(dataset):
    warnings.filterwarnings("ignore")
    """
    Compute piecewise quadratic regression features for each variable with X and Y,
    using three breakpoints to create four segments, and only using quadratic terms.
    """
    variables = dataset.columns.drop(["X", "Y"]).tolist()

    df = []
    for variable in variables:
        # Prepare data
        v = dataset[variable].values.reshape(-1, 1)
        X = dataset["X"].values.reshape(-1, 1)
        Y = dataset["Y"].values.reshape(-1, 1)

        # Create piecewise quadratic features with three breakpoints
        v_breakpoints = np.percentile(v, [25, 50, 75])
        X_breakpoints = np.percentile(X, [25, 50, 75])
        Y_breakpoints = np.percentile(Y, [25, 50, 75])

        v_piecewise = np.column_stack((
            v**2,
            np.maximum(v - v_breakpoints[0], 0)**2,
            np.maximum(v - v_breakpoints[1], 0)**2,
            np.maximum(v - v_breakpoints[2], 0)**2
        ))
        X_piecewise = np.column_stack((
            X**2,
            np.maximum(X - X_breakpoints[0], 0)**2,
            np.maximum(X - X_breakpoints[1], 0)**2,
            np.maximum(X - X_breakpoints[2], 0)**2
        ))
        Y_piecewise = np.column_stack((
            Y**2,
            np.maximum(Y - Y_breakpoints[0], 0)**2,
            np.maximum(Y - Y_breakpoints[1], 0)**2,
            np.maximum(Y - Y_breakpoints[2], 0)**2
        ))

        # Fit models
        model_v_X = LinearRegression(fit_intercept=False).fit(v_piecewise, X)
        model_v_Y = LinearRegression(fit_intercept=False).fit(v_piecewise, Y)
        model_X_v = LinearRegression(fit_intercept=False).fit(X_piecewise, v)
        model_Y_v = LinearRegression(fit_intercept=False).fit(Y_piecewise, v)

        # Store coefficients
        df.append({
            "variable": variable,
            "v~X_quadratic_coef1": model_v_X.coef_[0][0],
            "v~X_quadratic_coef2": model_v_X.coef_[0][1],
            "v~X_quadratic_coef3": model_v_X.coef_[0][2],
            "v~X_quadratic_coef4": model_v_X.coef_[0][3],
            "v~Y_quadratic_coef1": model_v_Y.coef_[0][0],
            "v~Y_quadratic_coef2": model_v_Y.coef_[0][1],
            "v~Y_quadratic_coef3": model_v_Y.coef_[0][2],
            "v~Y_quadratic_coef4": model_v_Y.coef_[0][3],
            "X~v_quadratic_coef1": model_X_v.coef_[0][0],
            "X~v_quadratic_coef2": model_X_v.coef_[0][1],
            "X~v_quadratic_coef3": model_X_v.coef_[0][2],
            "X~v_quadratic_coef4": model_X_v.coef_[0][3],
            "Y~v_quadratic_coef1": model_Y_v.coef_[0][0],
            "Y~v_quadratic_coef2": model_Y_v.coef_[0][1],
            "Y~v_quadratic_coef3": model_Y_v.coef_[0][2],
            "Y~v_quadratic_coef4": model_Y_v.coef_[0][3],
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    # Reorder columns
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

def sliding_window_linear_regression_estimate(dataset, X_col, y_col, window_size=None, step_size=None):
    warnings.filterwarnings("ignore")
    if window_size is None:
        window_size = len(dataset) // 3
    if step_size is None:
        step_size = len(dataset) // 6

    X = dataset[X_col].values
    y = dataset[y_col].values

    coefficients = []
    for start in range(0, len(dataset) - window_size + 1, step_size):
        end = start + window_size
        X_window = X[start:end].reshape(-1, 1)
        y_window = y[start:end]

        model = LinearRegression().fit(X_window, y_window)
        coefficients.append(model.coef_[0])

    return coefficients

def sliding_window_linear_regression_feature(dataset):
    warnings.filterwarnings("ignore")
    """
    Given a dataset, we compute sliding window linear regression features for each
    variable with X and Y, using a window size of 1/3 of the data and step size of 1/6.
    """
    variables = dataset.columns.drop(["X", "Y"])

    df = []
    for variable in variables:
        # For v ~ X
        v2X_coef = sliding_window_linear_regression_estimate(dataset, variable, "X")
        # For v ~ Y
        v2Y_coef = sliding_window_linear_regression_estimate(dataset, variable, "Y")
        # For X ~ v
        X2v_coef = sliding_window_linear_regression_estimate(dataset, "X", variable)
        # For Y ~ v
        Y2v_coef = sliding_window_linear_regression_estimate(dataset, "Y", variable)

        df.append({
            "variable": variable,
            "v~X_sliding_coef1": v2X_coef[0],
            "v~X_sliding_coef2": v2X_coef[1],
            "v~X_sliding_coef3": v2X_coef[2],
            "v~X_sliding_coef4": v2X_coef[3],
            "v~X_sliding_coef5": v2X_coef[4],
            "v~Y_sliding_coef1": v2Y_coef[0],
            "v~Y_sliding_coef2": v2Y_coef[1],
            "v~Y_sliding_coef3": v2Y_coef[2],
            "v~Y_sliding_coef4": v2Y_coef[3],
            "v~Y_sliding_coef5": v2Y_coef[4],
            "X~v_sliding_coef1": X2v_coef[0],
            "X~v_sliding_coef2": X2v_coef[1],
            "X~v_sliding_coef3": X2v_coef[2],
            "X~v_sliding_coef4": X2v_coef[3],
            "X~v_sliding_coef5": X2v_coef[4],
            "Y~v_sliding_coef1": Y2v_coef[0],
            "Y~v_sliding_coef2": Y2v_coef[1],
            "Y~v_sliding_coef3": Y2v_coef[2],
            "Y~v_sliding_coef4": Y2v_coef[3],
            "Y~v_sliding_coef5": Y2v_coef[4],
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

#"""岭回归"""
def ridge_regression_feature(dataset):
    warnings.filterwarnings("ignore")
    variables = dataset.columns.drop(["X", "Y"]).tolist()
    # 使用GridSearchCV来选择最佳的alpha值
    param_grid = {'alpha': np.logspace(-6, 6, 13)}

    # model1: Fit X, v ~ Y
    model1_features = ["X"] + variables
    d1 = Squared_term(dataset[model1_features], model1_features)
    # d1 = Interaction_term(d1, model1_features) # 掉分
    model1_features = d1.columns.tolist()
    scaler1 = StandardScaler()
    d1_scaled = scaler1.fit_transform(d1)
    model1 = GridSearchCV(Ridge(random_state=42), param_grid, cv=5)
    model1.fit(d1_scaled, dataset["Y"])
    model1_coefs = model1.best_estimator_.coef_.tolist()
    model1_dict = {name: coef for name, coef in zip(model1_features, model1_coefs)}

    # model2: Fit v ~ X
    model2_features = variables
    d2 = Squared_term(dataset[model2_features], model2_features)
    d2 = Interaction_term(d2, model2_features)
    # d2 = Cos_Sin_term(d2, model2_features)  # 掉分
    model2_features = d2.columns.tolist()
    scaler2 = StandardScaler()
    d2_scaled = scaler2.fit_transform(d2)
    model2 = GridSearchCV(Ridge(random_state=42), param_grid, cv=5)
    model2.fit(d2_scaled, dataset["X"])
    model2_coefs = model2.best_estimator_.coef_.tolist()
    model2_dict = {name: coef for name, coef in zip(model2_features, model2_coefs)}

    # # 获取最优的 alpha 值
    # best_alpha_model1 = model1.best_params_['alpha']   # 0.4730-0.4727
    # best_alpha_model2 = model2.best_params_['alpha']

    df = []
    for i, variable in enumerate(variables):
        df.append({
            "variable": variable,
            "v~Y_ridge_coefficient": model1_dict[variable],
            # "v_squared~Y_ridge_coefficient": model1_dict[f"{variable}_squared_term"],  # 掉分
            "v~X_ridge_coefficient": model2_dict[variable],
            # "v_squared~X_ridge_coefficient": model2_dict[f"{variable}_squared_term"],  # 掉分
            # "v_cos~X_ridge_coefficient": model2_dict[f"{variable}_cos_term"],  # 掉分
            # "v_sin~X_ridge_coefficient": model2_dict[f"{variable}_sin_term"],  # 掉分
            # "v~Y_ridge_alpha": best_alpha_model1,
            # "v~X_ridge_alpha": best_alpha_model2
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    df["X~Y_ridge_coefficient"] = model1_dict["X"]

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

###########################################################################
################################因果发现特征################################
###########################################################################
#"""因果发现算法"""
def ExactSearch_estimate(dataset, search_method='astar', use_path_extension=True,
                        use_k_cycle_heuristic=False, k=3, max_parents=None):
    warnings.filterwarnings("ignore")
    """
    使用Exact Search算法进行因果发现。

    参数:
    - dataset (pd.DataFrame): 输入的数据框，包含'X'、'Y'和其他协变量。
    - search_method (str): Exact Search方法，'astar'或'dp'。默认值为'astar'。
    - use_path_extension (bool): 是否使用路径扩展。默认值为True。
    - use_k_cycle_heuristic (bool): 是否使用k-cycle冲突启发式。仅适用于'astar'方法。默认值为False。
    - k (int): k-cycle启发式的参数。默认值为3。
    - max_parents (int or None): 节点的最大父节点数量。默认值为None。

    返回:
    - adj_df (pd.DataFrame): 因果图的邻接矩阵，格式为DataFrame。
    """

    # 1. 将数据框转换为numpy.ndarray
    data = dataset.values

    # 检验相关系数是否奇异，如果存在多重共线性，对存在多重共线性的变量添加随机扰动
    data = handle_multicollinearity(data)

    # 2. 定义X→Y的超级图
    d = dataset.shape[1]
    super_graph = np.ones((d, d), dtype=int)  # 初始化为全1，表示所有边默认允许
    columns = dataset.columns.tolist()
    x_idx = columns.index('X')
    y_idx = columns.index('Y')
    super_graph[y_idx][x_idx] = 0     # 禁止'Y'→'X'
    np.fill_diagonal(super_graph, 0)  # 禁止自环：确保对角线为0

    # 3. 运行Exact Search算法，使用指定的参数
    dag_est, search_stats = bic_exact_search(X=data, super_graph=super_graph, search_method=search_method,
        use_path_extension=use_path_extension, use_k_cycle_heuristic=use_k_cycle_heuristic,
        k=k, verbose=False, max_parents=max_parents)

    # 4. 将邻接矩阵转换为pandas DataFrame，并设置行列索引为原数据框的列名
    adj_df = pd.DataFrame(dag_est, index=dataset.columns, columns=dataset.columns)

    return adj_df

def ExactSearch_feature(dataset):
    warnings.filterwarnings("ignore")
    variables = dataset.columns.drop(["X", "Y"]).tolist()

    estimate_adj_df = ExactSearch_estimate(dataset)
    estimate_adj_df_dag = estimate_adj_df.astype(int)  # 转换为整型

    df = []
    for variable in variables:
        # 检查变量与'X'和'Y'之间的边
        v_to_X = estimate_adj_df_dag.loc[variable, 'X']
        X_to_v = estimate_adj_df_dag.loc['X', variable]
        v_to_Y = estimate_adj_df_dag.loc[variable, 'Y']
        Y_to_v = estimate_adj_df_dag.loc['Y', variable]
        X_to_Y = estimate_adj_df_dag.loc['X', 'Y']

        # # 检查是否存在中介路径
        # v_to_others_to_X = int(any(
        #     estimate_adj_df_dag.loc[variable, other] and estimate_adj_df_dag.loc[other, 'X']
        #     for other in variables if other != variable
        # ))
        # X_to_others_to_v = int(any(
        #     estimate_adj_df_dag.loc['X', other] and estimate_adj_df_dag.loc[other, variable]
        #     for other in variables if other != variable
        # ))
        # v_to_others_to_Y = int(any(
        #     estimate_adj_df_dag.loc[variable, other] and estimate_adj_df_dag.loc[other, 'Y']
        #     for other in variables if other != variable
        # ))
        # Y_to_others_to_v = int(any(
        #     estimate_adj_df_dag.loc['Y', other] and estimate_adj_df_dag.loc[other, variable]
        #     for other in variables if other != variable
        # ))

        df.append({
            "variable": variable,
            "ExactSearch(v,X)": v_to_X,
            "ExactSearch(X,v)": X_to_v,
            "ExactSearch(v,Y)": v_to_Y,
            "ExactSearch(Y,v)": Y_to_v,
            "ExactSearch(X,Y)": X_to_Y,
            # "ExactSearch(v,others,X)": v_to_others_to_X,
            # "ExactSearch(X,others,v)": X_to_others_to_v,
            # "ExactSearch(v,others,Y)": v_to_others_to_Y,
            # "ExactSearch(Y,others,v)": Y_to_others_to_v
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

def PC_estimate(dataset, alpha=0.05, indep_test='fisherz', kernel=None, stable=True,
               uc_rule=0, uc_priority=2, verbose=False, show_progress=False):
    warnings.filterwarnings("ignore")
    """
    使用PC算法进行因果发现，并应用先验知识。

    参数:
    - dataset (pd.DataFrame): 输入的数据框，包含'X'、'Y'和其他协变量。
    - alpha (float): 显著性水平，默认值为0.05。
    - indep_test (str): 独立性检验方法，默认值为'fisherz'。
    - kernel (str): 核函数类型，默认值为'linear'。
    - stable (bool): 是否运行稳定的骨架发现，默认值为True。
    - uc_rule (int): 未屏蔽碰撞点的定向规则，默认值为0。
    - uc_priority (int): 解决未屏蔽碰撞点冲突的优先规则，默认值为2。
    - verbose (bool): 是否打印详细输出，默认值为False。
    - show_progress (bool): 是否显示算法进度，默认值为False。

    返回:
    - adj_df (pd.DataFrame): 因果图的邻接矩阵，格式为pivot_table。
    """

    # 1. 将数据框转换为numpy.ndarray
    data = dataset.values

    # 检验相关系数是否奇异，如果存在多重共线性，对存在多重共线性的变量添加随机扰动
    data = handle_multicollinearity(data)

    # 2. 定义先验知识：'X' → 'Y'
    # 创建GraphNode对象
    try:
        node_X = GraphNode('X')
        node_Y = GraphNode('Y')
    except Exception as e:
        raise ValueError("确保数据框中包含名为'X'和'Y'的列。") from e

    # 初始化BackgroundKnowledge对象并添加先验知识
    bk = BackgroundKnowledge().add_required_by_node(node_X, node_Y)

    # 3. 配置核参数
    if indep_test == 'kci':
        if kernel is None:
            kernel = 'linear'
        if kernel == 'linear':
            kernel_kwargs = {
                'kernelX': 'Linear',
                'kernelY': 'Linear',
                'kernelZ': 'Linear',
                'approx': True,           # 使用伽玛近似
                'nullss': 1000,          # 原假设下模拟的样本量
            }
        elif kernel == 'polynomial':
            kernel_kwargs = {
                'kernelX': 'Polynomial',
                'kernelY': 'Polynomial',
                'kernelZ': 'Polynomial',
                'polyd': 3,               # 多项式次数设置为3
                'approx': True,           # 使用伽玛近似
                'nullss': 1000,          # 原假设下模拟的样本量
            }
        elif kernel == 'gaussian':
            kernel_kwargs = {
                'kernelX': 'Gaussian',
                'kernelY': 'Gaussian',
                'kernelZ': 'Gaussian',
                'est_width': 'empirical', # 使用经验宽度
                'approx': True,           # 使用伽玛近似
                'nullss': 1000,          # 原假设下模拟的样本量
            }
        elif kernel == 'mix':
            kernel_kwargs = {
                'kernelX': 'Polynomial',
                'kernelY': 'Polynomial',
                'kernelZ': 'Gaussian',     # Z使用高斯核
                'polyd': 3,                # 多项式次数设置为3
                'est_width': 'median',     # Z的高斯核带宽使用中位数技巧
                'approx': True,            # 使用伽玛近似
                'nullss': 1000,           # 原假设下模拟的样本量
            }
        else:
            raise ValueError(f'Unknown kernel: {kernel}')
    else:
        kernel_kwargs = {}

    # 4. 运行PC算法，传入先验知识
    cg = pc(data, alpha=alpha, indep_test=indep_test, stable=stable, uc_rule=uc_rule, uc_priority=uc_priority,
            background_knowledge=bk, verbose=verbose, show_progress=show_progress, **kernel_kwargs)

    # 5. 提取邻接矩阵
    adj_matrix = cg.G.graph
    # 6. 将邻接矩阵转换为pandas DataFrame，并设置行列索引为原数据框的列名
    adj_df = pd.DataFrame(adj_matrix, index=dataset.columns, columns=dataset.columns)

    return adj_df

def PC_feature(dataset):
    warnings.filterwarnings("ignore")
    variables = dataset.columns.drop(["X", "Y"]).tolist()

    estimate_adj_df_bidirectional = PC_estimate(dataset)  # 双向的估计因果图
    estimate_adj_df_dag = convert_bidirectional_to_dag(estimate_adj_df_bidirectional)  # 将双向图转换为有向图

    df = []
    for variable in variables:
        # 检查变量与'X'和'Y'之间的边
        v_to_X = estimate_adj_df_dag.loc[variable, 'X']
        X_to_v = estimate_adj_df_dag.loc['X', variable]
        v_to_Y = estimate_adj_df_dag.loc[variable, 'Y']
        Y_to_v = estimate_adj_df_dag.loc['Y', variable]
        X_to_Y = estimate_adj_df_dag.loc['X', 'Y']

        df.append({
            "variable": variable,
            "PC(v,X)": v_to_X,
            "PC(X,v)": X_to_v,
            "PC(v,Y)": v_to_Y,
            "PC(Y,v)": Y_to_v,
            "PC(X,Y)": X_to_Y
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

def FCI_estimate(dataset, alpha=0.05, indep_test='fisherz', kernel=None,
               depth=-1, max_path_length=-1, verbose=False, show_progress=False):
    warnings.filterwarnings("ignore")
    """
    使用FCI算法进行因果发现，并应用先验知识。

    参数:
    - dataset (pd.DataFrame): 输入的数据框，包含'X'、'Y'和其他协变量。
    - alpha (float): 显著性水平，默认值为0.05。
    - indep_test (str): 独立性检验方法，默认值为'fisherz'。
    - kernel (str): 核函数类型，默认值为'linear'。
    - verbose (bool): 是否打印详细输出，默认值为False。
    - show_progress (bool): 是否显示算法进度，默认值为False。

    返回:
    - adj_df (pd.DataFrame): 因果图的邻接矩阵，格式为pivot_table。
    """

    # 1. 将数据框转换为numpy.ndarray
    data = dataset.values

    # 检验相关系数是否奇异，如果存在多重共线性，对存在多重共线性的变量添加随机扰动
    data = handle_multicollinearity(data)

    # 2. 定义先验知识：'X' → 'Y'
    # 创建GraphNode对象
    try:
        node_X = GraphNode('X')
        node_Y = GraphNode('Y')
    except Exception as e:
        raise ValueError("确保数据框中包含名为'X'和'Y'的列。") from e

    # 初始化BackgroundKnowledge对象并添加先验知识
    bk = BackgroundKnowledge().add_required_by_node(node_X, node_Y)

    # 3. 配置核参数
    if indep_test == 'kci':
        if kernel is None:
            kernel = 'linear'
        if kernel == 'linear':
            kernel_kwargs = {
                'kernelX': 'Linear',
                'kernelY': 'Linear',
                'kernelZ': 'Linear',
                'approx': True,           # 使用伽玛近似
                'nullss': 1000,          # 原假设下模拟的样本量
            }
        elif kernel == 'polynomial':
            kernel_kwargs = {
                'kernelX': 'Polynomial',
                'kernelY': 'Polynomial',
                'kernelZ': 'Polynomial',
                'polyd': 3,               # 多项式次数设置为3
                'approx': True,           # 使用伽玛近似
                'nullss': 1000,          # 原假设下模拟的样本量
            }
        elif kernel == 'gaussian':
            kernel_kwargs = {
                'kernelX': 'Gaussian',
                'kernelY': 'Gaussian',
                'kernelZ': 'Gaussian',
                'est_width': 'empirical', # 使用经验宽度
                'approx': True,           # 使用伽玛近似
                'nullss': 1000,          # 原假设下模拟的样本量
            }
        elif kernel == 'mix':
            kernel_kwargs = {
                'kernelX': 'Polynomial',
                'kernelY': 'Polynomial',
                'kernelZ': 'Gaussian',     # Z使用高斯核
                'polyd': 3,                # 多项式次数设置为3
                'est_width': 'median',     # Z的高斯核带宽使用中位数技巧
                'approx': True,            # 使用伽玛近似
                'nullss': 1000,           # 原假设下模拟的样本量
            }
        else:
            raise ValueError(f'Unknown kernel: {kernel}')
    else:
        kernel_kwargs = {}

    # 4. 运行FCI算法，传入先验知识
    try:
        g, edges = fci(data,
                alpha=alpha,
                independence_test_method=indep_test,
                depth=depth,
                max_path_length=max_path_length,
                background_knowledge=bk,
                verbose=verbose,
                show_progress=show_progress,
                **kernel_kwargs
        )

        # 5. 提取邻接矩阵
        adj_matrix = g.graph
    except Exception as e:
        adj_matrix = np.zeros((data.shape[1], data.shape[1]))

    # 6. 将邻接矩阵转换为pandas DataFrame，并设置行列索引为原数据框的列名
    adj_df = pd.DataFrame(adj_matrix, index=dataset.columns, columns=dataset.columns)

    return adj_df

def FCI_feature(dataset):
    warnings.filterwarnings("ignore")
    variables = dataset.columns.drop(["X", "Y"]).tolist()

    estimate_adj_df_bidirectional = FCI_estimate(dataset)  # PAG
    estimate_adj_df_dag = estimate_adj_df_bidirectional.astype('int')

    df = []
    for variable in variables:
        # 检查变量与'X'和'Y'之间的边
        v_to_X = estimate_adj_df_dag.loc[variable, 'X']
        X_to_v = estimate_adj_df_dag.loc['X', variable]
        v_to_Y = estimate_adj_df_dag.loc[variable, 'Y']
        Y_to_v = estimate_adj_df_dag.loc['Y', variable]
        X_to_Y = estimate_adj_df_dag.loc['X', 'Y']

        df.append({
            "variable": variable,
            "FCI(v,X)": v_to_X,
            "FCI(X,v)": X_to_v,
            "FCI(v,Y)": v_to_Y,
            "FCI(Y,v)": Y_to_v,
            "FCI(X,Y)": X_to_Y
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    # # One-hot encode the edge types (-1, 0, 1, 2)
    # one_hot_columns = ["FCI(v,X)", "FCI(X,v)", "FCI(v,Y)", "FCI(Y,v)"]

    # for col in one_hot_columns:
    #     one_hot = pd.get_dummies(df[col], prefix=col)
    #     df = pd.concat([df, one_hot], axis=1)

    # # Remove original edge type columns after one-hot encoding
    # df = df.drop(columns=one_hot_columns)

    # Reorder columns:
    df = df[["dataset", "variable"] + [col for col in df.columns if col not in ["dataset", "variable"]]]

    return df

def GRaSP_estimate(dataset, score_func='local_score_BIC', maxP=None):
    warnings.filterwarnings("ignore")
    # 1. 将数据框转换为numpy.ndarray
    data = dataset.values
    # 检验相关系数是否奇异，如果存在多重共线性，对存在多重共线性的变量添加随机扰动
    data = handle_multicollinearity(data)

    parameters = {
        'kfold': 2,         # 2 折交叉验证
        'lambda': 0.01      # 正则化参数
    }

    # 2. 运行PC算法，传入先验知识
    G = grasp(
        data,
        score_func=score_func,
        maxP=maxP,
        parameters=parameters
    )

    # 3. 获取邻接矩阵
    adj_matrix = G.graph

    # 4. 将邻接矩阵转换为pandas DataFrame，并设置行列索引为原数据框的列名
    adj_df = pd.DataFrame(adj_matrix, index=dataset.columns, columns=dataset.columns)

    return adj_df

def GRaSP_feature(dataset):
    warnings.filterwarnings("ignore")
    variables = dataset.columns.drop(["X", "Y"]).tolist()

    estimate_adj_df = GRaSP_estimate(dataset)
    estimate_adj_df_dag = estimate_adj_df.astype(int)  # 将邻接矩阵转换为整数类型

    df = []
    for variable in variables:
        # 检查变量与'X'和'Y'之间的边
        v_to_X = estimate_adj_df_dag.loc[variable, 'X']
        X_to_v = estimate_adj_df_dag.loc['X', variable]
        v_to_Y = estimate_adj_df_dag.loc[variable, 'Y']
        Y_to_v = estimate_adj_df_dag.loc['Y', variable]
        X_to_Y = estimate_adj_df_dag.loc['X', 'Y']

        df.append({
            "variable": variable,
            "GRaSP(v,X)": v_to_X,
            "GRaSP(X,v)": X_to_v,
            "GRaSP(v,Y)": v_to_Y,
            "GRaSP(Y,v)": Y_to_v,
            "GRaSP(X,Y)": X_to_Y,
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

#"""DML"""
def DML_estimate(T_, Y_, X_, data):
    warnings.filterwarnings("ignore")
    # 设置处理变量、结果变量和控制变量
    T = data[T_].values
    Y = data[Y_].values
    X = data[X_].values

    # 定义 LassoCV 作为第一阶段模型
    model_t = LassoCV(random_state=42, n_jobs=None)
    model_y = LassoCV(random_state=42, n_jobs=None)

    # # 定义 "forest" 作为第二阶段模型
    # model_t = "forest"
    # model_y = "forest"

    # 初始化 CausalForestDML 使用自定义的估计器
    model = CausalForestDML(model_t=model_t, model_y=model_y,
        cv=4, n_estimators=36, n_jobs=None, random_state=42, inference=True)

    # # 初始化 LinearDML 使用自定义的估计器
    # model = LinearDML(model_t=model_t, model_y=model_y,
    #     cv=4, n_jobs=None, random_state=42, inference=True)

    model.fit(Y, T, X=X)

    # 平均边际处理效应的推断结果
    amte_inference = model.ate_inference(X=X)

    return amte_inference

def DML_feature(dataset):
    warnings.filterwarnings("ignore")
    variables = dataset.columns.drop(["X", "Y"])

    df = []
    for variable in variables:
        # 判断v-X的因果效应，设置variables中的其他v和Y为控制变量
        amte_inference1 = DML_estimate(variable, "X", ["Y"] + list(variables.drop(variable)), dataset)
        # 判断v-Y的因果效应，设置variables中的其他v和X为控制变量
        amte_inference2 = DML_estimate(variable, "Y", ["X"] + list(variables.drop(variable)), dataset)
        # 判断X-v的因果效应，设置variables中的其他v和Y为控制变量
        amte_inference3 = DML_estimate("X", variable, ["Y"] + list(variables.drop(variable)), dataset)
        # 判断Y-v的因果效应，设置variables中的其他v和X为控制变量
        amte_inference4 = DML_estimate("Y", variable, ["X"] + list(variables.drop(variable)), dataset)

        df.append({
            "variable": variable,
            "v~X_DML_AMTE": amte_inference1.mean_point,
            "v~X_DML_AMTE_zstat": amte_inference1.zstat(),
            "v~X_DML_AMTE_pvalue": amte_inference1.pvalue(),
            "v~X_DML_std_point": amte_inference1.std_point,
            "v~X_DML_stderr_point": amte_inference1.stderr_point,

            "v~Y_DML_AMTE": amte_inference2.mean_point,
            "v~Y_DML_AMTE_zstat": amte_inference2.zstat(),
            "v~Y_DML_AMTE_pvalue": amte_inference2.pvalue(),
            "v~Y_DML_std_point": amte_inference2.std_point,
            "v~Y_DML_stderr_point": amte_inference2.stderr_point,

            "X~v_DML_AMTE": amte_inference3.mean_point,
            "X~v_DML_AMTE_zstat": amte_inference3.zstat(),
            "X~v_DML_AMTE_pvalue": amte_inference3.pvalue(),
            "X~v_DML_std_point": amte_inference3.std_point,
            "X~v_DML_stderr_point": amte_inference3.stderr_point,

            "Y~v_DML_AMTE": amte_inference4.mean_point,
            "Y~v_DML_AMTE_zstat": amte_inference4.zstat(),
            "Y~v_DML_AMTE_pvalue": amte_inference4.pvalue(),
            "Y~v_DML_std_point": amte_inference4.std_point,
            "Y~v_DML_stderr_point": amte_inference4.stderr_point,
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

###########################################################################
##################################结构方程特征##############################
###########################################################################
def sem_features(dataset):
    warnings.filterwarnings("ignore", message="Sample covariance matrix is not PD")
    warnings.filterwarnings("ignore")
    """
    针对每个变量 v，构建八种 SEM 模型，计算模型拟合指标，生成特征。

    参数：
    - dataset: 包含 X, Y, v1, v2, ... 的 pandas DataFrame

    返回：
    - 包含 SEM 特征的 pandas DataFrame
    """
    variables = dataset.columns.drop(['X', 'Y'])
    df = []

    # 定义八种关系的 SEM 模型描述
    model_templates = {
        "Confounder": """
            X ~ a1*v
            Y ~ b1*X + b2*v
            X ~~ X
            v ~~ v
            Y ~~ Y
        """,
        "Collider": """
            v ~ a1*X + a2*Y
            Y ~ b1*X
            X ~~ X
            v ~~ v
            Y ~~ Y
        """,
        "Mediator": """
            v ~ a1*X
            Y ~ b1*v
            X ~~ X
            v ~~ v
            Y ~~ Y
        """,
        "Cause of X": """
            X ~ a1*v
            Y ~ b1*X
            X ~~ X
            v ~~ v
            Y ~~ Y
        """,
        "Cause of Y": """
            Y ~ a1*X + a2*v
            X ~~ X
            v ~~ v
            Y ~~ Y
        """,
        "Consequence of X": """
            v ~ a1*X
            Y ~ b1*X
            X ~~ X
            v ~~ v
            Y ~~ Y
        """,
        "Consequence of Y": """
            v ~ a1*Y
            Y ~ b1*X
            X ~~ X
            v ~~ v
            Y ~~ Y
        """,
        # "Independent": """
        #     Y ~ a1*X
        #     X ~~ X
        #     v ~~ v
        #     Y ~~ Y
        # """
    }

    # 对于每个变量 v，构建并拟合八种模型
    for variable in variables:
        # 存储每种模型的拟合指标
        fit_indices_list = []
        for label, model_desc_template in model_templates.items():
            # 替换模型描述中的变量名
            model_desc = model_desc_template.replace('v', variable)
            try:
                # 创建并拟合模型
                model = Model(model_desc)
                # 使用全局优化器，以提高模型拟合的稳定性, 关闭所有警告
                model.fit(dataset, solver='SLSQP')
                # 获取模型拟合指标
                # fit_indices = inspect(model)
                stats = semopy.calc_stats(model)
                # 提取常用的拟合指标

                fit_metrics = {
                    'AIC': stats['AIC'].loc['Value'],
                    'BIC': stats['BIC'].loc['Value'],
                    'CFI': stats['CFI'].loc['Value'],
                    'TLI': stats['TLI'].loc['Value'],
                    'RMSEA': stats['RMSEA'].loc['Value'],
                    # 'chi2': stats['chi2'].loc['Value'],
                    # 'chi2_baseline': stats['chi2 Baseline'].loc['Value'],
                    # 'NFI': stats['NFI'].loc['Value'],
                    # 'LogLik': stats['LogLik'].loc['Value'],
                }
            except Exception as e:
                # 如果模型无法收敛，设置拟合指标为缺失值
                # print(e)
                # print(model_desc_template)
                fit_metrics = {
                    'AIC': None,
                    'BIC': None,
                    'CFI': None,
                    'TLI': None,
                    'RMSEA': None,
                    # 'chi2': None,
                    # 'chi2_baseline': None,
                    # 'NFI': None,
                    # "logLik": None,
                }
            fit_metrics['Model'] = label
            fit_indices_list.append(fit_metrics)
        # 将拟合指标列表转换为 DataFrame
        fit_df = pd.DataFrame(fit_indices_list)
        fit_df['variable'] = variable
        df.append(fit_df)

    # 合并所有变量的结果
    result_df = pd.concat(df, ignore_index=True)

    # 将模型名称和变量名称组合，展开为列
    pivot_df = result_df.pivot(index='variable', columns='Model')
    pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
    pivot_df.reset_index(inplace=True)
    pivot_df['dataset'] = dataset.name

    # 返回结果 DataFrame
    return pivot_df

def Cloud_feature(dataset):
    variables = dataset.columns.drop(["X", "Y"])
    X = discretize_sequence(dataset['X'].values, ffactor=5)
    Y = discretize_sequence(dataset['Y'].values, ffactor=5)

    df = []
    for variable in variables:
        v = discretize_sequence(dataset[variable].values, ffactor=5)

        result_vX = Cloud(
            X=v,
            Y=X,
            n_candidates=2, # select a set of model candidates
            is_print=False, # print out inferred causal direction
            X_ndistinct_vals=11,
            Y_ndistinct_vals=11,
        )
        pred_vX, conf_vX = Cloud_output(result_vX)
        result_vY = Cloud(
            X=v,
            Y=Y,
            n_candidates=2, # select a set of model candidates
            is_print=False, # print out inferred causal direction
            X_ndistinct_vals=11,
            Y_ndistinct_vals=11,
        )
        pred_vY, conf_vY = Cloud_output(result_vY)
        df.append({
            "variable": variable,
            "Cloud_to(v,X)": result_vX[0][0],
            "Cloud_gets(v,X)": result_vX[1][0],
            # "Cloud_indep(v,X)": result_vX[2][0],
            # "Cloud_confounder(v,X)": result_vX[0][0],
            "Cloud_pred(v,X)": pred_vX,
            "Cloud_conf(v,X)": conf_vX,
            "Cloud_to(v,Y)": result_vY[0][0],
            "Cloud_gets(v,Y)": result_vY[1][0],
            # "Cloud_indep(v,Y)": result_vY[2][0],
            # "Cloud_confounder(v,Y)": result_vY[0][0],
            "Cloud_pred(v,Y)": pred_vY,
            "Cloud_conf(v,Y)": conf_vY,
        })

    result_XY = Cloud(
        X=X,
        Y=Y,
        n_candidates=4, # select a set of model candidates
        is_print=False, # print out inferred causal direction
        X_ndistinct_vals=21,
        Y_ndistinct_vals=21,
    )
    pred_XY, conf_XY = Cloud_output(result_XY)

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    df["Cloud_to(X,Y)"] = result_XY[0][0]
    df["Cloud_gets(X,Y)"] = result_XY[1][0]
    df["Cloud_indep(X,Y)"] = result_XY[2][0]
    df["Cloud_confounder(X,Y)"] = result_XY[3][0]
    df["Cloud_pred(X,Y)"] = pred_XY
    df["Cloud_conf(X,Y)"] = conf_XY

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df


def label(adjacency_matrix):
    """
    Given a graph as adjacency_matrix, create the class labels of each variable.
    """

    adjacency_graph, adjacency_label = create_graph_label()
    labels = get_labels(adjacency_matrix, adjacency_label)
    variables = adjacency_matrix.columns.drop(["X", "Y"])

    df = pd.DataFrame({
        "variable": variables,
        "label": [labels[variable] for variable in variables],
    })
    df["dataset"] = adjacency_matrix.name

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df


def create_some_columns(names_datasets, function):
    """
    Apply an embedding function to a list of datasets.
    """

    df = []
    for name, dataset in tqdm(names_datasets.items()):
        dataset = names_datasets[name]
        dataset.name = name

        try:
            df_dataset = function(dataset)
        except ValueError as e:
            print(name, e)
            raise NotImplementedError

        df_dataset["dataset"] = name
        df.append(df_dataset)

    df = pd.concat(df, axis="index").reset_index(drop=True)
    return df


def create_some_columns_parallel(names_datasets, function, n_jobs=-1):
    """
    Apply an embedding function to a list of datasets.

    Parallel version.
    """

    def f(name, dataset, function):
        dataset.name = name
        df_dataset = function(dataset)
        df_dataset["dataset"] = name
        return df_dataset

    df = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(f)(name, dataset, function)
        for name, dataset in tqdm(names_datasets.items())
    )

    df = pd.concat(df, axis="index").reset_index(drop=True)
    return df


def create_all_columns(functions_names_datasets, n_jobs=-1, create_dimension_feature = False):
    """
    given a dictionary of {function1:names, function2:names,...} apply
    the desired functions to the list of datasets and merge all of them
    in a single X_y_group dataframe.
    """

    columns = []
    if create_dimension_feature:
        dimension_feature = create_some_columns(functions_names_datasets[list(functions_names_datasets.keys())[0]], add_dimension_feature)
        columns.append(dimension_feature)

    for function, names_datasets in functions_names_datasets.items():
        print(f"set: {function.__name__}")

        if n_jobs != 1:
            feature_set = create_some_columns_parallel(names_datasets, function, n_jobs=n_jobs)
        else:
            feature_set = create_some_columns(names_datasets, function)

        columns.append(feature_set)

    # Merge all feature sets into a single dataframe:
    columns = functools.reduce(
        lambda left, right: pd.merge(left, right, on=["dataset", "variable"]),
        columns,
    )

    return columns


import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

import random
import typing
from collections import defaultdict


#"""训练的工具函数"""
def remove_columns_with_keywords(df, keywords):
    columns_to_drop = [col for col in df.columns if any(keyword in col for keyword in keywords)]
    return df.drop(columns=columns_to_drop)

def get_columns_with_keywords(df, keywords):
    columns_to_get = [col for col in df.columns if any(keyword in col for keyword in keywords)]
    return columns_to_get

def recovery_dataset_id(X_y_group_train):
    # 假设'dataset'列是需要转换的列
    X_y_group_train['dataset'] = X_y_group_train['dataset'].apply(lambda x: f'{int(x):05}')
    return X_y_group_train

def clean_feature_names(X):
    def clean_name(name):
        # 将空格替换为下划线
        name = name.replace(' ', '_')
        # 将逗号替换为下划线
        name = name.replace(',', '_')
        # 移除或替换其他特殊字符
        name = re.sub(r'[^\w\-]', '_', name)
        # 确保名称不以数字开头
        if name and name[0].isdigit():
            name = 'f_' + name
        # 移除连续的下划线
        name = re.sub(r'_+', '_', name)
        # 移除开头和结尾的下划线
        name = name.strip('_')
        return name

    X.columns = [clean_name(col) for col in X.columns]
    return X

def process_categorical_features(df, max_unique=10):
    """
    检测和处理数据框中的类别变量。

    参数：
    - df (pd.DataFrame): 输入的数据框。
    - max_unique (int): 判定为类别变量的最大唯一值数量。

    返回：
    - cat_idxs (list of int): 类别特征的索引。
    - cat_dims (list of int): 每个类别特征的模态数。
    - df (pd.DataFrame): 经过编码后的数据框。
    """
    cat_cols = [col for col in df.columns if df[col].nunique() <= max_unique]
    cat_dims = []
    cat_idxs = []

    for col in cat_cols:
        if col == 'augment':
            continue
        print(f"     ->->->处理类别特征: {col}，唯一值数量: {df[col].nunique()}")
        # 使用 LabelEncoder
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).fillna('NaN'))
        cat_dims.append(len(le.classes_))
        cat_idxs.append(df.columns.get_loc(col))

    return cat_idxs, cat_dims, df

def filter_features(X_test, model):
    """
    过滤测试数据集，只保留模型训练时使用的特征。

    参数:
    X_test : pandas.DataFrame 或 numpy.array
        需要进行预测的测试数据
    model : 已训练的模型
        包含 feature_names_in_ 属性的模型（如sklearn的大多数模型）

    返回:
    pandas.DataFrame 或 numpy.array
        只包含模型训练时使用的特征的测试数据
    """
    # if hasattr(model, 'feature_name_'):
    #     # 获取模型训练时使用的特征名称
    #     model_features = model.feature_name_
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_

        if isinstance(X_test, pd.DataFrame):
            print("测试数据类型为 DataFrame")
            # 对于DataFrame，我们可以直接使用列名
            common_features = list(set(X_test.columns) & set(model_features))
            missing_features = set(model_features) - set(X_test.columns)
            if missing_features:
                print(f"警告: 测试数据缺少 {len(missing_features)} 个训练时使用的特征: {missing_features}")
            extra_features = set(X_test.columns) - set(model_features)
            if extra_features:
                print(f"警告: 移除了 {len(extra_features)} 个在训练时未使用的特征: {extra_features}")
            return X_test[common_features]
        elif isinstance(X_test, np.ndarray):
            print("测试数据类型为 numpy array")
            # 对于numpy数组，我们假设特征的顺序与训练时相同
            if X_test.shape[1] > len(model_features):
                print(f"警告: 测试数据包含额外的特征。只使用前 {len(model_features)} 个特征。")
                return X_test[:, :len(model_features)]
            elif X_test.shape[1] < len(model_features):
                missing_count = len(model_features) - X_test.shape[1]
                print(f"错误: 测试数据的特征数 ({X_test.shape[1]}) 少于模型训练时的特征数 ({len(model_features)})")
                print(f"缺少的特征数量: {missing_count}")
                raise ValueError("特征数量不匹配")
            return X_test
    else:
        print("警告: 模型没有 feature_name_ 属性。无法验证特征。")
        return X_test

def align_features(X, model):
    """
    调整输入特征的顺序，使其与模型训练时的特征顺序一致。

    参数:
    X : pandas.DataFrame 或 numpy.ndarray
        需要调整顺序的输入特征
    model : 已训练的模型
        包含 feature_names_in_ 属性的模型（如sklearn的大多数模型）

    返回:
    pandas.DataFrame 或 numpy.ndarray
        特征顺序调整后的数据
    """
    # if not hasattr(model, 'feature_name_'):
    #     print("警告: 模型没有 feature_name_ 属性。无法调整特征顺序。")
    #     return X

    # model_features = model.feature_name_

    if not hasattr(model, 'feature_names_in_'):
        print("警告: 模型没有 feature_names_in_ 属性。无法调整特征顺序。")
        return X

    model_features = model.feature_names_in_

    if isinstance(X, pd.DataFrame):
        print("测试数据类型为 DataFrame")
        # 对于DataFrame，我们可以直接使用列名重新排序
        if set(X.columns) != set(model_features):
            raise ValueError("输入特征与模型特征不完全匹配。")
        return X.reindex(columns=model_features)

    elif isinstance(X, np.ndarray):
        print("测试数据类型为 numpy array")
        if X.shape[1] != len(model_features):
            raise ValueError("输入特征数量与模型特征数量不匹配。")

        # 对于numpy数组，我们需要创建一个映射来重新排序
        current_features = [f"feature_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=current_features)
        feature_mapping = dict(zip(current_features, X.columns if isinstance(X, pd.DataFrame) else model_features))
        df = df.rename(columns=feature_mapping)
        return df.reindex(columns=model_features).values

    else:
        raise ValueError("输入X必须是pandas.DataFrame或numpy.ndarray。")

def filter_train_data(
    X_train: typing.Dict[str, pd.DataFrame],
    y_train: typing.Dict[str, pd.DataFrame],
    n_samples: int = 50
) -> typing.Tuple[typing.Dict[str, pd.DataFrame], typing.Dict[str, pd.DataFrame]]:
    assert n_samples >= 8, "n_samples must be at least 8 to ensure representation of all dimensions"

    all_keys = list(X_train.keys())
    dimension_dict = defaultdict(list)

    # Group keys by their dimension (number of columns)
    for key in all_keys:
        dim = X_train[key].shape[1]
        dimension_dict[dim].append(key)

    selected_keys = []

    # Ensure at least one sample from each dimension
    for dim in range(3, 11):  # dimensions from 3 to 10
        if dimension_dict[dim]:
            selected_keys.append(random.choice(dimension_dict[dim]))
            dimension_dict[dim].remove(selected_keys[-1])

    # Fill the rest randomly
    remaining_keys = [key for keys in dimension_dict.values() for key in keys]
    additional_keys = random.sample(remaining_keys, min(n_samples - len(selected_keys), len(remaining_keys)))
    selected_keys.extend(additional_keys)

    # Shuffle the selected keys to randomize the order
    random.shuffle(selected_keys)

    # Filter X_train and y_train
    filtered_X_train = {k: X_train[k] for k in selected_keys}
    filtered_y_train = {k: y_train[k] for k in selected_keys}

    return filtered_X_train, filtered_y_train

def filter_test_data(
    X_test: typing.Dict[str, pd.DataFrame],
    n_samples: int = 50
) -> typing.Dict[str, pd.DataFrame]:
    assert n_samples >= 8, "n_samples must be at least 8 to ensure representation of all dimensions"

    all_keys = list(X_test.keys())
    dimension_dict = defaultdict(list)

    # Group keys by their dimension (number of columns)
    for key in all_keys:
        dim = X_test[key].shape[1]
        dimension_dict[dim].append(key)

    selected_keys = []

    # Ensure at least one sample from each dimension
    for dim in range(3, 11):  # dimensions from 3 to 10
        if dimension_dict[dim]:
            selected_keys.append(random.choice(dimension_dict[dim]))
            dimension_dict[dim].remove(selected_keys[-1])

    # Fill the rest randomly
    remaining_keys = [key for keys in dimension_dict.values() for key in keys]
    additional_keys = random.sample(remaining_keys, min(n_samples - len(selected_keys), len(remaining_keys)))
    selected_keys.extend(additional_keys)

    # Shuffle the selected keys to randomize the order
    random.shuffle(selected_keys)

    # Filter X_test
    filtered_X_test = {k: X_test[k] for k in selected_keys}

    return filtered_X_test


def check_online_data(X_train, y_train, local_datasets_describe):
    online_add_X_train = {}
    online_add_y_train = {}

    # 取云端数据集的Key
    dataset_keys = list(X_train.keys())
    # Initialize
    missing_keys_count = 0       # 云端数据集的Key不在本地数据集的Key中

    for key in tqdm(dataset_keys, desc="Checking", unit="dataset"):
        dataset = X_train[key]
        DAG = y_train[key]
        # 云端数据集的Key不在本地数据集的Key中
        if key not in local_datasets_describe:
            print(f"     ->Key not found in local dataset: {key}")
            missing_keys_count += 1
            online_add_X_train[key] = dataset
            online_add_y_train[key] = DAG

    # 打印统计信息
    print(f"     ->Statistics:")
    print(f"     ->Number of keys not found in the local dataset: {missing_keys_count}")
    return online_add_X_train, online_add_y_train

def create_submission(X_y_pred_test):
    """
    From the predicted test set, for each dataset, take predicted
    classes of all variables, create the adjacency matrix, then create
    the submission in the requested format.
    """

    submission = {}
    for name, prediction in tqdm(X_y_pred_test.groupby("dataset"), delay=10):
        variables_labels = prediction[["variable", "label_predicted"]].set_index("variable")
        variables = variables_labels.index.tolist()
        variables_all = ["X", "Y"] + variables

        adjacency_matrix = pd.DataFrame(index=variables_all, columns=variables_all)
        adjacency_matrix.index.name = "parent"
        adjacency_matrix[:] = 0
        adjacency_matrix.loc["X", "Y"] = 1

        for v in variables:
            l = variables_labels.loc[v].item()
            if l == "Cause of X":
                adjacency_matrix.loc[v, "X"] = 1
            elif l == "Cause of Y":
                adjacency_matrix.loc[v, "Y"] = 1
            elif l == "Consequence of X":
                adjacency_matrix.loc["X", v] = 1
            elif l == "Consequence of Y":
                adjacency_matrix.loc["Y", v] = 1
            elif l == "Confounder":
                adjacency_matrix.loc[v, "X"] = 1
                adjacency_matrix.loc[v, "Y"] = 1
            elif l == "Collider":
                adjacency_matrix.loc["X", v] = 1
                adjacency_matrix.loc["Y", v] = 1
            elif l == "Mediator":
                adjacency_matrix.loc["X", v] = 1
                adjacency_matrix.loc[v, "Y"] = 1
            elif l == "Confounder":
                pass

        for i in variables_all:
            for j in variables_all:
                submission[f'{name}_{i}_{j}'] = int(adjacency_matrix.loc[i, j])

    return submission

class PretrainedVotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, voting='soft', weights=None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights

    def fit(self, X, y=None):
        # 已经预训练，无需再训练
        return self

    def predict(self, X):
        if self.voting == 'soft':
            # 对于概率投票
            probas = np.asarray([clf.predict_proba(X) for clf in self.estimators])
            avg_proba = np.average(probas, axis=0, weights=self.weights)
            return np.argmax(avg_proba, axis=1)
        else:
            # 对于硬投票
            predictions = np.asarray([clf.predict(X) for clf in self.estimators]).T
            maj_vote = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)), axis=1, arr=predictions
            )
            return maj_vote


def train(
    X_train: typing.Dict[str, pd.DataFrame],
    y_train: typing.Dict[str, pd.DataFrame],
    # number_of_features: int,
    model_directory_path: str,
    # id_column_name: str,
    # prediction_column_name: str,
    # has_gpu: bool,
) -> None:
    # return
    # 设置版本（模型、数据和结果保存名称）
    version = 'traindata_25augment_final'
    X_y_group_train_pathname = os.path.join(model_directory_path, f"X_y_group_train_{version}.pkl")

    try:
        print(f"##### Loading {X_y_group_train_pathname}")
        # 读取本地数据
        X_y_group_train = pd.read_pickle(X_y_group_train_pathname)
        print('     ->已加载本地数据 X_y_group_train')
        if 'Unnamed_0' in X_y_group_train.columns:
            print("     ->'Unnamed_0' column found. It will be removed.")
            X_y_group_train.drop(columns=['Unnamed_0'], inplace=True)
        shape = X_y_group_train.shape
        columns = X_y_group_train.columns.tolist()
        print(f'     ->X_y_group_train形状: {shape} \n X_y_group_train列名: {columns}')


        print(f"##### Checking More Data in Online X_train")
        # 读取本地数据描述字典
        local_datasets_describe_path = os.path.join(model_directory_path, "local_datasets_describe_drop50.json")
        local_datasets_describe = json.load(open(local_datasets_describe_path, 'r', encoding='utf-8'))
        online_add_X_train, online_add_y_train = check_online_data(X_train, y_train, local_datasets_describe)


        print("##### Creating online_add_X_y_group_train")
        # 补充线上数据
        # # 小批量测试
        # online_add_X_train, online_add_y_train = filter_train_data(online_add_X_train, online_add_y_train, n_samples=20)
        # # 打印X_train中每个数据集的形状
        # for dataset_name, dataset in online_add_X_train.items():
        #     print(f"{dataset_name} 的形状: {dataset.shape}")

        # 数据增强
        online_add_X_train, online_add_y_train, failure_count = augment_data(online_add_X_train, online_add_y_train, augment_factor=7.0)
        print(f'     ->线上数据增强失败次数: {failure_count}')
        print(f'     ->线上数据增强后样本量: {len(online_add_X_train)}, {len(online_add_y_train)}')
        # # 打印X_train中每个数据集的形状
        # for dataset_name, dataset in online_add_X_train.items():
        #     print(f"{dataset_name} 的形状: {dataset.shape}")

        online_add_names_datasets_train = online_add_X_train
        online_add_names_graphs_train = online_add_y_train
        online_add_X_y_group_train = create_all_columns(
            {
                label: online_add_names_graphs_train,

                pearson_correlation: online_add_names_datasets_train,
                rolling_abs_pearson_correlation: online_add_names_datasets_train,
                spearman_correlation: online_add_names_datasets_train,
                kendall_correlation: online_add_names_datasets_train,
                mutual_information: online_add_names_datasets_train,
                conditional_mutual_information: online_add_names_datasets_train,
                distance_correlation: online_add_names_datasets_train,
                partial_correlation: online_add_names_datasets_train,
                copula_entropy: online_add_names_datasets_train,
                PPS_feature: online_add_names_datasets_train,

                linear_regression_feature: online_add_names_datasets_train,
                ridge_regression_feature: online_add_names_datasets_train,

                ExactSearch_feature: online_add_names_datasets_train,
                PC_feature: online_add_names_datasets_train,
                FCI_feature: online_add_names_datasets_train,
                GRaSP_feature: online_add_names_datasets_train,
                DML_feature: online_add_names_datasets_train,

                piecewise_linear_regression_feature: online_add_names_datasets_train,
                piecewise_quadratic_regression_feature: online_add_names_datasets_train,
                piecewise_quadratic_regression_feature_4_improved: online_add_names_datasets_train,
                sliding_window_linear_regression_feature: online_add_names_datasets_train,

                sem_features: online_add_names_datasets_train,
                grid_feature: online_add_names_datasets_train,
                discrete_statistic_features: online_add_names_datasets_train,
                Cloud_feature: online_add_names_datasets_train,
            },
            n_jobs=-1,
            create_dimension_feature=True,
        )
        online_add_X_y_group_train['MI(v,X)^2'] = online_add_X_y_group_train['MI(v,X)'] ** 2
        online_add_X_y_group_train['MI(v,Y)^2'] = online_add_X_y_group_train['MI(v,Y)'] ** 2
        online_add_X_y_group_train['MI(X,Y)^2'] = online_add_X_y_group_train['MI(X,Y)'] ** 2
        online_add_X_y_group_train['max(MI(v, others))^2'] = online_add_X_y_group_train['max(MI(v, others))'] ** 2
        online_add_X_y_group_train['min(MI(v, others))^2'] = online_add_X_y_group_train['min(MI(v, others))'] ** 2

        online_add_X_y_group_train = online_add_X_y_group_train.loc[:,~online_add_X_y_group_train.columns.duplicated()]
        print('     ->已生成线上补充数据 online_add_X_y_group_train')


        print("##### Checking different columns between online_add_X_y_group_train and X_y_group_train")
        # 检查online_add_X_y_group_train的列名和X_y_group_train的列名是否相同
        if online_add_X_y_group_train.columns.equals(X_y_group_train.columns):
            print('     ->online_add_X_y_group_train的列名和X_y_group_train的列名相同')
            print(f'     ->拼接前样本量: {X_y_group_train.shape}, {online_add_X_y_group_train.shape}')
            # 拼接
            X_y_group_train = pd.concat([X_y_group_train, online_add_X_y_group_train], ignore_index=True)
            print(f'     ->拼接后样本量: {X_y_group_train.shape}')
            # # 去重
            # X_y_group_train = X_y_group_train.drop_duplicates()
            # print(f'     ->去重后样本量: {X_y_group_train.shape}')
            shape = X_y_group_train.shape
            columns = X_y_group_train.columns.tolist()
            print('     ->线上训练数据 X_y_group_train 已生成')
            print(f'     ->X_y_group_train形状: {shape} \n X_y_group_train列名: {columns}')
        else:
            print('     ->online_add_X_y_group_train的列名和X_y_group_train的列名不同')
            print('!!!!!!!无法拼接本地和线上数据集，跳过训练，使用已经训练好的模型进行推理!!!!!!!')
            return

    except FileNotFoundError:
        print("##### Creating X_y_group_train")

        # # 小批量测试
        # X_train, y_train = filter_train_data(X_train, y_train, n_samples=20)

        # 数据增强
        X_train, y_train, failure_count = augment_data(X_train, y_train, augment_factor=3.0)
        print(f'     ->数据增强失败次数: {failure_count}')
        print(f'     ->数据增强后样本量: {len(X_train)}, {len(y_train)}')

        names_datasets_train = X_train
        names_graphs_train = y_train
        X_y_group_train = create_all_columns(
            {
                label: names_graphs_train,

                pearson_correlation: names_datasets_train,
                rolling_abs_pearson_correlation: names_datasets_train,
                spearman_correlation: names_datasets_train,
                kendall_correlation: names_datasets_train,
                mutual_information: names_datasets_train,
                conditional_mutual_information: names_datasets_train,
                distance_correlation: names_datasets_train,
                partial_correlation: names_datasets_train,
                copula_entropy: names_datasets_train,
                PPS_feature: names_datasets_train,

                linear_regression_feature: names_datasets_train,
                ridge_regression_feature: names_datasets_train,

                ExactSearch_feature: names_datasets_train,
                PC_feature: names_datasets_train,
                FCI_feature: names_datasets_train,
                GRaSP_feature: names_datasets_train,
                DML_feature: names_datasets_train,

                piecewise_linear_regression_feature: names_datasets_train,
                piecewise_quadratic_regression_feature: names_datasets_train,
                piecewise_quadratic_regression_feature_4_improved: names_datasets_train,
                sliding_window_linear_regression_feature: names_datasets_train,

                sem_features: names_datasets_train,
                grid_feature: names_datasets_train,
                discrete_statistic_features: names_datasets_train,
                Cloud_feature: names_datasets_train,
            },
            n_jobs=-1,
            create_dimension_feature=True,
        )
        X_y_group_train['MI(v,X)^2'] = X_y_group_train['MI(v,X)'] ** 2
        X_y_group_train['MI(v,Y)^2'] = X_y_group_train['MI(v,Y)'] ** 2
        X_y_group_train['MI(X,Y)^2'] = X_y_group_train['MI(X,Y)'] ** 2
        X_y_group_train['max(MI(v, others))^2'] = X_y_group_train['max(MI(v, others))'] ** 2
        X_y_group_train['min(MI(v, others))^2'] = X_y_group_train['min(MI(v, others))'] ** 2

        X_y_group_train = X_y_group_train.loc[:,~X_y_group_train.columns.duplicated()]

        shape = X_y_group_train.shape
        columns = X_y_group_train.columns.tolist()
        print('     ->线上训练数据 X_y_group_train 已生成')
        print(f'     ->X_y_group_train形状: {shape} \n X_y_group_train列名: {columns}')


    print("##### Adding numeric labels y")
    # 添加数值标签 y
    le = LabelEncoder()
    le.classes_ = np.array([
        'Confounder', 'Collider',
        'Mediator', 'Independent',
        'Cause of X', 'Consequence of X',
        'Cause of Y', 'Consequence of Y',
    ])
    X_y_group_train["y"] = le.transform(X_y_group_train["label"])
    # 重新排列列
    X_y_group_train = X_y_group_train[["dataset", "variable"] + X_y_group_train.columns.drop(["dataset", "variable", "label", "y"]).tolist() + ["label", "y"]]


    print("##### Data Preprocessing...")
    # 定义要删除的列
    blacklist = [
        "ttest(v,X)",
        "pvalue(ttest(v,X))<=0.05",
        "ttest(v,Y)",
        "pvalue(ttest(v,Y))<=0.05",
        "ttest(X,Y)",
        "pvalue(ttest(X,Y))<=0.05",
        "square_dimension",
        "max(PPS(v,others))",
        "TLI_Collider",
        "TLI_Confounder",
        "RMSEA_Collider",
        "RMSEA_Confounder",
    ]
    columns_to_drop = [col for col in blacklist if col in X_y_group_train.columns]
    X_y_group_train = X_y_group_train.drop(columns=columns_to_drop)
    print(f'     1.删除多余列后样本量: {X_y_group_train.shape}')
    # 填充缺失值
    X_y_group_train = X_y_group_train.replace([np.inf, -np.inf], np.nan)
    numeric_columns = X_y_group_train.select_dtypes(include=[np.number]).columns
    X_y_group_train[numeric_columns] = X_y_group_train[numeric_columns].fillna(X_y_group_train[numeric_columns].mean())
    print(f'     2.填充缺失值后样本量: {X_y_group_train.shape}')
    # 清理特征名称
    X_y_group_train = clean_feature_names(X_y_group_train)
    print(f'     3.清理特征名称后样本量: {X_y_group_train.shape}')


    print("##### Extracting X_train, y_train")
    X_y_group_train['raw_key'] = X_y_group_train['dataset'].str[-5:]
    all_key_list = X_y_group_train['raw_key'].unique().tolist()
    print(f'     ->原始key个数: {len(all_key_list)}')
    X_y_group_train['augment'] = np.where(X_y_group_train['dataset'].str.len() == 5, 0, 1)
    print(f'     ->增强key个数: {X_y_group_train["augment"].sum()}')
    X = X_y_group_train.drop(["variable", "dataset", "label", "y"], axis="columns")
    y = X_y_group_train["y"]

    # 处理类别特征
    cat_idxs, cat_dims, X = process_categorical_features(X)
    print(f'     ->类别特征索引 (cat_idxs): {cat_idxs}')
    print(f'     ->类别特征模态数 (cat_dims): {cat_dims}')
    print(f'     4.处理类别特征后样本量: {X.shape}, {y.shape}')


    print("##### Extracting X_train, y_train, and group")
    # # 分割all_key_list为训练集和测试集key
    # train_keys, test_keys = train_test_split(all_key_list, test_size=0.2, random_state=42)
    # # 定义掩码：X的'raw_key'列在train_keys中
    # mask_train = X.raw_key.str.endswith(tuple(train_keys))
    # # 定义掩码：X的'augment'列为0，且'raw_key'列在test_keys中
    # mask_test = (X.augment == 0) & (X.raw_key.str.endswith(tuple(test_keys)))
    # 定义掩码：X的'augment'列为0
    mask_rawdata = X.augment == 0
    X = X.drop(columns=['raw_key', 'augment'])
    # # 使用掩码分割数据集为训练集和测试集
    # X_train = X[mask_train]
    # X_test = X[mask_test]
    # y_train = y[mask_train]
    # y_test = y[mask_test]
    # # 定义掩码：X的'dataset'列的后五个字符在train_keys中
    # mask = X.raw_key.str.endswith(tuple(train_keys))
    # X = X.drop(columns=['raw_key'])
    # # 使用掩码分割数据集为训练集和测试集
    # X_train = X[mask]
    # X_test = X[~mask]
    # y_train = y[mask]
    # y_test = y[~mask]
    # # 分割数据集为训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.1, random_state=42, stratify=y
    # )
    # 直接完整训练
    X_train = X.copy()
    y_train = y.copy()
    print(f'     5.分割数据集后样本量: {X_train.shape}, {y_train.shape}')
    # print(f'     5.分割数据集后样本量: {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}')
    print(f'     ->y_train 唯一值: {np.unique(y_train)}')
    # print(f'     ->y_test 唯一值: {np.unique(y_test)}')


    print("##### Computing class weights")
    # classes = np.unique(y_train)
    # class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    # # 类别权重列表
    # class_weights = list(class_weights)
    # print(f'     ->分类目标权重列表: {class_weights}')
    # # 类别权重字典
    # weight_dict = dict(zip(classes, class_weights))
    # print(f'     ->分类目标权重字典: {weight_dict}')
    # # 样本权重
    # sample_weights = y_train.map(weight_dict)

    # 使用原始数据集的逆频率
    classes = np.unique(y[mask_rawdata])
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y[mask_rawdata])
    # 类别权重列表
    class_weights = list(class_weights)
    print(f'     ->分类目标权重列表: {class_weights}')
    # 类别权重字典
    weight_dict = dict(zip(classes, class_weights))
    print(f'     ->分类目标权重字典: {weight_dict}')
    # 样本权重
    sample_weights = y_train.map(weight_dict)


    print("##### Start training")
    early_stop = xgb.callback.EarlyStopping(
        rounds=20, save_best=True, maximize=False, metric_name='mlogloss'
    )
    xgb_model = XGBClassifier(
        n_estimators=7000,
        learning_rate=0.05,
        objective='multi:softmax',
        num_class=8,
        eval_metric='mlogloss',

        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=1,
        gamma=0.3,
        reg_lambda=18.0,
        reg_alpha=18.0,
        tree_method='gpu_hist',      # 如果没有GPU，可以使用 'hist'
        max_delta_step=0.5,

        verbosity=0,
        use_label_encoder=False,
        random_state=42,
        callbacks=[early_stop]
    )
    xgb_model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train)],
        verbose=100
    )
    pp = pprint.PrettyPrinter(indent=4)
    params_str = pp.pformat(xgb_model.get_params())
    print("##### Xgb的参数配置:")
    print(params_str)
    print("############################")
    print("     Xgb的性能展示:")
    print("############################")
    # 预测
    y_train_pred = xgb_model.predict(X_train)
    # y_test_pred = xgb_model.predict(X_test)
    # 计算平衡准确率
    train_score = balanced_accuracy_score(y_train, y_train_pred)
    # test_score = balanced_accuracy_score(y_test, y_test_pred)
    print(f'     ->训练集平衡准确率: {train_score:.6f}')
    # print(f'     ->测试集平衡准确率: {test_score:.6f}')
    # # 打印分类报告
    # print('     ->测试集分类报告:')
    # report = classification_report(y_test, y_test_pred)
    # print("\n" + report)
    # 保存
    joblib.dump(
        xgb_model,
        os.path.join(model_directory_path, f"xgb_model_1_{version}.joblib")
    )
    print('     5.Xgboost 1 训练完成')
    print("############################")
    print("\n")


    early_stop = xgb.callback.EarlyStopping(
        rounds=20, save_best=True, maximize=False, metric_name='mlogloss'
    )
    xgb_model = XGBClassifier(
        n_estimators=7500,
        learning_rate=0.05,
        objective='multi:softmax',
        num_class=8,
        eval_metric='mlogloss',

        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=1,
        gamma=0.3,
        reg_lambda=18.0,
        reg_alpha=18.0,
        tree_method='gpu_hist',      # 如果没有GPU，可以使用 'hist'
        max_delta_step=0.5,

        verbosity=0,
        use_label_encoder=False,
        random_state=77,
        callbacks=[early_stop]
    )
    xgb_model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train)],
        verbose=100
    )
    pp = pprint.PrettyPrinter(indent=4)
    params_str = pp.pformat(xgb_model.get_params())
    print("##### Xgb的参数配置:")
    print(params_str)
    print("############################")
    print("     Xgb的性能展示:")
    print("############################")
    # 预测
    y_train_pred = xgb_model.predict(X_train)
    # y_test_pred = xgb_model.predict(X_test)
    # 计算平衡准确率
    train_score = balanced_accuracy_score(y_train, y_train_pred)
    # test_score = balanced_accuracy_score(y_test, y_test_pred)
    print(f'     ->训练集平衡准确率: {train_score:.6f}')
    # print(f'     ->测试集平衡准确率: {test_score:.6f}')
    # # 打印分类报告
    # print('     ->测试集分类报告:')
    # report = classification_report(y_test, y_test_pred)
    # print("\n" + report)
    # 保存
    joblib.dump(
        xgb_model,
        os.path.join(model_directory_path, f"xgb_model_2_{version}.joblib")
    )
    print('     5.Xgboost 2 训练完成')
    print("############################")
    print("\n")


    early_stop = xgb.callback.EarlyStopping(
        rounds=20, save_best=True, maximize=False, metric_name='mlogloss'
    )
    xgb_model = XGBClassifier(
        n_estimators=8000,
        learning_rate=0.05,
        objective='multi:softmax',
        num_class=8,
        eval_metric='mlogloss',

        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=1,
        gamma=0.3,
        reg_lambda=18.0,
        reg_alpha=18.0,
        tree_method='gpu_hist',      # 如果没有GPU，可以使用 'hist'
        max_delta_step=0.5,

        verbosity=0,
        use_label_encoder=False,
        random_state=2,
        callbacks=[early_stop]
    )
    xgb_model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train)],
        verbose=100
    )
    pp = pprint.PrettyPrinter(indent=4)
    params_str = pp.pformat(xgb_model.get_params())
    print("##### Xgb的参数配置:")
    print(params_str)
    print("############################")
    print("     Xgb的性能展示:")
    print("############################")
    # 预测
    y_train_pred = xgb_model.predict(X_train)
    # y_test_pred = xgb_model.predict(X_test)
    # 计算平衡准确率
    train_score = balanced_accuracy_score(y_train, y_train_pred)
    # test_score = balanced_accuracy_score(y_test, y_test_pred)
    print(f'     ->训练集平衡准确率: {train_score:.6f}')
    # print(f'     ->测试集平衡准确率: {test_score:.6f}')
    # # 打印分类报告
    # print('     ->测试集分类报告:')
    # report = classification_report(y_test, y_test_pred)
    # print("\n" + report)
    # 保存
    joblib.dump(
        xgb_model,
        os.path.join(model_directory_path, f"xgb_model_3_{version}.joblib")
    )
    print('     5.Xgboost 3 训练完成')
    print("############################")
    print("\n")


    # train_pool = catboost.Pool(data=X_train, label=y_train, cat_features=cat_idxs)
    # test_pool = catboost.Pool(data=X_test, label=y_test, cat_features=cat_idxs)
    # cat_model = CatBoostClassifier(
    #     iterations=4000,
    #     learning_rate=0.05,
    #     classes_count=8,
    #     class_weights=class_weights,
    #     cat_features=cat_idxs,
    #     loss_function='MultiClass',
    #     eval_metric='Accuracy',

    #     depth=12,
    #     l2_leaf_reg=18.0,

    #     verbose=100,
    #     random_seed=42,
    #     early_stopping_rounds=20,
    #     task_type='GPU',
    #     devices='0:1',  # 如果使用GPU,指定GPU设备
    #     save_snapshot=False,
    #     train_dir=f"/tmp/catboost_info_{version}",
    #     leaf_estimation_method='Newton',  # 默认方法
    # )
    # cat_model.fit(
    #     train_pool,
    #     eval_set=test_pool,
    #     use_best_model=True
    # )
    # pp = pprint.PrettyPrinter(indent=4)
    # params_str = pp.pformat(cat_model.get_params())
    # print("##### Cat的参数配置:")
    # print(params_str)
    # print("############################")
    # print("     Cat的性能展示:")
    # print("############################")
    # # 预测
    # y_train_pred = cat_model.predict(X_train)
    # y_test_pred = cat_model.predict(X_test)
    # # 计算平衡准确率
    # train_score = balanced_accuracy_score(y_train, y_train_pred)
    # test_score = balanced_accuracy_score(y_test, y_test_pred)
    # print(f"     ->训练集平衡准确率: {train_score:.6f}")
    # print(f"     ->测试集平衡准确率: {test_score:.6f}")
    # # 打印分类报告
    # print("     ->测试集分类报告:")
    # report = classification_report(y_test, y_test_pred)
    # print("\n" + report)
    # # 保存
    # joblib.dump(
    #     cat_model,
    #     os.path.join(model_directory_path, f"cat_model_{version}.joblib")
    # )
    # print('     6.Catboost训练完成')
    # print("############################")
    # print("\n")


    # callbacks = [
    #     lgb.log_evaluation(period=100), lgb.early_stopping(stopping_rounds=20)
    # ]
    # lgb_model = lgb.LGBMClassifier(
    #     n_estimators=6000,
    #     learning_rate=0.05,
    #     class_weight=weight_dict,

    #     max_depth=6,
    #     num_leaves=21,
    #     min_child_samples=50,
    #     colsample_bytree=0.6,
    #     reg_alpha=18.0,
    #     reg_lambda=18.0,

    #     random_state=42,
    #     n_jobs=-1,
    #     device='cpu',
    #     verbosity=-1
    # )
    # lgb_model.fit(
    #     X_train, y_train,
    #     callbacks=callbacks,
    #     eval_set=[(X_train, y_train)],
    #     categorical_feature=cat_idxs
    # )
    # pp = pprint.PrettyPrinter(indent=4)
    # params_str = pp.pformat(lgb_model.get_params())
    # print("##### Lgb的参数配置:")
    # print(params_str)
    # print("############################")
    # print("     Lgb的性能展示:")
    # print("############################")
    # # 预测
    # y_train_pred = lgb_model.predict(X_train)
    # # y_test_pred = lgb_model.predict(X_test)
    # # 计算平衡准确率
    # train_score = balanced_accuracy_score(y_train, y_train_pred)
    # # test_score = balanced_accuracy_score(y_test, y_test_pred)
    # print(f"     ->训练集平衡准确率: {train_score:.6f}")
    # # print(f"     ->测试集平衡准确率: {test_score:.6f}")
    # # # 打印分类报告
    # # print("     ->测试集分类报告:")
    # # report = classification_report(y_test, y_test_pred)
    # # print("\n" + report)
    # # 保存
    # joblib.dump(
    #     lgb_model,
    #     os.path.join(model_directory_path, f"lgb_model_{version}.joblib")
    # )
    # print('     7.LightGBM训练完成')
    # print("############################")
    # print("\n")


    # hgb_model = HistGradientBoostingClassifier(
    #     max_iter=6000,
    #     learning_rate=0.05,
    #     class_weight=weight_dict, #'balanced',
    #     categorical_features=cat_idxs,

    #     max_depth=6,
    #     max_leaf_nodes=21,
    #     min_samples_leaf=50,
    #     l2_regularization=18.0,
    #     max_features=0.9,
    #     max_bins=255,

    #     random_state=42,
    #     early_stopping=False,
    #     # n_iter_no_change=20,
    #     # validation_fraction=0.1,
    #     verbose=100,
    # )
    # hgb_model.fit(
    #     X_train, y_train
    # )
    # pp = pprint.PrettyPrinter(indent=4)
    # params_str = pp.pformat(hgb_model.get_params())
    # print("##### Hgb的参数配置:")
    # print(params_str)
    # print("############################")
    # print("     Hgb的性能展示:")
    # print("############################")
    # # 预测
    # y_train_pred = hgb_model.predict(X_train)
    # y_test_pred = hgb_model.predict(X_test)
    # # 计算平衡准确率
    # train_score = balanced_accuracy_score(y_train, y_train_pred)
    # test_score = balanced_accuracy_score(y_test, y_test_pred)
    # print(f"     ->训练集平衡准确率: {train_score:.6f}")
    # print(f"     ->测试集平衡准确率: {test_score:.6f}")
    # # 打印分类报告
    # print("     ->测试集分类报告:")
    # report = classification_report(y_test, y_test_pred)
    # print("\n" + report)
    # # 保存
    # joblib.dump(
    #     hgb_model,
    #     os.path.join(model_directory_path, f"hgb_model_{version}_noes.joblib")
    # )
    # print('     8.HistGradientBoosting训练完成')
    # print("############################")
    # print("\n")


    # voting_clf = PretrainedVotingClassifier(
    #     estimators=[xgb_model, lgb_model],   # , cat_model
    #     voting='soft'
    # )
    # print("     Voting的性能展示:")
    # print("############################")
    # # 预测
    # y_train_pred = voting_clf.predict(X_train)
    # y_test_pred = voting_clf.predict(X_test)
    # # 计算平衡准确率
    # train_score = balanced_accuracy_score(y_train, y_train_pred)
    # test_score = balanced_accuracy_score(y_test, y_test_pred)
    # print(f"     ->训练集平衡准确率: {train_score:.6f}")
    # print(f"     ->测试集平衡准确率: {test_score:.6f}")
    # # 打印分类报告
    # print("     ->测试集分类报告:")
    # report = classification_report(y_test, y_test_pred)
    # print("\n" + report)
    # print('     8.Voting检查完成')
    # print("############################")
    # print("\n")


def infer(
    X_test: typing.Dict[str, pd.DataFrame],
    # number_of_features: int,
    model_directory_path: str,
    id_column_name: str,
    prediction_column_name: str,
    # has_gpu: bool,
    # has_trained: bool,
) -> pd.DataFrame:
    # 设置版本（模型、数据和结果保存名称）
    version = 'traindata_25augment_final'
    X_group_test_pathname = os.path.join(model_directory_path, f"X_group_test_{version}.pkl")

    try:
        print(f"##### Loading {X_group_test_pathname}")
        X_group_test = pd.read_pickle(X_group_test_pathname)
        if 'Unnamed_0' in X_group_test.columns:
            print("     ->'Unnamed_0' column found. It will be removed.")
            X_group_test.drop(columns=['Unnamed_0'], inplace=True)
        shape = X_group_test.shape
        columns = X_group_test.columns.tolist()
        print(f'     ->X_group_test形状: {shape} \n X_group_test列名: {columns}')
    except FileNotFoundError:
        print("##### Creating X_group_test")

        # # 小批量测试
        # X_test = filter_test_data(X_test, n_samples=20)

        names_datasets_test = X_test
        X_group_test = create_all_columns(
            {
                pearson_correlation: names_datasets_test,
                rolling_abs_pearson_correlation: names_datasets_test,
                spearman_correlation: names_datasets_test,
                kendall_correlation: names_datasets_test,
                mutual_information: names_datasets_test,
                conditional_mutual_information: names_datasets_test,
                distance_correlation: names_datasets_test,
                partial_correlation: names_datasets_test,
                copula_entropy: names_datasets_test,
                PPS_feature: names_datasets_test,

                linear_regression_feature: names_datasets_test,
                ridge_regression_feature: names_datasets_test,

                ExactSearch_feature: names_datasets_test,
                PC_feature: names_datasets_test,
                FCI_feature: names_datasets_test,
                GRaSP_feature: names_datasets_test,
                DML_feature: names_datasets_test,

                piecewise_linear_regression_feature: names_datasets_test,
                piecewise_quadratic_regression_feature: names_datasets_test,
                piecewise_quadratic_regression_feature_4_improved: names_datasets_test,
                sliding_window_linear_regression_feature: names_datasets_test,

                sem_features: names_datasets_test,
                grid_feature: names_datasets_test,
                discrete_statistic_features: names_datasets_test,
                Cloud_feature: names_datasets_test
            },
            n_jobs=-1,
            create_dimension_feature=True,
            )
        X_group_test['MI(v,X)^2'] = X_group_test['MI(v,X)'] ** 2
        X_group_test['MI(v,Y)^2'] = X_group_test['MI(v,Y)'] ** 2
        X_group_test['MI(X,Y)^2'] = X_group_test['MI(X,Y)'] ** 2
        X_group_test['max(MI(v, others))^2'] = X_group_test['max(MI(v, others))'] ** 2
        X_group_test['min(MI(v, others))^2'] = X_group_test['min(MI(v, others))'] ** 2

        X_group_test = X_group_test.loc[:,~X_group_test.columns.duplicated()]

        shape = X_group_test.shape
        columns = X_group_test.columns.tolist()
        print('     ->线上测试数据 X_group_test 已生成')
        print(f'     ->X_group_test形状: {shape} \n X_group_test列名: {columns}')


    print('##### Loading Models...')
    xgb_model_1 = joblib.load(os.path.join(model_directory_path, f"xgb_model_1_{version}.joblib"))
    xgb_model_2 = joblib.load(os.path.join(model_directory_path, f"xgb_model_2_{version}.joblib"))
    xgb_model_3 = joblib.load(os.path.join(model_directory_path, f"xgb_model_3_{version}.joblib"))
    # cat_model = joblib.load(os.path.join(model_directory_path, f"cat_model_{version}.joblib"))
    # lgb_model = joblib.load(os.path.join(model_directory_path, f"lgb_model_{version}.joblib"))

    print("##### Data Preprocessing...")
    # 定义要删除的列
    blacklist = [
        "ttest(v,X)",
        "pvalue(ttest(v,X))<=0.05",
        "ttest(v,Y)",
        "pvalue(ttest(v,Y))<=0.05",
        "ttest(X,Y)",
        "pvalue(ttest(X,Y))<=0.05",
        "square_dimension",
        "max(PPS(v,others))",
        "TLI_Collider",
        "TLI_Confounder",
        "RMSEA_Collider",
        "RMSEA_Confounder",
    ]
    columns_to_drop = [col for col in blacklist if col in X_group_test.columns]
    X_group_test = X_group_test.drop(columns=columns_to_drop)
    print(f'     1.删除多余列后样本量: {X_group_test.shape}')
    # 填充缺失值
    X_group_test = X_group_test.replace([np.inf, -np.inf], np.nan)
    numeric_columns = X_group_test.select_dtypes(include=[np.number]).columns
    X_group_test[numeric_columns] = X_group_test[numeric_columns].fillna(X_group_test[numeric_columns].mean())
    print(f'     2.填充缺失值后样本量: {X_group_test.shape}')
    # 清理特征名称
    X_group_test = clean_feature_names(X_group_test)
    print(f'     3.清理特征名称后样本量: {X_group_test.shape}')

    print("##### Extracting X_test")
    X = X_group_test.drop(["variable", "dataset"], axis="columns")

    # 处理类别特征
    cat_idxs, cat_dims, X = process_categorical_features(X)
    print(f'     ->类别特征索引 (cat_idxs): {cat_idxs}')
    print(f'     ->类别特征模态数 (cat_dims): {cat_dims}')

    X_test = X
    print(f'     4.处理类别特征后样本量: {X_test.shape}')

    # X = clean_feature_names(X)   # 前面已经有了
    X_test = filter_features(X_test, xgb_model_1)
    X_test = align_features(X_test, xgb_model_1)
    print(f'     5.对齐特征后样本量: {X_test.shape}')

    print("##### Predicting on X_test")
    voting_clf = PretrainedVotingClassifier(
        estimators=[xgb_model_1, xgb_model_2, xgb_model_3],   # , cat_model
        voting='soft'
    )
    y_predicted = voting_clf.predict(X_test)
    # y_predicted = lgb_model.predict(X_test)
    X_y_pred_test = X_group_test
    X_y_pred_test["y_predicted"] = y_predicted

    le = LabelEncoder()
    le.classes_ = np.array([
        'Confounder', 'Collider',
        'Mediator', 'Independent',
        'Cause of X', 'Consequence of X',
        'Cause of Y', 'Consequence of Y',
    ])
    X_y_pred_test["label_predicted"] = le.inverse_transform(y_predicted)

    submission = create_submission(X_y_pred_test)
    print('预测完成')

    return pd.DataFrame(
        submission.items(),
        columns=[
            id_column_name,
            prediction_column_name
        ]
    )


# """单模"""
# def create_submission(X_y_pred_test):
#     """
#     From the predicted test set, for each dataset, take predicted
#     classes of all variables, create the adjacency matrix, then create
#     the submission in the requested format.
#     """

#     submission = {}
#     for name, prediction in tqdm(X_y_pred_test.groupby("dataset"), delay=10):
#         variables_labels = prediction[["variable", "label_predicted"]].set_index("variable")
#         variables = variables_labels.index.tolist()
#         variables_all = ["X", "Y"] + variables

#         adjacency_matrix = pd.DataFrame(index=variables_all, columns=variables_all)
#         adjacency_matrix.index.name = "parent"
#         adjacency_matrix[:] = 0
#         adjacency_matrix.loc["X", "Y"] = 1

#         for v in variables:
#             l = variables_labels.loc[v].item()
#             if l == "Cause of X":
#                 adjacency_matrix.loc[v, "X"] = 1
#             elif l == "Cause of Y":
#                 adjacency_matrix.loc[v, "Y"] = 1
#             elif l == "Consequence of X":
#                 adjacency_matrix.loc["X", v] = 1
#             elif l == "Consequence of Y":
#                 adjacency_matrix.loc["Y", v] = 1
#             elif l == "Confounder":
#                 adjacency_matrix.loc[v, "X"] = 1
#                 adjacency_matrix.loc[v, "Y"] = 1
#             elif l == "Collider":
#                 adjacency_matrix.loc["X", v] = 1
#                 adjacency_matrix.loc["Y", v] = 1
#             elif l == "Mediator":
#                 adjacency_matrix.loc["X", v] = 1
#                 adjacency_matrix.loc[v, "Y"] = 1
#             elif l == "Confounder":
#                 pass

#         for i in variables_all:
#             for j in variables_all:
#                 submission[f'{name}_{i}_{j}'] = int(adjacency_matrix.loc[i, j])

#     return submission


# def infer(
#     X_test: typing.Dict[str, pd.DataFrame],
#     # number_of_features: int,
#     model_directory_path: str,
#     id_column_name: str,
#     prediction_column_name: str,
#     # has_gpu: bool,
#     # has_trained: bool,
# ) -> pd.DataFrame:
#     model = joblib.load(os.path.join(model_directory_path, "model.joblib"))

#     names_datasets_test = X_test
#     X_group_test = create_all_columns(
#         {
#             PPS_feature: names_datasets_test,
#             pearson_correlation: names_datasets_test,
#             mutual_information: names_datasets_test,
#             spearman_correlation: names_datasets_test,
#             kendall_correlation: names_datasets_test,
#             distance_correlation: names_datasets_test,
#             conditional_mutual_information: names_datasets_test,
#             partial_correlation: names_datasets_test,
#             linear_regression_feature: names_datasets_test,
#             ridge_regression_feature: names_datasets_test,
#             PC_feature: names_datasets_test,
#             ExactSearch_feature: names_datasets_test,
#             FCI_feature: names_datasets_test,
#             DML_feature: names_datasets_test,
#             copula_entropy: names_datasets_test,
#             piecewise_linear_regression_feature: names_datasets_test,
#             },
#             n_jobs=-1,
#             create_dimension_feature=True,
#         )
#     X_group_test['MI(v,X)^2'] = X_group_test['MI(v,X)'] ** 2
#     X_group_test['MI(v,Y)^2'] = X_group_test['MI(v,Y)'] ** 2
#     X_group_test['MI(X,Y)^2'] = X_group_test['MI(X,Y)'] ** 2
#     X_group_test['max(MI(v, others))^2'] = X_group_test['max(MI(v, others))'] ** 2
#     X_group_test['min(MI(v, others))^2'] = X_group_test['min(MI(v, others))'] ** 2

#     blacklist = ["ttest(v,X)", "pvalue(ttest(v,X))<=0.05", "ttest(v,Y)", "pvalue(ttest(v,Y))<=0.05", "ttest(X,Y)", "pvalue(ttest(X,Y))<=0.05"]
#     columns_to_drop = [col for col in blacklist if col in X_group_test.columns]

#     X_group_test = X_group_test.drop(columns=columns_to_drop)

#     numeric_columns = X_group_test.select_dtypes(include=[np.number]).columns

#     X_group_test[numeric_columns] = X_group_test[numeric_columns].fillna(X_group_test[numeric_columns].mean())

#     X_test = X_group_test.drop(columns=["dataset", "variable"])
#     y_predicted = model.predict(X_test)
#     X_y_pred_test = X_group_test
#     X_y_pred_test["y_predicted"] = y_predicted

#     le = LabelEncoder()
#     le.classes_ = np.array([
#         'Cause of X', 'Consequence of X', 'Confounder', 'Collider',
#         'Mediator', 'Independent', 'Cause of Y', 'Consequence of Y',
#     ])

#     X_y_pred_test["label_predicted"] = le.inverse_transform(y_predicted)

#     submission = create_submission(X_y_pred_test)

#     return pd.DataFrame(
#         submission.items(),
#         columns=[
#             id_column_name,
#             prediction_column_name
#         ]
#     )


#crunch.test(
#    no_determinism_check=True
#)

#print("Download this notebook and submit it to the platform: https://hub.crunchdao.com/competitions/causality-discovery/submit/via/notebook")
