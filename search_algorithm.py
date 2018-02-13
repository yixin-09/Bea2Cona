# -*- coding: utf-8 -*-
# In this file, three algorithm will be tested:
# Local, Global, and Hybrid Search
# Local: Hill Climbing algorithm
# Global: Genetic algorithm
# Hybrid
# fitness function: Relative error and Condition Number

# partition ways: rdistribution_partiion, fdistribution_partiion,

# Six situation for each algorithm
# fdistribution_partiion + condition + relative error
# fdistribution_partiion + relative error
# rdistribution_partiion + condition + relative error
# rdistribution_partiion + relative error
# no_partition + condition + relative error
# no_partition + relative error
from fun2search import *
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy import special
import time
from mpmath import *
import numpy as np
#import fdistan
#import gsl_fun
# global algorithm: evolution algorithm
# based on Relative_error_ulp
# partition: no partition, real partition, floating-point partition

def glob_search_fit1(partition,input_domain,min_flag,limit):
    input_list = partition(input_domain,0)
    times = limit[1]
    len_in = len(input_list)
    one_time = times/len(fpartition(input_domain,0))
    if len(input_list) == 1:
        popsize = int(min(times/1000.0,100))
        maxiter = int(times/popsize)
    else:
        popsize = int(max(times/(len_in*160),10))
        maxiter = int(times/(len_in*popsize))
    print popsize
    print maxiter
    iter_number = 0
    start_time = time.clock()
    res_l = []
    for i in input_list:
        ret = differential_evolution(glob_fitness_fun,i,popsize=popsize,maxiter=maxiter,polish=min_flag)
        res_l.append((1.0 / ret.fun, ret.x))
        iter_number = iter_number+ret.nfev
    sl_res = sorted(res_l,key=lambda x:x[0], reverse=True)
    t = time.clock() - start_time
    return sl_res[0][0],tuple(sl_res[0][1]),t,iter_number
#divide the input domain based on partition function
def glob_search_fit2(partition,input_domain,min_flag,limit):
    input_list = partition(input_domain,0)
    times = limit[1]*limit[3]
    len_in = len(input_list)
    if len(input_list) == 1:
        popsize = int(times/10000)
        maxiter = int(5000)
    else:
        popsize = int(max(times/(len_in*1000),10))
        maxiter = int(500)
    print popsize
    print maxiter
    res_l = []
    start_time = time.clock()
    iter_number = 0
    for i in input_list:
        ret = differential_evolution(glob_fitness_fun_cu,i,popsize=popsize,maxiter=maxiter,polish=min_flag)
        iter_number = iter_number+ret.nfev
        res_l.append((1.0 / glob_fitness_fun(ret.x), ret.x))
    sl_res = sorted(res_l, key=lambda x:x[0], reverse=True)
    t = time.clock() - start_time
    return sl_res[0][0], tuple(sl_res[0][1]), t,iter_number

def produce_small_interval(x,interval):
    new_interval = []
    for i,j in zip(x,interval):
        values=j[1]-j[0]
        new_interval.append([max(j[0],i-(0.1*values)),min(j[1],i+(0.1*values))])
    return new_interval
def glob_search_fit_hybrid(partition,input_domain,min_flag,limit):
    input_list = partition(input_domain, 0)
    times = limit[1] * limit[3]
    len_in = len(input_list)
    if len(input_list) == 1:
        popsize = int(min(times/20000.0,200))
        maxiter = int(5000)
    else:
        popsize = int(max(times/(len_in*2000),10))
        maxiter = int(500)
    print popsize
    print maxiter
    res_l = []
    start_time = time.clock()
    iter_number = 0
    for i in input_list:
        ret = differential_evolution(glob_fitness_fun_cu,i,popsize=popsize,maxiter=maxiter,polish=min_flag)
        res_l.append((1 / glob_fitness_fun(ret.x), ret.x,i))
        iter_number = iter_number + ret.nfev
    sl_res = sorted(res_l, key=lambda x:x[0],reverse=True)
    print time.clock()-start_time
    res_l = []
    lmax_x = []
    times = limit[1]
    len_in = len(input_list)
    if len(input_list) == 1:
        popsize = int(min(times/1000.0,50))
        maxiter = int(times/popsize)
    else:
        popsize = int(max(times/(len_in*320),5))
        maxiter = int(times/(len_in*popsize))
    iter_number = iter_number/limit[3]
    for i in sl_res:
        interval_around_x = produce_small_interval(i[1],i[2])
        print interval_around_x
        ret = differential_evolution(glob_fitness_fun,interval_around_x,popsize=popsize,maxiter=maxiter,polish=min_flag)
        iter_number = iter_number + ret.nfev
        res_l.append((1 / glob_fitness_fun(ret.x), ret.x))
        lmax_x.append(ret.x)
    sl_res = sorted(res_l, key=lambda x:x[0],reverse=True)
    t = time.clock() - start_time
    return sl_res[0][0], tuple(sl_res[0][1]), t , iter_number



def hybrid_search_fit_hybrid(partition,input_domain,min_flag,limit):
    input_list = partition(input_domain, 0)
    times = limit[1] * limit[3]
    len_in = len(input_list)
    if len(input_list) == 1:
        popsize = int(min(times/20000.0,200))
        maxiter = int(5000)
    else:
        popsize = int(max(times/(len_in*2000),10))
        maxiter = int(500)
    res_l = []
    start_time = time.clock()
    iter_number = 0
    print popsize
    print maxiter
    for i in input_list:
        ret = differential_evolution(glob_fitness_fun_cu,i,popsize=popsize,maxiter=maxiter,polish=min_flag)
        res_l.append((1 / glob_fitness_fun(ret.x), ret.x,i))
        iter_number = iter_number + ret.nfev
    sl_res = sorted(res_l, key=lambda x:x[0],reverse=True)
    print time.clock()-start_time
    res_l = []
    lmax_x = []
    maxfex = int(limit[1] / len(input_list))
    for i in sl_res:
        x = i[1]
        ret = minimize(glob_fitness_fun, x, method="L-BFGS-B",bounds=i[2], options={'maxiter': maxfex})
        iter_number = iter_number + ret.nfev
        res_l.append((1 / ret.fun, ret.x))
        lmax_x.append(ret.x)
    sl_res = sorted(res_l, key=lambda x:x[0],reverse=True)
    t = time.clock() - start_time
    return sl_res[0][0], tuple(sl_res[0][1]), t , iter_number
def hybrid_search_fit1(partition,input_domain,min_flag,limit):
    input_list = partition(input_domain, 0)
    times = limit[1]
    len_in = len(input_list)
    if len(input_list) == 1:
        popsize = int(min(times/1000.0,100))
        maxiter = int(times/popsize)
    else:
        popsize = int(max(times/(len_in*320),10))
        maxiter = int(times/(len_in*popsize))
    res_l = []
    start_time = time.clock()
    iter_number = 0
    for i in input_list:
        ret = differential_evolution(glob_fitness_fun,i,popsize=popsize,maxiter=maxiter,polish=min_flag)
        res_l.append((1.0/ret.fun , ret.x,i))
        iter_number = iter_number + ret.nfev
    sl_res = sorted(res_l, key=lambda x:x[0],reverse=True)
    print time.clock()-start_time
    res_l = []
    lmax_x = []
    maxfex = int(limit[1] / (2.0*len(input_list)))
    for i in sl_res:
        x = i[1]
        ret = minimize(glob_fitness_fun, x, method="L-BFGS-B",bounds=i[2], options={'maxiter': maxfex})
        iter_number = iter_number + ret.nfev
        res_l.append((1 / ret.fun, ret.x))
        lmax_x.append(ret.x)
    sl_res = sorted(res_l, key=lambda x:x[0],reverse=True)
    t = time.clock() - start_time
    return sl_res[0][0], tuple(sl_res[0][1]), t , iter_number
#local_search:
def local_search_fit1(partition,input_domain,limit):
    input_list = partition(input_domain, 0)
    res_l = []
    start_time = time.clock()
    maxfex = int(limit[1]/len(input_list))
    iter_number = 0
    for i in input_list:
        x = produce_one_input(i)
        ret = minimize(glob_fitness_fun,x,bounds=i,options={'maxiter':maxfex})
        iter_number = iter_number + ret.nfev
        res_l.append((1 / ret.fun, ret.x))
    sl_res = sorted(res_l,key=lambda x:x[0], reverse=True)
    t = time.clock() - start_time
    return sl_res[0][0], tuple(sl_res[0][1]), t,iter_number

def local_search_fit2(partition,input_domain,limit):
    input_list = partition(input_domain, 0)
    res_l = []
    start_time = time.clock()
    maxfex = int(limit[1]*limit[3]/len(input_list))
    iter_number = 0
    for i in input_list:
        x = produce_one_input(i)
        ret = minimize(glob_fitness_fun_cu,x,bounds=i,options={'maxiter':maxfex})
        iter_number = iter_number + ret.nfev
        res_l.append((1.0 / ret.fun, ret.x))
    sl_res = sorted(res_l, key=lambda x:x[0],reverse=True)
    t = time.clock() - start_time
    return sl_res[0][0], tuple(sl_res[0][1]), t,iter_number


def local_search_hybrid(partition,input_domain,limit):
    input_list = partition(input_domain, 0)
    res_l = []
    start_time = time.clock()
    maxfex = int(limit[1] * limit[3]*0.5 / len(input_list))
    print maxfex
    iter_number = 0
    for i in input_list:
        x = produce_one_input(i)
        ret = minimize(glob_fitness_fun_cu, x, bounds=i, options={'maxiter': maxfex})
        iter_number = iter_number + ret.nfev
        res_l.append((1.0 / ret.fun, ret.x,i))
    sl_res = sorted(res_l, key=lambda x:x[0],reverse=True)
    print sl_res
    res_l = []
    maxfex = int(limit[1]*0.5 / len(input_list))
    print maxfex
    iter_number = iter_number/limit[3]
    for i in sl_res:
        x = i[1]
        interval_around_x = produce_small_interval(i[1], i[2])
        ret = minimize(glob_fitness_fun, x,bounds=interval_around_x, options={'maxiter': maxfex})
        iter_number = iter_number + ret.nfev
        res_l.append((1 / ret.fun, ret.x))
    sl_res = sorted(res_l, key=lambda x:x[0],reverse=True)
    t = time.clock() - start_time
    return sl_res[0][0], tuple(sl_res[0][1]), t,iter_number

# random_search: random and binary random
def random_search_fit1(partition,input_domain,limit):
    input_list = partition(input_domain, 0)
    sl_res = []
    maxfex = int(limit[1] / len(input_list))
    print maxfex
    start_time = time.clock()
    for i in input_list:
        x = produce_n_input(i,pow(maxfex,1.0/len(i)))
        l = glob_fun_scalar(glob_fitness_fun, x)
        l = sorted(l, key=lambda x: x[0])
        sl_res.append((1.0/l[0][0],l[0][1]))
    sl_res = sorted(sl_res, key=lambda x: x[0], reverse=True)
    t = time.clock() - start_time
    return sl_res[0][0], tuple(sl_res[0][1]), t,limit[1]
def random_search_fit2(partition,input_domain,limit):
    input_list = partition(input_domain, 0)
    sl_res = []
    maxfex = int(limit[1]*limit[3] / len(input_list))
    print maxfex
    start_time = time.clock()
    for i in input_list:
        x = produce_n_input(i, pow(maxfex, 1.0 / len(i)))
        l = glob_fun_scalar(glob_fitness_fun_cu,x)
        l = sorted(l, key=lambda x: x[0])
        sl_res.append((1.0/glob_fitness_fun(l[0][1]),l[0][1]))
    sl_res = sorted(sl_res, key=lambda x: x[0], reverse=True)
    t = time.clock() - start_time
    return sl_res[0][0], tuple(sl_res[0][1]), t,limit[1]

def random_search_hybrid(partition,input_domain,limit):
    input_list = partition(input_domain, 0)
    sl_res = []
    maxfex = int(limit[1]*limit[3] / len(input_list))
    print maxfex
    start_time = time.clock()
    for i in input_list:
        x = produce_n_input(i, pow(maxfex, 1.0 / len(i)))
        l = glob_fun_scalar(glob_fitness_fun_cu,x)
        l = sorted(l, key=lambda x: x[0])
        for j in range(0,20):
            sl_res.append((1.0/glob_fitness_fun(l[j][1]),l[j][1]))
    sl_res = sorted(sl_res, key=lambda x: x[0], reverse=True)
    t = time.clock() - start_time
    return sl_res[0][0], tuple(sl_res[0][1]), t,limit[1]

# binary random search:
def binary_random_search(input_domain,limit):
    input_list = produce_new_confs(input_domain)
    # iteration_times = 10
    iteration = 0
    lmax = []
    max = 0.0
    tmp_conf = input_list
    times = int(limit[1]/10.0)
    start_time = time.clock()
    while iteration < 10:
        max = 0
        for i in input_list:
            x = produce_n_input(i,pow(times/len(input_list), 1.0 / len(i)))
            l = glob_fun_scalar(glob_fitness_fun,x)
            l = sorted(l,key=lambda x: x[0])
            tmp_max = 1.0/l[0][0]
            if tmp_max > max:
                tmp_conf = i
                max = tmp_max
                max_x = l[0][1]
        input_list = produce_new_confs(tmp_conf)
        iteration = iteration + 1
    t = time.clock() - start_time
    return max, max_x, t,limit[1]

def binary_random_search_fit2(input_domain,limit):
    input_list = produce_new_confs(input_domain)
    # iteration_times = 10
    iteration = 0
    lmax = []
    tmp_conf = input_list
    times = int(limit[1]*limit[3] / 10.0)
    start_time = time.clock()
    while iteration < 10:
        max = 0.0
        for i in input_list:
            x = produce_n_input(i, pow(times / len(input_list), 1.0 / len(i)))
            l = glob_fun_scalar(glob_fitness_fun_cu, x)
            l = sorted(l)
            tmp_max = 1.0 / l[0][0]
            if tmp_max > max:
                tmp_conf = i
                max = tmp_max
                max_x = l[0][1]
        input_list = produce_new_confs(tmp_conf)
        iteration = iteration + 1
    t = time.clock() - start_time
    return max, max_x, t,limit[1]



def binary_random_search_hybrid(input_domain,limit):
    input_list = produce_new_confs(input_domain)
    # iteration_times = 10
    iteration = 0
    lmax = []
    tmp_conf = input_list
    times = int(limit[1]*limit[3] / 10.0)
    start_time = time.clock()
    while iteration < 10:
        max = 0.0
        for i in input_list:
            x = produce_n_input(i, pow(times / len(input_list), 1.0 / len(i)))
            l = glob_fun_scalar(glob_fitness_fun_cu, x)
            l = sorted(l)
            tmp_l = [1.0/glob_fitness_fun(m[1]) for m in l[0:5]]
            tmp_max = max(tmp_l)
            if tmp_max > max:
                tmp_conf = i
                max = tmp_max
                max_x = l[0][1]
        input_list = produce_new_confs(tmp_conf)
        iteration = iteration + 1
    t = time.clock() - start_time
    return max, max_x, t,limit[1]



def hybrid_search():
    start_time = time.clock()
    res_l = []
    x_l = np.random.uniform(-8235490.6645, 108, 10)
    for x in x_l:
        ret = basinhopping(glob_fitness_fun_cu, x, niter=400, niter_success=40)
        res_l.append((1 / glob_fitness_fun(ret.x), ret.x))
    sl_res = sorted(res_l, reverse=True)
    print sl_res[0:10]
    print sl_res[0]
    print time.clock() - start_time




def test_globa_search(input_domain):
    glob_search_fit2(fpartition,input_domain)
    glob_search_fit2(rpartition,input_domain)
    #glob_search_fit2(nopartition,input_domain)
    glob_search_fit1(fpartition,input_domain)
    glob_search_fit1(rpartition,input_domain)
    #glob_search_fit1(nopartition,input_domain)
    glob_search_fit_hybrid(fpartition,input_domain)
    glob_search_fit_hybrid(rpartition,input_domain)
    #glob_search_fit_hybrid(nopartition, [[-823549.6645, 108]])

def print_fun_results(x):
    res1 = float(rf(x))
    res2 = float(fp(x))
    print res1
    print res2
    print res1-res2
    print fdistan_two(res1,res2)
    print special.airy(x)
    print special.ai_zeros(1)



#test_globa_search()
#glob_search_fit2(fpartition,[[-10,10]])
#print differential_evolution(glob_fitness_fun,([-823549.6645, 108],))
#glob_search_fit_hybrid(nopartition, [[-823549.6645, 108]])
#print_fun_results(-42.99484903)
#hybrid_search()
#local_search_fit1(rpartition,[[-10,10]])
#local_search_fit2(rpartition,[[-10,10]])
#local_search_fit2(fpartition,[[-10,10]])
def test_local(input_domain):
    local_search_hybrid(fpartition,input_domain)
    local_search_hybrid(rpartition,input_domain)
    local_search_hybrid(nopartition,input_domain)
#input_domain = [[-823549.6645, 108]]
#glob_search_fit2(fpartition,input_domain,True)
#random_search_fit1(nopartition,input_domain)
#random_search_fit2(nopartition,input_domain)
#binary_random_search_fit2(input_domain)
#print produce_one_input(([-823549.6645, 108],))
#test_globa_search([[-10,10]])
#print minimize(glob_fitness_fun, [(-294449.68550886482,)])
# Random: random or Binary search random test
#print glob_search_fit2(rpartition,[[-823549.6645, 108]],False,(60.032892, 29000, 0.9000010000000032, 66.70313921873397))
#local_search_fit1(rpartition,[[-823549.6645, 108]],(60.032892, 29000, 0.9000010000000032, 66.70313921873397))
