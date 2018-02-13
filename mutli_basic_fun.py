from mpmath import *
from scipy.misc import derivative
import numpy as np
import time
import fdistan
import xlwt
import pickle
import math
from sympy import symbols, diff
import itertools
from pygsl.testing import sf
from pathos.multiprocessing import ProcessingPool as Pool
from operator import mul
#from multiprocessing import Pool


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException
def test_gsl_fun(f,x):
    gsl_l = []
    for i in x:
        gsl_l.append((f(*i),i))
    return gsl_l
def test_mp_fun(f,x):
    mp_l = []
    for i in x:
        mp_l.append((float(f(*i)),i))
    return mp_l
def depart(in_min,in_max):
    tmp_l = []
    a = np.frexp(in_min)
    b = np.frexp(in_max)
    tmp_j = 0
    if (in_min<0)&(in_max>0):
        if in_min>=-1.0:
            tmp_l.append([in_min,0])
        else:
            for i in range(1,a[1]+1):
                tmp_i = np.ldexp(-0.5,i)
                tmp_l.append([tmp_i,tmp_j])
                tmp_j = tmp_i
            if in_min!=tmp_j:
                tmp_l.append([in_min,tmp_j])
        tmp_j = 0
        if in_max<=1.0:
            tmp_l.append([0,in_max])
        else:
            for i in range(1,b[1]+1):
                tmp_i = np.ldexp(0.5, i)
                tmp_l.append([tmp_j, tmp_i])
                tmp_j = tmp_i
            if in_max!=tmp_j:
                tmp_l.append([tmp_j,in_max])
    tmp_j = 0
    if (in_min < 0) & (0>=in_max):
        if in_min >= -1:
            tmp_l.append([in_min, in_max])
            return tmp_l
        else:
            if in_max > -1:
                tmp_l.append([-1,in_max])
                tmp_j = -1.0
                for i in range(2, a[1] + 1):
                    tmp_i = np.ldexp(-0.5, i)
                    tmp_l.append([tmp_i, tmp_j])
                    tmp_j = tmp_i
                if in_min != tmp_j:
                    tmp_l.append([in_min, tmp_j])
            else:
                if a[1]==b[1]:
                    tmp_l.append([in_min,in_max])
                    return tmp_l
                else:
                    tmp_j = np.ldexp(-0.5, b[1]+1)
                    tmp_l.append([tmp_j,in_max])
                    if tmp_j != in_min:
                        for i in range(b[1]+2,a[1]+1):
                            tmp_i = np.ldexp(-0.5, i)
                            tmp_l.append([tmp_i, tmp_j])
                            tmp_j = tmp_i
                        if in_min != tmp_j:
                            tmp_l.append([in_min, tmp_j])
    tmp_j = 0
    if (in_min >= 0) & (in_max > 0):
        if in_max<=1:
            tmp_l.append([in_min,in_max])
            return tmp_l
        else:
            if in_min <1:
                tmp_l.append([in_min,1])
                tmp_j = 1.0
                for i in range(2, b[1] + 1):
                    tmp_i = np.ldexp(0.5, i)
                    tmp_l.append([tmp_j, tmp_i])
                    tmp_j = tmp_i
                if in_max != tmp_j:
                    tmp_l.append([tmp_j, in_max])
            else:
                if a[1]==b[1]:
                    tmp_l.append([in_min, in_max])
                    return tmp_l
                else:
                    tmp_j = np.ldexp(0.5,a[1]+1)
                    tmp_l.append([in_min, tmp_j])
                    if tmp_j!=in_max:
                        for i in range(a[1]+2,b[1]+1):
                            tmp_i = np.ldexp(0.5,i)
                            tmp_l.append([tmp_j,tmp_i])
                            tmp_j = tmp_i
                        if in_max!=tmp_j:
                            tmp_l.append([tmp_j,in_max])
    return tmp_l

def partial_derivative(func,b, var=0, point=[]):
    args = point[:]
    diff = math.fabs(args[var] * 2.2204460492503131e-15)
    def wraps(x):
        args[var] = x
        return func(*args)
    #return derivative(wraps, point[var],dx = 1e-8)
    return (wraps(args[var] + diff) - b) / diff
def condition(a,b,f):
    #l_x = []
    #der_x = []
    tmp = 0
    for i in range(0,len(a)):
        #l_x.append(fdistan.ulp(a[i]))
        #der_x.append(partial_derivative(f,i,a))
        tmp= tmp + 2.2204460492503131e-16*a[i]*partial_derivative(f,b,i,a)
    ulp_b = b*2.2204460492503131e-16
    cond_num = math.fabs(tmp/ulp_b)
    return cond_num






def get_distan(f1,f2,x):
    m_x = float(f1(*x))
    f_x = f2(*x)
    return np.fabs(fdistan.fdistan(m_x,f_x))

def search_around(f1,f2,ulp_l):
    temp_err = 0.0
    tmp_x = 0.0
    for i in ulp_l[0:40]:
        mx = i[1]
        ds = get_distan(f1, f2, mx)
        if ds > temp_err:
            temp_err = ds
            tmp_x = mx
    return temp_err,tmp_x


def produce_interval(x,k):
    interval_l = []
    for i,j in zip(x,k):
        temp = 0.001 * np.fabs(j[1] - j[0])
        interval_l.append((i-temp,i+temp))
    return interval_l


def fine_search_tmp(in_l,f1,f2):
    var_l = []
    for k in in_l:
        var_l.append(sorted(np.random.uniform(k[0], k[1], 20)))
    input_l = []
    for element in itertools.product(*var_l):
        input_l.append(element)
    res_mp = test_mp_fun(f1, input_l)
    # res_mp = paral_proc_mppl(X,cof,200,1)
    res_d = test_gsl_fun(f2, input_l)
    # res_d = paral_proc_pl(X,cof,len(cof)-1,1)
    re_err_1 = [(distan_cal(op_f[0], float(op_r[0])), op_f[1]) for op_f, op_r in zip(res_d, res_mp)]
    error_l = sorted(re_err_1, reverse=True)
    return error_l[0][0], error_l[0][1]

def random_search(in_var,f1,f2,n):
    var_l = []
    l_var = []
    for i in in_var:
        tmp_l = depart(i[0], i[1])
        l_var.append(tmp_l)
    l_confs = []
    for element in itertools.product(*l_var):
        l_confs.append(element)
    input_l = []
    for k in l_confs:
        var_l = []
        for z in k:
            var_l.append(sorted(np.random.uniform(z[0], z[1], n)))
        for element in itertools.product(*var_l):
            input_l.append(element)
    res_mp = test_mp_fun(f1, input_l)
    # res_mp = paral_proc_mppl(X,cof,200,1)
    res_d = test_gsl_fun(f2, input_l)
    # res_d = paral_proc_pl(X,cof,len(cof)-1,1)
    re_err_1 = [(np.fabs(fdistan.fdistan(op_f[0], float(op_r[0]))), op_f[1]) for op_f, op_r in zip(res_d, res_mp)]
    error_l = sorted(re_err_1, reverse=True)
    return (error_l[0][0], error_l[0][1], in_var)

def test_two_fun(f1,f2,input_l):
    res_mp = test_mp_fun(f1, input_l)
    res_d = test_gsl_fun(f2, input_l)
    re_err_1 = [(np.fabs(fdistan.fdistan(op_f[0], float(op_r[0]))), op_f[1]) for op_f, op_r in zip(res_d, res_mp)]
    error_l = sorted(re_err_1, reverse=True)
    return (error_l[0][0], error_l[0][1])

def test_two_fun_fit2(f1,f2,input_l):
    #res_mp = test_mp_fun(f1, input_l)
    res_d = test_gsl_fun(f2, input_l)
    re_err_1 = [(condition(list(op_f[1]), op_f[0], f2), op_f[1]) for op_f in res_d]
    error_l = sorted(re_err_1, reverse=True)
    res = search_around(f1,f2,error_l)
    print res
    return res
def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]
def para_random_res(in_l,f1,f2,per,cpu_n):
    start_time = time.time()
    var_l = []
    lf1 = []
    lf2 = []
    p = Pool(cpu_n)
    for k in list(in_l):
        n = int(per*fdistan.fdistan(k[0],k[1]))
        print n
        var_l.append(sorted(np.random.uniform(k[0], k[1], n)))
    input_l = []
    for element in itertools.product(*var_l):
        inp = list(element)
        input_l.append(inp)
    input_l = chunks(input_l,8)
    lf1.extend([f1] * len(input_l))
    lf2.extend([f2] * len(input_l))
    tmp_res = p.map(test_two_fun, lf1, lf2, input_l)
    tmp_res = sorted(tmp_res, reverse=True)
    end_time = time.time() - start_time
    return (tmp_res[0][0], tmp_res[0][1], end_time)

def para_random_res_tlimit(in_l,f1,f2,limit_n,cpu_n):
    start_time = time.time()
    var_l = []
    lf1 = []
    lf2 = []
    p = Pool(cpu_n)
    n_l = []
    ts = 0
    i = 0
    for k in list(in_l):
        dis_i = int(fdistan.fdistan(k[0],k[1]))
        n_l.append(dis_i)
        ts = dis_i+ts
    print n_l
    ts = float(ts)
    tmp_n = [i/ts for i in n_l]
    print tmp_n
    tmp_mul = reduce(mul,tmp_n)
    print tmp_mul
    basic_k = pow((limit_n/tmp_mul),1.0/len(in_l))
    print basic_k
    j = 0
    for k in list(in_l):
        n = int(tmp_n[j]*basic_k)
        print n
        var_l.append(sorted(np.random.uniform(k[0], k[1], n)))
        j = j + 1
    input_l = []
    for element in itertools.product(*var_l):
        inp = list(element)
        input_l.append(inp)
    print len(input_l)
    input_l = chunks(input_l,8)
    lf1.extend([f1] * len(input_l))
    lf2.extend([f2] * len(input_l))
    tmp_res = p.map(test_two_fun, lf1, lf2, input_l)
    tmp_res = sorted(tmp_res, reverse=True)
    end_time = time.time() - start_time
    return (tmp_res[0][0], tmp_res[0][1], end_time)

def para_random_res_tlimit_fit2(in_l,f1,f2,limit_n,cpu_n):
    start_time = time.time()
    var_l = []
    lf1 = []
    lf2 = []
    p = Pool(cpu_n)
    n_l = []
    ts = 0
    i = 0
    for k in list(in_l):
        dis_i = int(fdistan.fdistan(k[0],k[1]))
        n_l.append(dis_i)
        ts = dis_i+ts
    ts = float(ts)
    tmp_n = [i/ts for i in n_l]
    tmp_mul = reduce(mul,tmp_n)
    print tmp_mul
    basic_k = pow((limit_n/tmp_mul),1.0/len(in_l))
    print basic_k
    j = 0
    for k in list(in_l):
        n = int(tmp_n[j]*basic_k)
        print n
        var_l.append(sorted(np.random.uniform(k[0], k[1], n)))
        j = j + 1
    input_l = []
    for element in itertools.product(*var_l):
        inp = list(element)
        input_l.append(inp)
    print len(input_l)
    input_l = chunks(input_l,8)
    lf1.extend([f1] * len(input_l))
    lf2.extend([f2] * len(input_l))
    tmp_res = p.map(test_two_fun_fit2, lf1, lf2, input_l)
    tmp_res = sorted(tmp_res, reverse=True)
    end_time = time.time() - start_time
    return (tmp_res[0][0], tmp_res[0][1], end_time)


def para_random_res_tmp(in_l,f1,f2,per):
    start_time = time.time()
    var_l = []
    for k in list(in_l):
        n = int(per*fdistan.fdistan(k[0],k[1]))
        print n
        var_l.append(sorted(np.random.uniform(k[0], k[1], n)))
    input_l = []
    vp = 0
    tmp_var_l =  []
    tmp_input_l = []
    for vars in var_l:
        tmp_var_l.append(vars)
        if vp == 1:
            for iters in itertools.product(*tmp_var_l):
                tmp_input_l.append(iters)
            tmp_var_l = []
            input_l.append(tmp_input_l)
            vp = 0
        else:
            vp = vp +1
    if tmp_var_l != []:
        input_l.append(tmp_var_l)
    p = Pool()
    lf1 = []
    lf2 = []
    re_err_1 =[]
    for inp in range(0,len(input_l[1][0]),8):
        one_test_l = []
        tmp_test_l = []
        for ti in range(0,8):
            if inp+ti < len(input_l[1][0]):
                for tnp in itertools.product(input_l[0], [input_l[1][0][inp + ti]]):
                    tmp_tnp = list(tnp[0])
                    tmp_tnp.append(tnp[1])
                    tmp_test_l.append(tmp_tnp)
                one_test_l.append(tmp_test_l)
        lf1.extend([f1]*len(one_test_l))
        lf2.extend([f2]*len(one_test_l))
        tmp_res = p.map(test_two_fun, lf1, lf2, one_test_l)
        tmp_res = sorted(tmp_res, reverse=True)
        re_err_1.append(tmp_res[0])
    error_l = sorted(re_err_1, reverse=True)
    end_time = time.time()-start_time
    return (error_l[0][0], error_l[0][1], end_time)

def random_res(in_l,f1,f2,tlimit,exe_time):
    start_time = time.time()
    len_inl = len(in_l)
    l_var = []
    in_var= in_l
    for i in in_var:
        tmp_l = depart(i[0], i[1])
        l_var.append(tmp_l)
    l_confs = []
    for element in itertools.product(*l_var):
        l_confs.append(element)
    len_confs = len(l_confs)
    print len_inl
    print len_confs*pow(20,len_inl)*tlimit/(exe_time)
    n = int(pow((len_confs*pow(20.0,float(len_inl))*tlimit/(exe_time)),1.0/float(len_inl)))
    print n
    var_l = []
    for k in list(in_l):
        var_l.append(sorted(np.random.uniform(k[0], k[1], n)))
    input_l = []
    for element in itertools.product(*var_l):
        input_l.append(element)
    res_mp = test_mp_fun(f1, input_l)
    # res_mp = paral_proc_mppl(X,cof,200,1)
    res_d = test_gsl_fun(f2, input_l)
    # res_d = paral_proc_pl(X,cof,len(cof)-1,1)
    re_err_1 = [(np.fabs(fdistan.fdistan(op_f[0], float(op_r[0]))), op_f[1]) for op_f, op_r in zip(res_d, res_mp)]
    error_l = sorted(re_err_1, reverse=True)
    end_time = time.time()-start_time
    return (error_l[0][0], error_l[0][1], end_time)
def fine_search(in_var,f1,f2,tlimit):
    b_start_time = time.time()
    l_var = []
    for i in in_var:
        tmp_l = binary_depart(i)
        l_var.append(tmp_l)
    ini_confs = []
    for element in itertools.product(*l_var):
        ini_confs.append(element)
    max_err = 0.0
    tmp_max = 0.0
    max_x =0.0
    m = 0
    p = 0
    tmp_time = 0.0
    temp_max = 0.0
    one_temp_time = 0.0
    tmp_x = 0.0
    while tlimit> tmp_time + one_temp_time:
        b_tmp_time = time.time()
        p = 0
        for c in ini_confs:
            p = p+1
            var_l = []
            for k in c:
                var_l.append(sorted(np.random.uniform(k[0], k[1], 10)))
            input_l = []
            for element in itertools.product(*var_l):
                input_l.append(element)
            res_mp = test_mp_fun(f1, input_l)
            # res_mp = paral_proc_mppl(X,cof,200,1)
            res_d = test_gsl_fun(f2, input_l)
            re_err_1 = [(distan_cal(op_f[0], float(op_r[0])), op_f[1]) for op_f, op_r in zip(res_d, res_mp)]
            error_l = sorted(re_err_1, reverse=True)
            max_err = error_l[0][0]
            max_x = error_l[0][1]
            if max_err>tmp_max:
                tmp_confs = produce_new_confs(c)
                tmp_max = max_err
                tmp_x = max_x
            if p==len(ini_confs):
                ini_confs = tmp_confs
        one_temp_time = time.time() - b_tmp_time
        tmp_time = time.time() - b_start_time
    return tmp_max, tmp_x



def pfulp_res_mutli(in_var,f1,f2,tlimit):
    mp.prec = 100
    start_time = time.time()
    l_var = []
    for i in in_var:
        tmp_l = depart(i[0], i[1])
        l_var.append(tmp_l)
    l_confs = []
    for element in itertools.product(*l_var):
        l_confs.append(element)
    next_tmp_l = []
    print len(l_confs)
    print l_confs
    for j in l_confs:
        var_l = []
        for k in j:
            var_l.append(sorted(np.random.uniform(k[0], k[1], 40)))
        input_l = []
        for element in itertools.product(*var_l):
            input_l.append(element)
        res_d = test_gsl_fun(f2, input_l)
        ulpx_f = [(condition(list(op_f[1]), op_f[0], f2), op_f[1])
                  for op_f in res_d]
        ulpx_f = sorted(ulpx_f, reverse=True)
        ds, max_x = search_around(f1, f2, ulpx_f)
        next_tmp_l.append((ds, max_x, j))
    tmp_time = time.time() - start_time
    print tmp_time
    k = len(next_tmp_l)
    if k > 10:
        k = min(int(len(next_tmp_l) / 2), 10)
    next_tmp_l = sorted(next_tmp_l, reverse=True)[0:k]
    print next_tmp_l[0]
    next_tmp_l_2 = []
    one_time = (tlimit-tmp_time)/(len(next_tmp_l))
    print one_time
    print tlimit
    print tmp_time
    print len(next_tmp_l)
    print next_tmp_l
    for i in next_tmp_l:
        gen_l = produce_interval(i[1], i[2])
        print gen_l
        max_l = fine_search(gen_l, f1, f2,one_time)
        next_tmp_l_2.append(max_l)
    next_tmp_l_2 = sorted(next_tmp_l_2, reverse=True)
    end_time = time.time() - start_time
    if len(next_tmp_l_2) == 0:
        return [[0.0, 0.0, [0.0, 0.0]], [0.0, 0.0, [0.0, 0.0]]], end_time
    return next_tmp_l_2[0], end_time
def test_pulp(f1,f2,input_l,cof):
    res_d = test_gsl_fun(f2, input_l)
    ulpx_f = [(condition(list(op_f[1]), op_f[0], f2), op_f[1])
              for op_f in res_d]
    ulpx_f = sorted(ulpx_f, reverse=True)
    ds, max_x = search_around(f1, f2, ulpx_f)
    return  (ds,max_x,cof)
def para_pfulp_res_mutli(in_var,f1,f2,tlimit,cpu_n):
    mp.prec = 100
    start_time = time.time()
    l_var = []
    for i in in_var:
        tmp_l = depart(i[0], i[1])
        l_var.append(tmp_l)
    l_confs = []
    for element in itertools.product(*l_var):
        l_confs.append(element)
    next_tmp_l = []
    print len(l_confs)
    print l_confs
    p = Pool(cpu_n)
    all_input_l = []
    lf1 = []
    lf2 = []
    for j in l_confs:
        var_l = []
        for k in j:
            var_l.append(sorted(np.random.uniform(k[0], k[1], 20)))
        input_l = []
        for element in itertools.product(*var_l):
            input_l.append(element)
        all_input_l.append(input_l)
    lf1.extend([f1] * len(all_input_l))
    lf2.extend([f2] * len(all_input_l))
    res = p.map(test_pulp, lf1, lf2, all_input_l,l_confs)
    next_tmp_l = res
    tmp_time = time.time() - start_time
    print tmp_time
    k = len(next_tmp_l)
    if k > 10:
        k = min(int(len(next_tmp_l) / 2), 10)
    next_tmp_l = sorted(next_tmp_l, reverse=True)[0:k]
    print next_tmp_l[0]
    next_tmp_l_2 = []
    one_time = (tlimit-tmp_time)/(k)
    print one_time
    print tlimit
    print len(next_tmp_l)
    print next_tmp_l
    time_list = []
    gen_l = []
    for i in next_tmp_l:
        tmp_gen_l = produce_interval(i[1], i[2])
        gen_l.append(tmp_gen_l)
    lf1.extend([f1] * len(gen_l))
    lf2.extend([f2] * len(gen_l))
    time_list.extend([one_time]*len(gen_l))
    print gen_l
    res = p.map(fine_search,gen_l,lf1,lf2,time_list)
    print res
    next_tmp_l_2 = res
    next_tmp_l_2 = sorted(next_tmp_l_2, reverse=True)
    print next_tmp_l_2
    end_time = time.time() - start_time
    if len(next_tmp_l_2) == 0:
        return [[0.0, 0.0, [0.0, 0.0]], [0.0, 0.0, [0.0, 0.0]]], end_time
    return next_tmp_l_2[0], end_time
def binary_depart(in_l):
    mid = in_l[0] + (in_l[1]-in_l[0])/2.0
    return ([in_l[0],mid],[mid,in_l[1]])

def produce_new_confs(in_var):
    l_var = []
    for i in in_var:
        tmp_l = binary_depart(i)
        l_var.append(tmp_l)
    new_confs = []
    for element in itertools.product(*l_var):
        new_confs.append(element)
    return new_confs


def distan_cal(a,b):
    return np.fabs(fdistan.fdistan(a,b))

def binary_config_test(f1,f2,c):
    var_l = []
    for k in c:
        var_l.append(sorted(np.random.uniform(k[0], k[1], 60)))

    input_l = []
    for element in itertools.product(*var_l):
        input_l.append(element)
    res_mp = test_mp_fun(f1, input_l)
    # res_mp = paral_proc_mppl(X,cof,200,1)
    res_d = test_gsl_fun(f2, input_l)
    re_err_1 = [(distan_cal(op_f[0], float(op_r[0])), op_f[1]) for op_f, op_r in zip(res_d, res_mp)]
    error_l = sorted(re_err_1, reverse=True)
    return (error_l[0][0],error_l[0][1],c)

def binary_res_time_limit(in_var,f1,f2,tlimit):
    b_start_time = time.time()
    mp.prec = 100
    l_var = []
    for i in in_var:
        tmp_l = binary_depart(i)
        l_var.append(tmp_l)
    ini_confs = []
    for element in itertools.product(*l_var):
        ini_confs.append(element)
    max_err = 0.0
    tmp_max = 0.0
    m = 0
    p = 0
    tmp_time = 0.0
    temp_max = 0.0
    one_temp_time = 0.0
    tmp_x = 0.0
    while tlimit> tmp_time+one_temp_time:
        b_tmp_time = time.time()
        p=0
        for c in ini_confs:
            p = p+1
            res = binary_config_test(f1,f2,c)
            max_err = res[0]
            max_x = res[1]
            if max_err>tmp_max:
                tmp_confs = produce_new_confs(c)
                tmp_x = max_x
                tmp_max = max_err
            if p==len(ini_confs):
                ini_confs = tmp_confs
        one_temp_time = time.time()-b_tmp_time
        tmp_time = time.time()-b_start_time
    end_time = time.time()-b_start_time
    return (tmp_max,tmp_x,end_time)

def para_binary_res_time_limit(in_var,f1,f2,tlimit,cpu_n):
    b_start_time = time.time()
    l_var = []
    for i in in_var:
        tmp_l = binary_depart(i)
        l_var.append(tmp_l)
    ini_confs = []
    for element in itertools.product(*l_var):
        ini_confs.append(element)
    max_err = 0.0
    tmp_max = 0.0
    m = 0
    tmp_time = 0.0
    temp_max = 0.0
    one_temp_time = 0.0
    tmp_x = 0.0
    p = Pool(cpu_n)
    while tlimit> tmp_time+one_temp_time:
        b_tmp_time = time.time()
        arg_l = []
        lf1 = []
        lf2 = []
        res = []
        for c in ini_confs:
            #p = Process(binary_config_test, args=(f1,f2,c))
            arg_l.append(c)
            lf1.append(f1)
            lf2.append(f2)
        for r in res:
            print r.get()
        res = p.map(binary_config_test,lf1,lf2,arg_l)
        res = sorted(res, reverse=True)
        ini_confs = produce_new_confs(res[0][2])
        tmp_x = res[0][1]
        tmp_max = res[0][0]
        one_temp_time = time.time()-b_tmp_time
        tmp_time = time.time()-b_start_time
    end_time = time.time()-b_start_time
    return (tmp_max,tmp_x,end_time)

def para_binary_pfulp_test(f1,f2,c):
    var_l = []
    for k in c:
        var_l.append(sorted(np.random.uniform(k[0], k[1], 10)))
    input_l = []
    for element in itertools.product(*var_l):
        input_l.append(element)
    res_d = test_gsl_fun(f2, input_l)
    ulpx_f = [(condition(list(op_f[1]), op_f[0], f2), op_f[1])
              for op_f in res_d]
    ulpx_f = sorted(ulpx_f, reverse=True)
    ds, max_x = search_around(f1, f2, ulpx_f)
    gen_l = produce_interval(max_x, c)
    max_err, max_x = fine_search_tmp(gen_l, f1, f2)
    return (max_err, max_x,c)

def para_binary_pfulp_res_mutli(in_var,f1,f2,tlimit,cpu_n):
    b_start_time = time.time()
    mp.prec = 100
    l_var = []
    for i in in_var:
        tmp_l = binary_depart(i)
        l_var.append(tmp_l)
    ini_confs = []
    for element in itertools.product(*l_var):
        ini_confs.append(element)
    tmp_max = 0.0
    tmp_time = 0.0
    tmp_x = 0.0
    one_temp_time = 0.0
    tmp_confs = ini_confs
    p = Pool(cpu_n)
    tmp_val = 0.0
    while tlimit > tmp_time+one_temp_time:
        b_tmp_time = time.time()
        arg_l = []
        lf1 = []
        lf2 = []
        for c in ini_confs:
            arg_l.append(c)
            lf1.append(f1)
            lf2.append(f2)
        res = p.map(para_binary_pfulp_test, lf1, lf2, arg_l)
        res = sorted(res, reverse=True)
        ini_confs = produce_new_confs(res[0][2])
        tmp_x = res[0][1]
        tmp_max = res[0][0]
        if (tmp_val==tmp_max):
            if dp == 10:
                break
            else:
                dp = dp + 1
        else:
            tmp_val = tmp_max
            dp = 0
        one_temp_time = time.time() - b_tmp_time
        tmp_time = time.time() - b_start_time
    end_time = time.time() - b_start_time
    return (tmp_max, tmp_x, end_time)

def binary_pfulp_res_mutli(in_var,f1,f2,tlimit):
    b_start_time = time.time()
    mp.prec = 100
    l_var = []
    for i in in_var:
        tmp_l = binary_depart(i)
        l_var.append(tmp_l)
    ini_confs = []
    for element in itertools.product(*l_var):
        ini_confs.append(element)
    tmp_max = 0.0
    tmp_time = 0.0
    tmp_x = 0.0
    one_temp_time = 0.0
    tmp_confs = ini_confs
    tmp_val = 0.0
    tmp = 0
    while tlimit > tmp_time+one_temp_time:
        b_tmp_time = time.time()
        dp = 0
        for c in ini_confs:
            dp = dp + 1
            var_l = []
            for k in c:
                var_l.append(sorted(np.random.uniform(k[0], k[1], 10)))
            input_l = []
            for element in itertools.product(*var_l):
                input_l.append(element)
            #res_mp = test_mp_fun(f1, input_l)
            res_d = test_gsl_fun(f2, input_l)
            ulpx_f = [(condition(list(op_f[1]), op_f[0], f2), op_f[1])
                      for op_f in res_d]
            ulpx_f = sorted(ulpx_f, reverse=True)
            ds, max_x = search_around(f1, f2, ulpx_f)
            gen_l = produce_interval(max_x, c)
            max_err,max_x = fine_search_tmp(gen_l, f1, f2)
            if max_err > tmp_max:
                tmp_confs = produce_new_confs(c)
                tmp_x = max_x
                tmp_max = max_err
            if dp == len(ini_confs):
                ini_confs = tmp_confs
        if (tmp_val == tmp_max):
            if tmp == 10:
                break
            else:
                tmp = tmp + 1
        else:
            tmp_val = tmp_max
            tmp = 0
        one_temp_time = time.time() - b_tmp_time
        tmp_time = time.time() - b_start_time
    end_time = time.time() - b_start_time
    return (tmp_max, tmp_x, end_time)


#laguerre_2 = lambda a,b: laguerre(2,a,b)
#print pfulp_res_mutli([[-50.0,50.0],[-50.0,50.0]],laguerre_2,sf.laguerre_2,140)
#print binary_pfulp_res_mutli([[0.0,50.0],[0.0,50.0]],laguerre_3,sf.laguerre_3,100)
#print pfulp_res_mutli([[0.0,50.0],[0.0,50.0]],hyp0f1,sf.hyperg_0F1,100)
#print pfulp_res_mutli([[0.0, 10.0],[0.0, 10.0],[0.0, 10.0], [-1.0, 1.0]],hyp2f1, sf.hyperg_2F1,100)
#print pfulp_res_mutli([[0.0,50.0],[0.0,50.0]],besselk, sf.bessel_Knu,60)
#print binary_pfulp_res_mutli([[0.0,50.0],[0.0,50.0]],hyp0f1,sf.hyperg_0F1,100)
#print binary_pfulp_res_mutli([[0.0,50.0],[0.0,50.0]],besselk, sf.bessel_Knu,60)
#print binary_pfulp_res_mutli([[0.0,50.0],[0.0,50.0]],bessely, sf.bessel_Ynu,60)
#print binary_pfulp_res_mutli([[0.0, 10.0],[0.0, 10.0],[0.0, 10.0], [-1.0, 1.0]],hyp2f1, sf.hyperg_2F1,100)
#print binary_res_time_limit([[0.0,50.0],[0.0,50.0]],hyp0f1,sf.hyperg_0F1,50)


#print produce_new_confs(([1,2],[3,4]))

#print binary_res_time_limit([[0.0,50.0],[0.0,50.0]],laguerre_3,sf.laguerre_3,300)
#print binary_res_time_limit( [[0.0, 10.0],[0.0, 10.0],[0.0, 10.0], [-1.0, 1.0]],hyp2f1, sf.hyperg_2F1,200)
#print binary_res_time_limit([[0.0,50.0],[0.0,50.0]],besselk, sf.bessel_Knu,80)