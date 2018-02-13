# -*- coding: utf-8 -*-
# basic functions for search_algorithm
from mpmath import *
#from gsl_fun import *
from scipy import special
import scipy.special as sc
import time
import numpy as np
import math
#import fdistan
import itertools
import struct
from scipy.misc import derivative
from pygsl.testing import sf
mp.prec = 200
r_airyai = np.frompyfunc(airyai,1,1)
f_airyai = lambda x: special.airy(x)[0]
gsl_airy_ai = lambda t: sf.airy_Ai(t, 1)
g_airyai = np.frompyfunc(gsl_airy_ai,1,1)
r_struve = np.frompyfunc(struveh,2,1)
f_struve = special.struve
r_besselj = np.frompyfunc(besselj,2,1)
f_besselj = special.jv
r_hyp1f1 = np.frompyfunc(hyp1f1,3,1)
f_hyp1f1 = special.hyp1f1
besselj0 = lambda t: besselj(0, t)
r_besselj0 = np.frompyfunc(j0,1,1)
f_besselj0 = special.j0
bessely1 = lambda t: bessely(1, t)
r_bessely1 = np.frompyfunc(bessely1,1,1)
f_bessely1 = special.y1
g_bessely1 = np.frompyfunc(sf.bessel_Y1,1,1)
r_sici = np.frompyfunc(ci,1,1)
f_sici = lambda x: special.sici(x)[0]
legendre3 = lambda t: legendre(3, t)
eval_legendre3 = lambda t: special.eval_legendre(3, t)
r_legendre3 = np.frompyfunc(legendre3,1,1)
f_legendre3 = eval_legendre3
g_legendre3 =np.frompyfunc(sf.legendre_P3,1,1)
rf_l = [r_airyai,r_struve,r_besselj,r_hyp1f1,r_bessely1,r_legendre3]
fp_l = [f_airyai,f_struve,f_besselj,f_hyp1f1,f_bessely1,f_legendre3]
fg_l = [g_airyai,g_bessely1,g_legendre3]
rg_l = [r_airyai,r_bessely1,r_legendre3]
gsl_l = []
rf = rg_l[1]
fp = fg_l[1]
input_domain_list = [[[-823549.6645, 108]],[[0, 40], [0, 1000]],[[0,125],[0,125]],[[0,10],[0,10],[0,10]], [[0, 1.7e+10]],[[-1.7976931348623157e+10, 1.7976931348623157e+10]]]
input_domain_list2 = [[[-823549.6645, 108]],[[0, 1.7e+10]],[[-1.7976931348623157e+10, 1.7976931348623157e+10]]]
name_list = ["airyai","struve","besselj","hyp1f1","bessely1","legendre3"]
name_list2 = ["gairyai","gbessely1","glegendre3"]
#add EAGT 12 gsl functions
eagt_12f_name=['gsl_airy_ai','gsl_sf_Chi','gsl_sf_Ci','gsl_sf_bessel_J0','gsl_sf_bessel_J1','gsl_sf_bessel_Y0','gsl_sf_bessel_Y1','gsl_sf_eta','gsl_sf_gamma','gsl_sf_legendre_P2','gsl_sf_legendre_P3','gsl_sf_lngamma']
eagt_12f_tmpn=['gsl_airy_ai','gsl_sf_bessel_Y1','gsl_sf_bessel_Y0','gsl_sf_bessel_J1','gsl_sf_bessel_J0','gsl_sf_Chi','gsl_sf_gamma','gsl_sf_Ci','gsl_sf_lngamma','gsl_sf_legendre_P2','gsl_sf_legendre_P3']
gsl_airy_ai = lambda t: sf.airy_Ai(t, 0)
bessely1 = lambda t: bessely(1, t)
bessely0 = lambda t: bessely(0, t)
besselj1 = lambda t: besselj(1, t)
besselj0 = lambda t: besselj(0, t)
legendre3 = lambda t: legendre(3, t)
legendre2 = lambda t: legendre(2, t)
pyairyai = lambda z: sc.airy(z)[0]
pybessely1 = sc.y1
pybessely0 = sc.y0
pybesselj1 = sc.j1
pybesselj0 = sc.j0
pychi = lambda z: sc.shichi(z)[1]
pyeta = lambda x: (1.0-math.pow(2.0,1.0-x))*sc.zeta(x,1)
pygamma = sc.gamma
pyci = lambda z:sc.sici(z)[1]
pyloggamma = lambda z: (sc.loggamma(z)).real
pylegendre3 = lambda x: sc.eval_legendre(3, x)
pylegendre2 = lambda x: sc.eval_legendre(2, x)
rf_12_l = [airyai,bessely1,bessely0,besselj1,besselj0,chi,altzeta,gamma,ci,loggamma,legendre3,legendre2]
gf_12_l = [gsl_airy_ai,sf.bessel_Y1,sf.bessel_Y0,sf.bessel_J1,sf.bessel_J0,sf.Chi,sf.eta,sf.gamma,sf.Ci,sf.lngamma,sf.legendre_P3,sf.legendre_P2]
pf_12_l = [pyairyai,pybessely1,pybessely0,pybesselj1,pybesselj0,pychi,pyeta,pygamma,pyci,pyloggamma,pylegendre3,pylegendre2]
input_domain_12 = [[[-823549.6645, 102]],[[1.0, 1.7e10]],[[1.0, 1.7e10]],[[0, 1.7e+100]],[[0, 1.7e+100]],[[0, 700]],[[-168, 100]],[[-168, 168]],[[1.0, 823549]],[[0.0, 1000.0]],[[-1.7976931348623157e+10, 1.7976931348623157e+10]],[[-1.7976931348623157e+10, 1.7976931348623157e+10]]]
input_domain_12_py = [[[-823549.6645, 102]],[[1.0, 1.7e10]],[[1.0, 1.7e10]],[[0, 1.7e+100]],[[0, 1.7e+100]],[[0, 700]],[[1, 100]],[[0, 168]],[[1.0, 823549]],[[0.0, 1000.0]],[[-1.7976931348623157e+10, 1.7976931348623157e+10]],[[-1.7976931348623157e+10, 1.7976931348623157e+10]]]
# The iteration number
n_r_iter = 20
mp.pre = 200
x = 1.2414480316687057e+40
print pybesselj0(x)
print sf.bessel_J0(x)
print besselj0(x)

def produce_one_input(i):
    var_l = []
    for k in i:
        var_l.append(sorted(np.random.uniform(k[0], k[1], 1)))
    input_l = []
    for element in itertools.product(*var_l):
        input_l.append(element)
    return input_l

def produce_n_input(i,n):
    var_l = []
    n = int(n)
    for k in i:
        var_l.append(sorted(np.random.uniform(k[0], k[1], n)))
    input_l = []
    for element in itertools.product(*var_l):
        input_l.append(element)
    return input_l




# fitness function:
# Relative_error_ulp
def glob_fitness_fun(x):
    res1 = float(rf(*x))
    res2 = float(fp(*x))
    if isnan(res2):
        res = 1.0
    else:
        #res = np.fabs(fdistan.fdistan(res1,res2))
        res = fdistan_two(res1,res2)
    if res != 0.0:
        res = 1.0 / res
    else:
        res = 1.0
    return res

def glob_fitness_fun_scalar(x):
    res1 = float(rf(*x))
    res2 = float(fp(*x))
    if isnan(res2):
        res = 1.0
    else:
        #res = np.fabs(fdistan.fdistan(res1,res2))
        res = fdistan_two(res1,res2)
    if res != 0.0:
        res = 1.0 / res
    else:
        res = 1.0
    return res

def glob_fun_scalar(f,x):
    gl_l = []
    for i in x:
        gl_l.append((f(i),i))
    return gl_l
def glob_fun_scalar_tmp(f,x):
    gl_l = []
    for i in x:
        gl_l.append((f(*i),i))
    return gl_l
# condition_ulp
def glob_fitness_fun_cu(x):
    res2 = float(fp(*x))
    res = condition(x,res2,fp)
    if res != 0.0:
        res = 1.0 / res
    else:
        res = 1.0
    return res
def floatToRawLongBits(value):
	return struct.unpack('Q', struct.pack('d', value))[0]
def longBitsToFloat(bits):
	return struct.unpack('d', struct.pack('Q', bits))[0]

def fdistan_two(a,b):
    fdistance = floatToRawLongBits(a) - floatToRawLongBits(b)
    return math.fabs(fdistance)

def fdistribution_partiion(in_min,in_max):
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

def rdistribution_partition(in_min,in_max):
    len_fp = len(fdistribution_partiion(in_min,in_max))
    one_len = (in_max-in_min)/float(len_fp)
    tmp_l = []
    tmp = in_min
    for i in range(len_fp):
        if tmp + one_len > in_max:
            tmp_l.append([tmp,in_max])
        else:
            tmp_l.append([tmp,tmp + one_len])
        tmp = tmp + one_len
    return tmp_l

def rpartition(input_domain,n):
    l_var = []
    for i in input_domain:
        tmp_l = rdistribution_partition(i[0], i[1])
        l_var.append(tmp_l)
    ini_confs = []
    for element in itertools.product(*l_var):
        ini_confs.append(element)
    return ini_confs
def fpartition(input_domain,n):
    l_var = []
    for i in input_domain:
        tmp_l = fdistribution_partiion(i[0], i[1])
        l_var.append(tmp_l)
    ini_confs = []
    for element in itertools.product(*l_var):
        ini_confs.append(element)
    return ini_confs
def nopartition(input_domain,n):
    return [tuple(input_domain),]

def binary_depart(in_l):
    mid = in_l[0] + (in_l[1]-in_l[0])/2.0
    return ([in_l[0],mid],[mid,in_l[1]])

def fbinary_depart(in_l):
    l = floatToRawLongBits(in_l[0])
    h = floatToRawLongBits(in_l[1])
    z = l/2.0+h/2.0
    z = longBitsToFloat(z)
    mid = z
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

def fproduce_new_confs(in_var):
    l_var = []
    for i in in_var:
        tmp_l = fbinary_depart(i)
        l_var.append(tmp_l)
    new_confs = []
    for element in itertools.product(*l_var):
        new_confs.append(element)
    return new_confs

def binary_partition(input_domain,n):
    return 0

def condition(a,b,f):
    i = b*2.2204460492503131e-16
    a = a[0]
    j = a*2.2204460492503131e-16
    y = math.fabs((f(a + j) - b) / (i))
    return float(y)
# fitness_function: relative_error_ulp, condition_ulp
def partial_derivative(func,b, var=0, point=[]):
    args = point[:]
    diff = math.fabs(args[var] * 2.2204460492503131e-14)
    if diff == 0:
        diff == 2.2204460492503131e-16
    def wraps(x):
        args[var] = x
        return func(*args)
    #return derivative(wraps, point[var],dx = 1e-8)
    tmp_der = (wraps(args[var] + diff) - b) / diff
    return tmp_der
def condition2(a,b,f):
    #l_x = []
    #der_x = []
    tmp = 0
    x = list(a)
    for i in range(0, len(x)):
        # l_x.append(fdistan.ulp(a[i]))
        # der_x.append(partial_derivative(f,i,a))
        tmp = tmp + 2.2204460492503131e-16 * x[i] * partial_derivative(f, b, i, x)
    ulp_b = b*2.2204460492503131e-16
    if ulp_b == 0:
        ulp_b = 2.2204460492503131e-16
    cond_num = math.fabs(tmp/ulp_b)
    return cond_num

def limit_time(times,input_domain):
    lens_part = len(fpartition(input_domain,0))
    fpart = fpartition(input_domain,0)
    nopart = nopartition(input_domain,0)
    times = len(fpart)*times
    es_time = time.clock()
    for i in nopart:
        x = produce_n_input(i,pow(times,1.0/len(i)))
        glob_fun_scalar(glob_fitness_fun,x)
    t1 = time.clock()-es_time
    es_time2 = time.clock()
    for i in nopart:
        x = produce_n_input(i,pow(times,1.0/len(i)))
        glob_fun_scalar(glob_fitness_fun_cu,x)
    t2 = time.clock()-es_time2
    return t1, times,t2,t1/t2

def limit_time2(times,input_domain):
    lens_part = len(fpartition(input_domain,0))
    fpart = fpartition(input_domain,0)
    nopart = nopartition(input_domain,0)
    es_time = time.clock()
    for i in fpart:
        x = produce_n_input(i,pow(times,1.0/len(i)))
        glob_fun_scalar(glob_fitness_fun,x)
    t1 = time.clock()-es_time
    es_time2 = time.clock()
    for i in fpart:
        x = produce_n_input(i,pow(times,1.0/len(i)))
        glob_fun_scalar(glob_fitness_fun_cu,x)
    t2 = time.clock()-es_time2
    return t1, lens_part*times,t2,t1/t2


def test_two_fun(rf,pf,input_domain):
    nopart = nopartition(input_domain, 0)
    es_time = time.clock()
    num = 1000
    for i in nopart:
        x = produce_n_input(i, num)
        glob_fun_scalar_tmp(rf, x)
    t1 = time.clock() - es_time
    es_time2 = time.clock()
    for i in nopart:
        x = produce_n_input(i, num)
        glob_fun_scalar_tmp(pf, x)
    t2 = time.clock() - es_time2
    return t1, t2, t1 / t2
#print test_two_fun(gamma,pygamma,[[0,168]])
#print relative_error_ulp(1,2)
# function example
# mpmath  SciPy
# airyai  airy
# airybi  airy
# ellipk  ellipk
# ellipfun(cd/
# testing when input out of input domain

#print limit_time(1,[[-823549.6645, 108]])
#print limit_time(10,[[-823549.6645, 108]])
#print glob_fitness_fun((-0.99753129014944353, 0.0054954190650676749))x
#x = [0.31257327,11.38414646]
#print 1.0/glob_fitness_fun(x)t
#print limit_time(100,[[0,125],[0,125]])
#print limit_time2(100,[[0,125],[0,125]])
