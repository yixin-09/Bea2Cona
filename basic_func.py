# -*- coding: utf-8 -*-
#from gsl_fun import *
from mpmath import *
from scipy.misc import derivative
import numpy as np
import fdistan
import time
import xlwt
import pickle
def load_object(fn):
    f = open(fn,'rb')
    data = pickle.load(f)
    f.close()
    return data
def test_gsl_fun(f,x):
    gsl_l = []
    for i in x:
        gsl_l.append((f(i),i))
    return gsl_l
def test_mp_fun(f,x):
    mp_l = []
    for i in x:
        mp_l.append((float(f(i)),i))
    return mp_l
def condition(a,b,f):
    i = b*2.2204460492503131e-16
    j = a*2.2204460492503131e-16
    der = derivative(f, a)
    if der == 0.0:
        y = fabs((f(a + j) - f(a)) / (i))
    else:
        y = fabs(der * j / (i))
    return float(y)

def binary_res(in_l,f1,f2,iter,n):
    #输入区间上下界
    in_max = in_l[1]
    in_min = in_l[0]
    max_err = 0.0
    #考虑迭代次数iter，每个区间取点数目n
    for i in range(0,iter):
        mid = (in_max-in_min)/2.0+in_min

        #计算区间1的最大误差
        #得到输入X
        X = sorted(np.random.uniform(in_min, mid, n))
        #得到高精度输出
        res_mp = test_mp_fun(f1,X)
        #res_mp = paral_proc_mppl(X,cof,200,1)
        #得到double输出
        res_d = test_gsl_fun(f2,X)
        #res_d = paral_proc_pl(X,cof,len(cof)-1,1)
        #根据公式，得到相对误差nmp.abs((op_f[0] - op_r[0]) / (nmp.nextafter(float(op_r[0]), float(op_r[0]) + 1) - float(op_r[0])))
        re_err_1=[(fdistan.fdistan(op_f[0],float(op_r[0])),op_f[1]) for op_f, op_r in zip(res_d, res_mp)]
        mxer = sorted(re_err_1, reverse=True)[0][0]
        #re_err_1 = [(nmp.abs((op_f[0]-op_r[0])/(nmp.nextafter(float(op_r[0]),float(op_r[0])+1)-float(op_r[0]))),op_f[1]) for op_f,op_r in zip(res_d,res_mp)]

        #计算区间2的最大误差
        # 得到输入X
        X = sorted(np.random.uniform(mid, in_max, n))
        # 得到高精度输出
        res_mp = test_mp_fun(f1, X)
        # res_mp = paral_proc_mppl(X,cof,200,1)
        # 得到double输出
        res_d = test_gsl_fun(f2, X)
        # res_d = paral_proc_pl(X,cof,len(cof)-1,1)
        # 根据公式，得到相对误差nmp.abs((op_f[0] - op_r[0]) / (nmp.nextafter(float(op_r[0]), float(op_r[0]) + 1) - float(op_r[0])))
        re_err_2 = [(fdistan.fdistan(op_f[0],float(op_r[0])),op_f[1])
                    for op_f, op_r in
                    zip(res_d, res_mp)]
        #re_err_2 = [(nmp.abs((op_f[0]-op_r[0])/(nmp.nextafter(float(op_r[0]),float(op_r[0])+1)-float(op_r[0]))),op_f[1]) for op_f,op_r in zip(res_d,res_mp)]
        #分别得到两个区间的最大误差：
        re_err_3 = sorted(re_err_2,reverse=True)
        x_1 = 0.0
        x_2 = 0.0
        erl_1 = 0.0
        for j in re_err_1:
            if j[0] > erl_1:
                erl_1 = j[0]
                x_1 = j[1]
        erl_2 = 0.0
        for j in re_err_2:
            if j[0] > erl_2:
                erl_2 = j[0]
                x_2 = j[1]
        if(erl_1>erl_2):
            #如果区间1的相对误差最大值更大，那么继续在区间1内搜索
            in_min = in_min
            in_max = mid
            max_err = erl_1
            out_min = in_min
            out_max = in_max
            out_x = x_1
        else:
            in_min = mid
            in_max = in_max
            max_err = erl_2
            out_min = in_min
            out_max = in_max
            out_x = x_2
    return (max_err,out_x,[out_min,out_max])
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
def pfulp_res(in_l,f1,f2,n):
    # 输入区间上下界
    time_begin = time.time()
    in_max = in_l[1]
    in_min = in_l[0]
    #直接将区间进行分段处理，分为n段
    mins=(in_max-in_min)/n
    tmp_l = depart(in_min,in_max)
    tmp = in_min
    max_err = 0.0
    next_tmp_l = []
    max_err_l = []
    #在每个分段区间内搜索最大值
    for j in tmp_l:
        max_err = 0
        max_x = 0
        #每段内取10*n个点
        X = sorted(np.random.uniform(j[0], j[1], 10*n))
        #得到double计算的结果
        # 得到double输出
        res_d = test_gsl_fun(f2,X)
        # 得到ulpx/ulpf(x)的近似值

        ulpx_f=[(condition(op_f[1],op_f[0],f2), op_f[1])
                  for op_f in res_d]
        #考虑评价标准：直接得到区间内最大值，并进入下一步筛选
        for i in ulpx_f:
            if i[0] > max_err:
                max_err = i[0]
                max_x = i[1]
        #保存max_err,max_x,j
        next_tmp_l.append((max_err,max_x,j))
    print (time.time() - time_begin)
    #按max_err对列表进行排序,并取前n/10个
    next_tmp_l = sorted(next_tmp_l,reverse=True)[0:int(n/10)]
    print next_tmp_l
    #重新对n/2个区间进行计算
    mins = mins/10.0
    next_tmp_l_2 = []
    for j in next_tmp_l:
        #对于max_err大于1的进行计算
        if j[0]>100:
            max_l = binary_res([j[1]-0.001, j[1]+0.001], f1,f2, 20, 30)
            next_tmp_l_2.append(max_l)
    next_tmp_l_2 = sorted(next_tmp_l_2,reverse=True)
    return next_tmp_l_2
#构建
def random_res(in_l,f1,f2,tlimit,exe_time):
    r_start_time = time.time()
    #输入区间上下界
    in_max = in_l[1]
    in_min = in_l[0]
    max_err = 0.0
    n = int(tlimit*1000/exe_time)
    # 得到输入X
    X = sorted(np.random.uniform(in_min, in_max, n))
    # 得到高精度输出
    res_mp = test_mp_fun(f1, X)
    # res_mp = paral_proc_mppl(X,cof,200,1)
    # 得到double输出
    res_d = test_gsl_fun(f2, X)
    # res_d = paral_proc_pl(X,cof,len(cof)-1,1)
    # 根据公式，得到相对误差nmp.abs((op_f[0] - op_r[0]) / (nmp.nextafter(float(op_r[0]), float(op_r[0]) + 1) - float(op_r[0])))
    re_err_1 = [(fdistan.fdistan(op_f[0], float(op_r[0])), op_f[1]) for op_f, op_r in zip(res_d, res_mp)]
    error_l = sorted(re_err_1,reverse=True)
    end_time = time.time()-r_start_time
    return (error_l[0][0],error_l[0][1],in_l,end_time)


def test_sig(f1,f2,x,y):
    d_x = f2(x,y)
    m_x = float(f1(x,y))
    print fdistan.fdistan(d_x,m_x)
    print d_x-m_x

domain_clausen = []

def to_two_pi(x):
    P1 = 4 * 7.85398125648498535156e-01
    P2 = 4 * 3.77489470793079817668e-08
    P3 = 4 * 2.69515142907905952645e-15
    TwoPi = 2*(P1 + P2 + P3)
    y = 2*floor(x/TwoPi)
    r = ((x - y*P1) - y*P2) - y*P3
    print r
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException


# output the results to excel
def out_to_excel(t_l):
    book = xlwt.Workbook()
    sheet = book.add_sheet("sheet1")
    sheet.write(0, 0, "functions")
    sheet.write(0, 1, "estimate absolute error")
    sheet.write(0, 2, "bin_max_relative error")
    sheet.write(0, 3, "bin absolute error")
    sheet.write(0, 4, "bin_input")
    sheet.write(0, 5, "ulp_max_relative error")
    sheet.write(0, 6, "ulp absolute error")
    sheet.write(0, 7, "ulp_input")
    sheet.write(0, 8, "rand_max_relative_error")
    sheet.write(0, 9, "rand absolute error")
    sheet.write(0, 10, "rand input")
    n = 1
    for t in t_l:
        sheet.write(n,0,t.name)
        sheet.write(n,1,t.e_err)
        sheet.write(n,2,t.t_abs_err)
        sheet.write(n,3, t.p_abs_err)
        sheet.write(n,4,t.t_res_err)
        sheet.write(n,5,t.p_res_err)
        sheet.write(n,6,t.p_res_err/t.t_res_err)
        sheet.write(n,7,t.input)
        n = n+1
    book.save("test_ICSE15.xls")


def distan_cal(a,b):
    return np.fabs(fdistan.fdistan(a,b))
def binary_res_time_limit(in_l,f1,f2,tlimit):
    b_start_time = time.time()
    #输入区间上下界
    in_max = in_l[1]
    in_min = in_l[0]
    max_err = 0.0
    basic_len = in_max - in_min
    m = 0
    interval_len = len(depart(in_min, in_max))
    n = interval_len*100
    #考虑迭代次数iter，每个区间取点数目n
    while 1>0:
        mid = (in_max-in_min)/2.0+in_min
        b_tmp_time = time.time()
        if n < 1000:
            n = 1000
        else:
            n = n / 2
        #计算区间1的最大误差
        #得到输入X
        X = sorted(np.random.uniform(in_min, mid, n))
        #得到高精度输出
        res_mp = test_mp_fun(f1,X)
        #res_mp = paral_proc_mppl(X,cof,200,1)
        #得到double输出
        res_d = test_gsl_fun(f2,X)
        #res_d = paral_proc_pl(X,cof,len(cof)-1,1)
        #根据公式，得到相对误差nmp.abs((op_f[0] - op_r[0]) / (nmp.nextafter(float(op_r[0]), float(op_r[0]) + 1) - float(op_r[0])))
        re_err_1=[(distan_cal(op_f[0],float(op_r[0])),op_f[1]) for op_f, op_r in zip(res_d, res_mp)]
        mxer = sorted(re_err_1, reverse=True)[0][0]
        #re_err_1 = [(nmp.abs((op_f[0]-op_r[0])/(nmp.nextafter(float(op_r[0]),float(op_r[0])+1)-float(op_r[0]))),op_f[1]) for op_f,op_r in zip(res_d,res_mp)]

        #计算区间2的最大误差
        # 得到输入X
        X = sorted(np.random.uniform(mid, in_max, n))
        # 得到高精度输出
        res_mp = test_mp_fun(f1, X)
        # res_mp = paral_proc_mppl(X,cof,200,1)
        # 得到double输出
        res_d = test_gsl_fun(f2, X)
        # res_d = paral_proc_pl(X,cof,len(cof)-1,1)
        # 根据公式，得到相对误差nmp.abs((op_f[0] - op_r[0]) / (nmp.nextafter(float(op_r[0]), float(op_r[0]) + 1) - float(op_r[0])))
        re_err_2 = [(distan_cal(op_f[0],float(op_r[0])),op_f[1])
                    for op_f, op_r in
                    zip(res_d, res_mp)]
        #re_err_2 = [(nmp.abs((op_f[0]-op_r[0])/(nmp.nextafter(float(op_r[0]),float(op_r[0])+1)-float(op_r[0]))),op_f[1]) for op_f,op_r in zip(res_d,res_mp)]
        #分别得到两个区间的最大误差：
        re_err_3 = sorted(re_err_2,reverse=True)
        x_1 = 0.0
        x_2 = 0.0
        erl_1 = 0.0
        for j in re_err_1:
            if j[0] > erl_1:
                erl_1 = j[0]
                x_1 = j[1]
        erl_2 = 0.0
        for j in re_err_2:
            if j[0] > erl_2:
                erl_2 = j[0]
                x_2 = j[1]
        temp_max = max_err
        if(erl_1>=erl_2):
            #如果区间1的相对误差最大值更大，那么继续在区间1内搜索
            in_min = in_min
            in_max = mid
            max_err = erl_1
            out_min = in_min
            out_max = in_max
            out_x = x_1
        else:
            in_min = mid
            in_max = in_max
            max_err = erl_2
            out_min = in_min
            out_max = in_max
            out_x = x_2
        if temp_max == max_err:
            m = m+1
            if m == 10:
                print end_time
                return (max_err, out_x, [out_min, out_max], end_time)
        else:
            m = 0
        tmp_one_time = time.time()-b_tmp_time
        end_time = time.time() - b_start_time
        if (tlimit)<end_time+tmp_one_time:
            print end_time
            return (max_err, out_x, [out_min, out_max], end_time)


def fine_search(in_l,f1,f2,tlimit):
    b_start_time = time.time()
    #输入区间上下界
    in_max = in_l[1]
    in_min = in_l[0]
    max_err = 0.0
    basic_len = in_max - in_min
    m = 0
    interval_len = len(depart(in_min, in_max))
    n = interval_len*100
    #考虑迭代次数iter，每个区间取点数目n
    while 1>0:
        mid = (in_max-in_min)/2.0+in_min
        b_tmp_time = time.time()
        if n/2 < 100:
            n = 100
        else:
            n = n/2
        #计算区间1的最大误差
        #得到输入X
        X = sorted(np.random.uniform(in_min, mid, n))
        #得到高精度输出
        res_mp = test_mp_fun(f1,X)
        #res_mp = paral_proc_mppl(X,cof,200,1)
        #得到double输出
        res_d = test_gsl_fun(f2,X)
        #res_d = paral_proc_pl(X,cof,len(cof)-1,1)
        #根据公式，得到相对误差nmp.abs((op_f[0] - op_r[0]) / (nmp.nextafter(float(op_r[0]), float(op_r[0]) + 1) - float(op_r[0])))
        re_err_1=[(distan_cal(op_f[0],float(op_r[0])),op_f[1]) for op_f, op_r in zip(res_d, res_mp)]

        X = sorted(np.random.uniform(mid, in_max, n))
        # 得到高精度输出
        res_mp = test_mp_fun(f1, X)
        # res_mp = paral_proc_mppl(X,cof,200,1)
        # 得到double输出
        res_d = test_gsl_fun(f2, X)
        # res_d = paral_proc_pl(X,cof,len(cof)-1,1)
        # 根据公式，得到相对误差nmp.abs((op_f[0] - op_r[0]) / (nmp.nextafter(float(op_r[0]), float(op_r[0]) + 1) - float(op_r[0])))
        re_err_2 = [(distan_cal(op_f[0],float(op_r[0])),op_f[1])
                    for op_f, op_r in
                    zip(res_d, res_mp)]
        x_1 = 0.0
        x_2 = 0.0
        erl_1 = 0.0
        for j in re_err_1:
            if j[0] > erl_1:
                erl_1 = j[0]
                x_1 = j[1]
        erl_2 = 0.0
        for j in re_err_2:
            if j[0] > erl_2:
                erl_2 = j[0]
                x_2 = j[1]
        temp_max = max_err
        if(erl_1>=erl_2):
            #如果区间1的相对误差最大值更大，那么继续在区间1内搜索
            in_min = in_min
            in_max = mid
            max_err = erl_1
            out_min = in_min
            out_max = in_max
            out_x = x_1
        else:
            in_min = mid
            in_max = in_max
            max_err = erl_2
            out_min = in_min
            out_max = in_max
            out_x = x_2
        if temp_max == max_err:
            m = m+1
            if m == 5:
                return (max_err, out_x, [out_min, out_max])
        else:
            m = 0
        tmp_one_time = time.time()-b_tmp_time
        b_end_time = time.time() - b_start_time
        if (tlimit)<b_end_time+tmp_one_time:
            return (max_err, out_x, [out_min, out_max])

def exprel_2(x):
    if x == 0 :
        return 1.0
    return 2*(expm1(x)-x)/power(x,2)

def synchrotron_1(x):
    return x*quad(lambda t: besselk(5.0/3.0,t),[x,inf])
def get_distan(f1,f2,x):
    m_x = float(f1(x))
    f_x = f2(x)
    return np.fabs(fdistan.fdistan(m_x,f_x))
# generate a small interval around x, the t would influnce the test time, 0.001 value not good for all function but enough
def produce_interval(x,k):
    a = 0.001*np.fabs(k[1]-k[0])
    return [x-a,x+a]

def pfulp_test_minus(in_l,f1,f2,tlimit,exe_time):
    #输入区间上下界
    in_max = in_l[1]
    in_min = in_l[0]
    max_err = 0.0
    n = int(tlimit*1000/exe_time)
    # 得到输入X
    X = sorted(np.random.uniform(in_min, in_max, n))
    # 得到高精度输出
    res_mp = test_mp_fun(f1, X)
    # res_mp = paral_proc_mppl(X,cof,200,1)
    # 得到double输出
    res_d = test_gsl_fun(f2, X)
    # res_d = paral_proc_pl(X,cof,len(cof)-1,1)
    # 根据公式，得到相对误差nmp.abs((op_f[0] - op_r[0]) / (nmp.nextafter(float(op_r[0]), float(op_r[0]) + 1) - float(op_r[0])))
    re_err_1 = [(fdistan.fdistan(op_f[0], float(op_r[0])), op_f[1]) for op_f, op_r in zip(res_d, res_mp)]
    error_l = sorted(re_err_1,reverse=True)
    return (error_l[0][0],error_l[0][1],in_l)
def search_around(f1,f2,ulp_l):
    temp_err = 0.0
    tmp_x = 0.0
    for i in ulp_l[0:100]:
        mx = i[1]
        ds = get_distan(f1, f2, mx)
        if ds > temp_err:
            temp_err = ds
            tmp_x = mx
    return temp_err,tmp_x

def pfulp_res_time_limit(in_l,f1,f2,re_max,tlimit,exe_time):
    start_time = time.time()
    # 输入区间上下界
    in_max = in_l[1]
    in_min = in_l[0]
    tmp_l = depart(in_min, in_max)
    next_tmp_l = []
    #在每个分段区间内搜索最大值
    for j in tmp_l:
        max_err = 0.0
        max_x = 0.0
        #每段内取10*n个点
        X = sorted(np.random.uniform(j[0], j[1], 3000))
        #得到double计算的结果
        # 得到double输出
        res_d = test_gsl_fun(f2,X)
        # 得到ulpx/ulpf(x)的近似值

        ulpx_f=[(condition(op_f[1],op_f[0],f2), op_f[1])
                  for op_f in res_d]
        ulpx_f = sorted(ulpx_f,reverse=True)
        #考虑评价标准：直接得到区间内最大值，并进入下一步筛选
        max_x = ulpx_f[0][1]
        ds,max_x = search_around(f1,f2,ulpx_f)
        next_tmp_l.append((ds, max_x, j))

    tmp_time = time.time() - start_time
    print tmp_time
    k =len(tmp_l)
    if k >15:
        k = min(int(len(tmp_l)/2),15)
    next_tmp_l = sorted(next_tmp_l,reverse=True)[0:k]
    #重新对n/2个区间进行计算
    next_tmp_l_2 = []
    for i in next_tmp_l:
        gen_l = produce_interval(float(i[1]),i[2])
        max_l = fine_search(gen_l, f1,f2,(tlimit-tmp_time)/k)
        next_tmp_l_2.append(max_l)
    next_tmp_l_2 = sorted(next_tmp_l_2,reverse=True)
    end_time = time.time()-start_time
    if len(next_tmp_l_2)==0:
        return [[0.0, 0.0, [0.0, 0.0]], [0.0, 0.0, [0.0, 0.0]]],end_time
    return next_tmp_l_2,end_time

def test_gsl_clausen():
    # run 15 minutes
    # besselj
    mp.prec = 53
    in_l = [0, 1700]
    besselj1 = lambda t: besselj(1, t)
    time_begin = time.time()
    print "*********"
    for i in range(0,9):
        print binary_res_time_limit(binary_res_time_limit, (in_l, besselj1, gsl_sf_bessel_J1, 1000), {}, 1)
    print "**********"
    print time.time() - time_begin
    time_begin = time.time()
    print pfulp_res_time_limit(in_l, besselj1, gsl_sf_bessel_J1, 10)
    print time.time() - time_begin

def pfun(f1,f2,inp):
    print f1(inp)
    print f2(inp)
    print diff(f1, inp)
    z = condition(inp, f2(inp), f2)
    print z
    ds = get_distan(f1, f2, inp)
    print ds
def get_result_list(f,x):
    tmp_l = []
    for i in x:
        tmp_l.append(float(f(i)))
    return tmp_l

#print get_distan(loggamma,gsl_sf_lngamma,2.0160486462175982)


# Airy function Ai(x), Ai'(x) and int_0^x Ai(t) dt on the real line



