from basic_func import *
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import xlrd
import math
import os
import re
def load_object(fn):
    f = open(fn,'rb')
    data = pickle.load(f)
    f.close()
    return data
def data_cv(data):
    return np.std(data)/np.mean(data)
#def re_build(l):
def plot_box(l):
    l = map(list, zip(*l))
    l_df = pd.DataFrame(l)
    std_l_df = l_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    std_l_df.boxplot()
    plt.show()

def plot_bar(e1,e2,e3,name_list):
    n_groups = len(name_list)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.25

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = plt.barh(index, e3, bar_width,
                     alpha=0.3,
                     lw = 3,
                     error_kw=error_config,
                     label='RAND')

    rects2 = plt.barh(index + bar_width, e2, bar_width,
                     alpha=0.6,
                     lw = 1,
                     error_kw=error_config,
                     label='BGRT')
    rects2 = plt.barh(index + 2*bar_width, e1, bar_width,
                      alpha=1.0,
                      error_kw=error_config,
                      label='EAGT')

    plt.xlabel('Bit error')
    plt.yticks(index+bar_width, name_list)
    plt.legend()

    plt.tight_layout()
    plt.savefig("bit_err.pdf", format="pdf")
    plt.show()


def data_process(name_1,name_2):
    bgrt_list = load_data(name_1,0)
    eagt_list = load_data(name_2,0)
    #re_b  = re_build(bgrt_list)
    plot_box(bgrt_list)
    plot_box(eagt_list)

def load_data(filename,sheet_n):
    data = xlrd.open_workbook(filename)
    table = data.sheets()[sheet_n]
    error_list = []
    input_list = []
    time_list =[]
    for i in range(1,table.nrows):
        error_list.append(float(table.row_values(i)[1]))
        input_list.append(str(table.row_values(i)[2]))
        time_list.append(float(table.row_values(i)[3]))
    return [error_list,input_list,time_list]

#data_process("res_6_10/gsl_sf_Ci_BGRT.xls","res_6_10/gsl_sf_Ci_GULP.xls")

#data_process("res_6_10/gsl_sf_bessel_Y0_BGRT.xls","res_6_10/gsl_sf_bessel_Y0_GULP.xls")
def excel_data_process_mutilple(filename,out):
    file_list = sorted(os.listdir(filename))
    data_list = []
    for i in file_list:
        data_list.append(load_data(filename+"/"+i,0))
    print len(data_list)
    book = xlwt.Workbook()
    sheet = book.add_sheet("sheet1")
    ln = 0
    for i,j in zip(file_list,data_list):
        fname = i
        mean_err = np.mean(j[0])
        median_err = np.median(j[0])
        mean_time = np.mean(j[2])
        sheet.write(ln,0,i)
        sheet.write(ln,1,mean_err)
        sheet.write(ln,2,median_err)
        sheet.write(ln,3,mean_time)
        sheet.write(ln,4,mean_err/mean_time)
        sheet.write(ln,5,np.log2(mean_err))
        ln = ln+1
    book.save(out)
def excel_data_process():
    stable_list = [0,1,2,5,6,7,11]
    name_gsl_list = ['gsl_airy_ai','gsl_sf_Chi','gsl_sf_Ci','gsl_sf_bessel_J0','gsl_sf_bessel_J1','gsl_sf_bessel_Y0','gsl_sf_bessel_Y1','gsl_sf_eta','gsl_sf_gamma','gsl_sf_legendre_P2','gsl_sf_legendre_P3','gsl_sf_lngamma']
    file_list = sorted(os.listdir("last_res"))
    data_list = []
    for i in file_list:
        fn = "last_res/"+i
        data_list.append(load_data(fn,0))
    print len(data_list)
    bgrt_res = []
    eagt_res = []
    for j in range(0,24,2):
        bgrt_res.append(data_list[j])
        eagt_res.append(data_list[j+1])
    print len(bgrt_res)
    print len(eagt_res)
    err_bgrt_list = []
    time_bgrt_list = []
    input_bgrt_list = []
    for k in bgrt_res:
        err_bgrt_list.append(k[0])
        time_bgrt_list.append(k[2])
        input_bgrt_list.append(k[1])
    err_eagt_list = []
    time_eagt_list = []
    input_eagt_list = []
    for k in eagt_res:
        err_eagt_list.append(k[0])
        time_eagt_list.append(k[2])
        input_eagt_list.append(k[1])
    random_list = sorted(os.listdir("res_random_last"))
    random_data = []
    for i in random_list:
        random_data.append(load_data("res_random_last/"+i,0))
    err_random_list = []
    input_random_list = []
    time_random_list = []
    for k in random_data:
        err_random_list.append(k[0])
        time_random_list.append(k[2])
        input_random_list.append(k[1])

    mean_bgrt_err = []
    median_bgrt_err = []
    for i in err_bgrt_list:
        mean_bgrt_err.append(np.mean(i))
        median_bgrt_err.append(np.median(i))
    mean_bgrt_time = []
    for i in time_bgrt_list:
        mean_bgrt_time.append(np.mean(i))
    mean_bgrt_input = []
    for i in input_bgrt_list:
        mean_bgrt_input.append(np.mean(i))

    mean_eagt_err = []
    median_eagt_err = []
    for i in err_eagt_list:
        mean_eagt_err.append(np.mean(i))
        median_eagt_err.append(np.median(i))
    mean_eagt_time = []
    for i in time_eagt_list:
        mean_eagt_time.append(np.mean(i))
    mean_eagt_input = []
    for i in input_eagt_list:
        mean_eagt_input.append(np.mean(i))

    mean_random_err = []
    median_random_err = []
    for i in err_random_list:
        mean_random_err.append(np.mean(i))
        median_random_err.append(np.median(i))
    mean_random_time = []
    for i in time_random_list:
        mean_random_time.append(np.mean(i))
    mean_random_input = []
    for i in input_random_list:
        mean_random_input.append(np.mean(i))
    book = xlwt.Workbook()
    sheet = book.add_sheet("sheet1")
    for i in range(0,36,3):
        sheet.write(i+1, 0, "BGRT")
        sheet.write(i+1, 1, mean_bgrt_err[i/3])
        sheet.write(i+1, 3, mean_bgrt_time[i/3])
        sheet.write(i+1, 5, mean_bgrt_err[i/3]/mean_bgrt_time[i/3])
        sheet.write(i+1, 4, mean_bgrt_err[i/3]/mean_random_err[i/3])
        sheet.write(i+1, 2, median_bgrt_err[i/3])


        sheet.write(i, 0, "EAGT")
        sheet.write(i, 1, mean_eagt_err[i/3])
        sheet.write(i, 3, mean_eagt_time[i/3])
        sheet.write(i, 5, mean_eagt_err[i/3]/mean_eagt_time[i/3])
        sheet.write(i, 4, mean_eagt_err[i/3]/mean_random_err[i/3])
        sheet.write(i, 2, median_eagt_err[i/3])

        sheet.write(i + 2, 0, "RAND")
        sheet.write(i + 2, 1, mean_random_err[i/3])
        sheet.write(i + 2, 3, mean_random_time[i/3])
        sheet.write(i + 2, 5, mean_random_err[i/3]/mean_random_time[i/3])
        sheet.write(i + 2, 4, mean_random_err[i/3]/mean_random_err[i/3])
        sheet.write(i + 2, 2, median_random_err[i/3])

    book.save("data_output.xls")

    name_list = []
    e1 = []
    e2 = []
    e3 = []
    rev_l = range(0, 12)
    rev_l.reverse()
    for i in rev_l:
        name_list.append(name_gsl_list[i])
        e1.append(np.log2(mean_eagt_err[i]))
        e2.append(np.log2(mean_bgrt_err[i]))
        e3.append(np.log2(mean_random_err[i]))
    plot_bar(e1, e2, e3, name_list)



def gammastar_ser(x):
    y = 1.0 / (x * x)
    c0 = 1.0 / 12.0
    c1 = -1.0 / 360.0
    c2 = 1.0 / 1260.0
    c3 = -1.0 / 1680.0
    c4 = 1.0 / 1188.0
    c5 = -691.0 / 360360.0
    c6 = 1.0 / 156.0
    c7 = -3617.0 / 122400.0
    ser = c0 + y * (c1 + y * (c2 + y * (c3 + y * (c4 + y * (c5 + y * (c6 + y * c7))))))
    print ser
    val = exp(ser / x)
    print gsl_sf_exp(ser/x)
    return val

#excel_data_process()
#plot_bar()


def fun(x):
    z = 0
    if x>0:
        tmp = x
        x = math.pow(x, 5)
        y=x-1
        #y=(tmp-1)*(tmp*(tmp+1)+1)*(math.pow(tmp,3)+1)
    else:
        d=x*x
        #y=(x-1)*(x+1)
        y = d-1
    while(z<1e4):
        #z=(x-y)*(x+y)
        z=x*x-y*y
        x=x*10.0+1.0
    y = y*z
    return y
def fun1(x):
    z = 0
    if x>0:
        tmp = x
        x = math.pow(x, 5)
        y=x-1
        #y=(tmp-1)*(tmp*(tmp+1)+1)*(math.pow(tmp,3)+1)
    else:
        d=x*x
        y=(x-1)*(x+1)
        #y = d-1
    while(z<1e4):
        z=(x-y)*(x+y)
        #z=x*x-y*y
        x=x*10.0+1.0
    y = y*z
    return y


def mp_fun(x):
    mp.dps = 100
    z = mpf(0.0)
    if x > 0.0:
        x= mp.power(x,5)
        y = fadd(x,-1)
    else:
        d = fmul(x,x)
        #y = fmul(fadd(x,-1),fadd(x,1))
        y=fsub(d,1)
    while(z<1e4):
        z = fmul(x,x)-fmul(y,y)
        x = fadd(fmul(x,10.0),1.0)
    y = y*z
    return y




def plot_test(f1,f2,a,name):
    X = sorted(np.random.uniform(a[0],a[1],25000))
    res_mp = test_mp_fun(f1, X)

    res_d = test_gsl_fun(f2, X)
    l = []
    #re_err_1 = [math.log(distan_cal(op_r[0],op_f[0])+1,2) for op_f, op_r in zip(res_d, res_mp)]
    re_err_1 = [np.fabs((op_f[0]-float(op_r[0]))/float(op_r[0])) for op_f, op_r in zip(res_d, res_mp)]
    plt.figure()
    plt.plot(X, re_err_1, c='k')
    varep = 1e-15
    plt.plot([-10,0],[varep,varep])
    plt.annotate(r'$\frac{\varepsilon_1}{10}=1e-15$',
                 xy=(-4,varep), xycoords='data',
                 xytext=(0, -25), textcoords='offset points', fontsize=20)

    matplotlib.rc('xtick', labelsize = 10)
    matplotlib.rc('ytick', labelsize = 20)
    plt.rc('ytick', labelsize=10)
    for label in plt.gca().yaxis.get_ticklabels():
        label.set_fontsize(15)
    for label in plt.gca().xaxis.get_ticklabels():
        label.set_fontsize(10)
    plt.ylabel("Relative Error",fontsize=12)
    plt.xlabel("Input",fontsize=12)
    #plt.yticks([0, 2e-7, 4e-7, 6e-7, 8e-7],['0','2e-7','4e-7','6e-7','8e-7'])
    #plt.yticks([0, 3e-14, 6e-14, 9e-14, 1e-13], ['0', '3e-14', '6e-14', '9e-14', '1e-13'])
    plt.savefig(name, format="eps")
    plt.show()

def plot_test2(f1,f2,a,name):
    X = sorted(np.random.uniform(a[0],a[1],1000))
    res_mp = test_mp_fun(f1, X)

    res_d = test_gsl_fun(f2, X)
    l = []
    #re_err_1 = [math.log(distan_cal(op_r[0],op_f[0])+1,2) for op_f, op_r in zip(res_d, res_mp)]
    re_err_1 = [np.fabs((op_f[0]-float(op_r[0]))/float(op_r[0])) for op_f, op_r in zip(res_d, res_mp)]
    plt.figure()
    plt.plot(X, re_err_1, c='k')
    varep = 1e-15
    plt.plot([0,100],[varep,varep])
    plt.annotate(r'$\frac{\varepsilon_2}{1e7}=1e-15$',
                 xy=(60,varep), xycoords='data',
                 xytext=(0,-25), textcoords='offset points', fontsize=20)
    #\frac{\varepsilon_2}{1e7}=1e-8
    matplotlib.rc('xtick', labelsize = 10)
    matplotlib.rc('ytick', labelsize = 20)
    plt.rc('ytick', labelsize=10)
    for label in plt.gca().yaxis.get_ticklabels():
        label.set_fontsize(15)
    for label in plt.gca().xaxis.get_ticklabels():
        label.set_fontsize(10)
    plt.ylabel("Relative Error",fontsize=12)
    plt.xlabel("Input",fontsize=12)
    #plt.yticks([0, 2e-7, 4e-7, 6e-7, 8e-7],['0','2e-7','4e-7','6e-7','8e-7'])
    #plt.yticks([0, 3e-14, 6e-14, 9e-14, 1e-13], ['0', '3e-14', '6e-14', '9e-14', '1e-13'])
    plt.savefig(name, format="eps")
    plt.show()

def test_da():
    t = -27.999999999999993

    print gammastar_ser(28.999999999993)

    f = lambda x: gsl_sf_zeta(x)
    f2 = lambda x: zeta(x)
    print derivative(f, t)
    print fabs((f(t + t * 2.2204460492503131e-16) - f(t)) / (f(t) * 2.2204460492503131e-16))
    print condition(t, f(t), f)
    print f(t)
    print f2(t)
    print fdistan.fdistan(f(t), float(f2(t)))

def F(x):
    if (x<0.0):
        y = x-1.0
    else:
        y = 1.0

    z = (y-x)*(y+x)

    return z

def F_1(x):
    mp.prec = 200
    return power(x-1,2)-power(x,2)
def F_2(x):
    mp.prec = 200
    return polyval([-1,0,1],x)

#plot_test(F_1,F,[-1e14, -0.1])
#plot_test(F_2,F,[0, 5])

#print pfulp_res_time_limit([0.8,2],F_2,F,0,10,0)


def F_3(x):
    while(x!=x+1):
        x = x+1e10
    print x
def f0(x):
    return 0
#print fun(100)
#print mp_fun(100)
sqr_cof = [1,0,-1.0]
def sqr1(x):
    return plvel(x,sqr_cof)

def sqr1_mp(x):
    mp.prec =200
    return mp_polyval(x,sqr_cof)

def exp_cond(op_r):
    return np.fabs(float(op_r[1]))
def sqrt_cond(op_r):
    return 0.5

def sin_cond(op_r):
    return np.fabs(((float(cos(op_r[1])))*float(op_r[1]))/float(op_r[0]))
def cos_cond(op_r):
    return np.fabs(((float(-sin(op_r[1])))*float(op_r[1]))/float(op_r[0]))
def sqr_cond(op_r):
    return np.fabs(((op_r[1]*2.0-1.0)*float(op_r[1]))/float(op_r[0]))
def plot_sqr(f1,f2,fc,a,name):
    X = sorted(np.random.uniform(a[0],a[1],10000))
    if 0.0 in X:
        X.remove(0)
    mp.prec = 200
    res_mp = test_mp_fun(f1, X)
    res_d = test_gsl_fun(f2, X)
    l = []
    #re_err_1 = [math.log(distan_cal(op_r[0],op_f[0])+1,2) for op_f, op_r in zip(res_d, res_mp)]
    re_err_1 = [np.fabs((op_f[0]-float(op_r[0]))/float(op_r[0])) for op_f, op_r in zip(res_d, res_mp)]
    condi = [fc(op_r) for op_r in res_mp]
    back_err = [np.fabs(r/c) for r, c in zip(re_err_1, condi)]
    condi_s = [np.fabs(d/1e15) for d in condi]
    re_err_1_s = [np.fabs(r*6e1) for r in re_err_1]
    plt.figure()
    #plt.plot(X, re_err_1_s, c='k')
    #plt.plot(X, condi_s,c='r')
    plt.plot(X, back_err, c='b')


    matplotlib.rc('xtick', labelsize = 10)
    matplotlib.rc('ytick', labelsize = 20)
    plt.rc('ytick', labelsize=10)
    for label in plt.gca().yaxis.get_ticklabels():
        label.set_fontsize(15)
    for label in plt.gca().xaxis.get_ticklabels():
        label.set_fontsize(10)
    plt.ylabel("Relative Error",fontsize=12)
    plt.xlabel("Input",fontsize=12)
    #plt.yticks([0, 2e-7, 4e-7, 6e-7, 8e-7],['0','2e-7','4e-7','6e-7','8e-7'])
    #plt.yticks([0, 3e-14, 6e-14, 9e-14, 1e-13], ['0', '3e-14', '6e-14', '9e-14', '1e-13'])
    plt.savefig(name, format="eps")
    plt.show()
def sqr_cond(op_r):
    return np.fabs(((op_r[1]*2.0-1.0)*float(op_r[1]))/float(op_r[0]))
def plot_ulp_sqr(f1,f2,fc,a,name):
    X = sorted(np.random.uniform(a[0],a[1],10000))
    if 0.0 in X:
        X.remove(0)
    mp.prec = 200
    res_mp = test_mp_fun(f1, X)
    res_d = test_gsl_fun(f2, X)
    l = []
    #re_err_1 = [math.log(distan_cal(op_r[0],op_f[0])+1,2) for op_f, op_r in zip(res_d, res_mp)]
    re_err_1 = [np.fabs(fdistan.fdistan(op_f[0], float(op_r[0]))) for op_f, op_r in zip(res_d, res_mp)]
    condi = [fc(op_r) for op_r in res_mp]
    back_err = [np.fabs(r/c) for r, c in zip(re_err_1, condi)]
    condi_s = [np.fabs(d/1e15) for d in condi]
    re_err_1_s = [np.fabs(r*6e1) for r in re_err_1]
    plt.figure()
    #plt.plot(X, re_err_1_s, c='k')
    #plt.plot(X, condi_s,c='r')
    plt.plot(X, back_err, c='b')


    matplotlib.rc('xtick', labelsize = 10)
    matplotlib.rc('ytick', labelsize = 20)
    plt.rc('ytick', labelsize=10)
    for label in plt.gca().yaxis.get_ticklabels():
        label.set_fontsize(15)
    for label in plt.gca().xaxis.get_ticklabels():
        label.set_fontsize(10)
    plt.ylabel("Relative Error",fontsize=12)
    plt.xlabel("Input",fontsize=12)
    #plt.yticks([0, 2e-7, 4e-7, 6e-7, 8e-7],['0','2e-7','4e-7','6e-7','8e-7'])
    #plt.yticks([0, 3e-14, 6e-14, 9e-14, 1e-13], ['0', '3e-14', '6e-14', '9e-14', '1e-13'])
    plt.savefig(name, format="eps")
    plt.show()


#mp.prec=300
#plot_sqr(sqr1_mp,sqr1,sqr_cond,[0.9,1.1],"sqr1")

#plot_sqr(exp,math.exp,exp_cond,[-10,-0.1],"sqr1")
#plot_sqr(sqrt,math.sqrt,sqrt_cond,[0,100],"sqr1")
#plot_sqr(sin,math.sin,sin_cond,[0,4],"sqr1")
# plot_sqr(cos,math.cos,cos_cond,[0,10],"sqr1")
#plot_test2(mp_fun,fun,[0,100],"F_2_ulp.eps")
#plot_test(mp_fun,fun,[-10,0],"F_1_ulp.eps")
#plot_test2(mp_fun,fun1,[0,100],"F_2_rep.eps")
#plot_test(mp_fun,fun1,[-10,0],"F_1_rep.eps")
#F_3(1e10)


def plt_erf_mp():
    X = sorted(np.random.uniform(-8, 8, 1000))
    res_mp = test_mp_fun(erf, X)

    res_d = test_gsl_fun(gsl_sf_erf, X)

    re_err_1 = [np.fabs((op_f[0] - float(op_r[0])) / float(op_r[0])) for op_f, op_r in zip(res_d, res_mp)]
    Y= re_err_1

    plt.figure()
    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    for label in plt.gca().yaxis.get_ticklabels():
        label.set_fontsize(15)
    for label in plt.gca().xaxis.get_ticklabels():
        label.set_fontsize(10)

    plt.plot(X, Y, c="k")
    plt.xlabel("Input", fontsize=16)
    plt.ylabel("Relative Error", fontsize=16)
    plt.savefig("erf_ulp.eps", format="eps")
    plt.show()
#excel_data_process()
#plt_erf_mp()

def plt_airy():
    ls = load_data('last_res/gsl_airy_ai_GULP.xls', 0)
    X = sorted(ls[1])
    res_mp = test_mp_fun(airyai, X)
    gsl_airy_ai = lambda t: gsl_sf_airy_Ai(t, 0)
    res_d = test_gsl_fun(gsl_airy_ai, X)
    l = []
    for i in res_d:
        l.append(i[0])
    re_err_1 = [np.fabs(fdistan.fdistan(op_f[0], float(op_r[0]))) for op_f, op_r in zip(res_d, res_mp)]


    plt.figure()
    font = {'family': 'normal',
            'weight': 'bold'}
    plt.rc('font', **font)
    matplotlib.rc('xtick', labelsize=12)
    matplotlib.rc('ytick', labelsize=12)
    plt.rc('ytick', labelsize=15)
    for label in plt.gca().yaxis.get_ticklabels():
        label.set_fontsize(15)
    for label in plt.gca().xaxis.get_ticklabels():
        label.set_fontsize(12)
    plt.scatter(X, re_err_1, c='k')

    plt.xlabel("Input", fontsize=12)
    plt.ylabel("ULP Error", fontsize=16)
    plt.savefig("airy_ulp.eps", format="eps",bbox_inches='tight')
    plt.show()

def plt_lngamma():
    X = sorted(np.random.uniform(0.0, 10, 100000))
    res_mp = test_mp_fun(loggamma, X)

    res_d = test_gsl_fun(gsl_sf_lngamma, X)
    l = []
    for i in res_d:
        l.append(i[0])
    re_err_1 = [np.fabs(fdistan.fdistan(op_f[0], float(op_r[0]))) for op_f, op_r in zip(res_d, res_mp)]
    font = {'family': 'normal',
            'weight': 'bold'}
    plt.rc('font', **font)
    plt.figure()
    matplotlib.rc('xtick', labelsize=12)
    matplotlib.rc('ytick', labelsize=12)
    plt.rc('ytick', labelsize=15)
    for label in plt.gca().yaxis.get_ticklabels():
        label.set_fontsize(15)
    for label in plt.gca().xaxis.get_ticklabels():
        label.set_fontsize(12)
    plt.plot(X, re_err_1, 'k')
    plt.xlabel("Input", fontsize=12)
    #plt.yticks([0, 1000, 2000, 3000], ['0', '1e3', '2e3', '3e3'])
    plt.ylabel("ULP Error", fontsize=16)
    plt.savefig("lngamma_ulp.eps", format="eps",bbox_inches='tight')
    plt.show()
#plt_lngamma()
#plt_airy()

excel_data_process_mutilple("20170823","data_out_multiple_20170823.xls")