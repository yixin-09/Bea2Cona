from basic_func import *
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import xlrd
import math
import os
import re

def plot_bar_alg(e1,e2,e3,e4,e5,e6,e7,e8,name_list,name):
    n_groups = len(name_list)
    plt.figure(figsize=(12, 8))
    index = np.arange(n_groups)
    bar_width = 0.1
    plt.barh(index, e1, bar_width,facecolor = 'lightskyblue',
                     label='global_fit1')
    plt.barh(index+bar_width, e2, bar_width,facecolor = 'yellowgreen',
             label='global_fit2')
    plt.barh(index+2*bar_width, e3, bar_width,facecolor = 'firebrick',
             label='hybrid_fit1')
    plt.barh(index+3*bar_width, e4, bar_width,facecolor = 'violet',
             label='hybrid_fit2')
    plt.barh(index + 4 * bar_width, e5, bar_width,facecolor = 'chartreuse',
             label='local_fit1')
    plt.barh(index + 5 * bar_width, e6, bar_width,facecolor = 'plum',
             label='local_fit2')
    plt.barh(index + 6 * bar_width, e7, bar_width,facecolor = 'gold',
             label='random_fit1')
    plt.barh(index + 7 * bar_width, e8, bar_width,facecolor = 'black',
             label='random_fit2')

    plt.xlabel(name)
    plt.yticks(index+4*bar_width, name_list)
    plt.legend(bbox_to_anchor=(0.9, 1), loc=2)
    plt.savefig(name+"_error.pdf", format="pdf")
    plt.show()

def plot_bar(e1,e2,e3,name_list,name):
    n_groups = len(name_list)

    index = np.arange(n_groups)
    bar_width = 0.25

    error_config = {'ecolor': '0.4'}

    plt.barh(index, e3, bar_width,
                     alpha=0.3,
                     lw = 3,
                     error_kw=error_config,
                     label='rpart')

    plt.barh(index + bar_width, e2, bar_width,
                     alpha=0.6,
                     lw = 1,
                     error_kw=error_config,
                     label='fpart')
    plt.barh(index + 2*bar_width, e1, bar_width,
                      alpha=1.0,
                      error_kw=error_config,
                      label='nopart')


    plt.xlabel(name)
    plt.yticks(index+bar_width, name_list)
    plt.legend(bbox_to_anchor=(1.05, 1),loc=2)

    plt.savefig(name+"_error.pdf", format="pdf")
    plt.show()
    plt.clf()

def load_data(filename,sheet_n):
    data = xlrd.open_workbook(filename)
    table = data.sheets()[sheet_n]
    error_list = []
    input_list = []
    time_list =[]
    for i in range(1,table.nrows):
        error_list.append(float(table.row_values(i)[1]))
        input_list.append(tuple(table.row_values(i)[2]))
        time_list.append(float(table.row_values(i)[3]))
    return [error_list,input_list,time_list]
def mean_error(input_list):
    mean_error_list = []
    for i in input_list:
        mean_error_list.append(np.mean(i))
    return mean_error_list
def excel_data_process():
    name_gsl_list = ["airyai","bessely1","legendre3","struve"]
    file_list = sorted(os.listdir("20171228"))
    data_list = []
    data_list2 = []
    data_list3 = []
    for i in file_list:
        fn = "20171228/"+i
        data_list.append(load_data(fn,0))
        data_list2.append(load_data(fn,1))
        data_list3.append(load_data(fn,2))
    print len(data_list2)
    gf1_err_nop = []
    gf1_err_fp = []
    gf1_err_rp = []
    ghy_err_nop = []
    ghy_err_fp = []
    ghy_err_rp = []
    hhy_err_nop = []
    hhy_err_fp = []
    hhy_err_rp = []
    hf1_err_nop = []
    hf1_err_fp = []
    hf1_err_rp = []
    lf1_err_nop = []
    lf1_err_fp = []
    lf1_err_rp = []
    lhy_err_nop = []
    lhy_err_fp = []
    lhy_err_rp = []
    rf1_err_nop = []
    rf1_err_fp = []
    rf1_err_rp = []
    rhy_err_nop = []
    rhy_err_fp = []
    rhy_err_rp = []
    for j in range(0, 32, 8):
        gf1_err_nop.append(data_list[j][0])
        gf1_err_fp.append(data_list2[j][0])
        gf1_err_rp.append(data_list3[j][0])
        ghy_err_nop.append(data_list[j+1][0])
        ghy_err_fp.append(data_list2[j+1][0])
        ghy_err_rp.append(data_list3[j+1][0])
        hf1_err_nop.append(data_list[j+2][0])
        hf1_err_fp.append(data_list2[j+2][0])
        hf1_err_rp.append(data_list3[j+2][0])
        hhy_err_nop.append(data_list[j + 3][0])
        hhy_err_fp.append(data_list2[j + 3][0])
        hhy_err_rp.append(data_list3[j + 3][0])
        lf1_err_nop.append(data_list[j + 4][2])
        lf1_err_fp.append(data_list2[j + 4][2])
        lf1_err_rp.append(data_list3[j + 4][2])
        lhy_err_nop.append(data_list[j + 5][2])
        lhy_err_fp.append(data_list2[j + 5][2])
        lhy_err_rp.append(data_list3[j + 5][2])
        rf1_err_nop.append(data_list[j + 6][2])
        rf1_err_fp.append(data_list2[j + 6][2])
        rf1_err_rp.append(data_list3[j + 6][2])
        rhy_err_nop.append(data_list[j + 7][2])
        rhy_err_fp.append(data_list2[j + 7][2])
        rhy_err_rp.append(data_list3[j + 7][2])
    name_list = []
    e1 = []
    e2 = []
    e3 = []
    rev_l = range(0, 4)

    mean_gf1_err_nop = mean_error(gf1_err_nop)
    mean_gf1_err_fp = mean_error(gf1_err_fp)
    mean_gf1_err_rp = mean_error(gf1_err_rp)
    for i in rev_l:
        name_list.append(name_gsl_list[i])
        e1.append(np.log2(mean_gf1_err_nop[i]))
        e2.append(np.log2(mean_gf1_err_fp[i]))
        e3.append(np.log2(mean_gf1_err_rp[i]))
    print name_list
    plot_bar(e1, e2, e3, name_list, "global_fit1")
    e1 = []
    e2 = []
    e3 = []
    name_list = []
    mean_ghy_err_nop = mean_error(ghy_err_nop)
    mean_ghy_err_fp = mean_error(ghy_err_fp)
    mean_ghy_err_rp = mean_error(ghy_err_rp)
    for i in rev_l:
        name_list.append(name_gsl_list[i])
        e1.append(np.log2(mean_ghy_err_nop[i]))
        e2.append(np.log2(mean_ghy_err_fp[i]))
        e3.append(np.log2(mean_ghy_err_rp[i]))
    plot_bar(e1, e2, e3, name_list, "global_hy")
    e1 = []
    e2 = []
    e3 = []
    name_list = []
    mean_hhy_err_nop = mean_error(hhy_err_nop)
    mean_hhy_err_fp = mean_error(hhy_err_fp)
    mean_hhy_err_rp = mean_error(hhy_err_rp)
    for i in rev_l:
        name_list.append(name_gsl_list[i])
        e1.append(np.log2(mean_hhy_err_nop[i]))
        e2.append(np.log2(mean_hhy_err_fp[i]))
        e3.append(np.log2(mean_hhy_err_rp[i]))
    plot_bar(e1, e2, e3, name_list, "hybrid_hy")
    e1 = []
    e2 = []
    e3 = []
    name_list = []
    mean_hf1_err_nop = mean_error(hf1_err_nop)
    mean_hf1_err_fp = mean_error(hf1_err_fp)
    mean_hf1_err_rp = mean_error(hf1_err_rp)
    for i in rev_l:
        name_list.append(name_gsl_list[i])
        e1.append(np.log2(mean_hf1_err_nop[i]))
        e2.append(np.log2(mean_hf1_err_fp[i]))
        e3.append(np.log2(mean_hf1_err_rp[i]))
    plot_bar(e1, e2, e3, name_list, "hybrid_fit1")
    e1 = []
    e2 = []
    e3 = []
    name_list = []
    mean_lf1_err_nop = mean_error(lf1_err_nop)
    mean_lf1_err_fp = mean_error(lf1_err_fp)
    mean_lf1_err_rp = mean_error(lf1_err_rp)
    for i in rev_l:
        name_list.append(name_gsl_list[i])
        e1.append(np.log2(mean_lf1_err_nop[i]))
        e2.append(np.log2(mean_lf1_err_fp[i]))
        e3.append(np.log2(mean_lf1_err_rp[i]))
    plot_bar(e1, e2, e3, name_list, "local_f1")
    e1 = []
    e2 = []
    e3 = []
    name_list = []
    mean_lhy_err_nop = mean_error(lhy_err_nop)
    mean_lhy_err_fp = mean_error(lhy_err_fp)
    mean_lhy_err_rp = mean_error(lhy_err_rp)
    for i in rev_l:
        name_list.append(name_gsl_list[i])
        e1.append(np.log2(mean_lhy_err_nop[i]))
        e2.append(np.log2(mean_lhy_err_fp[i]))
        e3.append(np.log2(mean_lhy_err_rp[i]))
    plot_bar(e1, e2, e3, name_list, "local_hy")
    e1 = []
    e2 = []
    e3 = []
    name_list = []
    mean_rf1_err_nop = mean_error(rf1_err_nop)
    mean_rf1_err_fp = mean_error(rf1_err_fp)
    mean_rf1_err_rp = mean_error(rf1_err_rp)
    for i in rev_l:
        name_list.append(name_gsl_list[i])
        e1.append(np.log2(mean_rf1_err_nop[i]))
        e2.append(np.log2(mean_rf1_err_fp[i]))
        e3.append(np.log2(mean_rf1_err_rp[i]))
    plot_bar(e1, e2, e3, name_list, "random_f1")
    e1 = []
    e2 = []
    e3 = []
    name_list = []
    mean_rhy_err_nop = mean_error(rhy_err_nop)
    mean_rhy_err_fp = mean_error(rhy_err_fp)
    mean_rhy_err_rp = mean_error(rhy_err_rp)
    for i in rev_l:
        name_list.append(name_gsl_list[i])
        e1.append(np.log2(mean_rhy_err_nop[i]))
        e2.append(np.log2(mean_rhy_err_fp[i]))
        e3.append(np.log2(mean_rhy_err_rp[i]))
    plot_bar(e1, e2, e3, name_list, "random_hy")
def excel_data_process_sf():
    name_gsl_list = ["nopartition","fpartition","rpartition"]
    name_fun_list = ["airyai","bessely1","legendre3","struve"]
    file_list = sorted(os.listdir("20171228"))
    data_list = []
    data_list2 = []
    data_list3 = []
    for i in file_list:
        fn = "20171228/"+i
        data_list.append(load_data(fn,0))
        data_list2.append(load_data(fn,1))
        data_list3.append(load_data(fn,2))
    print len(data_list2)
    gf1_err_nop = []
    gf1_err_fp = []
    gf1_err_rp = []
    ghy_err_nop = []
    ghy_err_fp = []
    ghy_err_rp = []
    hhy_err_nop = []
    hhy_err_fp = []
    hhy_err_rp = []
    hf1_err_nop = []
    hf1_err_fp = []
    hf1_err_rp = []
    lf1_err_nop = []
    lf1_err_fp = []
    lf1_err_rp = []
    lhy_err_nop = []
    lhy_err_fp = []
    lhy_err_rp = []
    rf1_err_nop = []
    rf1_err_fp = []
    rf1_err_rp = []
    rhy_err_nop = []
    rhy_err_fp = []
    rhy_err_rp = []
    for j in range(0, 32, 8):
        gf1_err_nop.append(data_list[j][0])
        gf1_err_fp.append(data_list2[j][0])
        gf1_err_rp.append(data_list3[j][0])
        ghy_err_nop.append(data_list[j+1][0])
        ghy_err_fp.append(data_list2[j+1][0])
        ghy_err_rp.append(data_list3[j+1][0])
        hf1_err_nop.append(data_list[j+2][0])
        hf1_err_fp.append(data_list2[j+2][0])
        hf1_err_rp.append(data_list3[j+2][0])
        hhy_err_nop.append(data_list[j + 3][0])
        hhy_err_fp.append(data_list2[j + 3][0])
        hhy_err_rp.append(data_list3[j + 3][0])
        lf1_err_nop.append(data_list[j + 4][0])
        lf1_err_fp.append(data_list2[j + 4][0])
        lf1_err_rp.append(data_list3[j + 4][0])
        lhy_err_nop.append(data_list[j + 5][0])
        lhy_err_fp.append(data_list2[j + 5][0])
        lhy_err_rp.append(data_list3[j + 5][0])
        rf1_err_nop.append(data_list[j + 6][0])
        rf1_err_fp.append(data_list2[j + 6][0])
        rf1_err_rp.append(data_list3[j + 6][0])
        rhy_err_nop.append(data_list[j + 7][0])
        rhy_err_fp.append(data_list2[j + 7][0])
        rhy_err_rp.append(data_list3[j + 7][0])
    gf1_time_nop = []
    gf1_time_fp = []
    gf1_time_rp = []
    ghy_time_nop = []
    ghy_time_fp = []
    ghy_time_rp = []
    hhy_time_nop = []
    hhy_time_fp = []
    hhy_time_rp = []
    hf1_time_nop = []
    hf1_time_fp = []
    hf1_time_rp = []
    lf1_time_nop = []
    lf1_time_fp = []
    lf1_time_rp = []
    lhy_time_nop = []
    lhy_time_fp = []
    lhy_time_rp = []
    rf1_time_nop = []
    rf1_time_fp = []
    rf1_time_rp = []
    rhy_time_nop = []
    rhy_time_fp = []
    rhy_time_rp = []
    for j in range(0, 32, 8):
        gf1_time_nop.append(data_list[j][2])
        gf1_time_fp.append(data_list2[j][2])
        gf1_time_rp.append(data_list3[j][2])
        ghy_time_nop.append(data_list[j + 1][2])
        ghy_time_fp.append(data_list2[j + 1][2])
        ghy_time_rp.append(data_list3[j + 1][2])
        hf1_time_nop.append(data_list[j + 2][2])
        hf1_time_fp.append(data_list2[j + 2][2])
        hf1_time_rp.append(data_list3[j + 2][2])
        hhy_time_nop.append(data_list[j + 3][2])
        hhy_time_fp.append(data_list2[j + 3][2])
        hhy_time_rp.append(data_list3[j + 3][2])
        lf1_time_nop.append(data_list[j + 4][2])
        lf1_time_fp.append(data_list2[j + 4][2])
        lf1_time_rp.append(data_list3[j + 4][2])
        lhy_time_nop.append(data_list[j + 5][2])
        lhy_time_fp.append(data_list2[j + 5][2])
        lhy_time_rp.append(data_list3[j + 5][2])
        rf1_time_nop.append(data_list[j + 6][2])
        rf1_time_fp.append(data_list2[j + 6][2])
        rf1_time_rp.append(data_list3[j + 6][2])
        rhy_time_nop.append(data_list[j + 7][2])
        rhy_time_fp.append(data_list2[j + 7][2])
        rhy_time_rp.append(data_list3[j + 7][2])
    name_list = []
    mean_gf1_err_nop = mean_error(gf1_err_nop)
    mean_gf1_err_fp = mean_error(gf1_err_fp)
    mean_gf1_err_rp = mean_error(gf1_err_rp)
    mean_ghy_err_nop = mean_error(ghy_err_nop)
    mean_ghy_err_fp = mean_error(ghy_err_fp)
    mean_ghy_err_rp = mean_error(ghy_err_rp)
    mean_hhy_err_nop = mean_error(hhy_err_nop)
    mean_hhy_err_fp = mean_error(hhy_err_fp)
    mean_hhy_err_rp = mean_error(hhy_err_rp)
    mean_hf1_err_nop = mean_error(hf1_err_nop)
    mean_hf1_err_fp = mean_error(hf1_err_fp)
    mean_hf1_err_rp = mean_error(hf1_err_rp)
    mean_lf1_err_nop = mean_error(lf1_err_nop)
    mean_lf1_err_fp = mean_error(lf1_err_fp)
    mean_lf1_err_rp = mean_error(lf1_err_rp)
    mean_lhy_err_nop = mean_error(lhy_err_nop)
    mean_lhy_err_fp = mean_error(lhy_err_fp)
    mean_lhy_err_rp = mean_error(lhy_err_rp)
    mean_rf1_err_nop = mean_error(rf1_err_nop)
    mean_rf1_err_fp = mean_error(rf1_err_fp)
    mean_rf1_err_rp = mean_error(rf1_err_rp)
    mean_rhy_err_nop = mean_error(rhy_err_nop)
    mean_rhy_err_fp = mean_error(rhy_err_fp)
    mean_rhy_err_rp = mean_error(rhy_err_rp)
    e1 = []
    e2 = []
    e3 = []
    e4 = []
    e5 = []
    e6 = []
    e7 = []
    e8 = []
    for j in range(0,4):
        e1 = []
        e2 = []
        e3 = []
        e4 = []
        e5 = []
        e6 = []
        e7 = []
        e8 = []
        e1.append(np.log2(mean_gf1_err_nop[j]))
        e1.append(np.log2(mean_gf1_err_fp[j]))
        e1.append(np.log2(mean_gf1_err_rp[j]))
        e2.append(np.log2(mean_ghy_err_nop[j]))
        e2.append(np.log2(mean_ghy_err_fp[j]))
        e2.append(np.log2(mean_ghy_err_rp[j]))
        e3.append(np.log2(mean_hf1_err_nop[j]))
        e3.append(np.log2(mean_hf1_err_fp[j]))
        e3.append(np.log2(mean_hf1_err_rp[j]))
        e4.append(np.log2(mean_hhy_err_nop[j]))
        e4.append(np.log2(mean_hhy_err_fp[j]))
        e4.append(np.log2(mean_hhy_err_rp[j]))
        e5.append(np.log2(mean_lf1_err_nop[j]))
        e5.append(np.log2(mean_lf1_err_fp[j]))
        e5.append(np.log2(mean_lf1_err_rp[j]))
        e6.append(np.log2(mean_lhy_err_nop[j]))
        e6.append(np.log2(mean_lhy_err_fp[j]))
        e6.append(np.log2(mean_lhy_err_rp[j]))
        e7.append(np.log2(mean_rf1_err_nop[j]))
        e7.append(np.log2(mean_rf1_err_fp[j]))
        e7.append(np.log2(mean_rf1_err_rp[j]))
        e8.append(np.log2(mean_rhy_err_nop[j]))
        e8.append(np.log2(mean_rhy_err_fp[j]))
        e8.append(np.log2(mean_rhy_err_rp[j]))
        plot_bar_alg(e1, e2, e3, e4, e5, e6, e7, e8, name_gsl_list, name_fun_list[j])


def excel_data_process_time():
    name_gsl_list = ["airyai","bessely1","legendre3","struve"]
    file_list = sorted(os.listdir("20171228"))
    data_list = []
    data_list2 = []
    data_list3 = []
    for i in file_list:
        fn = "20171228/"+i
        data_list.append(load_data(fn,0))
        data_list2.append(load_data(fn,1))
        data_list3.append(load_data(fn,2))
    print len(data_list2)
    gf1_time_nop = []
    gf1_time_fp = []
    gf1_time_rp = []
    ghy_time_nop = []
    ghy_time_fp = []
    ghy_time_rp = []
    hhy_time_nop = []
    hhy_time_fp = []
    hhy_time_rp = []
    hf1_time_nop = []
    hf1_time_fp = []
    hf1_time_rp = []
    lf1_time_nop = []
    lf1_time_fp = []
    lf1_time_rp = []
    lhy_time_nop = []
    lhy_time_fp = []
    lhy_time_rp = []
    rf1_time_nop = []
    rf1_time_fp = []
    rf1_time_rp = []
    rhy_time_nop = []
    rhy_time_fp = []
    rhy_time_rp = []
    for j in range(0, 32, 8):
        gf1_time_nop.append(data_list[j][2])
        gf1_time_fp.append(data_list2[j][2])
        gf1_time_rp.append(data_list3[j][2])
        ghy_time_nop.append(data_list[j+1][2])
        ghy_time_fp.append(data_list2[j+1][2])
        ghy_time_rp.append(data_list3[j+1][2])
        hf1_time_nop.append(data_list[j+2][2])
        hf1_time_fp.append(data_list2[j+2][2])
        hf1_time_rp.append(data_list3[j+2][2])
        hhy_time_nop.append(data_list[j + 3][2])
        hhy_time_fp.append(data_list2[j + 3][2])
        hhy_time_rp.append(data_list3[j + 3][2])
        lf1_time_nop.append(data_list[j + 4][2])
        lf1_time_fp.append(data_list2[j + 4][2])
        lf1_time_rp.append(data_list3[j + 4][2])
        lhy_time_nop.append(data_list[j+5][2])
        lhy_time_fp.append(data_list2[j+5][2])
        lhy_time_rp.append(data_list3[j+5][2])
        rf1_time_nop.append(data_list[j + 6][2])
        rf1_time_fp.append(data_list2[j + 6][2])
        rf1_time_rp.append(data_list3[j + 6][2])
        rhy_time_nop.append(data_list[j + 7][2])
        rhy_time_fp.append(data_list2[j + 7][2])
        rhy_time_rp.append(data_list3[j + 7][2])
    name_list = []
    e1 = []
    e2 = []
    e3 = []
    rev_l = range(0, 4)

    mean_gf1_time_nop = mean_error(gf1_time_nop)
    mean_gf1_time_fp = mean_error(gf1_time_fp)
    mean_gf1_time_rp = mean_error(gf1_time_rp)
    for i in rev_l:
        name_list.append(name_gsl_list[i])
        e1.append((mean_gf1_time_nop[i]))
        e2.append((mean_gf1_time_fp[i]))
        e3.append((mean_gf1_time_rp[i]))
    print name_list
    plot_bar(e1, e2, e3, name_list, "global_fit1")
    e1 = []
    e2 = []
    e3 = []
    name_list = []
    mean_ghy_time_nop = mean_error(ghy_time_nop)
    mean_ghy_time_fp = mean_error(ghy_time_fp)
    mean_ghy_time_rp = mean_error(ghy_time_rp)
    for i in rev_l:
        name_list.append(name_gsl_list[i])
        e1.append((mean_ghy_time_nop[i]))
        e2.append((mean_ghy_time_fp[i]))
        e3.append((mean_ghy_time_rp[i]))
    plot_bar(e1, e2, e3, name_list, "global_hy")
    e1 = []
    e2 = []
    e3 = []
    name_list = []
    mean_hhy_time_nop = mean_error(hhy_time_nop)
    mean_hhy_time_fp = mean_error(hhy_time_fp)
    mean_hhy_time_rp = mean_error(hhy_time_rp)
    for i in rev_l:
        name_list.append(name_gsl_list[i])
        e1.append((mean_hhy_time_nop[i]))
        e2.append((mean_hhy_time_fp[i]))
        e3.append((mean_hhy_time_rp[i]))
    plot_bar(e1, e2, e3, name_list, "hybrid_hy")
    e1 = []
    e2 = []
    e3 = []
    name_list = []
    mean_hf1_time_nop = mean_error(hf1_time_nop)
    mean_hf1_time_fp = mean_error(hf1_time_fp)
    mean_hf1_time_rp = mean_error(hf1_time_rp)
    for i in rev_l:
        name_list.append(name_gsl_list[i])
        e1.append((mean_hf1_time_nop[i]))
        e2.append((mean_hf1_time_fp[i]))
        e3.append((mean_hf1_time_rp[i]))
    plot_bar(e1, e2, e3, name_list, "hybrid_fit1")
    e1 = []
    e2 = []
    e3 = []
    name_list = []
    mean_lf1_time_nop = mean_error(lf1_time_nop)
    mean_lf1_time_fp = mean_error(lf1_time_fp)
    mean_lf1_time_rp = mean_error(lf1_time_rp)
    for i in rev_l:
        name_list.append(name_gsl_list[i])
        e1.append((mean_lf1_time_nop[i]))
        e2.append((mean_lf1_time_fp[i]))
        e3.append((mean_lf1_time_rp[i]))
    plot_bar(e1, e2, e3, name_list, "local_f1")
    e1 = []
    e2 = []
    e3 = []
    name_list = []
    mean_lhy_time_nop = mean_error(lhy_time_nop)
    mean_lhy_time_fp = mean_error(lhy_time_fp)
    mean_lhy_time_rp = mean_error(lhy_time_rp)
    for i in rev_l:
        name_list.append(name_gsl_list[i])
        e1.append((mean_lhy_time_nop[i]))
        e2.append((mean_lhy_time_fp[i]))
        e3.append((mean_lhy_time_rp[i]))
    plot_bar(e1, e2, e3, name_list, "local_hy")
    e1 = []
    e2 = []
    e3 = []
    name_list = []
    mean_rf1_time_nop = mean_error(rf1_time_nop)
    mean_rf1_time_fp = mean_error(rf1_time_fp)
    mean_rf1_time_rp = mean_error(rf1_time_rp)
    for i in rev_l:
        name_list.append(name_gsl_list[i])
        e1.append((mean_rf1_time_nop[i]))
        e2.append((mean_rf1_time_fp[i]))
        e3.append((mean_rf1_time_rp[i]))
    plot_bar(e1, e2, e3, name_list, "random_f1")
    e1 = []
    e2 = []
    e3 = []
    name_list = []
    mean_rhy_time_nop = mean_error(rhy_time_nop)
    mean_rhy_time_fp = mean_error(rhy_time_fp)
    mean_rhy_time_rp = mean_error(rhy_time_rp)
    for i in rev_l:
        name_list.append(name_gsl_list[i])
        e1.append((mean_rhy_time_nop[i]))
        e2.append((mean_rhy_time_fp[i]))
        e3.append((mean_rhy_time_rp[i]))
    plot_bar(e1, e2, e3, name_list, "random_hy")
file_list = sorted(os.listdir("20171228"))
print file_list
excel_data_process_sf()
#excel_data_process()