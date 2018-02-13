# global variables:
# calculation time: cal_tm = 1000s
# number of sampling points: num_sam_p = 100000
# iteration times: iter_n = 10
# default terminate condition: max will not change after three iterations

from search_algorithm import *
import xlwt
import fun2search

def output_err(t_l,t_l2,t_l3,name,name2,rseed):
    book = xlwt.Workbook()
    sheet = book.add_sheet("sheet1")
    sheet.write(0, 0, "functions")
    sheet.write(0, 1, "max_error")
    sheet.write(0, 2, "input")
    sheet.write(0, 3, "execute time")
    sheet.write(0, 4, "Random Seed")
    sheet.write(0, 5, "Iteration number")
    n = 1
    for t in t_l:
        sheet.write(n,0,name2)
        sheet.write(n,1,repr(t[0]))
        sheet.write(n,2,repr(t[1]))
        sheet.write(n,3,repr(t[2]))
        sheet.write(n,4,rseed[n-1])
        sheet.write(n, 5, repr(t[3]))
        n = n+1
    sheet = book.add_sheet("sheet2")
    sheet.write(0, 0, "functions")
    sheet.write(0, 1, "max_error")
    sheet.write(0, 2, "input")
    sheet.write(0, 3, "execute time")
    sheet.write(0, 4, "Random Seed")
    sheet.write(0, 5, "Iteration number")
    n = 1
    for t in t_l2:
        sheet.write(n, 0, name2)
        sheet.write(n, 1, repr(t[0]))
        sheet.write(n, 2, repr(t[1]))
        sheet.write(n, 3, repr(t[2]))
        sheet.write(n, 4, rseed[n - 1])
        sheet.write(n, 5, repr(t[3]))
        n = n + 1
    sheet = book.add_sheet("sheet3")
    sheet.write(0, 0, "functions")
    sheet.write(0, 1, "max_error")
    sheet.write(0, 2, "input")
    sheet.write(0, 3, "execute time")
    sheet.write(0, 4, "Random Seed")
    sheet.write(0, 5, "Iteration number")
    n = 1
    for t in t_l3:
        sheet.write(n, 0, name2)
        sheet.write(n, 1, repr(t[0]))
        sheet.write(n, 2, repr(t[1]))
        sheet.write(n, 3, repr(t[2]))
        sheet.write(n, 4, rseed[n - 1])
        sheet.write(n, 5, repr(t[3]))
        n = n + 1
    book.save(name+".xls")

def tmp_test(funcOfname,input_domain,times,repeatOftrials):
    randOfseed = np.random.random_integers(0,1e8,repeatOftrials)
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    limit = limit_time(times,input_domain)
    print limit
    # global search fitness1 fitness2 or fit1+fit2 and no partition,fpartition,rpartition
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
        tempResult = glob_search_fit1(nopartition, input_domain, False, limit)
        resultOflist.append(tempResult)
        tempResult2 = glob_search_fit1(fpartition, input_domain, False, limit)
        resultOflist2.append(tempResult2)
        tempResult3 = glob_search_fit1(rpartition, input_domain, False, limit)
        resultOflist3.append(tempResult3)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_glob_fit1", funcOfname, randOfseed)
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
        tempResult = glob_search_fit2(nopartition, input_domain, False, limit)
        resultOflist.append(tempResult)
        tempResult2 = glob_search_fit2(fpartition, input_domain, False, limit)
        resultOflist2.append(tempResult2)
        tempResult3 = glob_search_fit2(rpartition, input_domain, False, limit)
        resultOflist3.append(tempResult3)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_glob_fit2", funcOfname, randOfseed)
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
        tempResult = glob_search_fit_hybrid(nopartition, input_domain, False, limit)
        resultOflist.append(tempResult)
        tempResult2 = glob_search_fit_hybrid(fpartition, input_domain, False, limit)
        resultOflist2.append(tempResult2)
        tempResult3 = glob_search_fit_hybrid(rpartition, input_domain, False, limit)
        resultOflist3.append(tempResult3)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_glob_hybrid", funcOfname, randOfseed)
    # local search fitness1 fitness2 or fit1+fit2 and no partition,fpartition,rpartition
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
        tempResult = local_search_fit1(nopartition, input_domain, limit)
        resultOflist.append(tempResult)
        tempResult2 = local_search_fit1(fpartition, input_domain, limit)
        resultOflist2.append(tempResult2)
        tempResult3 = local_search_fit1(rpartition, input_domain, limit)
        resultOflist3.append(tempResult3)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_local_fit1", funcOfname, randOfseed)
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
        tempResult = local_search_fit2(nopartition, input_domain, limit)
        resultOflist.append(tempResult)
        tempResult2 = local_search_fit2(fpartition, input_domain, limit)
        resultOflist2.append(tempResult2)
        tempResult3 = local_search_fit2(rpartition, input_domain, limit)
        resultOflist3.append(tempResult3)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_local_fit2", funcOfname, randOfseed)
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
        tempResult = local_search_hybrid(nopartition, input_domain, limit)
        resultOflist.append(tempResult)
        tempResult2 = local_search_hybrid(fpartition, input_domain, limit)
        resultOflist2.append(tempResult2)
        tempResult3 = local_search_hybrid(rpartition, input_domain, limit)
        resultOflist3.append(tempResult3)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_local_hybrid", funcOfname, randOfseed)
    # random search fitness1 fitness2 or fit1+fit2 and no partition,fpartition,rpartition
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
        tempResult = random_search_fit1(nopartition, input_domain, limit)
        resultOflist.append(tempResult)
        tempResult2 = random_search_fit1(fpartition, input_domain, limit)
        resultOflist2.append(tempResult2)
        tempResult3 = random_search_fit1(rpartition, input_domain, limit)
        resultOflist3.append(tempResult3)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_random_fit1", funcOfname, randOfseed)
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
        tempResult = random_search_fit2(nopartition, input_domain, limit)
        resultOflist.append(tempResult)
        tempResult2 = random_search_fit2(fpartition, input_domain, limit)
        resultOflist2.append(tempResult2)
        tempResult3 = random_search_fit2(rpartition, input_domain, limit)
        resultOflist3.append(tempResult3)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_random_fit2", funcOfname, randOfseed)
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
        tempResult = binary_random_search(input_domain, limit)
        resultOflist.append(tempResult)
        resultOflist2.append(tempResult)
        resultOflist3.append(tempResult)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_brand_fit1", funcOfname, randOfseed)
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
        tempResult = binary_random_search_fit2(input_domain, limit)
        resultOflist.append(tempResult)
        resultOflist2.append(tempResult)
        resultOflist3.append(tempResult)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_brand_fit2", funcOfname, randOfseed)
    # hybrid search fitness1 fitness2 or fit1+fit2 and no partition,fpartition,rpartition
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
    tempResult = glob_search_fit1(nopartition, input_domain, True, limit)
    resultOflist.append(tempResult)
    tempResult2 = glob_search_fit1(fpartition, input_domain, True, limit)
    resultOflist2.append(tempResult2)
    tempResult3 = glob_search_fit1(rpartition, input_domain, True, limit)
    resultOflist3.append(tempResult3)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_hybrid_fit1", funcOfname, randOfseed)
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
        tempResult = glob_search_fit2(nopartition, input_domain, True, limit)
        resultOflist.append(tempResult)
        tempResult2 = glob_search_fit2(fpartition, input_domain, True, limit)
        resultOflist2.append(tempResult2)
        tempResult3 = glob_search_fit2(rpartition, input_domain, True, limit)
        resultOflist3.append(tempResult3)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_hybrid_fit2", funcOfname, randOfseed)
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
        tempResult = glob_search_fit_hybrid(nopartition, input_domain, True, limit)
        resultOflist.append(tempResult)
        tempResult2 = glob_search_fit_hybrid(fpartition, input_domain, True, limit)
        resultOflist2.append(tempResult2)
        tempResult3 = glob_search_fit_hybrid(rpartition, input_domain, True, limit)
        resultOflist3.append(tempResult3)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_hybrid_hybrid", funcOfname, randOfseed)

    # local search fitness1 fitness2 or fit1+fit2 and no partition,fpartition,rpartition
    # hybrid search fitness1 fitness2 or fit1+fit2 and no partition,fpartition,rpartition

    # random search fitness1 fitness2 or fit1+fit2 and no partition,fpartition,rpartition
    # binary random search fitness1 fitness2 or fit1+fit2

    print randOfseed


def tmp_test_new(funcOfname,input_domain,times,repeatOftrials):
    randOfseed = np.random.random_integers(0,1e8,repeatOftrials)
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    limit = limit_time(times,input_domain)
    print limit
    # global search fitness1 fitness2 or fit1+fit2 and no partition,fpartition,rpartition
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
        tempResult = glob_search_fit1(nopartition, input_domain, False, limit)
        resultOflist.append(tempResult)
        tempResult2 = glob_search_fit1(fpartition, input_domain, False, limit)
        resultOflist2.append(tempResult2)
        tempResult3 = glob_search_fit1(rpartition, input_domain, False, limit)
        resultOflist3.append(tempResult3)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_glob_fit1", funcOfname, randOfseed)
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
        tempResult = glob_search_fit_hybrid(nopartition, input_domain, False, limit)
        resultOflist.append(tempResult)
        tempResult2 = glob_search_fit_hybrid(fpartition, input_domain, False, limit)
        resultOflist2.append(tempResult2)
        tempResult3 = glob_search_fit_hybrid(rpartition, input_domain, False, limit)
        resultOflist3.append(tempResult3)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_glob_hybrid", funcOfname, randOfseed)
    # local search fitness1 fitness2 or fit1+fit2 and no partition,fpartition,rpartition
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
        tempResult = local_search_fit1(nopartition, input_domain, limit)
        resultOflist.append(tempResult)
        tempResult2 = local_search_fit1(fpartition, input_domain, limit)
        resultOflist2.append(tempResult2)
        tempResult3 = local_search_fit1(rpartition, input_domain, limit)
        resultOflist3.append(tempResult3)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_local_fit1", funcOfname, randOfseed)
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
        tempResult = local_search_hybrid(nopartition, input_domain, limit)
        resultOflist.append(tempResult)
        tempResult2 = local_search_hybrid(fpartition, input_domain, limit)
        resultOflist2.append(tempResult2)
        tempResult3 = local_search_hybrid(rpartition, input_domain, limit)
        resultOflist3.append(tempResult3)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_local_hybrid", funcOfname, randOfseed)
    # random search fitness1 fitness2 or fit1+fit2 and no partition,fpartition,rpartition
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
        tempResult = random_search_fit1(nopartition, input_domain, limit)
        resultOflist.append(tempResult)
        tempResult2 = random_search_fit1(fpartition, input_domain, limit)
        resultOflist2.append(tempResult2)
        tempResult3 = random_search_fit1(rpartition, input_domain, limit)
        resultOflist3.append(tempResult3)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_random_fit1", funcOfname, randOfseed)
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
        tempResult = random_search_hybrid(nopartition, input_domain, limit)
        resultOflist.append(tempResult)
        tempResult2 = random_search_hybrid(fpartition, input_domain, limit)
        resultOflist2.append(tempResult2)
        tempResult3 = random_search_hybrid(rpartition, input_domain, limit)
        resultOflist3.append(tempResult3)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_random_hybrid", funcOfname, randOfseed)
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
        tempResult = binary_random_search(input_domain, limit)
        resultOflist.append(tempResult)
        resultOflist2.append(tempResult)
        resultOflist3.append(tempResult)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_brand_fit1", funcOfname, randOfseed)
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    # global search fitness1 fitness2 or fit1+fit2 and no partition,fpartition,rpartition
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
        tempResult = hybrid_search_fit1(nopartition, input_domain, False, limit)
        resultOflist.append(tempResult)
        tempResult2 = hybrid_search_fit1(fpartition, input_domain, False, limit)
        resultOflist2.append(tempResult2)
        tempResult3 = hybrid_search_fit1(rpartition, input_domain, False, limit)
        resultOflist3.append(tempResult3)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_hybrid_fit1", funcOfname, randOfseed)
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    # global search fitness1 fitness2 or fit1+fit2 and no partition,fpartition,rpartition
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
        tempResult = hybrid_search_fit_hybrid(nopartition, input_domain, False, limit)
        resultOflist.append(tempResult)
        tempResult2 = hybrid_search_fit_hybrid(fpartition, input_domain, False, limit)
        resultOflist2.append(tempResult2)
        tempResult3 = hybrid_search_fit_hybrid(rpartition, input_domain, False, limit)
        resultOflist3.append(tempResult3)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_hybrid_hybrid", funcOfname, randOfseed)

def tmp_test_hybrid(funcOfname,input_domain,times,repeatOftrials):
    randOfseed = np.random.random_integers(0, 1e8, repeatOftrials)
    resultOflist = []
    resultOflist2 = []
    resultOflist3 = []
    limit = limit_time(times, input_domain)
    print limit
    # global search fitness1 fitness2 or fit1+fit2 and no partition,fpartition,rpartition
    for i in range(0, repeatOftrials):
        np.random.seed(randOfseed[i])
        tempResult = hybrid_search_fit1(nopartition, input_domain, False, limit)
        resultOflist.append(tempResult)
        tempResult2 = hybrid_search_fit1(fpartition, input_domain, False, limit)
        resultOflist2.append(tempResult2)
        tempResult3 = hybrid_search_fit1(rpartition, input_domain, False, limit)
        resultOflist3.append(tempResult3)
    output_err(resultOflist, resultOflist2, resultOflist3, funcOfname + "_hybrid_fit1", funcOfname, randOfseed)


#fpartition results
def test_f(x,input_domain):
    x = [3698096,3068993]
    sResult2list = []
    for i in range(0,2):
        print x[i]
        np.random.seed(x[i])
        tempRes = glob_search_fit1(nopartition, input_domain, False)
        sResult2list.append(tempRes)
if __name__ == "__main__":
    for i in range(0,12):
        print i
        mp.dps = 30
        fun2search.rf = rf_12_l[i]
        fun2search.fp = gf_12_l[i]
        input_domain = input_domain_12[i]
        print input_domain
        name = eagt_12f_name[i]
        limit = limit_time(100, input_domain)
        print limit
        print test_two_fun(fun2search.rf, fun2search.fp, input_domain)
        #tmp_test_new(name, input_domain, 1000, 3)
