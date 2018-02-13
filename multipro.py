from multiprocessing import Pool
import multiprocessing
import math
import time

def test_func(i):
    j = 0
    for x in xrange(10000):
        j += math.atan2(i, i)
    return j

if __name__ == '__main__':
    pool_size = multiprocessing.cpu_count()
    print pool_size
    var = range(500)
    p = Pool(processes=4)
    tb = time.time()
    var = p.map(test_func, var)
    print time.time() - tb
    tb2 = time.time()
    var = map(test_func, var)
    print time.time() - tb2