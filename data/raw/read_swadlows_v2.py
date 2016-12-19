import pylab as pl
import h5py
import cPickle
from scipy.optimize import leastsq
import scipy.signal as ss

labels = list(pl.zeros(16))
for i in xrange(16):
    labels[i] = str(i)

class Signal:
    def __init__(self):
        self.data = {}

s = Signal()
                
for i in xrange(4):
    s.data[h5py.File("Swadlow_LFP_v2.h5",'r').keys()[i]] = {}
    s.data[h5py.File("Swadlow_LFP_v2.h5",'r').keys()[i]]['raw_data'] = pl.array(h5py.File("Swadlow_LFP_v2.h5",'r')[h5py.File("Swadlow_LFP_v2.h5",'r').keys()[i]])
    s.data[h5py.File("Swadlow_LFP_v2.h5",'r').keys()[i]]['tvec'] = pl.array(h5py.File("Swadlow_LFP_v2.h5",'r')['t'])
    s.data[h5py.File("Swadlow_LFP_v2.h5",'r').keys()[i]]['dt'] = 0.025
    s.data[h5py.File("Swadlow_LFP_v2.h5",'r').keys()[i]]['labels'] = labels


f = open('Swadlows2002_1BN1.c', 'wb')
cPickle.dump(s.data['data_02_1BN1'], f, protocol=2)
f.close()
f = open('Swadlows2002_1BN2.c', 'wb')
cPickle.dump(s.data['data_02_1BN2'], f, protocol=2)
f.close()
f = open('Stoelzel2008_1E.c', 'wb')
cPickle.dump(s.data['data_08_1E'], f, protocol=2)
f.close()
f = open('Stoelzel2008_2A1.c', 'wb')
cPickle.dump(s.data['data_08_2A1'], f, protocol=2)
f.close()

