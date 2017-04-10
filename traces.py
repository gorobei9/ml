
import numpy as np
import matplotlib.pyplot as plt

def inds(n, sz=10):
    r = range(sz)
    v = [ int(float(i)*(n-1)/(sz-1)+.49) for i in r ]
    return v

def inds2(n):
    r = []
    i = 1
    while i < n/2:
        r.append(i)
        i = i * 2
    return r + [ -i for i in reversed(r) ]

class Trace(object):
    def __init__(self, inp, name, yFn=None, skip=0):
        self.inp = inp
        self.name = name
        self.yFn = yFn
        self.data = []
        self.mbData = []
        self.skip = skip
        self.fn = None
    def sample(self, v, sz=10):
        return v[inds(len(v), sz=sz)]
    def mapE(self, v):
        return v
    def addEpoch(self, v):
        v = self.mapE(v)
        if self.fn == '(pow2 bins)':
            v = v[inds2(len(v))]
        if self.mbData:
            v2 = np.mean(self.mbData)
            v = (v2, v)
            self.mbData = []
        self.data.append(v)
    def addBatch(self, v):
        pass
    def plot(self):
        if len(self.data) <= self.skip:
            return
        d = np.array(self.data[self.skip:])
        y = [ e+self.skip for e in range(len(d)) ]

        if self.fn == '(pow2 bins)':
            c = ('bgcm' * 10 + 'rk')[-len(d[0])/2:]
            c = c + ''.join(list(reversed(c)))

            if self.yFn == 'log':
                for i, cc in enumerate(c):
                    dd = d[:, i]
                    plt.semilogy(y, dd, color=cc)
            else:
                for i, cc in enumerate(c):
                    dd = d[:, i]
                    plt.plot(y, dd, color=cc)

        else:
            if self.yFn == 'log':
                plt.semilogy(y, d)
            else:
                plt.plot(y, d)

    def title(self):
        s = self.name
        if self.fn:
            s = s + ' ' + self.fn
        if self.yFn:
            s = s + ', log'
        if self.skip:
            s = s + ' (first %s skipped)' % self.skip
        return s
    
class ScalarTrace(Trace):
    def __init__(self, inp, name, **k):
        super(ScalarTrace, self).__init__(inp, name, **k)
    def addBatch(self, v):
        self.mbData.append(v)
  
class BiasTrace(Trace):
    def __init__(self, inp, name, **k):
        super(BiasTrace, self).__init__(inp, name, **k)
        self.fn = '(pow2 bins)'
    def mapE(self, v):
        v = np.sort(v.flatten())
        return v
    
class WeightTrace(Trace):
    def __init__(self, inp, name, **k):
        super(WeightTrace, self).__init__(inp, name, **k)
        self.fn = '(pow2 bins)'
    def mapE(self, v):
        v = np.sort(v.flatten())
        return v
        
class OutTrace(Trace):
    def __init__(self, inp, name, **k):
        super(OutTrace, self).__init__(inp, name, **k)
        self.fn = '(pow2 bins)'
    def mapE(self, v):
        v = np.sort(v.reshape([v.shape[0]]+[-1]))
        v = np.mean(v, axis=0)
        return v
    
class TopTrace(Trace):
    def __init__(self, inp, name, index=-1):
        self.index = index
        super(TopTrace, self).__init__(inp, name)
        self.fn = '(pow2 bins)'
    def mapE(self, v):
        v = np.sort(v.reshape([v.shape[0]]+[-1]))
        v = v[:,self.index]
        v = np.sort(v)
        return v


class Track(object):
    def __init__(self, trackables):
        self.trackables = trackables
        tFlat = []
        for t in trackables:
            tFlat.extend(t)
        self.tFlat = tFlat
    def nodes(self):
        return [ t.inp for t in self.tFlat ]
    def plotAll(self, width=20, height=12):
        trackables = self.trackables
        plt.figure(figsize=(width, height))
        c = max( [ len(t) for t in trackables ] )
        r = len(trackables)
        for y, t in enumerate(trackables):
            for x, p in enumerate(t):
                i = y * c + x + 1
                plt.subplot(r, c, i)
                plt.title(p.title())
                p.plot()
        plt.tight_layout()
        plt.show()
    def addBatch(self, vals):
        for t, v in zip(self.tFlat, vals):
            t.addBatch(v)
    def addEpoch(self, vals):
        for t, v in zip(self.tFlat, vals):
            t.addEpoch(v)
