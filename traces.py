
import numpy as np
import matplotlib.pyplot as plt

def inds(n, sz=10):
    r = range(sz)
    v = [ int(float(i)*(n-1)/(sz-1)+.49) for i in r ]
    return v

class Trace(object):
    def __init__(self, inp, name, yFn=None):
        self.inp = inp
        self.name = name
        self.yFn = yFn
        self.data = []
        self.mbData = []
    def sample(self, v, sz=10):
        return v[inds(len(v), sz=sz)]
    def addEpoch(self, v):
        if self.mbData:
            v2 = np.mean(self.mbData)
            v = (v2, v)
            self.mbData = []
        self.data.append(v)
    def addBatch(self, v):
        pass
    def plot(self):
        if self.yFn == 'log':
            plt.semilogy(self.data)
        else:
            plt.plot(self.data)
    
class ScalarTrace(Trace):
    def __init__(self, inp, name, **k):
        super(ScalarTrace, self).__init__(inp, name, **k)
    def addBatch(self, v):
        self.mbData.append(v)
        
class BiasTrace(Trace):
    def __init__(self, inp, name, **k):
        super(BiasTrace, self).__init__(inp, name, **k)
    def addEpoch(self, v):
        v = np.sort(v.flatten())
        v = self.sample(v)
        self.data.append(v)

class WeightTrace(Trace):
    def __init__(self, inp, name, **k):
        super(WeightTrace, self).__init__(inp, name, **k)
    def addEpoch(self, v):
        v = np.sort(v.flatten())
        v = self.sample(v, 40)
        self.data.append(v)
        
class SMTrace(Trace):
    def __init__(self, inp, name, **k):
        super(SMTrace, self).__init__(inp, name, **k)
    def addEpoch(self, v):
        v = np.sort(v)
        v = np.mean(v, axis=0)
        v = self.sample(v)
        self.data.append(v)

class SMTopTrace(Trace):
    def __init__(self, inp, name, index=-1):
        self.index = index
        super(SMTopTrace, self).__init__(inp, name)
    def addEpoch(self, v):
        v = np.sort(v)
        v = v[:,self.index]
        v = np.sort(v)
        v = self.sample(v)
        self.data.append(v)


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
                plt.title(p.name)
                p.plot()    
        plt.show()
    def addBatch(self, vals):
        for t, v in zip(self.tFlat, vals):
            t.addBatch(v)
    def addEpoch(self, vals):
        for t, v in zip(self.tFlat, vals):
            t.addEpoch(v)
