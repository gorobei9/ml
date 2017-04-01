
import tensorflow as tf



class Node(object):
    def __init__(self, inp):
        self.inp = inp
        self.prn()
        
    def prn(self):
        print '%40s: %8s     %s' % (self.__class__.__name__, self.nParam(), self.shapeStr())
        
    def out(self):
        return self._out
    def format(self):
        return self.inp.format()
    def dims(self):
        return self.inp.dims()
    def shape(self):
        return tuple([ self.dims()[v] for v in self.format()] )
    def shapeStr(self):
        ret = [ '%s:%s ' % (v, sz) for v, sz in zip(self.format(), self.shape()) ]
        return ''.join(ret)
    def nParam(self):
        return 0
            
class Datasets(Node):
    def __init__(self, name, raw):
        self.inp = None
        self.name = name
        super(Datasets, self).__init__(None)
        self.raw = raw
        self.test       = DataSet(raw.test)
        self.validation = DataSet(raw.validation)
        self.train      = DataSet(raw.train)
        self.x = tf.placeholder(tf.float32, shape=[None, self.shape()[-1]])
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.nLabels()])
        self._out = self.x
        
    def format(self):
        return 'NC'
    def dims(self):
        return { 'N': -1, 'C': 784 }
    def nLabels(self):
        return self.raw.train.labels.shape[-1]
    
    def info(self):
        print 'Dataset:', self.name
        print '  shape:', self.shapeStr()
        print '  labels:', self.nLabels()
        print '  test:      ', self.test.shape()
        print '  validation:', self.validation.shape()
        print '  train:     ', self.train.shape()
        
class DataSet(object):
    def __init__(self, inp):
        self.inp = inp
    def shape(self):
        return self.inp.images.shape
       

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

class Conv2d(Node):
    def __init__(self, inp, size, outChannels, stride=1):
        self.size = size
        self.outChannels = outChannels
        self.stride = stride
        self.ncIn = inp.dims()['C']
        self.W = weight_variable((size, size, self.ncIn, outChannels))
        super(Conv2d, self).__init__(inp)
        
        x = self.inp.out()
        W = self.W
        stride = self.stride
        self._out = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME') 

    def dims(self):
        d = self.inp.dims().copy()
        d['C'] = self.outChannels
        return d
          
    def nParam(self):
        return self.size * self.size * self.ncIn * self.outChannels
    
class Bias(Node):
    def __init__(self, inp):
        sz = inp.dims()['C']
        self.b = bias_variable([sz])
        super(Bias, self).__init__(inp)
    def out(self):
        return self.inp.out() + self.b
    def nParam(self):
        return self.inp.dims()['C']
    
class MaxPool(Node):
    def __init__(self, inp, size, stride=1):
        self.size = size
        self.stride = stride
        self.padding = 'SAME'
        super(MaxPool, self).__init__(inp)
        
        x = self.inp.out()
        size = self.size
        stride = self.stride
        padding = self.padding
        self._out = tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding=padding)
        
    def dims(self):
        d = self.inp.dims().copy()
        size = self.size
        d['W'] = d['W'] / self.stride
        d['H'] = d['H'] / self.stride
        return d

class Reshape(Node):
    def __init__(self, inp, channel, outChannels, outShape):
        super(Reshape, self).__init__(inp)
        inNode = self.inp
        self._out = tf.reshape(inNode.out(), self.shape())
        
    def format(self):
        return 'NWHC'
    def dims(self):
        return { 'N': -1, 'H': 28, 'W': 28, 'C': 1 }
    def shape(self):
        return tuple([ self.dims()[v] for v in self.format()] )

class Relu(Node):
    def __init__(self, inp):
        super(Relu, self).__init__(inp)
        self._out = tf.nn.relu(self.inp.out())
    def dims(self): 
        return self.inp.dims()
    def format(self):
        return self.inp.format()

class Softmax(Node):
    def __init__(self, inp):
        super(Softmax, self).__init__(inp)
        self._out = tf.nn.softmax(self.inp.out())
        
    def dims(self): 
        return self.inp.dims()
    def format(self):
        return self.inp.format()

class Flatten(Node):
    def __init__(self, inp):
        v = 1
        for n in inp.shape():
            v *= n
        self.sz = -v
        super(Flatten, self).__init__(inp)
        x = self.inp.out()
        self._out = tf.reshape(x, self.shape())
        
    def format(self):
        return 'NC'
    def dims(self):
        return { 'N': -1, 'C': self.sz }
    
class Linear(Node):
    def __init__(self, inp, sz):
        self.sz = sz
        self.inSz = inp.dims()['C']
        self.W = weight_variable([self.inSz, sz])
        super(Linear, self).__init__(inp)
        x = self.inp.out()
        self._out = tf.matmul(x, self.W)
        
    def format(self):
        return 'NC'
    def dims(self):
        return { 'N': -1, 'C': self.sz }
    def nParam(self):
        return self.inSz * self.sz
    
class Dropout(Node):
    def __init__(self, inp, keep_prob):
        self.keep_prob = keep_prob
        super(Dropout, self).__init__(inp)
        x = self.inp.out()
        self._out = tf.nn.dropout(x, self.keep_prob)
        
class Accuracy(Node):
    def __init__(self, inp, y_):
        super(Accuracy, self).__init__(inp)
        correct_prediction = tf.equal(tf.argmax(inp.out(), 1), tf.argmax(y_, 1))
        self._out = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    def dims(self):
        return { 'N': -1, 'C': 1 }

class CrossEntropy(Node):
    def __init__(self, inp, y_):
        super(CrossEntropy, self).__init__(inp)
        self._out = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(inp.out()), reduction_indices=[1]))
    def dims(self):
        return { 'N': -1, 'C': 1 }

class Mean(Node):
    def __init__(self, inp):
        super(Mean, self).__init__(inp)
        self._out = tf.reduce_mean(inp.out())
    def dims(self):
        return { 'N': 1, 'C': 1 }

class SoftmaxCrossEntropyWithLogits(Node):
    def __init__(self, inp, y_):
        super(SoftmaxCrossEntropyWithLogits, self).__init__(inp)
        self._out = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=inp.out())
    def dims(self):
        return { 'N': -1, 'C': 1 } # XXX
        
    
