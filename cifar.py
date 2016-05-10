import nengo

# load in the CIFAR-10 dataset
# available at http://www.cs.toronto.edu/~kriz/cifar.html
import cPickle
fo = open('cifar-10-batches-py/data_batch_1')
data = cPickle.load(fo)
fo.close()

def stim_func(t):
    index = int(t / 0.1)
    return data['data'][index % len(data['data'])]


import numpy as np
import base64
import PIL
import cStringIO
def display_func(t, x):
    values = x.reshape((3, 32, 32))
    values = values.transpose((1,2,0))
    
    values = values.astype('uint8')

    png = PIL.Image.fromarray(values)
    buffer = cStringIO.StringIO()
    png.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue())
    
    display_func._nengo_html_ = '''
        <svg width="100%%" height="100%%" viewbox="0 0 100 100">
        <image width="100%%" height="100%%"
               xlink:href="data:image/png;base64,%s" 
               style="image-rendering: pixelated;">
        </svg>''' % (''.join(img_str))
    
    
model = nengo.Network()
with model:
    
    stim = nengo.Node(stim_func)
    
    display = nengo.Node(display_func, size_in=32*32*3)
    nengo.Connection(stim, display, synapse=None)
    