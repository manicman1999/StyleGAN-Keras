#Imports
from PIL import Image
from math import floor
import numpy as np
import time
from functools import partial
from random import random

#Config Stuff
im_size = 256
latent_size = 512
BATCH_SIZE = 4
directory = "Rooms"
n_images = 2686
suff = 'jpg'
cmode = 'YCbCr'

""" For testing color space ranges
temp = Image.open("data/Earth/im (2).jpg").convert(cmode)
temp1 = np.array(temp, dtype='float32')

print(np.max(temp1[...,0]))
print(np.min(temp1[...,0]))
print(np.max(temp1[...,1]))
print(np.min(temp1[...,1]))
print(np.max(temp1[...,2]))
print(np.min(temp1[...,2]))
"""

#Style Z
def noise(n):
    return np.random.normal(0.0, 1.0, size = [n, latent_size])

#Noise Sample
def noiseImage(n):
    return np.random.uniform(0.0, 1.0, size = [n, im_size, im_size, 1])

#Get random samples from an array
def get_rand(array, amount):
    
    idx = np.random.randint(0, array.shape[0], amount)
    return array[idx]

#Import Images Function
def import_images(loc, flip = True, suffix = 'png'):
    
    out = []
    cont = True
    i = 1
    print("Importing Images...")
    
    while(cont):
        try:
            temp = Image.open("data/"+loc+"/im ("+str(i)+")."+suffix+"").convert(cmode)
            temp = temp.resize((im_size, im_size), Image.BICUBIC)
            temp1 = np.array(temp, dtype='float32') / 255
            out.append(temp1)
            if flip:
                out.append(np.flip(out[-1], 1))
            
            i = i + 1
        except:
            cont = False
        
    print(str(i-1) + " images imported.")
            
    return np.array(out)

#This is the REAL data generator, which can take images from disk and temporarily use them in your program.
#Probably could/should get optimized at some point
class dataGenerator(object):
    
    def __init__(self, loc, n, flip = True, suffix = 'png'):
        self.loc = "data/"+loc
        self.flip = flip
        self.suffix = suffix
        self.n = n
    
    def get_batch(self, amount):
        
        idx = np.random.randint(0, self.n - 1, amount) + 1
        out = []
        
        for i in idx:
            temp = Image.open(self.loc+"/im ("+str(i)+")."+self.suffix+"").convert(cmode)
            temp1 = np.array(temp, dtype='float32') / 255
            if self.flip and random() > 0.5:
                temp1 = np.flip(temp1, 1)
                
            out.append(temp1)
            
        
        return np.array(out)

        


#Imports for layers and models
from keras.layers import Conv2D, Dense, AveragePooling2D, LeakyReLU, Activation
from keras.layers import Reshape, UpSampling2D, Dropout, Flatten, Input, add, Cropping2D
from keras.models import model_from_json, Model
from keras.optimizers import Adam
from adamlr import Adam_lr_mult
import keras.backend as K

from AdaIN import AdaInstanceNormalization


#r1/r2 gradient penalty
def gradient_penalty_loss(y_true, y_pred, averaged_samples, weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    
    # weight * ||grad||^2
    # Penalize the gradient norm
    return K.mean(gradient_penalty * weight)

#Upsample, Convolution, AdaIN, Noise, Activation, Convolution, AdaIN, Noise, Activation
def g_block(inp, style, noise, fil, u = True):
    
    b = Dense(fil, kernel_initializer = 'he_normal', bias_initializer = 'ones')(style)
    b = Reshape([1, 1, fil])(b)
    g = Dense(fil, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(style)
    g = Reshape([1, 1, fil])(g)

    n = Conv2D(filters = fil, kernel_size = 1, padding = 'same', kernel_initializer = 'zeros', bias_initializer = 'zeros')(noise)
    
    if u:
        out = UpSampling2D(interpolation = 'bilinear')(inp)
        out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal', bias_initializer = 'zeros')(out)
    else:
        out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal', bias_initializer = 'zeros')(inp)

    out = add([out, n])
    out = AdaInstanceNormalization()([out, b, g])
    out = LeakyReLU(0.01)(out)
    
    b = Dense(fil, kernel_initializer = 'he_normal', bias_initializer = 'ones')(style)
    b = Reshape([1, 1, fil])(b)
    g = Dense(fil, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(style)
    g = Reshape([1, 1, fil])(g)

    n = Conv2D(filters = fil, kernel_size = 1, padding = 'same', kernel_initializer = 'zeros', bias_initializer = 'zeros')(noise)
    
    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal', bias_initializer = 'zeros')(out)
    out = add([out, n])
    out = AdaInstanceNormalization()([out, b, g])
    out = LeakyReLU(0.01)(out)
    
    return out

#Convolution, Activation, Pooling, Convolution, Activation
def d_block(inp, fil, p = True):
    
    route2 = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal', bias_initializer = 'zeros')(inp)
    route2 = LeakyReLU(0.01)(route2)
    if p:
        route2 = AveragePooling2D()(route2)
    route2 = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal', bias_initializer = 'zeros')(route2)
    out = LeakyReLU(0.01)(route2)
    
    return out

#This object holds the models
class GAN(object):
    
    def __init__(self, steps = 1, lr = 0.0001, decay = 0.00001):
        
        #Models
        self.D = None
        self.G = None
        self.S = None
        
        self.DM = None
        self.DMM = None
        self.AM = None
        self.MM = None
        
        #Config
        #Automatic Decay
        temp = (1 - decay) ** steps
        self.LR = lr * temp
        self.steps = steps
        
        #Calculate number of layers needed
        self.style_layers = 0
        
        #Init Models
        self.discriminator()
        self.generator()
        self.stylist()
        
    def discriminator(self):
        
        if self.D:
            return self.D
        
        inp = Input(shape = [im_size, im_size, 3])
        
        # Size
        x = d_block(inp, 16) #Size / 2
        x = d_block(x, 32) #Size / 4
        x = d_block(x, 64) #Size / 8
        
        if (im_size > 32):
            x = d_block(x, 128) #Size / 16
        
        if (im_size > 64):
            x = d_block(x, 192) #Size / 32
        
        if (im_size > 128):
            x = d_block(x, 256) #Size / 64
        
        if (im_size > 256):
            x = d_block(x, 384) #Size / 128
            
        if (im_size > 512):
            x = d_block(x, 512) #Size / 256
            
            
        x = Flatten()(x)
        
        x = Dense(128, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x)
        x = LeakyReLU(0.01)(x)
        
        x = Dropout(0.2)(x)
        x = Dense(1, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x)
        
        self.D = Model(inputs = inp, outputs = x)
        
        return self.D
    
    def generator(self):
        
        if self.G:
            return self.G
        
        inp_s = []
        ss = im_size
        while ss >= 4:
            inp_s.append(Input(shape = [512]))
            ss = int(ss / 2)
        
        self.style_layers = len(inp_s)
        
        #Get the noise image and crop for each size
        inp_n = Input(shape = [im_size, im_size, 1])
        noi = [Activation('linear')(inp_n)]
        curr_size = im_size
        while curr_size > 4:
            curr_size = int(curr_size / 2)
            noi.append(Cropping2D(int(curr_size/2))(noi[-1]))
        
        #Here do the actual generation stuff
        inp = Input(shape = [1])
        x = Dense(4 * 4 * im_size, kernel_initializer = 'ones', bias_initializer = 'zeros')(inp)
        x = Reshape([4, 4, im_size])(x)
        x = g_block(x, inp_s[0], noi[-1], im_size, u=False)
        
        if(im_size >= 1024):
            x = g_block(x, inp_s[-8], noi[7], 512) # Size / 64
        if(im_size >= 512):
            x = g_block(x, inp_s[-7], noi[6], 384) # Size / 64
        if(im_size >= 256):
            x = g_block(x, inp_s[-6], noi[5], 256) # Size / 32
        if(im_size >= 128):
            x = g_block(x, inp_s[-5], noi[4], 192) # Size / 16
        if(im_size >= 64):
            x = g_block(x, inp_s[-4], noi[3], 128) # Size / 8
            
        x = g_block(x, inp_s[-3], noi[2], 64) # Size / 4
        x = g_block(x, inp_s[-2], noi[1], 32) # Size / 2
        x = g_block(x, inp_s[-1], noi[0], 16) # Size
        
        x = Conv2D(filters = 3, kernel_size = 1, padding = 'same', activation = 'sigmoid', bias_initializer = 'zeros')(x)
        
        self.G = Model(inputs = inp_s + [inp_n, inp], outputs = x)
        
        return self.G
    
    def stylist(self):
        
        if self.S:
            return self.S
        
        #Mapping FC, I only used 5 fully connected layers instead of 8 for faster training
        inp_s = Input(shape = [latent_size])
        sty = Dense(512, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(inp_s)
        sty = LeakyReLU(0.01)(sty)
        sty = Dense(512, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(sty)
        sty = LeakyReLU(0.01)(sty)
        sty = Dense(512, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(sty)
        sty = LeakyReLU(0.01)(sty)
        sty = Dense(512, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(sty)
        
        self.S = Model(inputs = inp_s, outputs = sty)
        
        return self.S
    
    def AdModel(self):
        
        #D does not update
        self.D.trainable = False
        for layer in self.D.layers:
            layer.trainable = False
        
        #G does update
        self.G.trainable = True
        for layer in self.G.layers:
            layer.trainable = True
        
        #S does update
        self.S.trainable = True
        for layer in self.S.layers:
            layer.trainable = True
        
        #This model is simple sequential one with inputs and outputs
        gi = Input(shape = [latent_size])
        gs = self.S(gi)
        gi2 = Input(shape = [im_size, im_size, 1])
        gi3 = Input(shape = [1])
        
        gf = self.G(([gs] * self.style_layers) + [gi2, gi3])
        df = self.D(gf)
        
        self.AM = Model(inputs = [gi, gi2, gi3], outputs = df)
            
        learning_rate_multipliers = {}
        learning_rate_multipliers['model_3'] = 0.1
            
        self.AM.compile(optimizer = Adam_lr_mult(self.LR, beta_1 = 0, beta_2 = 0.99, decay = 0.00001, multipliers = learning_rate_multipliers), loss = 'mse')
        
        return self.AM
    
    def MixModel(self):
        
        #D does not update
        self.D.trainable = False
        for layer in self.D.layers:
            layer.trainable = False
        
        #G does update
        self.G.trainable = True
        for layer in self.G.layers:
            layer.trainable = True
            
        #S does update
        self.S.trainable = True
        for layer in self.S.layers:
            layer.trainable = True
        
        #This model is simple sequential one with inputs and outputs
        inp_s = []
        ss = []
        for _ in range(self.style_layers):
            inp_s.append(Input([latent_size]))
            ss.append(self.S(inp_s[-1]))
            
            
        gi2 = Input(shape = [im_size, im_size, 1])
        gi3 = Input(shape = [1])
        
        gf = self.G(ss + [gi2, gi3])
        df = self.D(gf)
        
        self.MM = Model(inputs = inp_s + [gi2, gi3], outputs = df)

        learning_rate_multipliers = {}
        learning_rate_multipliers['model_3'] = 0.1
            
        self.MM.compile(optimizer = Adam_lr_mult(self.LR, beta_1 = 0, beta_2 = 0.99, decay = 0.00001, multipliers = learning_rate_multipliers), loss = 'mse')
        
        return self.MM
    
    def DisModel(self):
        
        #D does update
        self.D.trainable = True
        for layer in self.D.layers:
            layer.trainable = True
        
        #G does not update
        self.G.trainable = False
        for layer in self.G.layers:
            layer.trainable = False
        
        #S does not update
        self.S.trainable = False
        for layer in self.S.layers:
            layer.trainable = False
        
        # Real Pipeline
        ri = Input(shape = [im_size, im_size, 3])
        dr = self.D(ri)
        
        # Fake Pipeline
        gi = Input(shape = [latent_size])
        gs = self.S(gi)
        gi2 = Input(shape = [im_size, im_size, 1])
        gi3 = Input(shape = [1])
        gf = self.G(([gs] * self.style_layers) + [gi2, gi3])
        df = self.D(gf)
        
        # Samples for gradient penalty
        # For r1 use real samples (ri)
        # For r2 use fake samples (gf)
        #da = self.D(ri)
        
        # Model With Inputs and Outputs
        self.DM = Model(inputs=[ri, gi, gi2, gi3], outputs=[dr, df, dr])
        
        # Create partial of gradient penalty loss
        # For r1, averaged_samples = ri
        # For r2, averaged_samples = gf
        # Weight of 10 typically works
        partial_gp_loss = partial(gradient_penalty_loss, averaged_samples = ri, weight = 50)
        
        #Compile With Corresponding Loss Functions
        self.DM.compile(optimizer=Adam(self.LR, beta_1 = 0, beta_2 = 0.99, decay = 0.00001), loss=['mse', 'mse', partial_gp_loss])
        
        return self.DM
    
    def MixModelD(self):
        
        #D does update
        self.D.trainable = True
        for layer in self.D.layers:
            layer.trainable = True
        
        #G does not update
        self.G.trainable = False
        for layer in self.G.layers:
            layer.trainable = False
        
        #S does not update
        self.S.trainable = False
        for layer in self.S.layers:
            layer.trainable = False
        
        #This model is simple sequential one with inputs and outputs
        inp_s = []
        ss = []
        for _ in range(self.style_layers):
            inp_s.append(Input([latent_size]))
            ss.append(self.S(inp_s[-1]))
            
            
        gi2 = Input(shape = [im_size, im_size, 1])
        gi3 = Input(shape = [1])
        
        gf = self.G(ss + [gi2, gi3])
        df = self.D(gf)
        
        ri = Input(shape = [im_size, im_size, 3])
        dr = self.D(ri)
        
        self.DMM = Model(inputs = [ri] + inp_s + [gi2, gi3], outputs=[dr, df, dr])
        
        partial_gp_loss = partial(gradient_penalty_loss, averaged_samples = ri, weight = 50)
            
        self.DMM.compile(optimizer=Adam(self.LR, beta_1 = 0, beta_2 = 0.99, decay = 0.00001), loss=['mse', 'mse', partial_gp_loss])
        
        return self.DMM
    
    def predict(self, inputs):
        
        for i in range(len(inputs) - 2):
            inputs[i] = self.S.predict(inputs[i])
            
        return self.G.predict(inputs, batch_size = 4)
        
        
from keras.datasets import cifar10
class WGAN(object):
    
    def __init__(self, steps = 1, lr = 0.0001, decay = 0.00001, silent = True):
        
        self.GAN = GAN(steps = steps, lr = lr, decay = decay)
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
        self.MixModel = self.GAN.MixModel()
        self.MixModelD = self.GAN.MixModelD()
        self.generator = self.GAN.generator()
        
        self.lastblip = time.clock()
        
        self.noise_level = 0
        
        #self.ImagesA = import_images(directory, True)
        self.im = dataGenerator(directory, n_images, suffix = suff, flip = True)
        #(self.im, _), (_, _) = cifar10.load_data()
        #self.im = np.float32(self.im) / 255
        
        self.silent = silent

        #Train Generator to be in the middle, not all the way at real. Apparently works better??
        self.ones = np.ones((BATCH_SIZE, 1), dtype=np.float32)
        self.zeros = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
        self.nones = -self.ones
        
        self.enoise = noise(8)
        self.enoiseImage = noiseImage(8)
        
        self.t = [[], []]
    
    def train(self):
        
        #Train Alternating
        t1 = time.clock()
        if self.GAN.steps % 10 <= 5:
            a = self.train_dis()
            t2 = time.clock()
            b = self.train_gen()
            t3 = time.clock()
        else:
            a = self.train_mix_d()
            t2 = time.clock()
            b = self.train_mix_g()
            t3 = time.clock()
            
        self.t[0].append(t2-t1)
        self.t[1].append(t3-t2)
        
        #Print info
        if self.GAN.steps % 20 == 0 and not self.silent:
            print("\n\nRound " + str(self.GAN.steps) + ":")
            print("D: " + str(a))
            print("G: " + str(b))
            s = round((time.clock() - self.lastblip) * 1000) / 1000
            print("T: " + str(s) + " sec")
            self.lastblip = time.clock()
            
            if self.GAN.steps % 100 == 0:
                print("TD: " + str(np.sum(self.t[0])))
                print("TG: " + str(np.sum(self.t[1])))
                
                self.t = [[], []]
                
            
            #Save Model
            if self.GAN.steps % 500 == 0:
                self.save(floor(self.GAN.steps / 10000))
            if self.GAN.steps % 1000 == 0:
                self.evaluate(floor(self.GAN.steps / 1000))
                self.evalMix(floor(self.GAN.steps / 1000))
                self.evalTrunc(floor(self.GAN.steps / 1000))
            
        
        self.GAN.steps = self.GAN.steps + 1
          
    def train_dis(self):
        
        #Get Data 
        #self.im.get_batch(BATCH_SIZE)
        #get_rand(self.im, BATCH_SIZE)
        train_data = [self.im.get_batch(BATCH_SIZE), noise(BATCH_SIZE), noiseImage(BATCH_SIZE), self.ones]
        
        #Train
        d_loss = self.DisModel.train_on_batch(train_data, [self.ones, self.nones, self.ones])
        
        return d_loss
    
    def train_mix_d(self):
        
        threshold = np.int32(np.random.uniform(0.0, self.GAN.style_layers, size = [BATCH_SIZE]))
        n1 = noise(BATCH_SIZE)
        n2 = noise(BATCH_SIZE)
        
        n = []
        
        for i in range(self.GAN.style_layers):
            n.append([])
            for j in range(BATCH_SIZE):
                if i < threshold[j]:
                    n[i].append(n1[j])
                else:
                    n[i].append(n2[j])
            n[i] = np.array(n[i])
        
        images = self.im.get_batch(BATCH_SIZE)
        
        #Train
        d_loss = self.MixModelD.train_on_batch([images] + n + [noiseImage(BATCH_SIZE), self.ones], [self.ones, self.nones, self.ones])
        
        return d_loss
       
    def train_gen(self):
        
        #Train
        g_loss = self.AdModel.train_on_batch([noise(BATCH_SIZE), noiseImage(BATCH_SIZE), self.ones], self.ones)
        
        return g_loss
    
    def train_mix_g(self):
        
        threshold = np.int32(np.random.uniform(0.0, self.GAN.style_layers, size = [BATCH_SIZE]))
        n1 = noise(BATCH_SIZE)
        n2 = noise(BATCH_SIZE)
        
        n = []
        
        for i in range(self.GAN.style_layers):
            n.append([])
            for j in range(BATCH_SIZE):
                if i < threshold[j]:
                    n[i].append(n1[j])
                else:
                    n[i].append(n2[j])
            n[i] = np.array(n[i])
                    
        
        #Train
        g_loss = self.MixModel.train_on_batch(n + [noiseImage(BATCH_SIZE), self.ones], self.ones)
        
        return g_loss
    
    def evaluate(self, num = 0): #8x8 images, bottom row is constant
        
        n = noise(56)
        n2 = noiseImage(56)
        
        im = self.GAN.predict(([n] * self.GAN.style_layers) + [n2, np.ones([56, 1])])
        im3 = self.GAN.predict(([self.enoise] * self.GAN.style_layers) + [self.enoiseImage, np.ones([8, 1])])
        
        r = []
        r.append(np.concatenate(im[:8], axis = 1))
        r.append(np.concatenate(im[8:16], axis = 1))
        r.append(np.concatenate(im[16:24], axis = 1))
        r.append(np.concatenate(im[24:32], axis = 1))
        r.append(np.concatenate(im[32:40], axis = 1))
        r.append(np.concatenate(im[40:48], axis = 1))
        r.append(np.concatenate(im[48:56], axis = 1))
        r.append(np.concatenate(im3[:8], axis = 1))
        
        c1 = np.concatenate(r, axis = 0)
        
        x = Image.fromarray(np.uint8(c1*255), mode = cmode)
        
        x.save("Results/i"+str(num)+"ii.jpg")
        
    
    def evalMix(self, num = 0):
        
        bn = noise(8)
        sn = noise(8)
        
        n = []
        for i in range(self.GAN.style_layers):
            n.append([])
        
        for i in range(8):
            for j in range(8):
                for l in range(0, int(self.GAN.style_layers/2)):
                    n[l].append(bn[i])
                for l in range(int(self.GAN.style_layers/2), self.GAN.style_layers):
                    n[l].append(sn[j])
        
        for i in range(self.GAN.style_layers):
            n[i] = np.array(n[i])
            
        
        im = self.GAN.predict(n + [noiseImage(64), np.ones([64, 1])])
        
        r = []
        r.append(np.concatenate(im[:8], axis = 1))
        r.append(np.concatenate(im[8:16], axis = 1))
        r.append(np.concatenate(im[16:24], axis = 1))
        r.append(np.concatenate(im[24:32], axis = 1))
        r.append(np.concatenate(im[32:40], axis = 1))
        r.append(np.concatenate(im[40:48], axis = 1))
        r.append(np.concatenate(im[48:56], axis = 1))
        r.append(np.concatenate(im[56:], axis = 1))
        c = np.concatenate(r, axis = 0)
        
        x = Image.fromarray(np.uint8(c*255), mode = cmode)
        
        x.save("Results/i"+str(num)+"mm.jpg")
        
    def evalTrunc(self, num = 0, trunc = 2.0, scale = 1, nscale = 0.8, custom_noise = np.array([0])):
        
        ss = self.GAN.S.predict(noise(2048), batch_size = 128)
        mean = np.mean(ss, axis = 0)
        std = np.std(ss, axis = 0)
        
        if custom_noise.shape[0] != 16:
            noi = noise(16)
        else:
            noi = custom_noise
        
        n = self.GAN.S.predict(noi)
        n2 = noiseImage(16) * nscale
        
        for i in range(n.shape[0]):
            n[i] = np.clip(n[i], mean - (std*trunc), mean + (std * trunc))
            
            if scale != 1:
                n[i] = (n[i] - mean) * scale + mean
        
        im = self.GAN.G.predict(([n] * self.GAN.style_layers) + [n2, np.ones([16, 1])])
        
        r = []
        r.append(np.concatenate(im[:4], axis = 1))
        r.append(np.concatenate(im[4:8], axis = 1))
        r.append(np.concatenate(im[8:12], axis = 1))
        r.append(np.concatenate(im[12:16], axis = 1))
        
        c1 = np.concatenate(r, axis = 0)
        
        x = Image.fromarray(np.uint8(c1*255), mode = cmode)
        
        x.save("Results/i"+str(num)+"tt.jpg")
    
    def saveModel(self, model, name, num): #Save a Model
        json = model.to_json()
        with open("Models/"+name+".json", "w") as json_file:
            json_file.write(json)
            
        model.save_weights("Models/"+name+"_"+str(num)+".h5")
        
    def loadModel(self, name, num): #Load a Model
        
        file = open("Models/"+name+".json", 'r')
        json = file.read()
        file.close()
        
        mod = model_from_json(json, custom_objects = {'AdaInstanceNormalization': AdaInstanceNormalization})
        mod.load_weights("Models/"+name+"_"+str(num)+".h5")
        
        return mod
    
    def save(self, num): #Save JSON and Weights into /Models/
        self.saveModel(self.GAN.S, "sty", num)
        self.saveModel(self.GAN.G, "gen", num)
        self.saveModel(self.GAN.D, "dis", num)
        

    def load(self, num): #Load JSON and Weights from /Models/
        steps1 = self.GAN.steps
        
        self.GAN = None
        self.GAN = GAN()

        #Load Models
        self.GAN.S = self.loadModel("sty", num)
        self.GAN.G = self.loadModel("gen", num)
        self.GAN.D = self.loadModel("dis", num)
        
        self.GAN.steps = steps1
        
        self.generator = self.GAN.generator()
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
        self.MixModel = self.GAN.MixModel()
        self.MixModelD = self.GAN.MixModelD()
        
        
        
if __name__ == "__main__":
    model = WGAN(lr = 0.0001, silent = False)
    
    while(True):
        model.train()


