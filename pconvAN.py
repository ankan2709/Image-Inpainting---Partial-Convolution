import numpy as np
import tensorflow as tf
from tensorflow import keras  as K
import tensorflow.keras.layers as L
from tensorflow.keras.layers import Conv2D, InputSpec
from matplotlib import pyplot as plt
from PIL import Image as img
from os import listdir
from os.path import isfile, join
import cv2
import seaborn as sns
import csv
import os
from customBN.CustomNorm import CustomNorm
from natsort import natsorted as ns

# from IPython import display

sns.color_palette()
p = sns.hls_palette(8, l=.5, s=.9)
sns.set_palette(p)

# partial convolution layer definition
class PConv2D(tf.keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        super(PConv2D, self).__init__(*args, **kwargs)
        self.input_spec = [InputSpec(ndim = 4), InputSpec(ndim = 4)]
        
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
            
        self.input_dim = input_shape[0][channel_axis]
        
        self.kernel_shape = self.kernel_size + (self.input_dim, self.filters)
        
        self.kernel = self.add_weight(shape = self.kernel_shape,
                                      initializer = self.kernel_initializer,
                                      name = 'image_kernel',
                                      regularizer = self.kernel_regularizer,
                                      constraint = self.kernel_constraint,
                                      trainable = True)
        
        self.window_size = self.kernel_size[0] * self.kernel_size[1]
        
        if self.use_bias:
            self.bias = self.add_weight(shape = self.filters,
                                        initializer = self.bias_initializer,
                                        name ='bias',
                                        regularizer = self.bias_regularizer,
                                        constraint = self.bias_constraint,
                                        trainable = True)
        else:
            self.bias = None
    
    def call(self, inputs):
        mask_output = K.backend.conv2d(
            inputs[1], tf.ones(self.kernel_shape),
            strides = self.strides,
            padding = self.padding,
            data_format = self.data_format,
            dilation_rate = self.dilation_rate
        )
        
        img_output = K.backend.conv2d(
            inputs[0] * inputs[1], self.kernel,
            strides = self.strides,
            padding = self.padding,
            data_format = self.data_format,
            dilation_rate = self.dilation_rate
        )
        
        mask_ratio = self.window_size / (mask_output + 1e-8)
        mask_output = K.backend.clip(mask_output, 0.0, 1.0)
        mask_ratio *= mask_output
        
        img_output = img_output * mask_ratio
        
        if self.use_bias:
            img_output = K.backend.bias_add(img_output, self.bias, data_format = self.data_format)
        
        if self.activation is not None:
            img_output = self.activation(img_output)
            
        return [img_output, mask_output]

# directory to store the results
results = 'ANpconvBatchsize_5_70epochs'
isExist = os.path.exists(results)

if not isExist:
  
  # Create a new directory because it does not exist 
  os.makedirs(results)

class PConvUNet():
    def __init__(self, image_size = 256, batch_size = 1, lr = 2e-4, m = 64):
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = lr
        self.ema = 0.999 # ema coeffitient to smooth out losses for visualization purposes
        self.cache_size = 4200 # 1000 ?how many images to load into memory == number of iterations per `epoch`
        self.optimizer = tf.keras.optimizers.Adam(lr)
        
        # the next lines should be adjusted for your particular needs
        # they should fill in two lists of files - for images and for masks
        
        self.train_img_folder = "processed_data/augmented/"
        self.train_msk_folder = "processed_data/train_mask_dir/"     # add the test masks?

        self.train_image_files = [f for f in listdir(self.train_img_folder) if isfile(join(self.train_img_folder, f))\
                           and f.endswith(".png")]
    
        
        self.train_mask_files  = [f for f in listdir(self.train_msk_folder) if isfile(join(self.train_msk_folder, f))\
                            and f.endswith(".png")]

        self.val_img_folder = "processed_data/val_rgbd_dir/"
        self.val_msk_folder = "processed_data/val_mask_dir/"

        self.val_image_files = [f for f in listdir(self.val_img_folder) if isfile(join(self.val_img_folder, f))\
                           and f.endswith(".png")]
    
        
        self.val_mask_files  = [f for f in listdir(self.val_msk_folder) if isfile(join(self.val_msk_folder, f))\
                            and f.endswith(".png")]



        
        self.train_losses = {} # dictionary to store losses
        self.val_losses = {}
        
        self.G = self.generator(m = m)
        self.G.build(input_shape = (None, image_size, image_size, 4))
        
        self.D = self.discriminator()
        self.D.build(input_shape = (None, image_size, image_size, 4))

        self.DD = self.discriminatorDepth()
        self.DD.build(input_shape = (None, image_size, image_size, 4))

    def load_model(self, fname):
        self.G.load_weights(fname)
        
    def save_model(self, fname):
        self.G.save_weights(fname)   

    # main training loop, if needed, it will also periodically show examples of outputs.
    

    def train(self, epochs = 1, plot_progress = False, plot_interval = 50):
        train_loss_values = []
        val_loss_values = []
        for e in range(epochs):
            self.train_load_data()
            self.validation_load_data()
            for i in range(self.cache_size):
                if i > 0:
                    print("[%d][%d/%d] Loss %f" % (e, i, self.cache_size, self.train_losses['total'][-1]), end = "\r")

                else:
                    print("[%d][%d/%d]" % (e, i, self.cache_size), end = "\r")

                x, y = self.train_get_batch()
            
                with tf.GradientTape() as gen_tape:
                    gen_output = self.G([x, y])

                    comp = x * y + (1 - y) * gen_output
                    
                    dis_real = self.D(x)
                    dis_fake = self.D(gen_output)
                    dis_comp = self.D(comp)

                    dd2_real = self.DD(x)
                    dd2_fake = self.DD(gen_output)
                    # dd2.comp = self.DD(comp)       # maybe this is not required as we can just compare fake and real
    
                    gen_losses, losses_dict = self.gen_loss(dis_real, dis_fake, dis_comp, dd2_real, dd2_fake, x, gen_output, y)

                    # train_loss_values.append(int(gen_losses))
                    
                    for k in losses_dict.keys():
                        if k in self.train_losses.keys():
                            self.train_losses[k].append(losses_dict[k].numpy() * (1 - self.ema) +\
                                                  self.train_losses[k][-1] * self.ema)
                        else:
                            self.train_losses[k] = [losses_dict[k].numpy()]

                    gen_grads = gen_tape.gradient(gen_losses, self.G.trainable_variables)
                    self.optimizer.apply_gradients(zip(gen_grads, self.G.trainable_variables))

                # if i % plot_interval == 0 and plot_progress:
                #                     # input RGBD, target RGBD, input RGB, input D, Generated RGB, Generated Depth
                #     self.plot_example(x[0] * y[0], x[0], x[0][:,:,:3] * y[0][:,:,:3], x[0][:,:,3] * y[0][:,:,3], gen_output[0][:,:,:3], gen_output[0][:,:,3], e, i) 


            train_loss_values.append(int(gen_losses))
            print("Training Loss", e, gen_losses.numpy())

            x_val, y_val = self.val_get_batch()
            gen_output_val = self.G([x_val, y_val])


            self.plot_example(x_val[0][:,:,:3], x_val[0][:,:,3], y_val[0][:,:,:3], x_val[0][:,:,:3] * y_val[0][:,:,:3], x_val[0][:,:,3] * y_val[0][:,:,3], gen_output_val[0][:,:,:3], gen_output_val[0][:,:,3], e)

            comp_val = x_val * y_val + (1 - y_val) * gen_output_val
                    
            dis_real_val = self.D(x_val)
            dis_fake_val = self.D(gen_output_val)
            dis_comp_val = self.D(comp_val)

            dd2_real_val = self.DD(x_val)
            dd2_fake_val = self.DD(gen_output_val)
            # dd2.comp_val = self.DD(comp)       # maybe this is not required as we can just compare fake and real

            gen_losses_val, losses_dict_val = self.gen_loss(dis_real_val, dis_fake_val, dis_comp_val, dd2_real_val, dd2_fake_val, x_val, gen_output_val, y_val)

            for k in losses_dict_val.keys():
                if k in self.val_losses.keys():
                    self.val_losses[k].append(losses_dict_val[k].numpy() * (1 - self.ema) +\
                                          self.val_losses[k][-1] * self.ema)
                else:
                    self.val_losses[k] = [losses_dict_val[k].numpy()]

            print("Validation Loss", e, self.val_losses['total'][-1])
            print("--")
            val_loss_values.append(int(gen_losses_val))

            if e % 10 == 0:
                self.save_model(f'ANpconvBatchsize_5_70epochs/models/generator_{e}_model.h5')
            
        return train_loss_values, val_loss_values
    
    # this routine loads random set of input images and masks into memory
    def train_load_data(self):
        self.images, self.masks = [], []

        while(len(self.images) < self.cache_size):
            r = np.random.randint(len(self.train_image_files))
            image = np.array(img.open(self.train_img_folder + self.train_image_files[r]), dtype = np.float32)
            h, w, _ = image.shape
            if h >= self.image_size and w >= self.image_size:
                self.images.append(image)

        while(len(self.masks) < self.cache_size):
            r = np.random.randint(len(self.train_mask_files))
            mask = np.array(img.open(self.train_msk_folder + self.train_mask_files[r]), dtype = np.float32)
            h, w, _ = mask.shape
            if h >= self.image_size and w >= self.image_size:
                self.masks.append(mask)


    def validation_load_data(self):
        self.val_images, self.val_masks = [], []

        while(len(self.val_images) < self.cache_size):
            r = np.random.randint(len(self.val_image_files))
            image = np.array(img.open(self.val_img_folder + self.val_image_files[r]), dtype = np.float32)
            h, w, _ = image.shape
            if h >= self.image_size and w >= self.image_size:
                self.val_images.append(image)

        while(len(self.val_masks) < self.cache_size):
            r = np.random.randint(len(self.val_mask_files))
            mask = np.array(img.open(self.val_msk_folder + self.val_mask_files[r]), dtype = np.float32)
            h, w, _ = mask.shape
            if h >= self.image_size and w >= self.image_size:
                self.val_masks.append(mask)
    
    
    # this routine gets random batch of images
    # it also draws some random circles, rectangles, lines and ellipses on each mask
    # so that in each image at least 5% is damaged
    def train_get_batch(self):
        masks_batch  = np.ones(shape = (self.batch_size, self.image_size, self.image_size, 4), dtype = np.float32)

        images_batch = np.ones(shape = (self.batch_size, self.image_size, self.image_size, 4), dtype = np.float32)
        
        for b in range(self.batch_size):
            while np.mean(masks_batch[b]) > 0.95:
                rm = np.random.randint(self.cache_size)
                er = np.random.randint(2, 20)
                w, h = self.image_size, self.image_size
                s = 20
                x = np.random.randint(w - self.image_size + 1)
                y = np.random.randint(h - self.image_size + 1)
                masks_batch[b] = self.val_masks[rm][x:x+self.image_size, y:y+self.image_size] / 255.0

                if np.random.random() > 0.5:
                    masks_batch[b] = cv2.erode(masks_batch[b], np.ones((er, er), np.uint8), iterations = 1)
                    masks_batch[b][masks_batch[b] > 0] = 1
                
                # for _ in range(np.random.randint(20)):
                #     cx = np.random.randint(w + 50) - 25
                #     cy = np.random.randint(h + 50) - 25
                #     radius = np.random.randint(3, s)
                #     cv2.circle(masks_batch[b], (cx, cy), radius, (0, 0, 0), -1)
                    
                # for _ in range(np.random.randint(20)):
                #     x1, y1 = np.random.randint(1, w + 50) - 25, np.random.randint(1, h + 50) - 25
                #     x2, y2 = np.random.randint(x1 - 2 * s, x1 + 2 * s), np.random.randint(y1 - 2 * s, y1 + 2 * s)
                #     cv2.rectangle(masks_batch[b], (x1, y1), (x2, y2), (0, 0, 0), -1)

                # for _ in range(np.random.randint(20)):
                #     x1, x2 = np.random.randint(1, w + 50) - 25, np.random.randint(1, w + 50) - 25
                #     y1, y2 = np.random.randint(1, h + 50) - 25, np.random.randint(1, h + 50) - 25
                #     thickness = np.random.randint(1, s)
                #     cv2.line(masks_batch[b], (x1, y1), (x2, y2), (0, 0, 0), thickness)

                # for _ in range(np.random.randint(20)):
                #     x1, y1 = np.random.randint(1, w), np.random.randint(1, h)
                #     s1, s2 = np.random.randint(1, w), np.random.randint(1, h)
                #     a1, a2, a3 = np.random.randint(3, 180), np.random.randint(3, 180), np.random.randint(3, 180)
                #     thickness = np.random.randint(3, s)
                #     cv2.ellipse(masks_batch[b], (x1, y1), (s1, s2), a1, a2, a3, (0, 0, 0), thickness)

            rl = np.random.randint(self.cache_size)
            w, h, _ = self.images[rl].shape
            x = np.random.randint(w - self.image_size + 1)
            y = np.random.randint(h - self.image_size + 1)
            images_batch[b] = self.images[rl][x:x+self.image_size, y:y+self.image_size] / 255.0
            
        return images_batch, masks_batch
    

    def val_get_batch(self):
        val_masks_batch  = np.ones(shape = (self.batch_size, self.image_size, self.image_size, 4), dtype = np.float32)

        val_images_batch = np.ones(shape = (self.batch_size, self.image_size, self.image_size, 4), dtype = np.float32)
        
        for b in range(self.batch_size):
            while np.mean(val_masks_batch[b]) > 0.95:
                rm = np.random.randint(self.cache_size)
                er = np.random.randint(2, 20)
                w, h = self.image_size, self.image_size
                s = 20
                x = np.random.randint(w - self.image_size + 1)
                y = np.random.randint(h - self.image_size + 1)
                val_masks_batch[b] = self.val_masks[rm][x:x+self.image_size, y:y+self.image_size] / 255.0

                if np.random.random() > 0.5:
                    val_masks_batch[b] = cv2.erode(val_masks_batch[b], np.ones((er, er), np.uint8), iterations = 1)
                    val_masks_batch[b][val_masks_batch[b] > 0] = 1

            rl = np.random.randint(self.cache_size)
            w, h, _ = self.val_images[rl].shape
            x = np.random.randint(w - self.image_size + 1)
            y = np.random.randint(h - self.image_size + 1)
            val_images_batch[b] = self.val_images[rl][x:x+self.image_size, y:y+self.image_size] / 255.0
            
        return val_images_batch, val_masks_batch



    # generator network. m - is a multiplyer for the number of channels for convolutions
    def generator(self, m = 64):
        kernel = 3
        stride = 2

        c = [1, 2, 4, 8, 16, 16, 16, 16, 16, 16, 16, 8, 4, 2, 1]
        filters = [i * m for i in c] + [4]

        l_in = L.Input(shape=(self.image_size, self.image_size, 4), name = "gen_input_image")
        m_in = L.Input(shape=(self.image_size, self.image_size, 4), name = "gen_input_mask")
        print(l_in.shape)

        # print('input', l_in.shape, m_in.shape)
        # print('encoder')
        #encoder
        ls, ms = [], []

        l, m = PConv2D(filters[0], 7, stride, activation = 'relu', padding = 'same')([l_in, m_in])
        print('input   ',l.shape, m.shape)
        
        ls.append(l)
        ms.append(m)

        for i in range(7):
            if i < 2: k = 5
            else: k = kernel
            l, m = PConv2D(filters[i + 1], k, stride, activation = 'relu', padding = 'same')([l, m])
            # print(l.shape)
            l = CustomNorm()(l)
            ls.append(l)
            ms.append(m)
            print(i,' encoder  ',l.shape, m.shape)
        # exit()
        ms = ms[::-1]   # reversing
        ls = ls[::-1]
        # print(l.shape)
        #decoder
        # print('decoder')
        for i in range(7):
            l = L.UpSampling2D(size = 2, interpolation = 'nearest')(l)
            print('UpSampling2D first', l)
            # here we need to do the attention between l and ls, m and ms

            l = L.Concatenate()([l, ls[i + 1]])
            print('concat', l.shape)
            m = L.UpSampling2D(size = 2, interpolation = 'nearest')(m)
            m = L.Concatenate()([m, ms[i + 1]])
            l, m = PConv2D(filters[i + 8], kernel, padding = 'same')([l, m])
            l = L.LeakyReLU(alpha = 0.2)(l)
            l = CustomNorm()(l)
            print(i, 'decoder', l.shape, m.shape)
            print("   ")
        # exit()
        l = L.UpSampling2D(size = 2, interpolation = 'nearest')(l)
        print(l.shape)
        l = L.Concatenate()([l, l_in])
        print(l.shape)
        m = L.UpSampling2D(size = 2, interpolation = 'nearest')(m)
        m = L.Concatenate()([m, m_in])
        l, m = PConv2D(filters[15], kernel, padding = 'same', activation = 'relu')([l, m])
        print(l.shape, m.shape)
        l = L.Conv2D(filters[15], kernel_size = 1, strides = 1, activation = 'sigmoid', name = 'output_image')(l)
        print('output', l.shape)
        exit()
    
        return K.Model(inputs = [l_in, m_in], outputs = l, name = "generator")
    
    # discriminator is a VGG16
    # you can change this function and load weights provided by TF 2.0
    def discriminator(self):
        mean = [0.485, 0.456, 0.406]     # for scaling VGG weights 
        stdev = [0.229, 0.224, 0.225]

        inputs = L.Input(shape=(self.image_size, self.image_size, 4))
        inputsRGB = inputs[:,:,:,:3]        

        processed = L.Lambda(lambda x: (x - mean) / stdev)(inputsRGB)

        vgg = K.applications.vgg16.VGG16(weights = 'imagenet', include_top = False, input_tensor = processed)
        # vgg.load_weights("./vgg16.h5", by_name = True)

        vgg.outputs = [vgg.layers[i].output for i in [4, 7, 11]]
        model = K.Model(inputs = inputs, outputs = vgg.outputs)
        model.trainable = False
        model.compile(loss='mse', optimizer='adam')

        return model


    def discriminatorDepth(self):
        mean = [0.485, 0.456, 0.406]     # for scaling VGG weights 
        stdev = [0.229, 0.224, 0.225]

        inputs = L.Input(shape=(self.image_size, self.image_size, 4))
        
        inputsDepth = inputs[:,:,:,3]
        inputsDepth = tf.stack([inputsDepth, inputsDepth, inputsDepth], axis=-1)

        processed = L.Lambda(lambda x: (x - mean) / stdev)(inputsDepth)

        vgg = K.applications.vgg16.VGG16(weights = 'imagenet', include_top = False, input_tensor = processed)
        # vgg.load_weights("./vgg16.h5", by_name = True)

        vgg.outputs = [vgg.layers[i].output for i in [4, 7, 11]]
        model = K.Model(inputs = inputs, outputs = vgg.outputs)
        model.trainable = False
        model.compile(loss='mse', optimizer='adam')

        return model        
    
    # this calculates all the losses described in the paper and returns total loss and a dictionary
    # with its separate components for logging
    def gen_loss(self, dis_output_real, dis_output_fake, dis_output_comp, dd2_output_real, dd2_output_fake,  target,\
             generated, mask, weights = [1, 6, 0.05, 120.0, 120.0, 0.1]):
    
        comp = target * mask + (1 - mask) * generated

        loss = 0
        d = {}

        # valid  (this is the mean absolute error)
        l = tf.reduce_mean(tf.abs(target * mask - generated * mask)) * weights[0]
        d['valid'] = l
        loss += l

        # hole   
        l = tf.reduce_mean(tf.abs(target * (1 - mask) - generated * (1 - mask))) * weights[1]
        d['hole'] = l
        loss += 6*l

        # perceptual losses
        for p in range(len(dis_output_real)):
            l  = tf.reduce_mean(tf.math.abs(dis_output_real[p] - dis_output_fake[p])) * weights[2]
            l += tf.reduce_mean(tf.math.abs(dis_output_comp[p] - dis_output_fake[p])) * weights[2]
            l = 0.05*l
        # perceptual loss for dd2, only using fake and real not comp
            l += 0.025*(tf.reduce_mean(tf.math.abs(dd2_output_real[p] - dd2_output_fake[p])) * weights[2])

            d['perceprual_' + str(p)] = l
            loss += l

        # style losses
        for p in range(len(dis_output_real)):
            b, w, h, c = dis_output_real[p].shape.as_list()

            r = tf.reshape(dis_output_real[p], [b, w * h, c])
            f = tf.reshape(dis_output_fake[p], [b, w * h, c])
            k = tf.reshape(dis_output_comp[p], [b, w * h, c])


            r = tf.keras.backend.batch_dot(r, r, axes = [1, 1])
            f = tf.keras.backend.batch_dot(f, f, axes = [1, 1])
            k = tf.keras.backend.batch_dot(k, k, axes = [1, 1])

            l = tf.reduce_sum(tf.math.abs(r - f) / c**3/ h / w) * weights[3]

            d['style_fake_' + str(p)] = l
            loss += 120*l
            l = tf.reduce_sum(tf.math.abs(r - k) / c**3/ h / w) * weights[4]
            d['style_comp_' + str(p)] = l
            loss += 120*l

        # style loss for dd2
        for p in range(len(dd2_output_real)):
            b, w, h, c = dd2_output_real[p].shape.as_list()

            r = tf.reshape(dd2_output_real[p], [b, w * h, c])
            f = tf.reshape(dd2_output_fake[p], [b, w * h, c])

            r = tf.keras.backend.batch_dot(r, r, axes = [1, 1])
            f = tf.keras.backend.batch_dot(f, f, axes = [1, 1])

            l = tf.reduce_sum(tf.math.abs(r - f) / c**3/ h / w) * weights[3]

            d['dd2_style_fake_' + str(p)] = l
            loss += 60 * l
            # no comp used

        # TV loss
        kernel = K.backend.ones(shape=(3, 3, mask.shape[3], mask.shape[3]))
        dilated_mask = K.backend.conv2d(1 - mask, kernel, data_format = 'channels_last', padding = 'same')
        dilated_mask = K.backend.cast(K.backend.greater(dilated_mask, 0), 'float32')
        TV = dilated_mask * comp

        l  = tf.reduce_mean(tf.abs(TV[:,1:,:,:] - TV[:,:-1,:,:])) * weights[5]
        l += tf.reduce_mean(tf.abs(TV[:,:,1:,:] - TV[:,:,:-1,:])) * weights[5]

        d['tv'] = 0.1*l
        loss += l

        d['total'] = loss

        return loss, d
    
    def load_model(self, fname):
        self.G.load_weights(fname)
        
    def save_model(self, fname):
        self.G.save_weights(fname)
    
    # plots example outputs during training


    def plot_example(self, inputRGB, inputDepth, inputMASK, inputRGB_masked, inputDepth_masked, genrgb, gendepth,e):
        plt.close()

#  x_val[0][:,:,:3], x_val[0][:,:,3], y_val[0][:,:,:3], x_val[0][:,:,:3] * y_val[0][:,:,:3], x_val[0][:,:,3] * y_val[0][:,:,3], gen_output_val[0][:,:,:3], gen_output_val[0][:,:,3], e 

        rgb_img = np.copy(inputRGB)
        back_img = np.array(img.open('checkerPattern.png'), dtype=np.float32)/255.0
        # gen_rgb = np.copy(genrgb)

        def checkerback(fore, back):
            src2 = back
            src1 = fore
            frontR = src1[:,:,0]
            frontG = src1[:,:,1]
            frontB = src1[:,:,2]
            backR = src2[:,:,0]
            backG = src2[:,:,1]
            backB = src2[:,:,2]
            r,c = np.where(frontR==0)
            frontR[(r,c)] = backR[(r,c)]
            r,c = np.where(frontG==0)
            frontG[(r,c)] = backG[(r,c)]
            r,c = np.where(frontB==0)
            frontB[(r,c)] = backB[(r,c)]
            merged = np.dstack((frontR,frontG,frontB))
            return merged

        merged = checkerback(rgb_img, back_img)

        input_mask = inputMASK
        check_masked_inputRGB = merged * input_mask
        masked_inputRGB = input_mask * inputRGB

        # gen_rb_check = checkerback(gen_rgb, back_img)

        fig, ax = plt.subplots(1, 6, sharex = True, figsize=(20, 5), dpi=150)

        ax[0].imshow(inputRGB)
        ax[0].set_title('Original RGB Image')
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)
        
        ax[1].imshow(inputDepth)
        ax[1].set_title('Original Depth Image')
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)

        ax[2].imshow(masked_inputRGB)
        ax[2].set_title('Masked Input RGB Image')
        ax[2].get_xaxis().set_visible(False)
        ax[2].get_yaxis().set_visible(False)

        # ax[3].imshow(check_masked_inputRGB)
        # ax[3].set_title('Checker pattern for background')
        # ax[3].get_xaxis().set_visible(False)
        # ax[3].get_yaxis().set_visible(False)
        
        ax[3].imshow(inputDepth_masked)
        ax[3].set_title('Masked Input Depth Image')
        ax[3].get_xaxis().set_visible(False)
        ax[3].get_yaxis().set_visible(False)
        
        ax[4].imshow(genrgb)
        ax[4].set_title('Generated RGB Image')
        ax[4].get_xaxis().set_visible(False)
        ax[4].get_yaxis().set_visible(False)

        ax[5].imshow(gendepth)
        ax[5].set_title('Generated Depth Image')
        ax[5].get_xaxis().set_visible(False)
        ax[5].get_yaxis().set_visible(False)

        plt.tight_layout()
        plt.savefig(f'ANpconvBatchsize_5_70epochs/epoch_iter_{e:05d}.png', bbox_inches='tight')

        # display.clear_output(wait = True)
        # display.display(plt.gcf())


    def predict(self, input_img, input_mask):

        image = np.array(input_img)
        mask = np.array(input_mask)

        output = self.G([image, mask])
        return output    


# def test_plot_example(inpRGBD, targetRGBD, inputRGB, inputDepth, genrgb, gendepth, i):
#     # input RGBD, target RGBD, input RGB, input D, Generated RGB, Generated Depth
#     plt.close()
#     fig, ax = plt.subplots(1, 6, sharex = True, figsize=(16.5, 16.5))
    
#     ax[0].imshow(inpRGBD)
#     ax[0].set_title('Input RGBD')
    
#     ax[1].imshow(targetRGBD)
#     ax[1].set_title('Target RGBD')

#     ax[2].imshow(inputRGB)
#     ax[2].set_title('Input RGB')
    
#     ax[3].imshow(inputDepth)
#     ax[3].set_title('Input Depth')
    
#     ax[4].imshow(genrgb)
#     ax[4].set_title('Generated RGB')

#     ax[5].imshow(gendepth)
#     ax[5].set_title('Generated Depth')

#     plt.savefig(f'ANpconvBatchsize_4_70epochs/test_image_hyper1_{i}.png', bbox_inches='tight')



model = PConvUNet(image_size = 256, batch_size = 4, lr = 0.001)


# train_loss_arr, val_loss_arr = model.train(epochs = 70, plot_progress = True)

# np.save('ANpconvBatchsize_5_70epochs/models/train_loss', train_loss_arr)
# np.save('ANpconvBatchsize_5_70epochs/models/val_loss', val_loss_arr)

# model.save_model("finalResults/monday2022res/generator_hyper1_100epochv2.h5")






######################################################################################
######################################################################################








model.load_model("ANpconvBatchsize_4_70epochs/models/generator_50_model.h5")
print(model)


# testing the model  

def plot_example_test(inputRGB, inputDepth, inputMASK, inputRGB_masked, inputDepth_masked, genrgb, gendepth,e):
    plt.close()

#  x_val[0][:,:,:3], x_val[0][:,:,3], y_val[0][:,:,:3], x_val[0][:,:,:3] * y_val[0][:,:,:3], x_val[0][:,:,3] * y_val[0][:,:,3], gen_output_val[0][:,:,:3], gen_output_val[0][:,:,3], e 

    rgb_img = np.copy(inputRGB)
    back_img = np.array(img.open('checkerPattern.png'), dtype=np.float32)/255.0
    # gen_rgb = np.copy(genrgb)

    def checkerback(fore, back):
        src2 = back
        src1 = fore
        frontR = src1[:,:,0]
        frontG = src1[:,:,1]
        frontB = src1[:,:,2]
        backR = src2[:,:,0]
        backG = src2[:,:,1]
        backB = src2[:,:,2]
        r,c = np.where(frontR==0)
        frontR[(r,c)] = backR[(r,c)]
        r,c = np.where(frontG==0)
        frontG[(r,c)] = backG[(r,c)]
        r,c = np.where(frontB==0)
        frontB[(r,c)] = backB[(r,c)]
        merged = np.dstack((frontR,frontG,frontB))
        return merged

    merged = checkerback(rgb_img, back_img)

    input_mask = inputMASK
    check_masked_inputRGB = merged * input_mask
    masked_inputRGB = input_mask * inputRGB

    # gen_rb_check = checkerback(gen_rgb, back_img)

    fig, ax = plt.subplots(1, 6, sharex = True, figsize=(20, 5), dpi=150)

    ax[0].imshow(inputRGB)
    ax[0].set_title('Original RGB Image')
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    
    ax[1].imshow(inputDepth)
    ax[1].set_title('Original Depth Image')
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)

    ax[2].imshow(masked_inputRGB)
    ax[2].set_title('Masked Input RGB Image')
    ax[2].get_xaxis().set_visible(False)
    ax[2].get_yaxis().set_visible(False)

    # ax[3].imshow(check_masked_inputRGB)
    # ax[3].set_title('Checker pattern for background')
    # ax[3].get_xaxis().set_visible(False)
    # ax[3].get_yaxis().set_visible(False)
    
    ax[3].imshow(inputDepth_masked)
    ax[3].set_title('Masked Input Depth Image')
    ax[3].get_xaxis().set_visible(False)
    ax[3].get_yaxis().set_visible(False)
    
    ax[4].imshow(genrgb)
    ax[4].set_title('Generated RGB Image')
    ax[4].get_xaxis().set_visible(False)
    ax[4].get_yaxis().set_visible(False)

    ax[5].imshow(gendepth)
    ax[5].set_title('Generated Depth Image')
    ax[5].get_xaxis().set_visible(False)
    ax[5].get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig(f'ANpconvBatchsize_4_70epochs/models/test_res/test_image_{i}.png', bbox_inches='tight')

#     # display.clear_output(wait = True)
#     # display.display(plt.gcf())


img_folder = 'processed_data/val_rgbd_dir/'
image_files = ns(listdir(img_folder))
print(len(image_files))

msk_folder = 'processed_data/test_masks/'
mask_files = ns(listdir(msk_folder))
print(len(mask_files))

test_res_image = []
test_depth_image = []
img_arr = []
depth_arr = []

for i in range(len(image_files)):
    print(i)
    # index = np.random.randint(0,49)

    imgX = np.array(img.open(img_folder + image_files[i]), dtype = np.float32)
    imgX = imgX / 255.0
    imgs = imgX[:,:,:3]
    depth = imgX[:,:,3]
    img_arr.append(imgs)
    depth_arr.append(depth)

    maskY = np.array(img.open(msk_folder + mask_files[i]), dtype = np.float32)
    maskY = maskY / 255.0

    res = model.predict(imgX[None, :, :, :], maskY[None, :, :, :])

    test_res_image.append(res[0][:,:,:3])
    test_depth_image.append(res[0][:,:,3])

    plot_example_test(imgX[:,:,:3], imgX[:,:,3], maskY[:,:,:3], imgX[:,:,:3] * maskY[:,:,:3], imgX[:,:,3]*maskY[:,:,3], res[0][:,:,:3], res[0][:,:,3], i)


test_res_image = np.array(test_res_image)
test_depth_image = np.array(test_depth_image)

img_arr = np.array(img_arr)
depth_arr = np.array(depth_arr)


np.save('ANpconvBatchsize_4_70epochs/models/test_res_array/test_res_images_array', test_res_image)
np.save('ANpconvBatchsize_4_70epochs/models/test_res_array/test_res_depths_array', test_depth_image)

np.save('ANpconvBatchsize_4_70epochs/models/test_res_array/images_array', img_arr)
np.save('ANpconvBatchsize_4_70epochs/models/test_res_array/depth_array', depth_arr)