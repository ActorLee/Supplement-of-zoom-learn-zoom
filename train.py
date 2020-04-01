#coding:utf-8

import os
import datetime
import yaml
import pickle
import rawpy
from PIL import Image
import numpy as np
import cv2
import random
import copy
import time
import scipy.stats as stats
import sys
import utils as utils

import net as net
import loss as losses
import tensorflow as tf

#tools
def print_lyw(s):
    sys.stdout.write('\r' + s)
    sys.stdout.flush()

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def raw2bgr(x):
    ret = x
    ret[0::2, 0::2] *= 2
    ret[1::2, 1::2] *= 2
    # ret = ret.clip(0, 1023)
    ret=ret-np.min(ret)
    ret = np.float32(ret) / ret.max() * 16383
    ret = np.uint16(ret)
    ret = cv2.cvtColor(ret, cv2.COLOR_BAYER_BG2BGR)

    ret = np.sqrt(np.float32(ret))
    ret = np.uint8(255 * (ret / ret.max()))
    return ret

def raw2gray(x):
    ret = x
    # ret[0::2, 0::2] *= 2
    # ret[1::2, 1::2] *= 2
    # ret = ret.clip(0, 1023)
    ret=np.maximum(ret, 0.0)
    ret = np.float32(ret) / ret.max() * 16383
    ret = np.uint16(ret)
    ret = cv2.cvtColor(ret, cv2.COLOR_BAYER_BG2GRAY)

    ret = np.sqrt(np.float32(ret))
    ret = np.uint8(255 * (ret / ret.max()))
    return ret

# --------------- network and loss function ---------------

def build_netwwork(input_raw, num_out_ch, up_ratio=4, up_type='deconv'):
    # set up the model
    with tf.variable_scope(tf.get_variable_scope()):
        output_rgb = net.SRResnet(input_raw, num_out_ch, up_ratio=up_ratio, reuse=False, up_type=up_type)

    return output_rgb

def build_unalign_loss(target_rgb, output_rgb, batch_size, rgb_tol, stride=1, align_loss_type='l1'):
    loss_l1 = []
    target_rgb_translated = []

    target_rgb_list = tf.split(target_rgb, batch_size, axis=0)
    output_rgb_list = tf.split(output_rgb, batch_size, axis=0)

    for target, output in zip(target_rgb_list, output_rgb_list):
        # compute_unalign_loss only support batch_size equal 1
        loss, target_translated = losses.compute_unalign_loss(output, target, tol=rgb_tol, losstype=align_loss_type, stride=stride) 

        target_rgb_translated.append(target_translated)
        loss_l1.append(loss)

    loss_l1 = tf.reduce_mean(loss_l1)
    target_rgb_translated = tf.concat(target_rgb_translated, axis=0)
    return loss_l1, target_rgb_translated

def build_cobi_loss(target_rgb, output_rgb, batch_size,up_ratio=4, w_spatial=0.5):
    loss_context_all = []
    loss_context_patch_all = []

    target_rgb_list = tf.split(target_rgb, batch_size, axis=0)
    output_rgb_list = tf.split(output_rgb, batch_size, axis=0)
    for target, output in zip(target_rgb_list, output_rgb_list):
        loss_context = losses.compute_contextual_loss(target, output, w_spatial=w_spatial)
        loss_context_all.append(loss_context)
        if up_ratio == 4:
            patch_sz = 10
        elif up_ratio == 8:
            patch_sz = 15
        else:
            error_message = 'Unexpected up_ratio value {}, must be 4 or 8.'.format(up_ratio)
            raise RuntimeError(error_message)
        loss_context_patch = losses.compute_patch_contextual_loss(target, output, patch_sz=patch_sz, rates=1, w_spatial=w_spatial)
        loss_context_patch_all.append(loss_context_patch)

    loss_context_mean=tf.reduce_mean(loss_context_all)
    loss_context_patch_mean=tf.reduce_mean(loss_context_patch_all)
    return loss_context_mean, loss_context_patch_mean


def build_cobi_loss_swap(target_rgb, output_rgb, up_ratio=4, w_spatial=0.5):
    loss_context = losses.compute_contextual_loss(output_rgb, target_rgb, w_spatial=w_spatial)
    if up_ratio == 4:
        patch_sz = 10
    elif up_ratio == 8:
        patch_sz = 15
    else:
        error_message = 'Unexpected up_ratio value {}, must be 4 or 8.'.format(up_ratio)
        raise RuntimeError(error_message)
    loss_context_patch = losses.compute_patch_contextual_loss(output_rgb, target_rgb, patch_sz=patch_sz, rates=1, w_spatial=w_spatial)
    return loss_context, loss_context_patch

# --------------- data function ---------------

NUM_PER_EPOCH = 1696

def remove_white_balance(input_raw, target_rgb, target_wb):
    target_rgb = copy.deepcopy(target_rgb)

    target_rgb[...,0] /= np.power(target_wb[0,0], 1/2.2)
    target_rgb[...,1] /= np.power(target_wb[0,1], 1/2.2)
    target_rgb[...,2] /= np.power(target_wb[0,3], 1/2.2)

    return input_raw, target_rgb

# PIL image format
def crop_raw2rgb_pair(raw, rgb, croph, cropw, rgb_tol=32, raw_tol=4, ratio=2):
    is_pad_h = False
    is_pad_w = False

    lower, upper = 0.1, 0.9
    mu, sigma = 0.5, 0.2
    rand_gen = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    rand_p = rand_gen.rvs(2)

    height_raw, width_raw = raw.shape[:2]
    height_rgb, width_rgb = rgb.shape[:2]
    if croph > height_raw*2*ratio or cropw > width_raw*2*ratio:
        print("Image too small to have the specified crop sizes.")
        return None, None
    croph_rgb = croph + rgb_tol * 2
    cropw_rgb = cropw + rgb_tol * 2
    croph_raw = int(croph/(ratio*2)) + raw_tol * 2  # add a small offset to deal with boudary case
    cropw_raw = int(cropw/(ratio*2)) + raw_tol * 2  # add a small offset to deal with boudary case

    if croph_rgb > height_rgb:
        sx_rgb = 0
        sx_raw = int(tol/2.)
        is_pad_h = True
        pad_h1_rgb = int((croph_rgb-height_rgb)/2)
        pad_h2_rgb = int(croph_rgb-height_rgb-pad_h1_rgb)
        pad_h1_raw = int(np.ceil(pad_h1_rgb/(2*ratio)))
        pad_h2_raw = int(np.ceil(pad_h2_rgb/(2*ratio)))
    else:
        sx_rgb = int((height_rgb - croph_rgb) * rand_p[0])
        sx_raw = max(0, int((sx_rgb + rgb_tol)/(2*ratio)) - raw_tol) # add a small offset to deal with boudary case

    if cropw_rgb > width_rgb:
        sy_rgb = 0
        sy_raw = int(tol/2.)
        is_pad_w = True
        pad_w1_rgb = int((cropw_rgb-width_rgb)/2)
        pad_w2_rgb = int(cropw_rgb-width_rgb-pad_w1_rgb)
        pad_w1_raw = int(np.ceil(pad_w1_rgb/(2*ratio)))
        pad_w2_raw = int(np.ceil(pad_w2_rgb/(2*ratio)))
    else:
        sy_rgb = int((width_rgb - cropw_rgb) * rand_p[1])
        sy_raw = max(0, int((sy_rgb + rgb_tol)/(2*ratio)) - raw_tol)

    raw_cropped = raw
    rgb_cropped = rgb
    if is_pad_h:
        print("Pad h with:", (pad_h1_rgb, pad_h2_rgb),(pad_h1_raw, pad_h2_raw))
        rgb_cropped = np.pad(rgb, pad_width=((pad_h1_rgb, pad_h2_rgb),(0, 0),(0,0)),
            mode='constant', constant_values=0)
        raw_cropped = np.pad(raw, pad_width=((pad_h1_raw, pad_h2_raw),(0, 0),(0,0)),
            mode='constant', constant_values=0)
    if is_pad_w:
        print("Pad w with:", (pad_w1_rgb, pad_w2_rgb),(pad_w1_raw, pad_w2_raw))
        rgb_cropped = np.pad(rgb, pad_width=((0, 0),(pad_w1_rgb, pad_w2_rgb),(0,0)),
            mode='constant', constant_values=0)
        raw_cropped = np.pad(raw, pad_width=((0, 0),(pad_w1_raw, pad_w2_raw),(0,0)),
            mode='constant', constant_values=0)
    raw_cropped = raw_cropped[sx_raw:sx_raw+croph_raw, sy_raw:sy_raw+cropw_raw,...]
    rgb_cropped = rgb_cropped[sx_rgb:sx_rgb+croph_rgb, sy_rgb:sy_rgb+cropw_rgb,...]

    return raw_cropped, rgb_cropped

def parse_zoom_tfrecord(example_proto):
    features = {
        'input_raw': tf.FixedLenFeature(shape=[], dtype=tf.string),
        'input_raw_width': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
        'input_raw_height': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
        'input_rgb': tf.FixedLenFeature(shape=[], dtype=tf.string),
        'target_rgb': tf.FixedLenFeature(shape=[], dtype=tf.string),
        'target_wb': tf.FixedLenFeature(shape=[], dtype=tf.string),
    }
    parsed_example = tf.parse_single_example(serialized=example_proto, features=features)
    input_rgb = tf.image.decode_png(parsed_example['input_rgb'], dtype=tf.uint8)
    input_rgb = tf.cast(input_rgb, dtype=tf.float32) / 255.0

    width = tf.squeeze(parsed_example['input_raw_width'])
    height = tf.squeeze(parsed_example['input_raw_height'])

    input_raw = tf.decode_raw(parsed_example['input_raw'], out_type=tf.float32)
    input_raw = tf.reshape(input_raw, [height, width, 4])

    target_rgb = tf.image.decode_png(parsed_example['target_rgb'], dtype=tf.uint8)
    target_rgb = tf.cast(target_rgb, dtype=tf.float32) / 255.0

    target_wb = tf.decode_raw(parsed_example['target_wb'], out_type=tf.float32)
    target_wb = tf.reshape(target_wb, [1, 4])

    return input_raw, target_rgb, target_wb

# --------------- multigpu support function ---------------

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = [g for g, _ in grad_and_vars]
        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads 

def feed_all_gpu(input_raw_list, target_rgb_list, batch_size_per_gpu, batch_input_raw, batch_target_rgb):
    assert(len(input_raw_list) == len(target_rgb_list))
    feed_dict = {}
    for i, (input_raw, target_rgb) in enumerate(zip(input_raw_list, target_rgb_list)):
        start_pos = i * batch_size_per_gpu
        stop_pos = (i + 1) * batch_size_per_gpu
        feed_dict[input_raw] = batch_input_raw[start_pos:stop_pos, :, :, :]
        feed_dict[target_rgb] = batch_target_rgb[start_pos:stop_pos, :, :, :]
    return feed_dict

def train():

    train_path = '/data/zoom-learn-zoom/train/'
    up_ratio = 4
    img_size = 256
    rgb_tol = 40 # 8
    raw_tol = 0
    stride = 2
    batch_size = 10
    w_spatial = 0.9


    loss_type = 'combine' # l1 or cx or combine
    max_epoch = 300
    init_learning_rate = 3e-7
    decay_rate = 0.98
    decay_epoch=2
    save_epoch = 2
    check_train_data_epoch=2
    display_step = 1

    finetune_training = True

    ex_base_dir = "/data/experiments/zoom/tfrecord_ws0.7_lr8e-5_dr0.98_decay_epoch2_rgbtol40_combine_stride2/combine_ws0.9_finetune/"
    train_img_save_path = ex_base_dir + "train_out_img"
    ckpt_path_l1 = ex_base_dir + 'l1_loss_train_model/model.ckpt'
    ckpt_path_cobi = ex_base_dir + 'Cobi_loss_train_model_finetune_from_cobi_19_10_25/model.ckpt'

    restore_dir_path_cobi = ex_base_dir + 'Cobi_loss_train_model/'
    restore_dir_path_l1 = '/data/experiments/zoom/tfrecord_ws0.7_lr8e-5_dr0.98_decay_epoch2_rgbtol40_combine_stride2/combine_ws0.9_finetune/Cobi_loss_train_model_finetune_from_cobi_19_10_24/'
    # restore_dir_path_l1 = '/data/experiments/zoom/tfrecord_ws0.5_lr8e-5_dr0.98_decay_epoch2_rgbtol32_combine/Cobi_loss_train_model/'
    # restore_dir_path_l1 = ex_base_dir + 'l1_loss_train_model/'

    # tb_path=ex_base_dir + 'tb_train/'
    num_gpus = 5

    loss_unalign_weight = 0.0
    loss_context_weight = 1.0
    loss_context_patch_weight = 1.0

    check_dir(train_img_save_path)
    check_dir(restore_dir_path_l1)
    check_dir(restore_dir_path_cobi)





    # reset rgb_tol to zero when in CX mode
    if loss_type.lower() == 'cx':
        rgb_tol = 0

    if not batch_size % num_gpus == 0:
        print('Error batch_size={} and num_gpus={} setting. Batch_size must be divisibled by num_gpus'.format(batch_size, num_gpus))
        exit()

    remove_white_balance_tf = lambda x, y, z: tf.py_func(remove_white_balance, [x, y, z], [tf.float32, tf.float32])
    crop_raw2rgb_pair_py = lambda x, y: crop_raw2rgb_pair(x, y, croph=img_size, cropw=img_size, rgb_tol=rgb_tol, raw_tol=raw_tol, ratio=up_ratio)
    crop_raw2rgb_pair_tf = lambda x, y: tf.py_func(crop_raw2rgb_pair_py, [x, y], [tf.float32, tf.float32])

    # create dataset
    dataset = tf.data.TFRecordDataset(['/data/dataset/zoom_data/data_upsample4x_idshift3.tfrecord'])
    dataset = dataset.map(parse_zoom_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(remove_white_balance_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(crop_raw2rgb_pair_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(NUM_PER_EPOCH)
    dataset = dataset.repeat(-1).batch(batch_size).prefetch(2*batch_size)

    iterator = dataset.make_initializable_iterator()
    next_one_element = iterator.get_next()

    # create optimizer
    batch_num_per_epoch = int(NUM_PER_EPOCH / batch_size)
    global_step = tf.train.get_or_create_global_step()

    # decay step
    decay_step = batch_num_per_epoch*decay_epoch

    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, decay_step, decay_rate, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
    
    models = []
    rgb_size = img_size + 2 * rgb_tol
    raw_size = img_size / (2 * up_ratio) + 2 * raw_tol
    batch_size_per_gpu = int(batch_size / num_gpus)
    with tf.variable_scope(tf.get_variable_scope()):
        for gpu_id in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_id):
                # build network and loss
                input_raw = tf.placeholder(dtype=tf.float32, shape=[batch_size_per_gpu, raw_size, raw_size, 4], name='input_raw') 
                target_rgb = tf.placeholder(dtype=tf.float32, shape=[batch_size_per_gpu, rgb_size, rgb_size, 3], name='target_rgb')

                output_rgb = build_netwwork(input_raw, num_out_ch=3, up_ratio=up_ratio)
                objDict = {}
                objDict['output_rgb'] = output_rgb
                objDict['input_raw'] = input_raw
                objDict['target_rgb'] = target_rgb
                if raw_tol > 0:
                    output_rgb = output_rgb[:,int(raw_tol/2)*(up_ratio*4):-int(raw_tol/2)*(up_ratio*4),
                        int(raw_tol/2)*(up_ratio*4):-int(raw_tol/2)*(up_ratio*4),:]

                loss_unalign = tf.constant(0.0)
                loss_context = tf.constant(0.0)
                loss_context_patch = tf.constant(0.0)

                if loss_type.lower() == 'cx':
                    loss_context, loss_context_patch = build_cobi_loss(target_rgb, output_rgb,batch_size_per_gpu,up_ratio=up_ratio, w_spatial=w_spatial)
                    #loss_context, loss_context_patch = build_cobi_loss_swap(target_rgb, output_rgb, up_ratio=up_ratio, w_spatial=w_spatial)
                elif loss_type.lower() == 'l1':
                    loss_unalign_weight=1.0
                    loss_unalign, _ = build_unalign_loss(target_rgb, output_rgb, batch_size_per_gpu, rgb_tol=rgb_tol, stride=stride)
                elif loss_type.lower() == 'combine':
                    loss_unalign, target_rgb_translated = build_unalign_loss(target_rgb, output_rgb, batch_size_per_gpu, rgb_tol=rgb_tol, stride=stride)
                    loss_context, loss_context_patch = build_cobi_loss(target_rgb_translated, output_rgb, batch_size_per_gpu,up_ratio=up_ratio, w_spatial=w_spatial)
                    #loss_context, loss_context_patch = build_cobi_loss_swap(target_rgb_translated, output_rgb, up_ratio=up_ratio, w_spatial=w_spatial)
                else:
                    error_message = 'Unknown loss type {}'.format(loss_type)
                    raise RuntimeError(error_message)
                 
                total_loss = loss_unalign_weight* loss_unalign \
                    + loss_context_weight* loss_context + loss_context_patch_weight * loss_context_patch

                tf.get_variable_scope().reuse_variables()

                train_var_list = [var for var in tf.trainable_variables() if 'vgg' not in var.name]
                gradients = optimizer.compute_gradients(total_loss, var_list=train_var_list)
                models.append((input_raw, target_rgb, output_rgb, loss_unalign, loss_context, loss_context_patch, total_loss, gradients))

    input_raw_list, target_rgb_list, output_rgb_list, loss_unalign_list, loss_context_list, loss_context_patch_list, total_loss_list, gradients_list = zip(*models)
    average_loss_unalign_op = tf.reduce_mean(loss_unalign_list)
    average_loss_context_op = tf.reduce_mean(loss_context_list)
    average_loss_context_patch_op = tf.reduce_mean(loss_context_patch_list)
    average_total_loss_op = tf.reduce_mean(total_loss_list)

    # apply gradients
    average_gradients_op = average_gradients(gradients_list)
    train_op = optimizer.apply_gradients(average_gradients_op, global_step=global_step)

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()

    # start training
    step = 0
    epoch = 0
    all_loss_epoch = 0.
    with tf.Session(config=tf.ConfigProto(allow_soft_placement = True)) as sess:

        sess.run(init_op)

        if finetune_training:
            restore_ckpt_path = tf.train.latest_checkpoint(restore_dir_path_l1)
            if not restore_ckpt_path is None:
                restore_ckpt_path = str(restore_ckpt_path)
                try:
                    saver.restore(sess, restore_ckpt_path)
                    print("Restore model for {}.".format(restore_ckpt_path))
                except:
                    print("load model error!!!")
                    exit()
            else:
                print("no load model!!!")
                exit()

        sess.run(global_step.assign(0))
        sess.run(iterator.initializer)
        start_time = time.time()
        # writer = tf.summary.FileWriter(tb_path, sess.graph)
        while epoch < max_epoch:




            batch_input_raw, batch_target_rgb = sess.run(next_one_element)

            feed_dict = feed_all_gpu(input_raw_list, target_rgb_list, batch_size_per_gpu, batch_input_raw, batch_target_rgb)

            out_objDict, train_op_, lr, loss, loss_unalign, loss_cx, loss_cx_patch = sess.run(
                [objDict,train_op, learning_rate, average_total_loss_op, average_loss_unalign_op, average_loss_context_op, average_loss_context_patch_op],
                feed_dict=feed_dict)

            info='Batch {}/{} Epoch {}/{} total_loss={:.8f} loss_unalign={:.8f} loss_context={:.8f} loss_context_patch={:.8f} lr={:.4e}'.format(
                step+1, batch_num_per_epoch, epoch+1, max_epoch, loss, loss_unalign, loss_cx*loss_context_weight, loss_cx_patch*loss_context_patch_weight, lr)
            print_lyw(info)
            all_loss_epoch=all_loss_epoch+loss
            # lr_tb=tf.summary.scalar('learning rate', lr)
            # loss_tb=tf.summary.scalar('total_loss', loss)
            # loss_unalign_tb=tf.summary.scalar('loss_unalign', loss_unalign)
            # loss_cx_tb=tf.summary.scalar('loss_context', loss_cx)
            # loss_cx_patch_tb=tf.summary.scalar('loss_context_patch', loss_cx_patch)
            #
            # writer.add_summary(lr_tb, epoch*batch_num_per_epoch+step)
            # writer.add_summary(loss_tb, epoch * batch_num_per_epoch + step)
            # writer.add_summary(loss_unalign_tb, epoch * batch_num_per_epoch + step)
            # writer.add_summary(loss_cx_tb, epoch * batch_num_per_epoch + step)
            # writer.add_summary(loss_cx_patch_tb, epoch * batch_num_per_epoch + step)
            step += 1


            if (step >= batch_num_per_epoch):
                avg_loss_epoch=all_loss_epoch / step
                end_time = time.time()
                print('\r Epoch {}/{} epoch avg loss= {:.8f} total_loss= {:.8f} loss_unalign= {:.8f} loss_context= {:.8f} loss_context_patch= {:.8f} lr= {:.4e} time= {:.2f}s'.format(
                        epoch + 1, max_epoch, avg_loss_epoch,loss, loss_unalign, loss_cx*loss_context_weight, loss_cx_patch*loss_context_patch_weight,
                        lr, (end_time - start_time)),end="\n")
                # avg_loss_tb=tf.summary.scalar('epoch avg loss', avg_loss_epoch/step)
                # writer.add_summary(avg_loss_tb, epoch)
                step=0
                all_loss_epoch=0
                start_time=time.time()
                if (epoch+1) % save_epoch == 0 or epoch==0:
                    if finetune_training:
                        save_path = saver.save(sess, ckpt_path_cobi, global_step=epoch+1)
                    else:
                        save_path = saver.save(sess, ckpt_path_l1, global_step=epoch+1)
                    # print("Model saved in file: %s" % save_path)



                # if epoch % check_train_data_epoch == 0:
                #     wb_rgb = out_objDict["output_rgb"][0, ...]
                #     # print(wb_rgb.max(),wb_rgb.min())
                #     # print("Saving outputs ... ")
                #     output_rgb = Image.fromarray(np.uint8(utils.clipped(wb_rgb) * 255))
                #     output_rgb.save("%s/%s_output_rgb.png" % (train_img_save_path, epoch))
                #     # output_rgb = output_rgb.resize((int(output_rgb.width * resize_ratio),
                #     #                                 int(output_rgb.height * resize_ratio)), Image.ANTIALIAS)
                #     # output_rgb_tb=tf.summary.image('output_rgb', output_rgb, 2)
                #     # writer.add_summary(output_rgb_tb, epoch)
                #
                #     wb_rgb = out_objDict["target_rgb"][0, ...]
                #     # print(wb_rgb.max(), wb_rgb.min())
                #     target_rgb = Image.fromarray(np.uint8(utils.clipped(wb_rgb) * 255))
                #     target_rgb.save("%s/%s_target_rgb.png" % (train_img_save_path, epoch))
                #     # output_rgb = output_rgb.resize((int(output_rgb.width * resize_ratio),
                #     #                                 int(output_rgb.height * resize_ratio)), Image.ANTIALIAS)
                #     # target_rgb_tb=tf.summary.image('target_rgb', target_rgb, 2)
                #     # writer.add_summary(target_rgb_tb, epoch)
                #
                #     wb_rgb = out_objDict["input_raw"][0, ...]
                #     # print(wb_rgb.max(), wb_rgb.min())
                #     wb_rgb = utils.reshape_back_raw(wb_rgb)
                #     input_raw = raw2gray(wb_rgb)
                #     cv2.imwrite("%s/%s_input_raw.png" % (train_img_save_path, epoch), input_raw)
                #     # input_raw_tb=tf.summary.image('input_raw', input_raw, 2)
                #     # writer.add_summary(input_raw_tb, epoch)
                #     # print("Saving outputs ... ")




                epoch += 1


        # save final model
        if finetune_training:
            save_path = saver.save(sess, ckpt_path_cobi, global_step=epoch+1)
        else:
            save_path = saver.save(sess, ckpt_path_l1, global_step=epoch+1)
        # save_path = saver.save(sess, ckpt_path, global_step=step)
        print("Final model saved in file: %s" % save_path)
         
 
if __name__ == '__main__':
    train()

