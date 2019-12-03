import cv2
import random
import glob
import os

def get_neg_samples(foldername, savapath):
    cnt = 0
    imgs = []
    labels = []
    f = open('neg.txt')
    filenames = glob.iglob(os.path.join(foldername, '*'))
    for filename in filenames:
        print('filename:' + filename)
        src = cv2.imread(filename, 1)
        
        if(src.cols >= 64) & (src.rows >= 128):
            x = random.uniform(0, src.cols - 64)
            y = random.uniform(0, src.rows - 128)
            
            imgRoi = src(cv2.Rect(x, y, 64, 128))
            imgs.append(imgRoi)
            saveName = savapath + 'neg' + str(cnt) + '.jpg'
            cv2.imwrite(saveName, imgRoi)

            label = 'neg' + str(cnt) + '.jpg'
            labels.append(label)
            label = label + '\n'
            f.write(label)
            cnt += 1

        return imgs, labels

def read_neg_samples(foldername):
    imgs = []
    labels = []
    neg_cnt = 0
    filenames = glob.iglob(os.path.join(foldername, '*'))
    for filename in filenames:
        src = cv2.imread(filename, 1)
        imgs.append(src)
        labels.append(-1)
        neg_cnt += 1

    return imgs, labels

def get_pos_samples(foldername, savepath):
    cnt = 0
    imgs = []
    labels = []
    f = open('pos.txt')
    filenames = glob.iglob(os.path.join(foldername, '*'))
    for filename in filenames:
        print('filename:', filename)
        src = cv2.imread(filename)
        imgRoi = src(cv2.Rect(16, 16, 64, 128))
        imgs.append(imgRoi)
        savename = savepath + 'pos' + str(cnt) + '.jpg'
        cv2.imwrite(savename, imgRoi)

        label = 'pos' + str(cnt) + '.jpg'
        labels.append(label)    
        f.write(label)
        cnt += 1

    return imgs, labels

def read_pos_samples(foldername):
    imgs = []
    labels = []
    pos_cnt = 0
    filenames = glob.iglob(os.path.join(foldername, '*'))

    for filename in filenames:
        src = cv2.imread(filename)
        imgs.append(src)
        labels.append(1)
        pos_cnt += 1

    return imgs, labels