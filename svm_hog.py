import cv2
import numpy as np
import random
import os

def load_images(dirname, dir, amout = 80):
    img_list = []
    file = open(dirname)
    img_names = file.readlines()    

    for img_name in img_names:   

        img_name = img_name.strip('\n')
        img_list.append(cv2.imread(os.getcwd() + "\\" + dir + r'\\' + img_name))
        amout -= 1

        if amout <= 0:
            break

    print("li")    
    return img_list

 



def sample_neg(full_neg_lst, neg_list, size):
    random.seed(1)
    width, height = size[1], size[0]
    for i in range(len(full_neg_lst)):

        for j in range(80):

            y = int(random.random() * (len(full_neg_lst[i]) - height))
            x = int(random.random() * (len(full_neg_lst[i][0]) - width))
            neg_list.append(full_neg_lst[i][y:y + height, x:x + width])

    print("sn")
    return neg_list

 

 

# wsize: 处理图片大小

def computeHOGs(img_lst, gradient_lst, wsize=(128, 64)):

    hog = cv2.HOGDescriptor()

    # hog.winSize = wsize

    for i in range(len(img_lst)):

        if img_lst[i].shape[1] >= wsize[1] and img_lst[i].shape[0] >= wsize[0]:

            roi = img_lst[i][(img_lst[i].shape[0] - wsize[0]) // 2: (img_lst[i].shape[0] - wsize[0]) // 2 + wsize[0], \

                  (img_lst[i].shape[1] - wsize[1]) // 2: (img_lst[i].shape[1] - wsize[1]) // 2 + wsize[1]]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            gradient_lst.append(hog.compute(gray))

    print("ch")

    # return gradient_lst

 

 

def get_svm_detector(svm):

    sv = svm.getSupportVectors()

    rho, _, _ = svm.getDecisionFunction(0)

    sv = np.transpose(sv)

    print("gsd")

    return np.append(sv, [[-rho]], 0)

 

 

# 主程序

# 第一步：计算HOG特征

neg_list = []

pos_list = []

gradient_lst = []

labels = []

hard_neg_list = []

svm = cv2.ml.SVM_create()

pos_list = load_images(r'.\pos\pos.lst', 'pos')

full_neg_lst = load_images(r'.\full_neg\full_neg.lst', 'full_neg')
sample_neg(full_neg_lst, neg_list, [128, 64])

print(len(neg_list))

computeHOGs(pos_list, gradient_lst)

[labels.append(+1) for _ in range(len(pos_list))]

computeHOGs(neg_list, gradient_lst)

[labels.append(-1) for _ in range(len(neg_list))]

 

# 第二步：训练SVM

svm.setCoef0(0)

svm.setCoef0(0.0)

svm.setDegree(3)

criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)

svm.setTermCriteria(criteria)

svm.setGamma(0)

svm.setKernel(cv2.ml.SVM_LINEAR)

svm.setNu(0.5)####0.5

svm.setP(0.1)  

svm.setC(0.01)

svm.setType(cv2.ml.SVM_EPS_SVR)

svm.train(np.array(gradient_lst), cv2.ml.ROW_SAMPLE, np.array(labels))

 

# 第三步：加入识别错误的样本，进行第二轮训练

hog = cv2.HOGDescriptor()
'''(winSize=(227,227), 
                        blockSize=(16,16), 
                        blockStride=(8,8), 
                        cellSize=(8,8), 
                        nbins=9, 
                        derivAperture=1, 
                        winSigma = 4., 
                        histogramNormType=0, 
                        L2HysThreshord=2.0000000000000001e-01, 
                        gammaCorrection = 0, 
                        nlevels=64)'''

hard_neg_list.clear()

hog.setSVMDetector(get_svm_detector(svm))

for i in range(len(full_neg_lst)):

    rects, wei = hog.detectMultiScale(full_neg_lst[i], winStride=(4, 4),padding=(8, 8), scale=1.05)

    for (x,y,w,h) in rects:

        hardExample = full_neg_lst[i][y:y+h, x:x+w]

        hard_neg_list.append(cv2.resize(hardExample,(64, 128)))

computeHOGs(hard_neg_list, gradient_lst)

[labels.append(-1) for _ in range(len(hard_neg_list))]

svm.train(np.array(gradient_lst), cv2.ml.ROW_SAMPLE, np.array(labels))

 

 

# 第四步：保存训练结果

hog.setSVMDetector(get_svm_detector(svm))

hog.save('myHogDector.xml')

 
