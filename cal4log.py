'''This coding for getting result(acc@m,iou, etc.) intuitively from log'''

import numpy as np

#topn
def compute_IoU_recall_top_n(topn, IoU, iuo_record):
    yes=0
    for i in range(len(iuo_record)):
        #for n in range(topn):
        if iuo_record[i]>=IoU:
            yes=yes+1
    acc=yes/len(iuo_record)

    return acc

def mIoU(iou_recode):
     return np.mean(iou_recode)

#acc
def read_log_and_cal_all(path):
    cal=list()
    f=open(path)
    #The first line is empty
    f.readline()
    for line in f:
        r=float(line.split()[2][7:12])
        cal.append(r)

    IoU_thresh=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for k in range(len(IoU_thresh)):
         IoU=IoU_thresh[k]
         acc=compute_IoU_recall_top_n(1,IoU,cal)
         print('acc@',IoU,acc)

    print('mIoU:',mIoU(cal))

def read_log_and_cal_n(cut,path):
    cal_n=list()
    f=open(path)
    #The first line is empty
    f.readline()
    for line in f:
        n=int(line.split()[1].split('-')[2])
        if n==cut:
            r=float(line.split()[2][7:12])
            cal_n.append(r)

    IoU_thresh=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for k in range(len(IoU_thresh)):
         IoU=IoU_thresh[k]
         acc=compute_IoU_recall_top_n(1,IoU,cal_n)
         print('acc@',IoU,acc)

    print('mIoU: ',mIoU(cal_n))



#Read the log to get the result
if __name__ == '__main__':
    #STRL_log
    path='./log/log_RL_all'
    #acc
    print('--------- acc')
    read_log_and_cal_all(path)
    
    for cut in [64,128,256,512]:
        print('---------',cut)
        read_log_and_cal_n(cut,path)

            
