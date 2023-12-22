import os
import shutil
import subprocess


def datasetcheck(src,dst):
    srclist=os.listdir(src)
    dstlist=os.listdir(dst)

    count=0

    for f in srclist:
        fname=f.split('.')[0]+'.txt'
        if (not fname in dstlist):
            print('del ',f)
            os.remove(src+f)
            count+=1
    return count

def detectValidCheck(resultdir,checkdir,postive=True):
    srclist=os.listdir(resultdir)
    dstlist=os.listdir(checkdir)

    count=0

    for f in dstlist:
        fname=f
        # print(f,f in srclist)
        # print(srclist)
        # print(f.split('.')[-1]=='jpg')
        if not f.split('.')[-1]=='jpg':
            continue
        if postive==True:   #保留正样本
            if (not fname in srclist):
                print('del ',f)
                os.remove(checkdir+f)
                count+=1
        else:           #保留负样本
            if (fname in srclist):
                print('del ',f)
                os.remove(checkdir+f)
                count+=1

    return count


def copyfile(src,dst):
    # src="1.txt.py"
    # dst="2.txt.py"
    cmd='cp %s %s' % (src, dst)

    print(cmd)

    status = subprocess.call(cmd, shell=True)

    if status != 0:
        if status < 0:
            print("Killed by signal", status)
        else:
            print("Command failed with return code - ", status)
    else:
        print('Execution of %s passed!\n' % cmd)

def gendataset(path,train_p=6,val_p=3,test_p=1):
    #分配数据集
    trainlist=[0,2,4,6,8,9]
    vallist=[1,3,7]
    testlist=[5]

    fdir=os.listdir(path)
    i=0
    dstpah='/usr/code/yolovt-master/data/jianhuadataset/'
    #dstpah='D:\\JianhuaOCR\\yolov5-master\\data\\jianhuadataset\\'
    dspf=''
    dsplabelf=''
    for f in fdir:
        lname=f.split('.')[-1]
        if(lname=='jpg'):
            i+=1
            if(i%10 in trainlist):
                dspf='images/train/'+f
                dsplabelf='labels/train/'+f.replace('.jpg','.txt')
            elif(i%10 in vallist):
                dspf='images/val/'+f
                dsplabelf='labels/val/'+f.replace('.jpg','.txt')
            elif(i%10 in testlist):
                dspf='images/test/'+f
                dsplabelf='labels/test/'+f.replace('.jpg','.txt')
                # print("copy img to train:",lname)


        #shutil.copyfile(path+f,dstpah+dspf)
        # comstr='copy '+ path+f+ ' ' + dstpah+dspf
        # print(comstr)
        # os.system(comstr)
        #copyfile(path+f,dstpah+dspf)
        #print(path+f)
        if(os.path.exists(path+f.replace('.jpg','.txt'))):
            print(path+f.replace('.jpg','.txt'))
            print(dstpah+dsplabelf)
            print(os.path.exists(dstpah))
            
            #shutil.copyfile(path+f.replace('.jpg','.txt'),dstpah+dsplabelf)
            # comstr='copy '+ path+f.replace('.jpg','.txt') +' ' + dstpah+dsplabelf
            # print(comstr)
            # os.system(comstr)
            copyfile(path+f.replace('.jpg','.txt'),dstpah+dsplabelf)
            
        else:
            print("label file not found:", path+f.replace('.jpg','.txt'))

if __name__ == "__main__":
    #gendataset('/usr/code/jx_labeled/2_2/')
    # gendataset('D:\\JianhuaOCR\\OCR\\oriimg\\11\\2_chen_selected\\')

    '''
    count=datasetcheck('../data/jianhuadataset/images/train/','../data/jianhuadataset/labels/train/')
    print(count)

    count=datasetcheck('../data/jianhuadataset/images/test/','../data/jianhuadataset/labels/test/')
    print(count)
    
    count=datasetcheck('../data/jianhuadataset/images/val/','../data/jianhuadataset/labels/val/')
    print(count)
    '''
    resultdir='F:/JianhuaOCR/3a_n/detected/'
    checkdir='F:/JianhuaOCR/3a_n/'

    resultdir='F:/JianhuaOCR/2_chen_selected/'
    checkdir='F:/JianhuaOCR/2_chen_selected/2_chen_select_n/'

    resultdir='F:/JianhuaOCR/2_jx_labeled/2_n_detected/'
    checkdir='F:/JianhuaOCR/2_jx_labeled/2_n_detected-p/'


    count=detectValidCheck(resultdir,checkdir,postive=True)
    print(count)
