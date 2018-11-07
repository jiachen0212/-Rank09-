#coding=utf-8
import os

class BatchRename():

    def __init__(self):    #新文件夹的路径.
        self.path = '/home/lenovo/256Gdisk/tainchi/enhance1118_4/train/images/training/'
        '''
        /home/lenovo/2Tdisk/Wkyao/_/juesai/m/2_15_4/
        /home/lenovo/2Tdisk/Wkyao/_/juesai/m/2_17_3/
        /home/lenovo/2Tdisk/Wkyao/_/juesai/m/2_17_4/
        
        '''

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        for item in filelist:
            if item.endswith('_sce.npy'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), item[:-8] + '_2_sce.npy')
                try:
                    os.rename(src, dst)
                    #print 'converting %s to %s ...' % (src, dst)
                except:
                    continue

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()
