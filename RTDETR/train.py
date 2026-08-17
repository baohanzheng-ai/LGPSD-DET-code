import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR



if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rtdetr/rtdetr-r18.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='RTDETR/LGPSD-DET.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                workers=4,
                # device='0,1',
                # resume='',
                patience=30,
                project='runs/train',
                name='exp',
                )