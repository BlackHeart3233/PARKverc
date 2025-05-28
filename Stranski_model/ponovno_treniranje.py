from ultralytics import YOLO

import torch
''' //preveri ce mas CUDA namesceno
print(torch.__version__)
print(torch.version.cuda)        # verzija CUDA, ki jo PyTorch uporablja
print(torch.cuda.is_available()) # ali je GPU na voljo
print(torch.cuda.device_count()) # koliko GPU-jev zazna
'''

if __name__ == '__main__':
    from ultralytics import YOLO

    model = YOLO("../runs/detect/train2/weights/best.pt")

    model.train(
        data="Stranski_model/yolo_data_2/split_train_2/data.yaml",
        epochs=30,
        imgsz=640,
        device=0,            # GPU: 0, CPU: 'cpu'
        resume=False,
        project="Stranski_model",  # kam shrani rezultate
        name="fine_tuned_3"          # ime mape z rezultati
    )

#finetuned 3 ma 30 epoches
#fintuned 2 ma 20 epoches