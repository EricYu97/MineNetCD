import torch
import transformers
import numpy as np
from torch.utils import data
from torch import nn
import os
from PIL import Image

from datasets import load_dataset, load_from_disk
import torchvision.transforms as tfs

from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger

from models.upernet import UperNetForSemanticSegmentation

import argparse

miou_list=[]
f1_list=[]

def args():
    parser = argparse.ArgumentParser(description='MaskCD Testing Arguments')
    parser.add_argument('--model', type=str, default='ericyu/minenetcd-upernet-VSSM-B-ST-Diff-Pretrained-ChannelMixing-Dropout', help='model id')
    args = parser.parse_args()
    return args

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

class ChangeDetectionDataset(data.Dataset):
    def __init__(self,dataset,transform=None) -> None:
        super().__init__()
        self.dataset=dataset
        self.transform=transform
    def __len__(self):
        return(len(self.dataset))
    def __getitem__(self, index):
        imageA=self.transform(self.dataset[index]["imageA"])
        imageB=self.transform(self.dataset[index]["imageB"])
        label=tfs.ToTensor()(self.dataset[index]["label"])
        label=torch.cat([label],dim=0)
        return imageA,imageB,label,index


def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

def main(model_name,dataset_name,model_id):
    print(f'testing {model_id}')
    print(transformers.__file__)

    logger = get_logger(__name__)
    accelerator=Accelerator()
    device=accelerator.device
    batch_size=10

    dataset=load_dataset(dataset_name)
    logger.info(dataset,main_process_only=True)

    test_ds=dataset["test"]

    transform=tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=ADE_MEAN,std=ADE_STD),
    ])

    test_dataset=ChangeDetectionDataset(test_ds,transform=transform)

    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = UperNetForSemanticSegmentation.from_pretrained(model_id, ignore_mismatched_sizes=True)
    model = model.to(device)

    model, test_dataloader=accelerator.prepare(model,test_dataloader)
    model.eval()

    TP,TN,FP,FN=0,0,0,0 
    os.makedirs(f"./test_predictions/{model_name}/", exist_ok=True)
    for i, batch in enumerate(tqdm(test_dataloader,disable=not accelerator.is_local_main_process, miniters=20)):
        with torch.no_grad():

            imageA,imageB, labels, index=batch
            pixel_values=torch.cat([imageA,imageB],dim=0)

            outputs = model(
                    pixel_values=pixel_values
                )
            predicted_segmentation_maps=torch.nn.Softmax(dim=1)(outputs.logits)
            predicted_segmentation_maps=torch.argmax(predicted_segmentation_maps,dim=1)
            tp,fp,tn,fn=confusion(predicted_segmentation_maps,labels.squeeze())
            TP+=tp
            TN+=tn
            FP+=fp
            FN+=fn
            
            img_idx=index

            for i in range(len(predicted_segmentation_maps)):
                segmentation_map=predicted_segmentation_maps[i]

                predictions=segmentation_map.squeeze().unsqueeze(0)

                segmentation_map = Image.fromarray((255*segmentation_map).cpu().numpy().astype(np.uint8))
                
                segmentation_map.save(os.path.join(f"./test_predictions/{model_name}/"+str(int(img_idx[i]))+".png"))

    OA=(TP+TN)/(TP+TN+FP+FN)
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    f1=2*TP/(2*TP+FP+FN)
    cIoU=TP/(TP+FP+FN)

    ts_metrics_list=torch.FloatTensor([OA,f1,precision,recall, cIoU]).cuda().unsqueeze(0)

    ts_eval_metric_gathered=accelerator.gather(ts_metrics_list)

    final_metric=torch.mean(ts_eval_metric_gathered, dim=0)
    accelerator.print(f'Accuracy={final_metric[0]}, mF1={final_metric[1]}, Precision={final_metric[2]}, Recall={final_metric[3]}, cIoU={final_metric[4]}')

def convert_dict_to_tensor_dict(ori_dict):
    tensor_dict={}
    for key, values in ori_dict.items():
        if isinstance(values,list):
            tensor_dict.update({key,torch.FloatTensor(values)})
        else:
            tensor_dict.update({key,torch.FloatTensor([values])})
    return tensor_dict


if __name__=="__main__":
    args=args()
    main(model_name=args.model,dataset_name="HZDR-FWGEL/MineNetCD256",model_id=args.model)