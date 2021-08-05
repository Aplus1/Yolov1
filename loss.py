import torch
from utils import intersection_over_union
import torch.nn as nn

class YoloLoss(nn.Module):

    def __init__(self,S=7,B=2,C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5


    def forward(self,predictions,target):
        predictions = predictions.reshape(-1,self.S,self.S,self.C + self.B * 5)

        # S x S X C X B * 5  (S = 7 B = 2 C = 20)
        #prediction => (c1,c2,c3..................,c19,c20,Pc,x,y,w,h) --> shape => (C * B * 5)

        iou_b1 = intersection_over_union(predictions[...,21:25],target[...,21:25])
        iou_b2 = intersection_over_union(predictions[...,26:30],target[...,21:25])
        ious = torch.cat(iou_b1.unsqueeze(0),iou_b2.unsqueeze(0),dim=0)

        iou_maxes,bestbox = torch.max(ious,dim=0)

        # in this paper this is the Iobj_i
        exists_box = target[...,20].unsqueeze(3) #take the Pc from the prediction

        #=========================#
        #     BOX CORDINATES      #
        #=========================#

        box_prediction = exists_box * (
        #this is for the second box (if second box has higher IOU the the max returns 1 else it retures 0 that why we used (1 - bestbox) bestbox would be zero if the first box has higher IOU
            (bestbox * predictions[...,26:30]) + (1 - bestbox) * predictions[...,21:25]
        )

        box_targets = exists_box * predictions[...,21:25]

        #this is where we take the sqrt for the Width and Height for the image
        box_prediction[...,2:4] = torch.sign(box_prediction[...,2:4]) * torch.sqrt(torch.abs(box_prediction[...,2:4] + 1e-6 ))
        box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4])
        box_loss = self.mse(
            torch.flatten(box_prediction,end_dim=-2),
            torch.flatten(box_targets,end_dim=-2),
        )

        # =========================#
        #       OBJECT LOSS        #
        # =========================#

        pred_box = (
            bestbox * predictions[...,25:26] + (1 - bestbox) * predictions[...,20:21]
        )

        obj_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[...,20:21])
        )

        # =========================#
        #     NO OBJECT LOSS       #
        # =========================#

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[...,20:21],start_dim=1),
            torch.flatten((1 - exists_box) * target[...,20:21],start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # =========================#
        #       CLASS LOSS         #
        # =========================#

        class_loss = self.mse(
            torch.flatten((exists_box * predictions[...,:20]),end_dim=-2),
            torch.flatten((exists_box * target[...,:20]),end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss,
            + obj_loss,
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss