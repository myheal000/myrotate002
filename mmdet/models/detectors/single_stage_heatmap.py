import torch.nn as nn
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
import os


def draw_feature_map1(features, img_path, save_dir = './work_dirs/feature_map/',name = None):
    '''
    :param features: 特征层。可以是单层，也可以是一个多层的列表
    :param img_path: 测试图像的文件路径
    :param save_dir: 保存生成图像的文件夹
    :return:
    '''
    img = cv2.imread(img_path)      #读取文件路径
    i=0
    if isinstance(features,torch.Tensor):   # 如果是单层
        features = [features]       # 转为列表
    for featuremap in features:     # 循环遍历
        heatmap = featuremap_2_heatmap1(featuremap)	#主要是这个，就是取特征层整个的求和然后平均，归一化
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
        heatmap0 = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式,0-255,heatmap0显示红色为关注区域，如果用heatmap则蓝色是关注区域
        heatmap = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
        plt.imshow(heatmap0)  # ,cmap='gray' ，这里展示下可视化的像素值
        # plt.imshow(superimposed_img)  # ,cmap='gray'
        plt.close()	#关掉展示的图片
        # 下面是用opencv查看图片的
        # cv2.imshow("1",superimposed_img)
        # cv2.waitKey(0)     #这里通过安键盘取消显示继续运行。
        # cv2.destroyAllWindows()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(save_dir, name + str(i) + '.png'), superimposed_img) #superimposed_img：保存的是叠加在原图上的图，也可以保存过程中其他的自己看看
        print(os.path.join(save_dir, name + str(i) + '.png'))
        i = i + 1

def featuremap_2_heatmap1(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    # heatmap = feature_map[:,0,:,:]*0    #
    heatmap = feature_map[:1, 0, :, :] * 0 #取一张图片,初始化为0
    for c in range(feature_map.shape[1]):   # 按通道
        heatmap+=feature_map[:1,c,:,:]      # 像素值相加[1,H,W]
    heatmap = heatmap.cpu().numpy()    #因为数据原来是在GPU上的
    heatmap = np.mean(heatmap, axis=0) #计算像素点的平均值,会下降一维度[H,W]

    heatmap = np.maximum(heatmap, 0)  #返回大于0的数[H,W]
    heatmap /= np.max(heatmap)      #/最大值来设置透明度0-1,[H,W]
    #heatmaps.append(heatmap)

    return heatmap
def feature_map_channel(features,img_path,save_dir = 'work_dirs/feature_map',name = 'noresbnsie2ltft_'):
	# 随便定义a,b,c,d去取对应的特征层，把通道数变换到最后一个维度，将计算的环境剥离由GPU变成CPU，tensor变为numpy
    a = torch.squeeze(features[0][:1, :, :, :], dim=0).permute(1, 2, 0).detach().cpu().numpy()
    b = torch.squeeze(features[1][:1, :, :, :], dim=0).permute(1, 2, 0).detach().cpu().numpy()
    c = torch.squeeze(features[2][:1, :, :, :], dim=0).permute(1, 2, 0).detach().cpu().numpy()
    d = torch.squeeze(features[3][:1, :, :, :], dim=0).permute(1, 2, 0).detach().cpu().numpy()
    img = cv2.imread(img_path)
    for j,x in enumerate([d]):
    				# x.shape[-1]：表示所有通道数，不想可视化这么多，可以自己写对应的数量
        for i in range(x.shape[-1]): 
            heatmap = x[:, :, i]
            # heatmap = np.maximum(heatmap, 0) #一个通道应该不用归一化了
            # heatmap /= np.max(heatmap)
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            heatmap0 = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式,0-255,heatmap0显示红色为关注区域，如果用heatmap则蓝色是关注区域
            heatmap = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)  
            superimposed_img = heatmap * 0.4 + img  # 将热力图应用于原始图像
            # plt.figure()  # 展示
            # plt.title(str(j))
            # plt.imshow(heatmap0) #, cmap='gray'
            # # plt.savefig(os.path.join(save_dir,  name+str(j)+str(i) + '.png'))
            # plt.close()
            cv2.imwrite(os.path.join(save_dir, name + str(j)+str(i) + '.png'), superimposed_img)

@DETECTORS.register_module()
class SingleStageDetectorHp(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetectorHp, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(SingleStageDetectorHp, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    #def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
     #   x = self.backbone(img)
      #  if self.with_neck:
       #     x = self.neck(x)
        #return x
    def extract_feat(self, img,img_metas):
        """Directly extract features from the backbone+neck."""
        imgpath = img_metas[0]['filename']  # 主要是要图片的原始路径 
        print(imgpath)
        x = self.backbone(img)
        draw_feature_map1(x,imgpath,name='inputs_') #特征层，图片路径，保存的文件名
        #feature_map_channel(x,imgpath,name='chanel_')
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetectorHp, self).forward_train(img, img_metas)
        x = self.extract_feat(img, img_metas)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            np.ndarray: proposals
        """
        x = self.extract_feat(img,img_metas)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation"""
        raise NotImplementedError
