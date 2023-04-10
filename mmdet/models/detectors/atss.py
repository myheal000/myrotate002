from ..builder import DETECTORS
#from .single_stage import SingleStageDetector
from .single_stage_heatmap import SingleStageDetectorHp


@DETECTORS.register_module()
#class ATSS(SingleStageDetector):

 #   def __init__(self,
      #           backbone,
       #          neck,
        #         bbox_head,
         #        train_cfg=None,
          #       test_cfg=None,
          #       pretrained=None):
          #super(ATSS, self).__init__(backbone, neck, bbox_head, train_cfg,
         #                          test_cfg, pretrained)
class ATSS(SingleStageDetectorHp):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(ATSS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)

