
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
import copy

from torchvision.utils import save_image
from deformable_detr.py import DeformableDETR as D




def get_box_from_object_queries(enc_outputs_class, enc_outputs_coord_unact)

    hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)
    outputs_classes = []
    outputs_coords = []
    
    outputs_class = self.class_embed[lvl](hs[lvl])


    for lvl in range(hs.shape[0]):
        if lvl == 0:
            reference = init_reference
        else:
            reference = inter_references[lvl - 1]
        reference = inverse_sigmoid(reference)
        outputs_class = self.class_embed[lvl](hs[lvl])
        tmp = self.bbox_embed[lvl](hs[lvl])
        if reference.shape[-1] == 4:
            tmp += reference
        else:
            assert reference.shape[-1] == 2
            tmp[..., :2] += reference
        outputs_coord = tmp.sigmoid()
        outputs_classes.append(outputs_class)
        outputs_coords.append(outputs_coord)
    outputs_class = torch.stack(outputs_classes)
    outputs_coord = torch.stack(outputs_coords)

    out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
    if self.aux_loss:
        out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

    if self.two_stage:
        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}