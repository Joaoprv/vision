import torch
import torch.nn as nn
from backbone import CSPDarknet53
from blocks import SPPBlock
from neck import PANet
from head import YOLOHead


class YOLOv4(nn.Module):
    """
    Complete YOLOv4 model
    """
    def __init__(self, num_classes=80):
        super(YOLOv4, self).__init__()
        
        # Backbone
        self.backbone = CSPDarknet53()
        
        # Neck - SPP and PAN
        self.spp = SPPBlock(1024, 512)
        self.pan = PANet()
        
        # Detection heads for different scales
        self.head_small = YOLOHead(128, num_classes)   # For small objects (large feature map)
        self.head_medium = YOLOHead(256, num_classes)  # For medium objects
        self.head_large = YOLOHead(512, num_classes)   # For large objects (small feature map)
        
        # Anchor boxes for different scales (normalized to [0,1])
        # These should be customized based on your dataset
        self.anchors = [
            # Small scale anchors (for detecting large objects)
            [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
            # Medium scale anchors
            [(0.07, 0.15), (0.18, 0.29), (0.32, 0.19)],
            # Large scale anchors (for detecting small objects)
            [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]
        ]
    
    def forward(self, x):
        # Get features from backbone
        route_1, route_2, route_3 = self.backbone(x)
        
        # Apply SPP to the last layer
        spp_out = self.spp(route_3)
        
        # Apply Path Aggregation Network
        small_scale, medium_scale, large_scale = self.pan(route_1, route_2, spp_out)
        
        # Apply detection heads
        output_small = self.head_small(small_scale)
        output_medium = self.head_medium(medium_scale)
        output_large = self.head_large(large_scale)
        
        return [output_small, output_medium, output_large]
    
    def transform_predictions(self, predictions, input_size, threshold=0.5):
        """
        Transform raw predictions to bounding boxes
        
        Args:
            predictions: List of tensors from forward pass
            input_size: Tuple (height, width) of original input image
            threshold: Confidence threshold for object detection
            
        Returns:
            List of detected objects with format:
            [batch_idx, class_idx, confidence, x1, y1, x2, y2]
        """
        device = predictions[0].device
        batch_size = predictions[0].shape[0]
        all_detections = []
        
        for batch_idx in range(batch_size):
            batch_detections = []
            
            # Process each scale (small, medium, large)
            for scale_idx, pred in enumerate(predictions):
                # Get current prediction for this batch
                pred_batch = pred[batch_idx]
                
                # Reshape to [num_anchors, grid_height, grid_width, 5 + num_classes]
                grid_height, grid_width = pred_batch.shape[1:3]
                pred_batch = pred_batch.permute(1, 2, 0).reshape(
                    grid_height, grid_width, self.num_anchors, 5 + self.num_classes
                )
                
                # Extract values
                x = torch.sigmoid(pred_batch[..., 0])  # Center x, normalized to cell [0,1]
                y = torch.sigmoid(pred_batch[..., 1])  # Center y, normalized to cell [0,1]
                w = pred_batch[..., 2]  # Width, normalized to input size
                h = pred_batch[..., 3]  # Height, normalized to input size
                conf = torch.sigmoid(pred_batch[..., 4])  # Object confidence
                cls_prob = torch.sigmoid(pred_batch[..., 5:])  # Class probabilities
                
                # Create grid offsets
                grid_x = torch.arange(grid_width, device=device).repeat(grid_height, 1)
                grid_y = torch.arange(grid_height, device=device).unsqueeze(1).repeat(1, grid_width)
                
                # Get anchors for this scale
                anchors = torch.tensor(self.anchors[scale_idx], device=device)
                
                # Calculate bounding box coordinates
                pred_boxes = torch.zeros_like(pred_batch[..., :4])
                pred_boxes[..., 0] = (x + grid_x.unsqueeze(2)) / grid_width  # Normalize x to [0,1]
                pred_boxes[..., 1] = (y + grid_y.unsqueeze(2)) / grid_height  # Normalize y to [0,1]
                pred_boxes[..., 2] = torch.exp(w) * anchors[:, 0].unsqueeze(0).unsqueeze(0)  # Normalize w to [0,1]
                pred_boxes[..., 3] = torch.exp(h) * anchors[:, 1].unsqueeze(0).unsqueeze(0)  # Normalize h to [0,1]
                
                # Convert to corners format (x1, y1, x2, y2)
                x1 = (pred_boxes[..., 0] - pred_boxes[..., 2] / 2) * input_size[1]
                y1 = (pred_boxes[..., 1] - pred_boxes[..., 3] / 2) * input_size[0]
                x2 = (pred_boxes[..., 0] + pred_boxes[..., 2] / 2) * input_size[1]
                y2 = (pred_boxes[..., 1] + pred_boxes[..., 3] / 2) * input_size[0]
                
                # Find objects with confidence above threshold
                mask = conf > threshold
                
                # Get class with highest probability for each detection
                class_scores, class_idx = torch.max(cls_prob, dim=-1)
                
                # Get final confidence score
                score = conf * class_scores
                
                # Process detections
                for i, j, k in zip(*torch.where(mask)):
                    detection = [
                        batch_idx,              # Batch index
                        class_idx[i, j, k],     # Class index
                        score[i, j, k],         # Confidence score
                        x1[i, j, k],            # x1
                        y1[i, j, k],            # y1
                        x2[i, j, k],            # x2
                        y2[i, j, k]             # y2
                    ]
                    batch_detections.append(detection)
            
            if batch_detections:
                # Convert to tensor
                batch_detections = torch.tensor(batch_detections, device=device)
                
                # Apply non-maximum suppression
                # This is a simplified version - in practice, you'd want a more efficient implementation
                keep = []
                while batch_detections.size(0) > 0:
                    # Get detection with highest confidence
                    best_idx = torch.argmax(batch_detections[:, 2])
                    best_det = batch_detections[best_idx]
                    keep.append(best_det)
                    
                    # Compute IoU with other detections
                    if batch_detections.size(0) > 1:
                        # Extract remaining detections
                        other_dets = torch.cat([
                            batch_detections[:best_idx],
                            batch_detections[best_idx+1:]
                        ])
                        
                        # Only compare detections of the same class
                        same_class = other_dets[:, 1] == best_det[1]
                        if not torch.any(same_class):
                            break
                            
                        other_dets = other_dets[same_class]
                        
                        # Compute IoU
                        best_area = (best_det[5] - best_det[3]) * (best_det[6] - best_det[4])
                        other_areas = (other_dets[:, 5] - other_dets[:, 3]) * (other_dets[:, 6] - other_dets[:, 4])
                        
                        # Calculate intersection area
                        x1 = torch.max(best_det[3], other_dets[:, 3])
                        y1 = torch.max(best_det[4], other_dets[:, 4])
                        x2 = torch.min(best_det[5], other_dets[:, 5])
                        y2 = torch.min(best_det[6], other_dets[:, 6])
                        
                        w = torch.clamp(x2 - x1, min=0)
                        h = torch.clamp(y2 - y1, min=0)
                        
                        inter = w * h
                        iou = inter / (best_area + other_areas - inter)
                        
                        # Keep boxes with IoU < 0.45
                        nms_mask = iou < 0.45
                        batch_detections = other_dets[nms_mask]
                    else:
                        break
                
                all_detections.extend(keep)
        
        return all_detections
    
def compute_loss(predictions, targets, model):
    """
    Compute YOLOv4 loss function
    
    Args:
        predictions: List of tensors from model forward pass
        targets: List of ground truth boxes [batch_idx, class_idx, x, y, w, h]
        model: YOLOv4 model for anchor information
    
    Returns:
        Total loss and dictionary with loss components
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    device = predictions[0].device
    lambda_coord = 5.0
    lambda_noobj = 0.5
    
    total_loss = torch.tensor(0.0, device=device)
    obj_loss = torch.tensor(0.0, device=device)
    noobj_loss = torch.tensor(0.0, device=device)
    box_loss = torch.tensor(0.0, device=device)
    class_loss = torch.tensor(0.0, device=device)
    
    # Get anchors and stride information from model
    anchors = model.module.anchors if hasattr(model, 'module') else model.anchors
    strides = model.module.strides if hasattr(model, 'module') else model.strides
    
    # Process each prediction layer (typically 3 layers for small, medium, large objects)
    for layer_idx, pred in enumerate(predictions):
        # Get grid size for this prediction layer
        batch_size, _, grid_h, grid_w = pred.shape
        
        # Reshape predictions to [batch, num_anchors, grid_h, grid_w, 5+num_classes]
        # where 5 represents x, y, w, h, objectness
        num_anchors = len(anchors[layer_idx])
        num_classes = pred.shape[1] // num_anchors - 5
        pred = pred.view(batch_size, num_anchors, 5 + num_classes, grid_h, grid_w)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()
        
        # Sigmoid activation for xy, objectness and class predictions
        pred_xy = torch.sigmoid(pred[..., 0:2])  # Center x, y
        pred_wh = torch.exp(pred[..., 2:4])  # Width, height
        pred_obj = torch.sigmoid(pred[..., 4:5])  # Objectness
        pred_cls = torch.sigmoid(pred[..., 5:])  # Class probabilities
        
        # Calculate stride for this layer
        stride = strides[layer_idx]
        
        # Add grid offsets to xy predictions
        grid_y, grid_x = torch.meshgrid([torch.arange(grid_h, device=device),
                                        torch.arange(grid_w, device=device)])
        grid_xy = torch.stack((grid_x, grid_y), dim=2).view(
            1, 1, grid_h, grid_w, 2).float()
        
        # Final predictions relative to image dimensions
        pred_xy = (pred_xy + grid_xy) * stride
        pred_wh = pred_wh * anchors[layer_idx].view(1, num_anchors, 1, 1, 2) * stride
        
        # Process targets for this layer
        target_mask = (targets[:, 0] == layer_idx)
        if target_mask.sum() == 0:
            # No targets for this layer, only compute no-object loss
            noobj_loss += lambda_noobj * F.binary_cross_entropy(
                pred_obj.view(-1), 
                torch.zeros_like(pred_obj.view(-1)),
                reduction='sum'
            )
            continue
            
        # Extract targets for this layer
        layer_targets = targets[target_mask]
        
        # Assign targets to specific anchors and grid cells
        # Each target gets assigned to the best-matching anchor
        t_batch_idx = layer_targets[:, 0].long()
        t_class_idx = layer_targets[:, 1].long()
        t_box = layer_targets[:, 2:6]  # x, y, w, h normalized to [0, 1]
        
        # Convert normalized coordinates to absolute values
        t_xy = t_box[:, 0:2] * torch.tensor([grid_w, grid_h], device=device)
        t_wh = t_box[:, 2:4] * torch.tensor([grid_w, grid_h], device=device)
        
        # Calculate which grid cell each target belongs to
        t_grid_xy = t_xy.long()
        t_grid_x, t_grid_y = t_grid_xy[:, 0], t_grid_xy[:, 1]
        
        # Calculate IoU between targets and anchors to find best anchor
        anchor_shapes = torch.tensor(anchors[layer_idx], device=device) / stride
        t_wh_tensor = t_wh.unsqueeze(1)
        anchor_shapes_tensor = anchor_shapes.unsqueeze(0)
        
        # Calculate intersection area for IoU
        w_min = torch.min(t_wh_tensor[:, :, 0], anchor_shapes_tensor[:, :, 0])
        h_min = torch.min(t_wh_tensor[:, :, 1], anchor_shapes_tensor[:, :, 1])
        intersect_area = w_min * h_min
        
        # Calculate union area for IoU
        t_area = t_wh_tensor[:, :, 0] * t_wh_tensor[:, :, 1]
        anchor_area = anchor_shapes_tensor[:, :, 0] * anchor_shapes_tensor[:, :, 1]
        union_area = t_area + anchor_area - intersect_area
        
        # Calculate IoU
        iou = intersect_area / union_area
        best_anchor_indices = torch.max(iou, dim=1)[1]
        
        # Now we have the assigned grid cells and anchors for each target
        # We can calculate losses for these grid cells and anchors
        
        # Initialize target tensors
        obj_mask = torch.zeros_like(pred_obj)
        noobj_mask = torch.ones_like(pred_obj)
        t_xy_offset = torch.zeros((len(layer_targets), 2), device=device)
        t_wh_offset = torch.zeros((len(layer_targets), 2), device=device)
        t_cls = torch.zeros((len(layer_targets), num_classes), device=device)
        
        # Fill target tensors
        for i, (b_idx, a_idx, gy, gx, cls_idx) in enumerate(
                zip(t_batch_idx, best_anchor_indices, t_grid_y, t_grid_x, t_class_idx)):
            if gx < grid_w and gy < grid_h:  # Make sure target is within grid bounds
                obj_mask[b_idx, a_idx, gy, gx] = 1
                noobj_mask[b_idx, a_idx, gy, gx] = 0
                
                # Calculate target xy offset within grid cell
                t_xy_offset[i] = t_xy[i] - torch.tensor([gx, gy], device=device)
                
                # Calculate target wh relative to anchor
                t_wh_offset[i] = torch.log(t_wh[i] / anchor_shapes[a_idx] + 1e-16)
                
                # One-hot encoding for class
                if cls_idx < num_classes:
                    t_cls[i, cls_idx] = 1
        
        # Compute CIoU loss for box regression
        # Get predictions corresponding to targets
        target_indices = torch.nonzero(obj_mask.view(-1), as_tuple=True)[0]
        if len(target_indices) > 0:
            # Extract target boxes and predicted boxes
            pred_boxes = torch.cat([
                pred_xy.view(-1, 2)[target_indices],
                pred_wh.view(-1, 2)[target_indices]
            ], dim=1)
            
            # Convert to [x1, y1, x2, y2] format for CIoU
            pred_x1y1 = pred_boxes[:, :2] - pred_boxes[:, 2:] / 2
            pred_x2y2 = pred_boxes[:, :2] + pred_boxes[:, 2:] / 2
            pred_boxes_xyxy = torch.cat([pred_x1y1, pred_x2y2], dim=1)
            
            target_boxes = torch.cat([
                (t_grid_xy + t_xy_offset) * stride,
                torch.exp(t_wh_offset) * anchor_shapes[best_anchor_indices] * stride
            ], dim=1)
            
            # Convert to [x1, y1, x2, y2] format for CIoU
            target_x1y1 = target_boxes[:, :2] - target_boxes[:, 2:] / 2
            target_x2y2 = target_boxes[:, :2] + target_boxes[:, 2:] / 2
            target_boxes_xyxy = torch.cat([target_x1y1, target_x2y2], dim=1)
            
            # Calculate CIoU loss
            ciou_loss = bbox_ciou(pred_boxes_xyxy, target_boxes_xyxy)
            box_loss += (lambda_coord * (1.0 - ciou_loss).sum())
            
            # Objectness loss with IoU weighting
            obj_loss += F.binary_cross_entropy(
                pred_obj.view(-1)[target_indices],
                obj_mask.view(-1)[target_indices] * ciou_loss.detach().clamp(0),
                reduction='sum'
            )
            
            # Classification loss
            if num_classes > 1:
                class_loss += F.binary_cross_entropy(
                    pred_cls.view(-1, num_classes)[target_indices],
                    t_cls,
                    reduction='sum'
                )
        
        # No-object loss
        noobj_loss += lambda_noobj * F.binary_cross_entropy(
            pred_obj.view(-1)[noobj_mask.view(-1) == 1],
            torch.zeros_like(pred_obj.view(-1)[noobj_mask.view(-1) == 1]),
            reduction='sum'
        )
    
    # Normalize losses by batch size
    batch_size = predictions[0].size(0)
    obj_loss /= batch_size
    noobj_loss /= batch_size
    box_loss /= batch_size
    class_loss /= batch_size
    total_loss = obj_loss + noobj_loss + box_loss + class_loss
    
    return total_loss, {
        'obj_loss': obj_loss,
        'noobj_loss': noobj_loss,
        'box_loss': box_loss,
        'class_loss': class_loss
    }


def bbox_ciou(boxes1, boxes2):
    """
    Calculate CIoU (Complete IoU) between two sets of bounding boxes
    
    Args:
        boxes1: First set of boxes in format [x1, y1, x2, y2]
        boxes2: Second set of boxes in format [x1, y1, x2, y2]
        
    Returns:
        CIoU value between 0 and 1 (1 being perfect match)
    """
    # Extract coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    
    # Calculate area of boxes
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    # Calculate intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    # Intersection area
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Union Area
    union_area = b1_area + b2_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-16)
    
    # Get enclosing box
    enclose_x1 = torch.min(b1_x1, b2_x1)
    enclose_y1 = torch.min(b1_y1, b2_y1)
    enclose_x2 = torch.max(b1_x2, b2_x2)
    enclose_y2 = torch.max(b1_y2, b2_y2)
    
    # Diagonal length of enclosing box squared
    c_square = torch.pow(enclose_x2 - enclose_x1, 2) + torch.pow(enclose_y2 - enclose_y1, 2)
    
    # Center distance squared
    b1_center_x = (b1_x1 + b1_x2) / 2
    b1_center_y = (b1_y1 + b1_y2) / 2
    b2_center_x = (b2_x1 + b2_x2) / 2
    b2_center_y = (b2_y1 + b2_y2) / 2
    center_dist_square = torch.pow(b1_center_x - b2_center_x, 2) + torch.pow(b1_center_y - b2_center_y, 2)
    
    # v term (aspect ratio consistency)
    w1 = b1_x2 - b1_x1
    h1 = b1_y2 - b1_y1
    w2 = b2_x2 - b2_x1
    h2 = b2_y2 - b2_y1
    
    v = (4 / (torch.pi ** 2)) * torch.pow(
        torch.atan(w2 / (h2 + 1e-16)) - torch.atan(w1 / (h1 + 1e-16)), 2
    )
    
    # Alpha for balancing v term
    alpha = v / (1 - iou + v + 1e-16)
    
    # CIoU
    ciou = iou - (center_dist_square / (c_square + 1e-16) + alpha * v)
    
    return ciou