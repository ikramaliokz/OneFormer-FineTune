from transformers import AutoProcessor, AutoModelForUniversalSegmentation
from PIL import Image
import torch
import cv2
import numpy as np
# processor = AutoProcessor.from_pretrained("oneformer_ade20k_swin_tiny")
# model = AutoModelForUniversalSegmentation.from_pretrained("oneformer_ade20k_swin_tiny")
# model.eval()
def normalize_bounding_boxes(bboxes, image_width, image_height):
    normalized_bboxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        x_min_norm = x_min / image_width if image_width != 0 else 0
        y_min_norm = y_min / image_height if image_height != 0 else 0
        x_max_norm = x_max / image_width if image_width != 0 else 0
        y_max_norm = y_max / image_height if image_height != 0 else 0
        normalized_bboxes.append([x_min_norm, y_min_norm, x_max_norm, y_max_norm])
    return normalized_bboxes

def oneformer_results(img_path, model, processor):
    # processor = AutoProcessor.from_pretrained("oneformer_ade20k_swin_tiny")
    # model = AutoModelForUniversalSegmentation.from_pretrained("oneformer_ade20k_swin_tiny")
    # model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # this disables the text encoder and hence enables to do forward passes
    # without passing text_inputs
    model.model.is_training = False

    image = Image.open(img_path)
    # prepare image for the model
    inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")

    for k,v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(k,v.shape)

        # forward pass (no need for gradients at inference time)
        with torch.no_grad():
            inputs = inputs.to(device)
            model = model.to(device)
            outputs = model(**inputs)

    # postprocessing
    semantic_segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    semantic_segmentation = semantic_segmentation //69
    # Convert tensor to NumPy array
    binary_mask_np = semantic_segmentation.cpu().numpy().astype(np.uint8)
    # Find contours
    contours, _ = cv2.findContours(binary_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract bounding boxes from contours
    bounding_boxes = [(x, y, x + w, y + h) for x, y, w, h in (cv2.boundingRect(contour) for contour in contours)]
    shape = image.size
    bounding_boxes = normalize_bounding_boxes(bounding_boxes,shape[0],shape[1])
    
    return semantic_segmentation, bounding_boxes

# img_path = 'panoptic_dataset/val2017/val_0_0_jpg.rf.0e6869d3c5830d8a48349245f0fe845d.jpg'
# oneformer_results(img_path, model, processor)