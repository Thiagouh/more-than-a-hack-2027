"""Person instance segmentation adapter; no team labels or truth inputs."""
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
from .pilot import sha256


class YoloPeople:
    def __init__(self, checkpoint, *, device='mps', imgsz=960, confidence=.25):
        from ultralytics import YOLO
        import ultralytics
        self.model = YOLO(str(checkpoint))
        if self.model.task != 'segment':
            raise ValueError('Person masks require a YOLO instance-segmentation checkpoint')
        self.options = dict(device=device, imgsz=imgsz, conf=confidence,
                            classes=[0], verbose=False)
        if self.model.names.get(0) != 'person':
            raise ValueError('Checkpoint class zero must be person')
        self.environment = dict(backend='ultralytics-person-segmentation',
            ultralytics_version=ultralytics.__version__, checkpoint=str(checkpoint),
            checkpoint_sha256=sha256(Path(self.model.ckpt_path)), **self.options,
            mask_method='original_image_polygon_rasterization')

    def segment(self, image):
        started = time.perf_counter()
        result = self.model.predict(image, **self.options)[0]
        masks = []
        if result.masks is not None:
            for polygon in result.masks.xy:
                canvas = Image.new('1', image.size)
                if len(polygon) >= 3:
                    ImageDraw.Draw(canvas).polygon([tuple(p) for p in polygon], fill=1)
                masks.append(np.asarray(canvas, dtype=bool))
        return masks, dict(elapsed_s=time.perf_counter()-started,
                           confidences=result.boxes.conf.cpu().tolist())


class YoloBoxes:
    """Boxes are observations; bottom centers are explicitly approximate feet."""
    def __init__(self, checkpoint, *, device='mps', imgsz=640, confidence=.25):
        from ultralytics import YOLO
        import ultralytics
        self.model=YOLO(str(checkpoint))
        if self.model.task!='detect' or self.model.names.get(0)!='person':
            raise ValueError('COCO person detection checkpoint required')
        self.options=dict(device=device,imgsz=imgsz,conf=confidence,classes=[0],verbose=False)
        self.environment=dict(backend='ultralytics-person-boxes',ultralytics_version=ultralytics.__version__,
            checkpoint=str(checkpoint),checkpoint_sha256=sha256(Path(self.model.ckpt_path)),**self.options)

    def detect(self,image):
        started=time.perf_counter()
        result=self.model.predict(image,**self.options)[0]
        return result.boxes.xyxy.cpu().tolist(),dict(elapsed_s=time.perf_counter()-started,
            confidences=result.boxes.conf.cpu().tolist())
