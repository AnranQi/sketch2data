from ultralytics import YOLO
 
# Load the model.
model = YOLO('.\pretrained_yolov8n_model\yolov8n.pt')
# Training.
results = model.train(
   data='.\\thoughts_detection_dataset\\thoughts.yaml',
   imgsz=1169,
   epochs=20,
   batch=64,
   name='thoughts_yolov8',
   workers=0
)

