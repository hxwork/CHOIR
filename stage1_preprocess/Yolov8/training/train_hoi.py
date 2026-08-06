from ultralytics import YOLO

# load pretrained model
# model = YOLO("models/wilor_hand_detector.pt")
model = YOLO("runs/fine_tune5/weights/last.pt")

# Run from the Yolov8 root with: python -m training.train_hoi
results = model.train(cfg="training/train_hoi_args.yaml")
