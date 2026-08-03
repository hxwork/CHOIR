from ultralytics import YOLO

# load pretrained model
model = YOLO("models/wilor_hand_detector.pt")

# Run from the Yolov8 root with: python -m training.train
results = model.train(cfg="training/train_args.yaml")
