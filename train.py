from ultralytics import YOLO

model = YOLO("yolo11m.pt") 
results = model.train(
    data="dataset/data.yaml",
    epochs=1000,
    imgsz=640,
    batch=16,
    device="0",
    workers=2,
    project="logo_id",
    name="yolov11_run",
    patience=50,
    exist_ok=True,
    
    # Optimizer and learning rate settings
    optimizer='AdamW',      # AdamW optimizer (better convergence)
    lr0=0.001,              # Initial learning rate (lower for AdamW)
    lrf=0.01,               # Final learning rate (1% of initial)
    momentum=0.937,         # Momentum/beta1
    weight_decay=0.0005,    # Weight decay for regularization
    warmup_epochs=3.0,      # Warmup epochs
    warmup_momentum=0.8,    # Warmup initial momentum
    warmup_bias_lr=0.1,     # Warmup initial bias learning rate
    
    # --- Geometric Augmentations ---
    degrees=15.0,        # Rotate image +/- 15 degrees
    translate=0.2,       # Shift image +/- 20% vertically/horizontally
    scale=0.6,           # Scale image gain +/- 60%
    shear=0.0,           # Shear angle (usually keep 0)
    flipud=0.0,          # Flip up-down (0.0 = off, 0.5 = 50% chance)
    fliplr=0.5,          # Flip left-right (standard is 50%)
    
    # --- Color & Noise Augmentations ---
    hsv_h=0.02,          # Adjust Hue (Color) +/- 2%
    hsv_s=0.7,           # Adjust Saturation +/- 70% (Simulates dull/vibrant cams)
    hsv_v=0.4,           # Adjust Value (Brightness) +/- 40% (Simulates lighting)
    
    # --- Advanced YOLO Specific ---
    mosaic=1.0,          # 100% chance to stitch 4 images together (Crucial!)
    mixup=0.1,           # 10% chance to blend 2 images (Good for large datasets)
    copy_paste=0.1,      # Copy-paste objects into other images (segmentation only)
    
    # --- Optimization ---
    close_mosaic=10      # Turn OFF Mosaic aug for the final 10 epochs for precision
)


