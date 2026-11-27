from ultralytics import YOLO

# โหลดไฟล์ weights ของคุณ
model = YOLO(r"E:\RMUTL\object DET 101\yolo\model\3best.pt")

# Export เป็นไฟล์ ONNX
model.export(format="onnx", dynamic=True)