from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")
model.info()

# Use the model
results = model("./test.jpg")

# Save the results
results[0].save("yolo11n-result.jpg")

# Export to onnx with simplify
model.export(format='onnx', simplify=True,nms=True,opset=18)


