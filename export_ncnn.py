# In a separate script, not your main app
from ultralytics import YOLO

# Load your custom-trained model
model = YOLO('/home/hllxrd/chinhcachep/best_lowres.pt')

# Export to NCNN format
model.export(format='ncnn')

# This will create a 'best_lowres_ncnn_model' directory.
# You will use the files inside this directory for inference.