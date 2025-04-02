import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Trash-Net"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def trash_classification(image):
    """Predicts the category of waste material in the image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "cardboard", 
        "1": "glass", 
        "2": "metal", 
        "3": "paper", 
        "4": "plastic", 
        "5": "trash"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=trash_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Trash Classification",
    description="Upload an image to classify the type of waste material."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
