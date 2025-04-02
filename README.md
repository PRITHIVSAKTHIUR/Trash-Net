
![11.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/zmvA8U-wg82APftWpVX1w.png)

# **Trash-Net**  

> **Trash-Net** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for a single-label classification task. It is designed to classify images of waste materials into different categories using the **SiglipForImageClassification** architecture.  

The model categorizes images into six classes:  
- **Class 0:** "cardboard"  
- **Class 1:** "glass"  
- **Class 2:** "metal"  
- **Class 3:** "paper"  
- **Class 4:** "plastic"  
- **Class 5:** "trash"

```py
Classification Report:
              precision    recall  f1-score   support

   cardboard     0.9912    0.9739    0.9825       806
       glass     0.9564    0.9641    0.9602      1002
       metal     0.9523    0.9744    0.9632       820
       paper     0.9520    0.9848    0.9681      1188
     plastic     0.9835    0.9274    0.9546       964
       trash     0.9127    0.9161    0.9144       274

    accuracy                         0.9626      5054
   macro avg     0.9580    0.9568    0.9572      5054
weighted avg     0.9631    0.9626    0.9626      5054
```

![download (1).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/gl4jGVduxcQQi2FrqzL1D.png)

# **Run with Transformers🤗**  

```python
!pip install -q transformers torch pillow gradio
```

```python
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
```  

# **Intended Use:**  

The **Trash-Net** model is designed to classify waste materials into different categories. Potential use cases include:  

- **Waste Management:** Assisting in automated waste sorting and recycling.  
- **Environmental Monitoring:** Identifying and categorizing waste in public spaces.  
- **Educational Purposes:** Teaching waste classification and sustainability.  
- **Smart Cities:** Enhancing waste disposal systems through AI-driven classification. 
