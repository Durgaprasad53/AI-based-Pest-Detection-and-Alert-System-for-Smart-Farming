from flask import Flask, request, render_template
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
app = Flask(__name__)
# Step 1: Load the YOLOv8 model
model = YOLO('pest.pt')

# Step 2: Define scientific-to-common name mappings
pest_suggestions = {
    "Adulto": "Introduce natural predators or use pheromone traps to control Adulto pests.",
    "aphid": "Spray neem oil or insecticidal soap to control aphid populations.",
    "Black-Grass-Caterpillar": "Apply Bacillus thuringiensis (Bt) or handpick visible caterpillars.",
    "Cerambycidae_larvae": "Use pheromone traps and practice good field sanitation.",
    "citricola scale": "Prune heavily infested branches and introduce natural enemies like ladybugs.",
    "Cnidocampa_flavescens(Walker_pupa)": "Remove pupae manually and use natural predators.",
    "Coconut-black-headed-caterpillar": "Spray neem-based pesticides and destroy affected leaves.",
    "Cricket": "Use bait traps and keep the environment dry.",
    "Diamondback-moth": "Apply insecticidal soaps or Bacillus thuringiensis (Bt).",
    "Drosicha_contrahens_female": "Spray insecticides or manually remove the pests.",
    "Erthesina_fullo_nymph-2": "Use chemical pesticides or introduce natural predators.",
    "Grasshopper": "Apply insecticidal baits or introduce natural enemies.",
    "Hyphantria_cunea_larvae": "Apply Bacillus thuringiensis (Bt) and remove affected leaves.",
    "Hyphantria_cunea_pupa": "Destroy pupae and use insecticides if necessary.",
    "Latoia_consocia_Walker_larvae": "Prune affected areas and apply organic pesticides.",
    "Leaf-eating-caterpillar": "Handpick caterpillars and use neem-based sprays.",
    "Sericinus_montela_larvae": "Use Bacillus thuringiensis (Bt) and encourage natural predators like wasps."
}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    if not file:
        return "No file uploaded", 400

    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)

    results = model.predict(source=image, conf=0.3)

    detection_results = ""
    if not results or len(results[0].boxes) == 0:
        detection_results = "No pests detected in the image."
    else:
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                pest_name = model.names[class_id]
                suggestion = pest_suggestions.get(pest_name, "No specific suggestion available.")

                detection_results += f"Detected: {pest_name} <br>"
                detection_results += f"Suggested Action: {suggestion}<br><br>"

    return detection_results


if __name__ == '__main__':
    app.run(debug=True)
