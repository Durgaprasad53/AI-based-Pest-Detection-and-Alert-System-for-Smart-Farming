from ultralytics import YOLO
import cv2

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

# Step 4: Run predictions on the image
results = model.predict(
    source='images/pest_6.jpeg',  # Path to the uploaded image
    conf=0.5,  # Confidence threshold
    save=False  # Don't save default output
)

# Step 5: Process results and manually overlay common names and suggestions
for result in results:
    img = result.orig_img.copy()  # Copy original image for visualization
    for box in result.boxes:
        class_id = int(box.cls[0])  # Extract class ID
        confidence = float(box.conf[0])  # Extract confidence score

        # Get scientific name from the model and replace with common name
        class_name = model.names[class_id]


        # Get pest control suggestion
        suggestion = pest_suggestions.get(class_name, "No specific suggestion available.")

        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Print detection results with suggestions
        print(f"Detected: {class_name}, Confidence: {confidence:.2f}")
        print(f"Suggested Action: {suggestion}\n")

    # Save and display the modified image
    cv2.imwrite("output_custom_names_suggestions.jpg", img)
    cv2.imshow("Detections", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
