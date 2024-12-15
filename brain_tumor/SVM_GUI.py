import cv2
import tkinter as tk
from tkinter import filedialog
import joblib
from PIL import Image, ImageTk

# Load the PCA model and SVM model
pca_model = joblib.load('pca_model.joblib')
svm_model = joblib.load('svm_model.joblib')


# Define GUI
class TumorPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Prediction")

        self.label = tk.Label(root, text="Upload MRI Scan Image:")
        self.label.pack(pady=10)

        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()

        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=20)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.predict_tumor(file_path)

    def predict_tumor(self, image_path):
        # Read and preprocess the image
        img = cv2.imread(image_path, 0)
        img = cv2.resize(img, (200, 200))
        img_flat = img.reshape(1, -1) / 255

        # Apply PCA transformation
        img_pca = pca_model.transform(img_flat)

        # Make prediction using SVM model
        prediction = svm_model.predict(img_pca)

        # Display the image with the result
        self.display_image(image_path)

        # Display the prediction result
        result_text = "Result: " + ("No Tumor" if prediction == 0 else "Pituitary Tumor")
        self.result_label.config(text=result_text)

    def display_image(self, image_path):
        img = Image.open(image_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)

        panel = tk.Label(root, image=img)
        panel.image = img
        panel.pack()


# Create the GUI window
root = tk.Tk()
app = TumorPredictionGUI(root)
root.mainloop()
