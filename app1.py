# --- Full Streamlit App with Grad-CAM + PDF Report ---

import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
import zipfile
import os
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchcam.methods import SmoothGradCAMpp
from torchvision.models.resnet import ResNet
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Class labels
class_names = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

referral_guide = {
    0: "\U0001F7E2 **No Diabetic Retinopathy (No DR)**\n- No immediate referral required.\n- Suggest routine screening every 12 months.",
    1: "\U0001F7E1 **Mild DR Detected**\n- Recommend eye exam within 6–12 months.\n- Early signs present. Lifestyle & glycemic control are critical.",
    2: "\U0001F7E0 **Moderate DR Detected**\n- Referral to ophthalmologist in 3–6 months.\n- Moderate changes found. Monitor for worsening. Timely check-up important.",
    3: "\U0001F534 **Severe DR Detected**\n- Refer within 1 month to retinal specialist.\n- High risk of vision-threatening complications.",
    4: "\U0001F6A8 **Proliferative DR (PDR)**\n- **URGENT referral** to retina specialist within 1 week.\n- Risk of permanent vision loss. Laser or anti-VEGF treatment likely."
}

@st.cache_resource
def load_model():
    import torch.serialization
    torch.serialization.add_safe_globals({ResNet: ResNet})
    model_path = hf_hub_download(
        repo_id="sakshamkr1/ResNet50-APTOS-DR",
        filename="diabetic_retinopathy_full_model.pth"
    )
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()
    return model

model = load_model()
cam_extractor = SmoothGradCAMpp(model, target_layer="layer4")

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0)
    tensor.requires_grad_()  # Enable gradients for Grad-CAM
    return tensor

def generate_gradcam(input_tensor, predicted_class, raw_image):
    _ = model(input_tensor)  # Forward pass
    activation_map = cam_extractor(predicted_class, model(input_tensor))[0].squeeze().cpu().numpy()
    activation_map = cv2.resize(activation_map, raw_image.size)
    heatmap = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(raw_image), 0.5, heatmap_color, 0.5, 0)
    return Image.fromarray(overlay)

def generate_pdf(image, gradcam_img, prediction, confidence, referral, filename):
    pdf_path = os.path.join(tempfile.gettempdir(), filename)
    c = canvas.Canvas(pdf_path, pagesize=A4)
    c.setFont("Helvetica", 14)
    c.drawString(50, 800, f"Prediction: {prediction}")
    c.drawString(50, 780, f"Confidence: {confidence:.2f}%")
    c.drawString(50, 760, f"Referral Advice: {referral.splitlines()[0]}")
    
    image_path = os.path.join(tempfile.gettempdir(), "uploaded_image.jpg")
    grad_path = os.path.join(tempfile.gettempdir(), "gradcam.jpg")
    image.save(image_path)
    gradcam_img = gradcam_img.resize((1800, 1800))
    gradcam_img.save(grad_path)

    c.drawImage(image_path, 50, 480, width=200, height=200)
    c.drawImage(grad_path, 300, 480, width=200, height=200)

    # Grad-CAM color explanation legend
    c.setFont("Helvetica", 10)
    c.drawString(50, 460, "Grad-CAM Color Legend:")
    c.setFillColorRGB(1, 0, 0)
    c.drawString(60, 445, "■")
    c.setFillColorRGB(0, 0, 0)
    c.drawString(75, 445, "Red: Most important region")
    c.setFillColorRGB(1, 0.65, 0)
    c.drawString(60, 430, "■")
    c.setFillColorRGB(0, 0, 0)
    c.drawString(75, 430, "Orange/Yellow: Moderately important")
    c.setFillColorRGB(0, 0, 1)
    c.drawString(60, 415, "■")
    c.setFillColorRGB(0, 0, 0)
    c.drawString(75, 415, "Blue: Least relevant area")
    c.save()
    return pdf_path

# Streamlit App Layout
st.set_page_config(page_title="👁️ DR Detection (ResNet‑50)", layout="wide")

# Sidebar
with st.sidebar:
    st.title("")
    st.markdown("""
    **Diabetic Retinopathy Detector**  
    Built using **ResNet‑50** model trained on **APTOS 2019** dataset.  
    ---
    🔍 **Model**: ResNet‑50  
    📊 **Classes**: 5  
    🧪 **Modes**: Single & Batch Prediction
    """)

# Title and Tabs
st.title("👁️ Diabetic Retinopathy Detection")
tabs = st.tabs(["🔬 Classify Single Image", "📂 Batch Prediction", "📈 Model Overview", "🧾 How It Works"])

# ------------------------
# SINGLE IMAGE TAB
# ------------------------
with tabs[0]:
    st.markdown("Upload a **retinal fundus image** and click **Predict** to classify into one of the 5 DR stages.")

    uploaded_file = st.file_uploader("Upload Fundus Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=False, width=4000)

        if st.button("🔍 Predict"):
            with st.spinner("Running model and generating Grad-CAM..."):
                input_tensor = preprocess_image(image)
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                pred = torch.argmax(probs).item()
                confidence = probs[pred].item()

                gradcam_img = generate_gradcam(input_tensor, pred, image)

            st.info(f"**Confidence:** {confidence * 100:.2f}%")
            st.warning(referral_guide[pred])

            # Show confidence chart directly after prediction
            fig, ax = plt.subplots(figsize=(2.5, 2))
            bar = ax.bar(["Confidence"], [probs[pred].item() * 100], color='#1f77b4', width=0.5)
            ax.set_ylim(0, 100)
            ax.set_title("🔎 Model Confidence", fontsize=13)
            ax.set_ylabel("Confidence (%)", fontsize=11)
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            for rect in bar:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 15),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, weight='bold', color='black')
            fig.tight_layout()
            st.pyplot(fig)

            

            

            st.subheader("Grad-CAM Explanation")
            st.markdown("""
            Grad-CAM (Gradient-weighted Class Activation Mapping) is a visual explanation technique
            that highlights the important regions in the retinal image used by the deep learning model to make its decision.
            This helps build trust by showing what the model is attending to.
            """)
            st.image(gradcam_img, caption="Model Attention (Grad-CAM)", use_column_width=False, width=1800)

            st.markdown("""
            <small>
            🟥 <b>Red</b>: Region with highest attention (model strongly focused here)  
            🟧 <b>Orange/Yellow</b>: Moderately important regions  
            🟦 <b>Blue</b>: Least relevant or ignored areas
            </small>
            """, unsafe_allow_html=True)

            pdf_path = generate_pdf(image, gradcam_img, class_names[pred], confidence * 100, referral_guide[pred], "report.pdf")
            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download PDF Report", f, file_name="dr_report.pdf", mime="application/pdf")

# ------------------------
# MODEL OVERVIEW TAB
# ------------------------
with tabs[2]:
    st.markdown("""
    ### 📈 Model Overview
    - **Architecture**: ResNet‑50
    - **Pretrained On**: APTOS 2019 Diabetic Retinopathy Dataset
    - **Image Size**: 224 × 224 pixels
    - **Input Channels**: RGB (3)
    - **Loss Function**: CrossEntropyLoss
    - **Optimizer**: Adam
    - **Augmentation**: Resize, Normalize (ImageNet mean/std)
    - **Explainability**: SmoothGradCAM++ for visual interpretation

    This model predicts one of five diabetic retinopathy stages:
    - 0 → No DR
    - 1 → Mild
    - 2 → Moderate
    - 3 → Severe
    - 4 → Proliferative DR
    """)

# ------------------------
# HOW IT WORKS TAB
# ------------------------
with tabs[3]:
    st.markdown("""
    ### 🧾 How It Works

1. **Image Upload**:  
   Upload a single retinal image or a ZIP of multiple images for analysis.
   
   2. **Preprocessing**:  
   All images are resized to 224×224 pixels and normalized using ImageNet statistics to match model input.

3. **Prediction**:  
   The ResNet‑50 model outputs a probability score for each class using softmax.
   
   6. **Downloadable Outputs**:  
   - Single image: PDF with confidence + Grad-CAM
   - Batch mode: CSV summary and bar chart per image**:  
   - Single image: PDF with prediction + Grad-CAM
   - Batch mode: CSV of predictions, optional bar chart and confusion matrix
       - PDF report for single image.
       - CSV summary and visualizations (bar chart, confusion matrix) for batch.
    """)

# ------------------------
# BATCH IMAGE TAB
# ------------------------
with tabs[1]:
    st.markdown("Upload a **.zip file** containing multiple retina images (JPG/PNG). Each image will be analyzed.")

    zip_file = st.file_uploader("Upload ZIP File", type=["zip"])

    if zip_file:
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            results = []
            st.info("Predicting and generating Grad-CAM for all images...")

            for filename in os.listdir(temp_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(temp_dir, filename)
                    img = Image.open(path).convert("RGB")
                    input_tensor = preprocess_image(img)

                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                    pred = torch.argmax(probs).item()
                    confidence = probs[pred].item()
                    referral = referral_guide[pred].splitlines()[0].strip()

                    gradcam_img = generate_gradcam(input_tensor, pred, img)

                    grad_path = os.path.join(temp_dir, f"gradcam_{filename}")
                    gradcam_img.save(grad_path)

                    results.append({
                        "Filename": filename,
                                                "Confidence (%)": round(confidence * 100, 2),
                        "Referral Advice": referral
                    })

            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results as CSV", data=csv, file_name="dr_predictions.csv", mime="text/csv")

            # Optional visualizations
            if st.checkbox("📊 Show Confidence Per Image (Bar Chart)"):
                for row in results:
                    fig, ax = plt.subplots()
                    ax.bar(["Confidence"], [row['Confidence (%)']], color='green')
                    ax.set_ylim(0, 100)
                    ax.set_title(f"{row['Filename']} - Confidence")
                    ax.set_ylabel("Confidence (%)")
                    st.pyplot(fig)

            if st.checkbox("🧮 Show Confusion Matrix"):
                from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
                y_true = df['Prediction']
                y_pred = df['Prediction']  # Placeholder (replace with actual if available)
                cm = confusion_matrix(y_true, y_pred, labels=list(class_names.values()))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names.values()))
                fig, ax = plt.subplots(figsize=(8, 6))
                disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
                st.pyplot(fig)
