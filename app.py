from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import torch
import traceback

from src.preprocess_image import load_image, preprocess_image
from src.vision_agent import VisionAgent
from src.history_agent import HistoryAgent
from src.diagnosis_agent import DiagnosisAgent
from src.ocr_agent import OCRAgent
from src.anatomy_agent import AnatomyAgent
from src.pretrained_classifiers import ChestClassifier, BrainClassifier

app = FastAPI(title="Agentic Medical Diagnosis API")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data/uploads", exist_ok=True)

# Initialize Agents Globally
try:
    print("Initializing Vision Agent (ImageNet pretrained weights)...")
    vision_agent = VisionAgent(pretrained=True)
    vision_agent.eval()

    print("Initializing Pre-trained Chest Classifier (TorchXRayVision)...")
    chest_classifier = ChestClassifier()

    print("Initializing Pre-trained Brain Classifier (HuggingFace ViT)...")
    brain_classifier = BrainClassifier()

    print("Initializing Anatomy Agent...")
    anatomy_agent = AnatomyAgent()

    print("Initializing History Agent...")
    history_agent = HistoryAgent("./chroma_db_test")

    print("Initializing Diagnosis Agent (Upgraded to Qwen2.5-0.5B-Instruct for low RAM requirements)...")
    diagnosis_agent = DiagnosisAgent(model_id="Qwen/Qwen2.5-0.5B-Instruct")

    print("Initializing OCR Agent...")
    ocr_agent = OCRAgent()
except Exception as e:
    print(f"Warning: Model initialization failed (perhaps missing packages during setup): {e}")

@app.post("/api/diagnose")
async def diagnose(
    patient_id: str = Form("patient_xyz"),
    symptoms: str = Form(...),
    dicom_file: UploadFile = File(None),
    prescription_file: UploadFile = File(None)
):
    try:
        # OCR Processing for Prescriptions
        if prescription_file:
            rx_location = f"data/uploads/rx_{prescription_file.filename}"
            with open(rx_location, "wb") as buffer:
                shutil.copyfileobj(prescription_file.file, buffer)
            extracted_text = ocr_agent.extract_text(rx_location)
            
            # Inject OCR text into the semantic RAG database
            history_agent.add_patient_history(patient_id, f"prescription_{prescription_file.filename}", f"Handwritten Prescription Note: {extracted_text}")
            
            # Append OCR text into the direct symptoms context so the LLM generation prioritizes it
            symptoms += f"\n[AI Auto-Read Prescription]: {extracted_text}"

        # 1. Image Processing
        vision_features = None
        detected_anatomy = "chest"
        if dicom_file:
            file_location = f"data/uploads/{dicom_file.filename}"
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(dicom_file.file, buffer)
            
            # Load and Extract Anatomy
            anatomy_result = anatomy_agent.detect_anatomy(file_location)
            detected_anatomy = anatomy_result["anatomy"]
            print(f"-> Detected Anatomy: {detected_anatomy} (Confidence: {anatomy_result['confidence']:.2f})")

            # Dynamic Routing: If it's a piece of paper, route to OCR instead of DenseNet!
            if "prescription" in detected_anatomy:
                print("-> Prescription detected! Bypassing DenseNet Vision Model and Routing to OCR Agent...")
                extracted_text = ocr_agent.extract_text(file_location)
                
                # Inject OCR text into the semantic RAG database
                history_agent.add_patient_history(patient_id, f"prescription_main_{dicom_file.filename}", f"Handwritten Prescription Note: {extracted_text}")
                
                # Append OCR text into the direct symptoms context so the LLM reads it
                symptoms += f"\n[AI Auto-Read Prescription]: {extracted_text}"
                
                # Provide an empty dummy tensor to satisfy the fusion architecture without crashing
                vision_features = torch.zeros((1, 1024))
            else:
                img_array = load_image(file_location)
                tensor_img = preprocess_image(img_array)
                with torch.no_grad():
                    vision_features = vision_agent(tensor_img)
        
        # 2. History RAG
        retrieved_text, text_emb = history_agent.retrieve_context(patient_id, symptoms)
        
        # 3. Fusion
        fused_tensor = None
        if vision_features is not None:
            fused_tensor = history_agent.fuse_features(vision_features, text_emb)
        else:
            return {"error": "An image (DICOM, PNG, JPG, etc.) is required for multimodal diagnosis."}
            
        # 4. Get findings from pre-trained classifier (real validated probabilities)
        branch_key = detected_anatomy.lower()
        if "chest" in branch_key:
            print("-> Using TorchXRayVision ChestClassifier (AUC 0.77-0.90)...")
            findings = chest_classifier.predict(file_location)
        elif "brain" in branch_key:
            print("-> Using HuggingFace BrainClassifier (Accuracy 96.4%)...")
            findings = brain_classifier.predict(file_location)
        else:
            # Knee / Arm / Hand — use ImageNet-pretrained features + MLP branch
            print(f"-> Using ImageNet-pretrained DenseNet + {branch_key} MLP branch...")
            findings = diagnosis_agent.extract_findings_from_tensor(fused_tensor, detected_anatomy)

        # 5. LLM Report Generation
        try:
            report_data = diagnosis_agent.generate_report(retrieved_text, symptoms, fused_tensor, detected_anatomy)
        except NameError:
            report_data = {"generated_report": "MOCK REPORT: LLM generation bypassed due to resource constraints."}
        
        # Save to local JSON
        import json
        from datetime import datetime
        os.makedirs("data", exist_ok=True)
        report_entry = {
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "anatomy_detected": detected_anatomy,
            "symptoms_described": symptoms,
            "ai_visual_probabilities": findings,
            "final_llm_report": report_data["generated_report"]
        }
        with open("data/diagnosis_reports.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(report_entry, default=str) + "\n")
        
        return {
            "status": "success",
            "detected_anatomy": detected_anatomy,
            "findings": findings,
            "report": report_data["generated_report"],
            "retrieved_context": retrieved_text
        }
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"ERROR DETAILS:\n{error_details}")
        raise HTTPException(status_code=500, detail=str(error_details))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
