import os
from src.preprocess_text import clean_medical_text

def test_text_pipeline():
    print("--- Testing Text Preprocessing Pipeline ---")
    sample_history = "Patient isn't feeling well. He presents with mild chest pain @ rest!! Reports couldn't sleep."
    cleaned_text = clean_medical_text(sample_history)
    print(f"Original Text:\n{sample_history}\n")
    print(f"Cleaned and Tokenized Text:\n{cleaned_text}\n")
    print("-------------------------------------------\n")

def test_image_pipeline():
    print("--- Testing Image Preprocessing Pipeline ---")
    try:
        dicom_path = "data/sample_images/CT_small.dcm"
        print(f"Loading real DICOM file: {dicom_path}")
        
        # Load and normalize
        from src.preprocess_image import load_dicom_image, preprocess_image
        img_array = load_dicom_image(dicom_path)
        tensor_img = preprocess_image(img_array)
        
        print(f"Successfully processed real DICOM image!")
        print(f"Normalized Image Tensor Shape: {tensor_img.shape}")
    except Exception as e:
        print(f"Failed to process DICOM: {e}")
    print("-------------------------------------------\n")

from src.vision_agent import VisionAgent
from src.history_agent import HistoryAgent
import torch

def test_vision_pipeline():
    print("--- Testing Vision Agent Pipeline ---")
    try:
        # Initialize model
        print("Loading pre-trained DenseNet121 Model...")
        # Using pretrained=False just for the rapid unit test so it doesn't download weights yet
        model = VisionAgent(pretrained=False) 
        model.eval()
        
        # Create a dummy image tensor (mocking the output of preprocess_image.py)
        dummy_tensor = torch.randn(1, 3, 224, 224)
        print(f"Input Tensor Shape (from preprocess_image.py): {dummy_tensor.shape}")
        
        # Extract features
        with torch.no_grad():
            features = model(dummy_tensor)
            
        print(f"Vision Agent Feature Output Shape: {features.shape}")
        print("Model successfully extracted a 1024-dimensional continuous vector!")
        return features
    except Exception as e:
        print(f"Vision Pipeline Failed: {e}")
        return None
    print("-------------------------------------------\n")

def test_history_pipeline(vision_features_tensor):
    print("--- Testing History Agent (RAG) Pipeline ---")
    try:
        agent = HistoryAgent("./chroma_db_test")
        
        # Add past medical notes for a mock patient
        print("Inserting Patient History into ChromaDB Vector Store...")
        agent.add_patient_history("patient_xyz", "note_1", "Patient diagnosed with asthma in 2015. Prescribed Albuterol.")
        agent.add_patient_history("patient_xyz", "note_2", "Patient broke ankle in 2020. Surgery performed.")
        
        # Query based on current symptom
        query = "Patient currently experiencing wheezing and shortness of breath."
        print(f"\nCurrent Symptom Query: '{query}'")
        
        # RAG Retrieval
        retrieved_text, text_emb = agent.retrieve_context("patient_xyz", query)
        print(f"RAG Retrieved Context: {retrieved_text}")
        print(f"Textual Embedding vector shape: {text_emb.shape}")
        
        # Multi-modal fusion
        if vision_features_tensor is not None:
            fused_representation = agent.fuse_features(vision_features_tensor, text_emb)
            print(f"\nSuccessfully fused Vision Features (1024) + Text Features (384)!")
            print(f"Final Multi-Modal Fusion Tensor Shape: {fused_representation.shape}") # Should be [1, 1408]
            return fused_representation, retrieved_text, query
    
    except Exception as e:
        print(f"History Pipeline Failed: {e}")
        return None, None, None
    print("-------------------------------------------\n")
    return None, None, None

from src.diagnosis_agent import DiagnosisAgent

def test_diagnosis_pipeline(fused_tensor, retrieved_text, current_symptoms):
    print("--- Testing Diagnosis Agent Pipeline ---")
    try:
        if fused_tensor is None:
            print("Cannot test Diagnosis Agent without the fused tensor.")
            return

        agent = DiagnosisAgent()
        report_data = agent.generate_report(retrieved_text, current_symptoms, fused_tensor)
        
        print("\n=== FINAL GENERATED MEDICAL REPORT ===")
        print(report_data["generated_report"])
        print("======================================\n")
    except Exception as e:
        print(f"Diagnosis Pipeline Failed: {e}")
    print("-------------------------------------------\n")

if __name__ == "__main__":
    print("Agentic Autonomous Medical Diagnosis System - Initialized\n")
    test_text_pipeline()
    test_image_pipeline()
    vision_features = test_vision_pipeline()
    fused_tensor, retrieved_text, query = test_history_pipeline(vision_features)
    test_diagnosis_pipeline(fused_tensor, retrieved_text, query)
