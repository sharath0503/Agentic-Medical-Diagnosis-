from src.history_agent import HistoryAgent

def demonstrate_rag_history():
    print("Initializing Vector Database (ChromaDB)...")
    # This connects to the exact same database the web app uses
    agent = HistoryAgent("./chroma_db_test")
    
    patient_id = "patient_xyz"
    
    # --- STEP 1: Add Historical Data ---
    print("\n--- STEP 1: Saving Past Medical History ---")
    past_note = "Patient was diagnosed with Type 2 Diabetes 5 years ago. Currently taking Metformin."
    agent.add_patient_history(patient_id, note_id="clinic_visit_2021", raw_note=past_note)
    
    # --- STEP 2: Future Diagnosis Retrieval ---
    print("\n--- STEP 2: Semantic Retrieval During a New Visit ---")
    # Imagine 3 years later, the patient comes in with a new symptom:
    current_symptoms = "Patient presents with tingling in the feet and blurry vision."
    print(f"Current Symptoms Input: '{current_symptoms}'")
    
    # The system automatically searches the database for relevant history!
    retrieved_text, text_embedding = agent.retrieve_context(patient_id, current_query=current_symptoms)
    
    print(f"\n-> AI Retrieved Context: '{retrieved_text}'")
    print(f"-> This text is now injected into the LLM Prompt so the AI knows about the Diabetes!")
    print(f"-> Embedding Shape for Neural Network Fusion: {text_embedding.shape}")

if __name__ == "__main__":
    demonstrate_rag_history()
