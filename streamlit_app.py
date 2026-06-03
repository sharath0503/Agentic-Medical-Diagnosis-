"""
streamlit_app.py
Autonomous Multi-Modal Medical Diagnosis System — Streamlit Frontend
Run with: .\\venv\\Scripts\\streamlit.exe run streamlit_app.py
"""

import streamlit as st
import torch
import os
import sys

mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

# Force UTF-8 encoding for stdout/stderr on Windows to prevent
# 'charmap' codec errors when any loaded module prints Unicode characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────────────────
# Page Config — must be FIRST streamlit call
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Medical AI",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #0d1117 100%); }
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-title { text-align: center; color: #94a3b8; font-size: 1rem; margin-bottom: 2rem; }
    .anatomy-badge {
        display: inline-block;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white; font-weight: 700; font-size: 1rem;
        padding: 0.4rem 1.2rem; border-radius: 999px; margin-bottom: 1rem;
    }
    .report-box {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(96, 165, 250, 0.3);
        border-radius: 12px; padding: 1.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem; line-height: 1.7; color: #e2e8f0; white-space: pre-wrap;
    }
    .section-header {
        font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 2px; color: #60a5fa; margin-bottom: 0.5rem;
    }
    .prob-high { color: #f87171; font-weight: 700; }
    .prob-med  { color: #fbbf24; font-weight: 700; }
    .prob-low  { color: #34d399; font-weight: 700; }
    #MainMenu  { visibility: hidden; }
    footer     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Agent Loader — cached, no Streamlit calls inside
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_agents():
    from src.vision_agent import VisionAgent
    from src.history_agent import HistoryAgent
    from src.diagnosis_agent import DiagnosisAgent
    from src.ocr_agent import OCRAgent
    from src.anatomy_agent import AnatomyAgent
    from src.preprocess_image import load_image, preprocess_image

    from src.pretrained_classifiers import ChestClassifier, BrainClassifier

    vision    = VisionAgent(pretrained=True)
    vision.eval()
    anatomy   = AnatomyAgent()
    history   = HistoryAgent("./chroma_db_test")
    diagnosis = DiagnosisAgent(model_id="Qwen/Qwen2.5-0.5B-Instruct")
    ocr       = OCRAgent()
    chest_cls = ChestClassifier()
    brain_cls = BrainClassifier()

    return {
        "vision": vision, "anatomy": anatomy, "history": history,
        "diagnosis": diagnosis, "ocr": ocr,
        "chest_cls": chest_cls, "brain_cls": brain_cls,
        "load_image": load_image, "preprocess_image": preprocess_image
    }


# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Patient Input")
    st.markdown("---")
    patient_id = st.text_input("Patient ID", value="001",
                               help="Use IDs 001–010 for seeded patients.")
    st.markdown("**Seeded Patients:**")
    seeded = {
        "001": "Arjun — Diabetes",           "002": "Sunita — Heart Disease",
        "003": "Ravi — COPD",                "004": "Meena — Cancer",
        "005": "Vijay — Disc Herniation",    "006": "Priya — Arthritis",
        "007": "Anand — Stroke",             "008": "Lakshmi — CKD",
        "009": "Karthik — ACL Tear",         "010": "Rekha — Hypothyroidism",
        "011": "Harish — Parkinson's",       "012": "Divya — Asthma",
        "013": "Suresh — Liver Cirrhosis",   "014": "Anitha — Schizophrenia",
        "015": "Dinesh — Sickle Cell",       "016": "Kavitha — Psoriasis + PsA",
        "017": "Gopal — Alzheimer's",        "018": "Fatima — Tuberculosis",
        "019": "Rajesh — Gout",              "020": "Nandini — Epilepsy",
    }
    for pid, label in seeded.items():
        st.caption(f"`{pid}` — {label}")
    st.markdown("---")
    st.markdown("**System Status**")
    st.success("Streamlit Running")
    st.info("Local JSON Storage: ./data/mongo_fallback_reports.json")
    st.info("ChromaDB: ./chroma_db_test")


# ─────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Autonomous Medical Diagnosis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multi-Agent AI Pipeline · Multi-Modal Fusion · RAG-Augmented Clinical Reporting</div>', unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────────────────
# Main Layout — always rendered, agents loaded on demand
# ─────────────────────────────────────────────────────────
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### Diagnosis Input")

    patient_description = st.text_area(
        "Patient Description",
        placeholder="e.g. Patient complains of knee pain and difficulty walking for the past 3 months...",
        height=130
    )
    scan_file = st.file_uploader(
        "Upload Medical Scan (.dcm, .jpg, .png)",
        type=["dcm", "jpg", "jpeg", "png"]
    )
    prescription_file = st.file_uploader(
        "Upload Handwritten Prescription (Optional)",
        type=["jpg", "jpeg", "png"]
    )

    run_btn = st.button(
        "Run Agentic Analysis",
        use_container_width=True,
        disabled=(not scan_file or not patient_description)
    )


with col_output:
    st.markdown("### AI Generated Report")

    if not run_btn:
        st.info("Fill in the Patient Description, upload a scan, and click **Run Agentic Analysis** to begin.")

    if run_btn and scan_file and patient_description:
        os.makedirs("data/uploads", exist_ok=True)
        symptoms = patient_description

        # Load agents (cached after first call)
        with st.spinner("Loading AI Agents (first run only)..."):
            try:
                agents = load_agents()
            except Exception as e:
                st.error(f"Failed to load agents: {e}")
                st.stop()

        with st.status("Running Agentic Pipeline...", expanded=True) as status:

            # Step 0: Optional Prescription OCR
            if prescription_file:
                st.write("Reading handwritten prescription (OCR)...")
                rx_path = f"data/uploads/rx_{prescription_file.name}"
                with open(rx_path, "wb") as f:
                    f.write(prescription_file.getbuffer())
                extracted = agents["ocr"].extract_text(rx_path)
                agents["history"].add_patient_history(
                    patient_id, f"prescription_{prescription_file.name}",
                    f"Handwritten Prescription: {extracted}"
                )
                symptoms += f"\n[AI Auto-Read Prescription]: {extracted}"
                st.write(f"  OCR: {extracted[:80]}...")

            # Step 1: Anatomy Detection
            st.write("Detecting anatomical region (CLIP)...")
            scan_path = f"data/uploads/{scan_file.name}"
            with open(scan_path, "wb") as f:
                f.write(scan_file.getbuffer())

            anatomy_result  = agents["anatomy"].detect_anatomy(scan_path)
            detected_anatomy = anatomy_result["anatomy"]
            confidence       = anatomy_result["confidence"]
            st.write(f"  Detected: **{detected_anatomy}** ({confidence*100:.1f}% confidence)")

            # Step 1b: Route
            if "prescription" in detected_anatomy:
                st.write("Prescription detected — routing to OCR Agent...")
                extracted = agents["ocr"].extract_text(scan_path)
                agents["history"].add_patient_history(
                    patient_id, f"prescription_main_{scan_file.name}",
                    f"Handwritten Prescription: {extracted}"
                )
                symptoms += f"\n[AI Auto-Read Prescription]: {extracted}"
                vision_features = torch.zeros((1, 1024))
                st.write("  OCR text injected into context.")
            else:
                st.write("Extracting visual features (DenseNet121)...")
                img_array  = agents["load_image"](scan_path)
                tensor_img = agents["preprocess_image"](img_array)
                with torch.no_grad():
                    vision_features = agents["vision"](tensor_img)
                st.write(f"  Feature vector: {tuple(vision_features.shape)}")

            # Step 2: RAG Retrieval
            st.write("Retrieving patient history (ChromaDB RAG)...")
            retrieved_text, text_emb = agents["history"].retrieve_context(patient_id, symptoms)
            st.write(f"  Retrieved: {retrieved_text[:80]}..." if retrieved_text else "  No prior history found.")

            # Step 3: Fusion
            st.write("Fusing image + text embeddings...")
            fused_tensor = agents["history"].fuse_features(vision_features, text_emb)
            st.write(f"  Fused tensor: {tuple(fused_tensor.shape)}")

            # Step 4: Get findings from pre-trained classifier
            branch_key = detected_anatomy.lower()
            if "chest" in branch_key:
                st.write("Using TorchXRayVision ChestClassifier (AUC 0.77-0.90)...")
                findings = agents["chest_cls"].predict(scan_path)
            elif "brain" in branch_key:
                st.write("Using HuggingFace BrainClassifier (Accuracy 96.4%)...")
                findings = agents["brain_cls"].predict(scan_path)
            else:
                st.write(f"Using ImageNet DenseNet + {branch_key.title()} MLP branch...")
                findings = agents["diagnosis"].extract_findings_from_tensor(fused_tensor, detected_anatomy)

            # Step 5: LLM Report
            st.write("Generating clinical report (Qwen2.5)...")
            report_data = agents["diagnosis"].generate_report(
                retrieved_text, symptoms, fused_tensor, detected_anatomy
            )

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
            st.write("  Saved to local JSON file.")

            status.update(label="Analysis Complete", state="complete", expanded=False)

        # ── Results ──
        st.markdown(f'<div class="anatomy-badge">{detected_anatomy.title()} — {confidence*100:.0f}% Confidence</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Visual Neural Findings</div>', unsafe_allow_html=True)
        for condition, prob in findings.items():
            pct   = prob * 100
            color = "prob-high" if pct > 60 else ("prob-med" if pct > 35 else "prob-low")
            st.markdown(f'<span style="color:#cbd5e1">{condition}</span> &nbsp; <span class="{color}">{pct:.1f}%</span>', unsafe_allow_html=True)
            st.progress(float(prob))

        st.markdown("---")
        st.markdown('<div class="section-header">Clinical Report</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="report-box">{report_data["generated_report"]}</div>', unsafe_allow_html=True)

        with st.expander("Retrieved Patient History (RAG Context)"):
            st.write(retrieved_text if retrieved_text else "No history found.")


# ─────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small style='color:#475569'>Autonomous Multi-Modal Medical Diagnosis System &nbsp;·&nbsp; "
    "CLIP · DenseNet121 · Qwen2.5-0.5B · EasyOCR · ChromaDB · Local JSON</small></center>",
    unsafe_allow_html=True
)
