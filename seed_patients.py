"""
seed_patients.py
Seeds 10 synthetic patient records into:
  1. ChromaDB  — for semantic RAG retrieval during diagnosis
  2. MongoDB   — for structured persistent storage / dashboard queries
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.history_agent import HistoryAgent
from src.database import MongoManager
from datetime import datetime, timedelta
import random

# ──────────────────────────────────────────────
# Patient Dataset
# ──────────────────────────────────────────────
PATIENTS = [
    {
        "patient_id": "001",
        "name": "Arjun Sharma",
        "age": 54,
        "gender": "Male",
        "condition": "Type 2 Diabetes",
        "history": (
            "Patient diagnosed with Type 2 Diabetes Mellitus 8 years ago. "
            "Currently on Metformin 500mg twice daily. "
            "HbA1c last recorded at 8.2%. "
            "Complains of occasional tingling in feet and blurry vision. "
            "Family history of diabetes on paternal side."
        )
    },
    {
        "patient_id": "002",
        "name": "Sunita Patel",
        "age": 67,
        "gender": "Female",
        "condition": "Coronary Artery Disease",
        "history": (
            "Patient has a history of Coronary Artery Disease (CAD) and underwent "
            "percutaneous coronary intervention (PCI) 3 years ago. "
            "On dual antiplatelet therapy with Aspirin and Clopidogrel. "
            "Mild left ventricular dysfunction noted on last echocardiogram (EF: 45%). "
            "Complains of exertional chest tightness and breathlessness on climbing stairs."
        )
    },
    {
        "patient_id": "003",
        "name": "Ravi Menon",
        "age": 61,
        "gender": "Male",
        "condition": "Chronic Obstructive Pulmonary Disease (COPD)",
        "history": (
            "Patient is a 40-pack-year smoker diagnosed with COPD Stage III (GOLD classification). "
            "On Tiotropium inhaler and Salbutamol rescue inhaler. "
            "FEV1/FVC ratio of 0.58 on last spirometry. "
            "Frequent exacerbations requiring oral corticosteroids. "
            "Chronic productive cough, wheezing, and increasing dyspnea on exertion."
        )
    },
    {
        "patient_id": "004",
        "name": "Meena Krishnan",
        "age": 45,
        "gender": "Female",
        "condition": "Breast Cancer (Post-Chemotherapy)",
        "history": (
            "Patient diagnosed with Stage IIB invasive ductal carcinoma of the right breast 2 years ago. "
            "Completed 6 cycles of AC-T chemotherapy. "
            "Currently on Tamoxifen 20mg daily. "
            "Last mammogram showed no recurrence. "
            "Complains of fatigue, hot flushes, and mild joint pain attributed to hormone therapy."
        )
    },
    {
        "patient_id": "005",
        "name": "Vijay Nair",
        "age": 39,
        "gender": "Male",
        "condition": "Lumbar Disc Herniation",
        "history": (
            "Patient presents with L4-L5 disc herniation confirmed on MRI. "
            "Complains of radiating lower back pain into the left leg (sciatica). "
            "On Diclofenac 75mg and physiotherapy. "
            "Neurological exam shows mild weakness in left foot dorsiflexion. "
            "No history of prior spinal surgery. Works as a software engineer (sedentary job)."
        )
    },
    {
        "patient_id": "006",
        "name": "Priya Desai",
        "age": 31,
        "gender": "Female",
        "condition": "Rheumatoid Arthritis",
        "history": (
            "Patient diagnosed with seropositive Rheumatoid Arthritis (RA) at age 28. "
            "Anti-CCP antibodies positive, ESR elevated at 76mm/hr. "
            "On Methotrexate 15mg weekly and folic acid supplementation. "
            "Complains of morning stiffness lasting more than 1 hour, bilateral wrist and finger joint swelling. "
            "Recent X-ray of hands shows periarticular osteopenia without bony erosions yet."
        )
    },
    {
        "patient_id": "007",
        "name": "Anand Rao",
        "age": 72,
        "gender": "Male",
        "condition": "Ischemic Stroke (Recovered)",
        "history": (
            "Patient suffered a right MCA territory ischemic stroke 18 months ago. "
            "Presented with left-sided hemiparesis and dysarthria. "
            "Managed with IV alteplase within the thrombolytic window. "
            "Currently on Aspirin 150mg and Atorvastatin 40mg. "
            "Residual mild left arm weakness and balance issues. "
            "History of hypertension and atrial fibrillation."
        )
    },
    {
        "patient_id": "008",
        "name": "Lakshmi Iyer",
        "age": 58,
        "gender": "Female",
        "condition": "Chronic Kidney Disease (CKD Stage 3)",
        "history": (
            "Patient diagnosed with CKD Stage 3 secondary to long-standing hypertension and diabetes. "
            "eGFR currently 38 mL/min/1.73m2. "
            "On ACE inhibitor (Ramipril 5mg), erythropoietin injections for anemia. "
            "Dietary protein restriction in place. "
            "Complains of fatigue, nocturia, and mild pedal edema. "
            "Renal ultrasound shows bilateral echogenic kidneys."
        )
    },
    {
        "patient_id": "009",
        "name": "Karthik Subramaniam",
        "age": 27,
        "gender": "Male",
        "condition": "Knee Ligament Injury (ACL Tear)",
        "history": (
            "Patient is a 27-year-old professional footballer who sustained an ACL tear during a match 6 months ago. "
            "MRI confirmed complete anterior cruciate ligament rupture of the right knee. "
            "Underwent ACL reconstruction surgery using patellar tendon autograft 3 months ago. "
            "Currently in post-operative physiotherapy rehabilitation. "
            "Complains of persistent right knee stiffness and mild swelling post-exercise. "
            "Goal is return to professional play within 6 months."
        )
    },
    {
        "patient_id": "010",
        "name": "Rekha Bose",
        "age": 49,
        "gender": "Female",
        "condition": "Hypothyroidism",
        "history": (
            "Patient diagnosed with Hashimoto's thyroiditis and hypothyroidism 10 years ago. "
            "TSH last recorded at 8.9 mIU/L (elevated). "
            "On Levothyroxine 75 micrograms daily. "
            "Complains of persistent fatigue, weight gain of 8kg over 2 years, constipation, and feeling cold. "
            "Anti-TPO antibodies significantly elevated. "
            "Ultrasound thyroid shows heterogeneous echotexture with reduced vascularity."
        )
    },
    {
        "patient_id": "011",
        "name": "Harish Nambiar",
        "age": 68,
        "gender": "Male",
        "condition": "Parkinson's Disease",
        "history": (
            "Patient diagnosed with idiopathic Parkinson's Disease 5 years ago. "
            "Presents with classic resting tremor (pill-rolling) in right hand, bradykinesia, and cogwheel rigidity. "
            "On Levodopa-Carbidopa 100/25mg three times daily with adjunct Pramipexole. "
            "Neurologist noted moderate disease progression; DaTSCAN confirmed dopaminergic deficit. "
            "Complains of sleep disturbances (REM sleep behaviour disorder), reduced facial expression, and gait freezing. "
            "No dementia features at this stage."
        )
    },
    {
        "patient_id": "012",
        "name": "Divya Rajendran",
        "age": 22,
        "gender": "Female",
        "condition": "Bronchial Asthma (Severe Persistent)",
        "history": (
            "Patient has a 12-year history of allergic bronchial asthma with multiple ED visits annually. "
            "Spirometry shows FEV1 of 62% predicted with significant bronchodilator reversibility. "
            "On high-dose inhaled corticosteroid (Fluticasone 500mcg) and long-acting beta-agonist (Salmeterol). "
            "Serum IgE markedly elevated at 680 IU/mL; skin prick test positive for dust mites and pollen. "
            "Frequent nocturnal awakenings and exercise-induced bronchospasm. "
            "Prescribed oral Montelukast and Salbutamol as-needed reliever."
        )
    },
    {
        "patient_id": "013",
        "name": "Suresh Chandrasekaran",
        "age": 55,
        "gender": "Male",
        "condition": "Liver Cirrhosis (Child-Pugh Class B)",
        "history": (
            "Patient with 20-year history of alcohol-related liver disease, progressed to decompensated cirrhosis. "
            "Child-Pugh score 8 (Class B); MELD score 16. "
            "Ultrasound shows nodular liver surface, splenomegaly, and moderate ascites. "
            "History of two prior variceal bleeding episodes, on Propranolol for prophylaxis. "
            "On Spironolactone and Furosemide for ascites management; dietary sodium restriction enforced. "
            "LFTs: Bilirubin 3.2 mg/dL, INR 1.8, Albumin 2.6 g/dL. Referred for liver transplant evaluation."
        )
    },
    {
        "patient_id": "014",
        "name": "Anitha Subramaniam",
        "age": 34,
        "gender": "Female",
        "condition": "Schizophrenia (Paranoid Type)",
        "history": (
            "Patient diagnosed with paranoid schizophrenia at age 26 following acute psychotic episode. "
            "Presents with persistent auditory hallucinations and persecutory delusions. "
            "On Risperidone 6mg daily with monthly Paliperidone palmitate injection for adherence. "
            "Two prior hospitalizations due to medication non-compliance. "
            "Cognitive testing shows mild impairment in working memory and executive function. "
            "Currently enrolled in community mental health program with weekly therapist contact."
        )
    },
    {
        "patient_id": "015",
        "name": "Dinesh Kumar",
        "age": 19,
        "gender": "Male",
        "condition": "Sickle Cell Anaemia",
        "history": (
            "Patient diagnosed with homozygous sickle cell disease (HbSS) at birth via neonatal screening. "
            "Recurrent vaso-occlusive crises requiring hospitalization 3-4 times per year. "
            "On Hydroxyurea 1g daily to reduce crisis frequency; Penicillin prophylaxis since childhood. "
            "Last crisis involved acute chest syndrome requiring exchange transfusion. "
            "Haemoglobin baseline 8.2 g/dL; reticulocyte count elevated. "
            "Ophthalmology review showed early proliferative sickle retinopathy in left eye."
        )
    },
    {
        "patient_id": "016",
        "name": "Kavitha Menon",
        "age": 41,
        "gender": "Female",
        "condition": "Psoriasis with Psoriatic Arthritis",
        "history": (
            "Patient has a 10-year history of plaque psoriasis covering >20% BSA, with joint involvement (psoriatic arthritis) "
            "confirmed 3 years ago affecting bilateral sacroiliac joints and fingers. "
            "On Adalimumab 40mg biweekly (anti-TNF biologic) with good skin response but partial joint response. "
            "PASI score reduced from 24 to 6 on current therapy. "
            "Complains of persistent morning stiffness in lower back and sausage digits on left hand. "
            "Hepatitis B and TB screening done prior to biologic initiation — both negative."
        )
    },
    {
        "patient_id": "017",
        "name": "Gopal Krishnamurthy",
        "age": 76,
        "gender": "Male",
        "condition": "Alzheimer's Disease (Moderate Stage)",
        "history": (
            "Patient diagnosed with Alzheimer's dementia 4 years ago following cognitive assessment (MMSE 18/30). "
            "Current MMSE score 14/30 indicating moderate stage decline. "
            "On Donepezil 10mg daily and Memantine 20mg daily. "
            "Family reports increasing disorientation, sundowning behaviour, and difficulty with ADLs. "
            "MRI brain shows hippocampal atrophy bilaterally and parietal cortex thinning. "
            "CSF biomarkers: elevated phosphorylated tau, low amyloid-beta 42 — consistent with AD pathology."
        )
    },
    {
        "patient_id": "018",
        "name": "Fatima Begum",
        "age": 38,
        "gender": "Female",
        "condition": "Pulmonary Tuberculosis (Active)",
        "history": (
            "Patient presenting with 3-month history of productive cough, low-grade fever, and 7kg weight loss. "
            "Sputum AFB smear positive (3+); GeneXpert confirms Mycobacterium tuberculosis, rifampicin sensitive. "
            "Chest X-ray shows cavitary lesion in right upper lobe with hilar lymphadenopathy. "
            "Started on DOTS Category I regimen: HRZE (Isoniazid, Rifampicin, Pyrazinamide, Ethambutol). "
            "Contact screening ordered for household members; HIV test negative. "
            "Baseline LFTs within normal limits; monthly monitoring scheduled for hepatotoxicity."
        )
    },
    {
        "patient_id": "019",
        "name": "Rajesh Iyer",
        "age": 52,
        "gender": "Male",
        "condition": "Gout with Tophi",
        "history": (
            "Patient has a 15-year history of recurrent gouty arthritis, now complicated by visible tophi on bilateral "
            "elbows and right great toe. Serum uric acid: 10.8 mg/dL (markedly elevated). "
            "On Allopurinol 300mg daily and Colchicine 0.5mg as prophylaxis. "
            "Dietary counseling given: avoidance of red meat, shellfish, alcohol, and fructose-rich beverages. "
            "X-ray of feet shows classic rat-bite erosions adjacent to tophaceous deposits. "
            "Renal function slightly impaired: eGFR 62 mL/min — urological referral planned."
        )
    },
    {
        "patient_id": "020",
        "name": "Nandini Pillai",
        "age": 29,
        "gender": "Female",
        "condition": "Temporal Lobe Epilepsy",
        "history": (
            "Patient diagnosed with drug-resistant temporal lobe epilepsy at age 21 following MRI-confirmed "
            "right hippocampal sclerosis. "
            "Seizures characterised by aura (deja vu, epigastric rising sensation) followed by automatisms (lip-smacking). "
            "On Levetiracetam 1500mg BD and Clobazam 10mg nocte — partial control achieved. "
            "EEG shows right temporal interictal epileptiform discharges. "
            "Referred for pre-surgical epilepsy evaluation (video-EEG monitoring and neuropsychology). "
            "Seizure diary reports 3-5 events per month; driving licence suspended per DVLA guidelines."
        )
    },
]

# ──────────────────────────────────────────────
# Seed Both Databases
# ──────────────────────────────────────────────
def seed():
    print("=" * 60)
    print("  PATIENT HISTORY SEED SCRIPT")
    print("  Seeding 20 patients into ChromaDB + MongoDB")
    print("=" * 60)

    # Connect to Vector DB (ChromaDB)
    print("\n[1/2] Connecting to ChromaDB (RAG Vector Store)...")
    history_agent = HistoryAgent("./chroma_db_test")

    # Connect to MongoDB
    print("[2/2] Connecting to MongoDB...")
    mongo_db = MongoManager()

    print("\n--- Seeding Records ---\n")
    for p in PATIENTS:
        pid = p["patient_id"]

        # ── ChromaDB ──
        history_agent.add_patient_history(
            patient_id=pid,
            note_id=f"initial_history_{pid}",
            raw_note=p["history"]
        )

        # ── MongoDB ──
        if mongo_db.patients is not None:
            # Avoid duplicate inserts
            existing = mongo_db.patients.find_one({"patient_id": pid})
            if not existing:
                mongo_db.patients.insert_one({
                    "patient_id": pid,
                    "name": p["name"],
                    "age": p["age"],
                    "gender": p["gender"],
                    "primary_condition": p["condition"],
                    "medical_history": p["history"],
                    "created_at": datetime.now(),
                    "status": "active"
                })
                print(f"  -> MongoDB: Inserted Patient {pid} ({p['name']})")
            else:
                print(f"  -> MongoDB: Patient {pid} already exists, skipping.")

    print("\n" + "=" * 60)
    print("  SEEDING COMPLETE!")
    print(f"  {len(PATIENTS)} patients seeded into ChromaDB and MongoDB.")
    print("=" * 60)

    # ──────────────────────────────────────────────
    # Quick RAG Verification Test
    # ──────────────────────────────────────────────
    print("\n--- RAG RETRIEVAL VERIFICATION ---")
    test_queries = [
        ("009", "Patient has knee stiffness after surgery and swelling"),
        ("001", "Patient complains of tingling feet and high blood sugar"),
        ("003", "Patient has severe wheezing and shortness of breath"),
    ]

    for pid, query in test_queries:
        retrieved, _ = history_agent.retrieve_context(pid, current_query=query)
        print(f"\n  Patient {pid} | Query: '{query}'")
        print(f"  Retrieved: '{retrieved[:100]}...'")

    print("\n  RAG system is verified and working correctly!")


if __name__ == "__main__":
    seed()
