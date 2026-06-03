import sys

with open('C:/Users/shara/DS/Agentic_Medical_Diagnosis/app.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith('from src.ocr_agent import OCRAgent'):
        new_lines.append(line)
        new_lines.append('from src.anatomy_agent import AnatomyAgent\n')
    elif line.startswith('    print("Initializing History Agent...")'):
        new_lines.append('    print("Initializing Anatomy Agent...")\n')
        new_lines.append('    anatomy_agent = AnatomyAgent()\n\n')
        new_lines.append(line)
    elif 'vision_features = None' in line and '1. Image Processing' in new_lines[-1]:
        new_lines.append('        vision_features = None\n')
        new_lines.append('        detected_anatomy = "chest"\n')
    elif 'img_array = load_image(file_location)' in line:
        new_lines.append('            anatomy_result = anatomy_agent.detect_anatomy(file_location)\n')
        new_lines.append('            detected_anatomy = anatomy_result["anatomy"]\n')
        new_lines.append('            print(f"-> Detected Anatomy: {detected_anatomy} (Confidence: {anatomy_result[\\"confidence\\"]:.2f})")\n\n')
        new_lines.append(line)
    elif 'findings = diagnosis_agent.extract_findings_from_tensor(fused_tensor)' in line:
        new_lines.append('            findings = diagnosis_agent.extract_findings_from_tensor(fused_tensor, detected_anatomy)\n')
    elif '"status": "success",' in line:
        new_lines.append(line)
        new_lines.append('            "detected_anatomy": detected_anatomy,\n')
    else:
        new_lines.append(line)

with open('C:/Users/shara/DS/Agentic_Medical_Diagnosis/app.py', 'w') as f:
    f.writelines(new_lines)

print('app.py updated successfully!')
