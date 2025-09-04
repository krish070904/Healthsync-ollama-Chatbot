# app/safety.py
import re
from typing import Optional

EMERGENCY_TRIGGERS = [
    r"crushing chest pain",
    r"severe shortness of breath",
    r"\bsuicid(e|al)\b",
    r"uncontrolled bleeding",
    r"stroke symptoms",
    r"anaphylax(is|tic)",
    r"child under 3 months with fever",
]

EMERGENCY_MESSAGE = (
    "This might be an emergency. Please seek immediate in-person medical care or call your local emergency number. "
    "I cannot provide crisis or emergency medical advice here."
)

DISCLAIMER = (
    "I'm an educational assistant, not a substitute for professional diagnosis or treatment. "
    "Always consult a licensed clinician for decisions about medical care."
)

# front gate: check user text for emergencies
def check_emergency(text: str) -> Optional[str]:
    t = text.lower()
    for pattern in EMERGENCY_TRIGGERS:
        if re.search(pattern, t):
            return EMERGENCY_MESSAGE
    return None

# back gate: check model output for forbidden actionable instructions
FORBIDDEN_OUTPUT_PATTERNS = [
    r"\bprescribe\b", r"\bdiagnos(e|is|ing)\b", r"\bstart taking\b", r"\brecommended dose\b",
    r"\badminister\b", r"\bimmediately take\b", r"\bcall 911\b", r"\bcall your doctor\b"
]
def check_model_output(out_text: str) -> Optional[str]:
    t = out_text.lower()
    for p in FORBIDDEN_OUTPUT_PATTERNS:
        if re.search(p, t):
            return "I can't provide treatment or dosing instructions. Please consult a licensed clinician. " + DISCLAIMER
    return None
