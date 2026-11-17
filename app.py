import json
import re
import fitz  # PyMuPDF
import streamlit as st
from mlx_lm import generate, load

st.set_page_config(page_title="Local Lease Trap Auditor", page_icon="🔍", layout="wide")
st.title("🔍 Private Local Lease & Rental Trap Finder")
st.caption("100% Offline on Your Mac • Powered by Fine-Tuned Qwen3.5-9B (NJ, CA, MA)")

SYSTEM_PROMPT = (
    "You are a strict residential lease auditor specializing in CA, NJ, and MA tenant law. "
    "Evaluate the lease clause for the specified state and output valid JSON only with keys: "
    "risk_level (ILLEGAL | PREDATORY | STANDARD), trap_category, statute_cited, explanation, negotiation_counter."
)


@st.cache_resource
def load_auditor():
    return load("./fused-lease-auditor")


model, tokenizer = load_auditor()

state = st.selectbox("Select Property State", ["NJ", "CA", "MA"])
show_debug = st.checkbox("Show raw model outputs (debug)", value=False)
uploaded_pdf = st.file_uploader("Upload Lease Agreement (PDF)", type=["pdf"])


def extract_clauses(pdf_bytes: bytes) -> list[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = "\n".join(page.get_text("text") for page in doc)
    # Split on numbered clauses (e.g., "1. ", "2. ") OR blank lines
    parts = re.split(r"(?=\n?\s*\d+\.\s+[A-Z])|\n\s*\n", full_text)
    return [p.strip() for p in parts if len(p.strip()) > 60 and not p.strip().startswith("RESIDENTIAL LEASE")]


if uploaded_pdf and st.button("Audit Lease Offline", type="primary"):
    clauses = extract_clauses(uploaded_pdf.read())
    st.info(f"Scanning {len(clauses)} lease clauses locally on your M3 Pro...")
    progress = st.progress(0)
    flagged = []

    for idx, clause in enumerate(clauses, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"State: {state}\nClause: {clause}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        raw_out = generate(model, tokenizer, prompt=prompt, max_tokens=600, verbose=False)

        if show_debug:
            with st.expander(f"Debug Clause #{idx}"):
                st.text(clause)
                st.code(raw_out)

        # Strip any <think> block and extract JSON object
        cleaned = re.sub(r"<think>.*?</think>", "", raw_out, flags=re.DOTALL).strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                if result.get("risk_level") in ("ILLEGAL", "PREDATORY"):
                    result["original_clause"] = clause
                    flagged.append(result)
            except json.JSONDecodeError:
                pass
        progress.progress(idx / len(clauses))

    st.success(f"Audit Complete! Found {len(flagged)} potential traps out of {len(clauses)} clauses.")
    for item in flagged:
        badge = "🚨 ILLEGAL CLAUSE" if item["risk_level"] == "ILLEGAL" else "⚠️ PREDATORY / NEGOTIABLE"
        with st.expander(f"{badge} — {item.get('trap_category', 'Lease Trap')}", expanded=True):
            st.markdown(f"**Statute:** `{item.get('statute_cited', 'N/A')}`")
            st.write(f"**Why It's Risky:** {item.get('explanation', '')}")
            st.code(item["original_clause"], language="text")
            st.markdown("**Suggested Email / Counter-Proposal to Landlord:**")
            st.info(item.get("negotiation_counter", ""))
