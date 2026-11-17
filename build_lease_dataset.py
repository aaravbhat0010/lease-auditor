import json
import random
import time
from openai import OpenAI
from pydantic import BaseModel, Field

# Connects directly to your local Ollama server — $0 cost, no API key needed
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

TEACHER_MODEL = "qwen3:14b"

SYSTEM_PROMPT = (
    "You are a strict residential lease auditor specializing in CA, NJ, and MA tenant law. "
    "Evaluate the lease clause for the specified state and output valid JSON only with keys: "
    "risk_level (ILLEGAL | PREDATORY | STANDARD), trap_category, statute_cited, explanation, negotiation_counter."
)

STATE_PROFILES = {
    "CA": {
        "name": "California",
        "laws": [
            "Security deposits capped at 1 month's rent for most landlords (Civ. Code § 1950.5 / AB 12); 2 months max only for small landlords owning <= 2 properties totaling <= 4 units.",
            "Security deposit must be returned within 21 calendar days with an itemized statement (Civ. Code § 1950.5(g)).",
            "Non-refundable security deposits, non-refundable cleaning fees, or non-refundable pet deposits are strictly illegal (Civ. Code § 1950.5(m)).",
            "Landlord must give at least 24 hours written notice before non-emergency entry, or 48 hours for initial move-out inspection (Civ. Code § 1954).",
            "Late fees must be reasonably related to actual damages caused by late payment; arbitrary punitive flat fees or daily penalties are unenforceable (Civ. Code § 1671).",
            "Implied warranty of habitability cannot be waived by 'as-is' clauses (Civ. Code § 1941.1 / § 1953)."
        ]
    },
    "NJ": {
        "name": "New Jersey",
        "laws": [
            "Security deposit cannot exceed 1.5 times one month's rent (N.J.S.A. 46:8-21.2); annual increases to deposit cannot exceed 10%.",
            "Security deposit plus accrued interest must be returned within 30 days of lease termination, or 5 business days if displaced by fire/flood (N.J.S.A. 46:8-21.1).",
            "Landlord must deposit security funds in a separate interest-bearing bank account and notify tenant of bank name/address within 30 days (N.J.S.A. 46:8-19).",
            "Senior citizens and certain subsidized tenants are entitled to a mandatory 5-business-day grace period before late fees apply (N.J.S.A. 2A:42-6.1).",
            "Landlord cannot enter without tenant consent or reasonable advance notice (typically 1 day for inspections/repairs under N.J.A.C. 5:10-5.1) except in emergencies.",
            "Under the Anti-Eviction Act (N.J.S.A. 2A:18-61.1), residential landlords in most properties cannot evict without statutory 'good cause' even after a lease expires."
        ]
    },
    "MA": {
        "name": "Massachusetts",
        "laws": [
            "At lease signing, landlord can only collect first month's rent, last month's rent, a security deposit up to 1 month's rent, and lock/key cost—no extra move-in, amenity, or pet deposits above 1 month (M.G.L. c. 186, § 15B(1)(b)).",
            "Landlord cannot charge any late fee or interest on unpaid rent until rent is at least 30 days past due (M.G.L. c. 186, § 15B(1)(c)).",
            "Security deposit must be held in a separate, interest-bearing escrow account in a Massachusetts bank with a receipt given within 30 days (M.G.L. c. 186, § 15B(3)(a)).",
            "Security deposit must be returned within 30 days of tenancy end, and deductions cannot be made for ordinary wear and tear (M.G.L. c. 186, § 15B(4)).",
            "Landlord may only enter an apartment for specific statutory reasons (inspect, make repairs, show to prospective tenants/buyers) with reasonable notice (M.G.L. c. 186, § 15B(1)(a)).",
            "Waivers of the implied warranty of habitability or tenant statutory rights under Chapter 186 are void and unenforceable (M.G.L. c. 186, § 15B(8))."
        ]
    }
}

class LeaseClauseExample(BaseModel):
    clause: str = Field(description="Realistic residential lease paragraph, 2-5 sentences long.")
    risk_level: str = Field(description="Must be exactly ILLEGAL, PREDATORY, or STANDARD.")
    trap_category: str = Field(description="Short category like 'Security Deposit Cap', 'Illegal Late Fee', 'Entry Notice', 'Auto-Renewal Trap', or 'Standard Boilerplate'.")
    statute_cited: str = Field(description="Exact state statute citation if ILLEGAL, 'N/A (Legal but unfavorable)' if PREDATORY, or 'N/A' if STANDARD.")
    explanation: str = Field(description="Clear 2-sentence plain-English explanation for a renter.")
    negotiation_counter: str = Field(description="Polite, actionable redline text to send to the landlord, or 'No change needed.' if STANDARD.")

class BatchResponse(BaseModel):
    examples: list[LeaseClauseExample]

def generate_batch(state: str, risk_level: str, batch_idx: int) -> list[dict]:
    profile = STATE_PROFILES[state]
    rent = random.randint(1500, 4800)

    if risk_level == "ILLEGAL":
        directive = (
            f"Create 5 realistic lease clauses set in {profile['name']} ({state}) where monthly rent is ${rent}. "
            f"Each clause MUST directly violate one of these {state} statutes (including math violations like deposits exceeding the state cap): "
            f"{' | '.join(profile['laws'])}. Vary writing styles from formal corporate property-management legalese to sloppy DIY landlord templates."
        )
    elif risk_level == "PREDATORY":
        directive = (
            f"Create 5 realistic lease clauses set in {profile['name']} ({state}) where monthly rent is ${rent} that are technically LEGAL under {state} law "
            f"but are predatory, one-sided traps renters should negotiate (e.g., 90-day auto-renewal lock-ins, tenant paying the first $250 of every appliance repair, "
            f"joint-and-several roommate traps, exorbitant subletting fees, or forced arbitration). Set statute_cited to 'N/A (Legal but unfavorable)'."
        )
    else:
        directive = (
            f"Create 5 realistic, completely fair and legally compliant residential lease clauses set in {profile['name']} ({state}) where monthly rent is ${rent} "
            f"(e.g., compliant deposit caps, proper 24h/reasonable entry notice, standard utility split, compliant late fee rules, standard noise/pet rules). "
            f"Set risk_level to 'STANDARD', statute_cited to 'N/A', and negotiation_counter to 'No change needed.'."
        )

    for attempt in range(3):
        try:
            response = client.beta.chat.completions.parse(
                model=TEACHER_MODEL,
                messages=[
                    {"role": "system", "content": "You generate high-precision legal synthetic training data in strict JSON."},
                    {"role": "user", "content": directive}
                ],
                response_format=BatchResponse,
                temperature=0.8,
            )
            parsed = next(iter(response.choices)).message.parsed
            formatted = []
            for ex in parsed.examples:
                assistant_payload = {
                    "risk_level": risk_level,
                    "trap_category": ex.trap_category,
                    "statute_cited": ex.statute_cited,
                    "explanation": ex.explanation,
                    "negotiation_counter": ex.negotiation_counter,
                }
                formatted.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"State: {state}\nClause: {ex.clause.strip()}"},
                        {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)}
                    ]
                })
            return formatted
        except Exception as e:
            print(f"[Retry {attempt+1}] {state}-{risk_level} batch {batch_idx} failed: {e}", flush=True)
            time.sleep(1)
    return []

def main():
    # 34 batches x 5 examples = 170 per bucket x 9 buckets = 1,530 total examples
    tasks = []
    for state in ["CA", "NJ", "MA"]:
        for risk in ["ILLEGAL", "PREDATORY", "STANDARD"]:
            for b in range(34):
                tasks.append((state, risk, b))

    all_records = []
    print(f"Generating ~{len(tasks) * 5} synthetic lease clauses locally via Ollama ({TEACHER_MODEL})...", flush=True)

    for idx, (s, r, b) in enumerate(tasks, 1):
        batch_records = generate_batch(s, r, b)
        all_records.extend(batch_records)
        print(f"Progress: {idx}/{len(tasks)} batches ({len(all_records)} examples saved)", flush=True)

    random.shuffle(all_records)
    split_idx = int(len(all_records) * 0.9)
    train_data = all_records[:split_idx]
    val_data = all_records[split_idx:]

    with open("train.jsonl", "w", encoding="utf-8") as f:
        for row in train_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open("val.jsonl", "w", encoding="utf-8") as f:
        for row in val_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nDone! Saved {len(train_data)} rows to train.jsonl and {len(val_data)} rows to val.jsonl.", flush=True)

if __name__ == "__main__":
    main()
