#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Ranking System
Team: akulasanith18

Approach: Multi-signal hybrid scorer
- Skill matching (keyword + proficiency-weighted)
- Career quality signals (product co. experience, no pure services, NLP/IR background)
- Behavioral availability signals (recency, response rate, notice period)
- Honeypot detection (impossible profiles)
- Location & relocation weighting
"""

import json
import csv
import re
import sys
import argparse
from datetime import datetime, date
from pathlib import Path

# ─── JD-derived constants ───────────────────────────────────────────────────

REFERENCE_DATE = date(2026, 6, 4)  # Dataset freeze date

# Core required skills from JD (must-haves)
CORE_SKILLS = {
    # Embeddings & retrieval
    "sentence-transformers", "sentence transformers", "embeddings", "vector embeddings",
    "openai embeddings", "bge", "e5", "dense retrieval", "semantic search",
    # Vector DBs / hybrid search
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "vector database", "vector search", "hybrid search",
    # Ranking & retrieval systems
    "ranking", "information retrieval", "recommendation system", "retrieval",
    "learning to rank", "bm25", "reranking", "re-ranking",
    # LLMs
    "llm", "large language model", "llms", "rag", "retrieval augmented generation",
    "fine-tuning", "fine tuning", "lora", "qlora", "peft",
    # Evaluation
    "ndcg", "mrr", "map", "a/b testing", "ab testing", "evaluation framework",
    # Python
    "python",
    # NLP
    "nlp", "natural language processing", "transformers", "bert", "gpt",
    # ML broad
    "machine learning", "deep learning", "neural network", "pytorch", "tensorflow",
    "xgboost", "scikit-learn",
}

# Nice-to-have skills
BONUS_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning llms", "huggingface",
    "xgboost", "lightgbm", "learning to rank",
    "distributed systems", "kafka", "spark",
    "open source", "github", "arxiv",
    "hr tech", "recruiting", "talent", "marketplace",
    "inference optimization", "triton", "onnx",
    "langchain", "llamaindex",  # ok if paired with real experience
}

# Hard disqualifiers from JD
DISQ_INDUSTRIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree", "l&t infotech",
}
# Pure services if ALL career is at these — partial ok
SERVICES_COMPANIES = DISQ_INDUSTRIES | {
    "ibm", "deloitte", "pwc", "kpmg", "ey", "ernst", "oracle consulting",
    "zensar", "persistent", "birlasoft",
}

CV_SKILL_DOMAINS = {"computer vision", "image classification", "object detection",
                    "speech recognition", "tts", "asr", "robotics"}

PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "ncr", "gurugram", "gurgaon", "hyderabad",
    "mumbai", "bengaluru", "bangalore", "india"
}

PRODUCT_INDUSTRIES = {
    "fintech", "edtech", "healthtech", "saas", "e-commerce", "ecommerce",
    "internet", "software", "ai", "ml", "data", "analytics", "tech",
    "startup", "platform", "marketplace", "media", "gaming", "telecom",
}


def days_since(date_str: str) -> int:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (REFERENCE_DATE - d).days
    except Exception:
        return 999


def normalize(text: str) -> str:
    return text.lower().strip()


def skill_names(candidate: dict) -> set:
    return {normalize(s["name"]) for s in candidate.get("skills", [])}


def skill_map(candidate: dict) -> dict:
    """name -> {proficiency, endorsements, duration_months}"""
    m = {}
    for s in candidate.get("skills", []):
        m[normalize(s["name"])] = s
    return m


def is_honeypot(candidate: dict) -> bool:
    """Detect impossible profiles."""
    p = candidate["profile"]
    
    # Check: company founded after experience
    for job in candidate.get("career_history", []):
        try:
            start = datetime.strptime(job["start_date"], "%Y-%m-%d").date()
            dur = job.get("duration_months", 0)
            if dur > (REFERENCE_DATE - start).days / 30 + 12:
                return True
        except Exception:
            pass
    
    # Check: expert in 10+ skills but 0 duration months on all
    skills = candidate.get("skills", [])
    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    zero_dur_experts = [s for s in expert_skills if s.get("duration_months", 1) == 0]
    if len(zero_dur_experts) >= 8:
        return True
    
    # Check: total claimed experience >> actual career history
    yoe = p.get("years_of_experience", 0)
    total_career_months = sum(
        j.get("duration_months", 0) for j in candidate.get("career_history", [])
    )
    if yoe > 3 and total_career_months < yoe * 6:  # Less than half claimed
        # Only flag if also has suspicious skills
        if len(expert_skills) > 8:
            return True
    
    return False


def score_candidate(candidate: dict) -> tuple[float, str]:
    """Returns (score 0-1, reasoning string)."""
    
    p = candidate["profile"]
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    sm = skill_map(candidate)
    snames = skill_names(candidate)
    
    score = 0.0
    reasons = []
    
    # ── 0. Honeypot check ────────────────────────────────────────────────
    if is_honeypot(candidate):
        return (0.0, "Honeypot: impossible profile signals detected.")
    
    # ── 1. Core skill match (max 35 pts) ─────────────────────────────────
    core_matches = snames & CORE_SKILLS
    
    # Also check career description text
    career_text = " ".join(
        normalize(j.get("description", "") + " " + j.get("title", ""))
        for j in career
    ).lower()
    summary_text = normalize(p.get("summary", "") + " " + p.get("headline", ""))
    all_text = career_text + " " + summary_text
    
    text_core_hits = sum(1 for kw in CORE_SKILLS if kw in all_text)
    
    # Proficiency-weighted skill score
    skill_score = 0.0
    PROF_WEIGHT = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}
    for name, data in sm.items():
        if name in CORE_SKILLS:
            pw = PROF_WEIGHT.get(data.get("proficiency", "intermediate"), 0.5)
            dur = min(data.get("duration_months", 0), 60) / 60  # cap at 5yr
            skill_score += pw * (0.6 + 0.4 * dur)
    
    # Normalize: assume 8 core skills is excellent
    skill_score = min(skill_score / 8.0, 1.0)
    
    # Text coverage bonus
    text_bonus = min(text_core_hits / 12, 1.0) * 0.3
    
    raw_skill = min(skill_score * 0.7 + text_bonus + len(core_matches)/20 * 0.3, 1.0)
    skill_pts = raw_skill * 35
    score += skill_pts
    
    if core_matches:
        top_skills = list(core_matches)[:5]
        reasons.append(f"{len(core_matches)} core skills: {', '.join(top_skills)}")
    
    # ── 2. Career quality (max 30 pts) ────────────────────────────────────
    career_pts = 0.0
    
    yoe = p.get("years_of_experience", 0)
    
    # YoE in sweet spot 5-9 years
    if 5 <= yoe <= 9:
        career_pts += 8
        reasons.append(f"{yoe}y exp (ideal 5-9)")
    elif 4 <= yoe < 5:
        career_pts += 5
    elif 9 < yoe <= 12:
        career_pts += 5
    elif 3 <= yoe < 4:
        career_pts += 2
    
    # Product company experience (not pure services)
    product_months = 0
    services_months = 0
    has_any_product = False
    for job in career:
        company_lower = normalize(job.get("company", ""))
        industry_lower = normalize(job.get("industry", ""))
        dur = job.get("duration_months", 0)
        
        is_services = any(s in company_lower for s in SERVICES_COMPANIES)
        is_product = any(kw in industry_lower for kw in PRODUCT_INDUSTRIES) and not is_services
        
        if is_product:
            product_months += dur
            has_any_product = True
        if is_services:
            services_months += dur
    
    total_months = max(sum(j.get("duration_months", 0) for j in career), 1)
    product_ratio = product_months / total_months
    services_ratio = services_months / total_months
    
    if product_ratio > 0.6:
        career_pts += 10
        reasons.append("Mostly product company background")
    elif product_ratio > 0.3:
        career_pts += 6
        reasons.append("Mixed product/services background")
    elif services_ratio > 0.8:
        career_pts -= 5  # Penalty for pure services
        reasons.append("Mostly services background (JD explicitly disfavors)")
    
    # Has shipped ranking/retrieval/search/recommendation systems
    shipped_keywords = [
        "ranking", "retrieval", "recommendation", "search", "vector",
        "embedding", "rag", "matching", "ranker", "retriev"
    ]
    shipped_count = sum(1 for kw in shipped_keywords if kw in career_text)
    if shipped_count >= 3:
        career_pts += 8
        reasons.append("Shipped ranking/retrieval/search systems")
    elif shipped_count >= 1:
        career_pts += 4
    
    # Penalize if ONLY computer vision / speech / robotics
    cv_skills = snames & CV_SKILL_DOMAINS
    if cv_skills and len(cv_skills) > 3 and not core_matches - CV_SKILL_DOMAINS:
        career_pts -= 4
        reasons.append("Primarily CV/speech domain (limited NLP/IR)")
    
    # Production experience signal (scale mentions)
    prod_keywords = ["production", "deployed", "real users", "scale", "latency",
                     "inference", "serving", "api", "million"]
    prod_count = sum(1 for kw in prod_keywords if kw in career_text)
    if prod_count >= 4:
        career_pts += 4
        reasons.append("Strong production/deployment experience")
    elif prod_count >= 2:
        career_pts += 2
    
    career_pts = max(0, min(career_pts, 30))
    score += career_pts
    
    # ── 3. Behavioral / availability signals (max 20 pts) ─────────────────
    avail_pts = 0.0
    
    # Recency of activity
    days_inactive = days_since(signals.get("last_active_date", "2020-01-01"))
    if days_inactive <= 30:
        avail_pts += 5
    elif days_inactive <= 90:
        avail_pts += 3
    elif days_inactive <= 180:
        avail_pts += 1
    else:
        avail_pts -= 2  # Ghost candidate
        reasons.append(f"Inactive {days_inactive}d (availability risk)")
    
    # Open to work
    if signals.get("open_to_work_flag", False):
        avail_pts += 3
    
    # Notice period (prefer <30d, penalize >90d)
    notice = signals.get("notice_period_days", 60)
    if notice <= 30:
        avail_pts += 4
        reasons.append(f"Notice {notice}d (ideal)")
    elif notice <= 60:
        avail_pts += 2
    elif notice > 90:
        avail_pts -= 1
        reasons.append(f"Notice {notice}d (long)")
    
    # Response rate
    rr = signals.get("recruiter_response_rate", 0)
    if rr >= 0.7:
        avail_pts += 4
    elif rr >= 0.5:
        avail_pts += 2
    elif rr < 0.15:
        avail_pts -= 2
        reasons.append(f"Low response rate ({rr:.0%})")
    
    # Interview completion
    icr = signals.get("interview_completion_rate", 0.5)
    if icr >= 0.8:
        avail_pts += 2
    elif icr < 0.4:
        avail_pts -= 1
    
    # Verified identity
    if signals.get("verified_email") and signals.get("verified_phone"):
        avail_pts += 1
    
    # Saved by recruiters (social proof)
    saved = signals.get("saved_by_recruiters_30d", 0)
    if saved >= 5:
        avail_pts += 1
    
    avail_pts = max(0, min(avail_pts, 20))
    score += avail_pts
    
    # ── 4. GitHub / external validation (max 8 pts) ───────────────────────
    github_score = signals.get("github_activity_score", -1)
    if github_score >= 70:
        score += 8
        reasons.append(f"Strong GitHub activity ({github_score:.0f})")
    elif github_score >= 40:
        score += 5
    elif github_score >= 10:
        score += 2
    # -1 = no GitHub linked, neutral
    
    # ── 5. Location fit (max 7 pts) ───────────────────────────────────────
    location = normalize(p.get("location", "") + " " + p.get("country", ""))
    
    if any(loc in location for loc in PREFERRED_LOCATIONS):
        score += 7
        loc_str = p.get("location", "")
        if loc_str not in [r for r in reasons if "loc" in r.lower()]:
            reasons.append(f"India-based ({p.get('location','')})")
    elif signals.get("willing_to_relocate", False):
        score += 3
        reasons.append("Willing to relocate")
    else:
        score += 0  # Outside India, won't relocate
    
    # ── Normalize to 0-1 ─────────────────────────────────────────────────
    # Max possible: 35+30+20+8+7 = 100
    score_norm = max(0.0, min(score / 100.0, 1.0))
    
    # Build reasoning
    title = p.get("current_title", "")
    yoe_str = f"{yoe:.1f}yr"
    base_reason = f"{title}; {yoe_str}; "
    reason_text = base_reason + "; ".join(reasons[:4]) if reasons else base_reason + "profile reviewed"
    # Cap at ~200 chars
    reason_text = reason_text[:200]
    
    return (score_norm, reason_text)


def rank_candidates(candidates_path: str, out_path: str):
    print(f"Loading candidates from {candidates_path}...")
    
    scored = []
    total = 0
    honeypots = 0
    
    with open(candidates_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            total += 1
            
            sc, reason = score_candidate(c)
            if sc == 0.0 and "Honeypot" in reason:
                honeypots += 1
            
            scored.append((sc, c["candidate_id"], reason, c))
            
            if total % 10000 == 0:
                print(f"  Processed {total:,}...")
    
    print(f"Total: {total:,} | Honeypots detected: {honeypots}")
    
    # Sort: descending score, then ascending candidate_id for ties
    scored.sort(key=lambda x: (-x[0], x[1]))
    
    top100 = scored[:100]
    
    # Validate non-increasing scores and fix tie-breaks
    rows = []
    for rank_idx, (sc, cid, reason, cand) in enumerate(top100):
        rank = rank_idx + 1
        rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(sc, 4),
            "reasoning": reason
        })
    
    # Write CSV
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id","rank","score","reasoning"])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nTop 10 preview:")
    for r in rows[:10]:
        print(f"  #{r['rank']} {r['candidate_id']} score={r['score']} | {r['reasoning'][:80]}")
    
    print(f"\nSubmission written to: {out_path}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", default="candidates.jsonl", help="Path to candidates.jsonl")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    args = parser.parse_args()
    
    rank_candidates(args.candidates, args.out)
