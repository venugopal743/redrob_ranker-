# Redrob Intelligent Candidate Ranking — akulasanith18

**Team:** Chunchu Venugopal & Sanith Akula  
**Challenge:** Intelligent Candidate Discovery & Ranking  
**Approach:** Multi-signal hybrid scoring (no GPU, no LLM API, CPU-only, <5 min)

---

## Architecture

### Why not pure semantic embeddings?

The JD explicitly warns against keyword-matching traps. A simple embedding cosine similarity over skill names would rank "Marketing Manager who listed RAG in skills" above "6yr ML engineer who built retrieval systems". Our approach reads **what candidates actually did**, not just what they listed.

### Scoring Components (total 100 pts)

| Component | Max pts | What it measures |
|-----------|---------|-----------------|
| Core skill match | 35 | Proficiency-weighted intersection with JD-critical skills (embeddings, vector DBs, ranking, NLP, LLMs, Python) + career text coverage |
| Career quality | 30 | YoE in 5–9yr sweet spot, product vs services background, shipped ranking/search/retrieval systems, production scale signals |
| Behavioral signals | 20 | Recency (last active), open-to-work flag, notice period, recruiter response rate, interview completion, verified identity |
| GitHub activity | 8 | External contribution signal (open-source evidence the JD explicitly values) |
| Location fit | 7 | India-based (Pune/Noida/Delhi/Bangalore/Hyderabad preferred), or willing to relocate |

### Key design decisions

**JD-aware disqualifiers** applied as score penalties:
- Pure IT services background (TCS/Wipro/Infosys/etc.) → −5 pts
- Primarily CV/Speech domain without NLP/IR → −4 pts
- Ghost candidates (>180 days inactive) → −2 pts availability
- Low recruiter response rate (<15%) → −2 pts

**Honeypot detection** (forced to score 0):
- Duration claimed > time since company founded
- Expert proficiency on 8+ skills with 0 months usage
- YoE claim >>2× actual career history months with suspiciously many expert skills

**Availability as multiplier**: A perfect-on-paper candidate inactive for 6 months drops ~7 pts, pushing them out of top-10 even with strong skills.

**Career text analysis**: Beyond skill tags, we scan job description narratives for evidence of shipped retrieval/ranking/recommendation systems — catching candidates whose skills section uses different terminology than their actual work.

---

## Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run ranking (full 100K pool)
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
Runtime: ~60–90 seconds on a modern CPU. Memory: <2 GB.

### Validate output
```bash
python validate_submission.py submission.csv
```

---

## Files

```
.
├── rank.py                     # Main ranking script (single entry point)
├── validate_submission.py      # Official challenge validator (included)
├── submission.csv              # Our ranked top-100 output
├── submission_metadata.yaml    # Portal metadata
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Reproduce in one command
```bash
python rank.py --candidates /path/to/candidates.jsonl --out submission.csv
```

No pre-computation step required. No embeddings to pre-build. No model weights to download.

---

## Scoring philosophy

We optimized for NDCG@10 (50% weight in the composite) by ensuring the absolute top candidates are genuine fits — senior ML engineers at product companies with real retrieval/ranking/embedding production experience, active on the platform, in India with manageable notice periods. The long tail (ranks 50–100) uses the same scoring but naturally includes adjacent profiles.
