import spacy
from typing import Dict, Any, List
from state import ResumeState

# Lazy-load spaCy model
_nlp = None

def get_nlp_model():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            import sys
            print("Downloading spaCy model 'en_core_web_sm'...")
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], capture_output=True)
            _nlp = spacy.load("en_core_web_sm")
    return _nlp

def extract_syntactic_phrases(text: str) -> List[str]:
    nlp = get_nlp_model()
    doc = nlp(text[:8000])  # Limit length for performance
    phrases = []
    
    # Extract active verb-noun relations (e.g. developed software, managed team)
    for token in doc:
        if token.pos_ in ("VERB", "AUX"):
            children = [w for w in token.children if w.dep_ in ("dobj", "attr", "prep", "acomp")]
            for child in children:
                phrase = f"{token.text} {child.text}"
                if child.dep_ == "prep":
                    pobjs = [w for w in child.children if w.dep_ == "pobj"]
                    if pobjs:
                        phrase += f" {pobjs[0].text}"
                phrases.append(phrase.lower().strip())
    
    # Remove duplicates preserving order
    seen = set()
    deduped = []
    for p in phrases:
        if p not in seen and len(p) > 3:
            seen.add(p)
            deduped.append(p)
    return deduped[:25]

def extract_ner_entities(text: str) -> Dict[str, List[str]]:
    nlp = get_nlp_model()
    doc = nlp(text[:8000])
    entities = {"PERSON": [], "ORG": [], "DATE": [], "GPE": []}
    
    for ent in doc.ents:
        if ent.label_ in entities:
            # Clean entity name
            clean_name = ent.text.strip().replace("\n", " ")
            entities[ent.label_].append(clean_name)
            
    # Deduplicate entities
    for label in entities:
        seen = set()
        deduped = []
        for name in entities[label]:
            if name.lower() not in seen and len(name) > 1:
                seen.add(name.lower())
                deduped.append(name)
        entities[label] = deduped[:12]
        
    return entities

def get_ngrams(text: str, n: int) -> List[str]:
    nlp = get_nlp_model()
    # Basic tokenization and stop word removal
    doc = nlp(text.lower()[:8000])
    words = [token.text for token in doc if not token.is_stop and not token.is_punct and len(token.text) > 1]
    
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i:i+n])
        ngrams.append(ngram)
    return ngrams

def compute_ngram_overlaps(resume_text: str, jd_text: str) -> Dict[str, Any]:
    # Compute n-grams
    resume_uni = set(get_ngrams(resume_text, 1))
    jd_uni = set(get_ngrams(jd_text, 1))
    uni_matched = list(resume_uni.intersection(jd_uni))
    
    resume_bi = set(get_ngrams(resume_text, 2))
    jd_bi = set(get_ngrams(jd_text, 2))
    bi_matched = list(resume_bi.intersection(jd_bi))
    
    resume_tri = set(get_ngrams(resume_text, 3))
    jd_tri = set(get_ngrams(jd_text, 3))
    tri_matched = list(resume_tri.intersection(jd_tri))
    
    bi_overlap_pct = len(bi_matched) / max(len(jd_bi), 1)
    tri_overlap_pct = len(tri_matched) / max(len(jd_tri), 1)
    
    return {
        "unigrams_matched": uni_matched[:15],
        "bigrams_matched": bi_matched[:15],
        "trigrams_matched": tri_matched[:15],
        "bigram_overlap_pct": round(bi_overlap_pct, 3),
        "trigram_overlap_pct": round(tri_overlap_pct, 3)
    }

def classify_text_domain(text: str) -> Dict[str, Any]:
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        domains = [
            "Software Engineering & IT",
            "Data Science & AI",
            "Marketing & Sales",
            "Finance & Accounting",
            "Human Resources",
            "Healthcare & Medical",
            "Operations & Management",
            "Creative & Design"
        ]
        
        text_emb = model.encode(text[:2000], convert_to_tensor=True)
        domain_embs = model.encode(domains, convert_to_tensor=True)
        scores = util.cos_sim(text_emb, domain_embs)[0]
        
        # Get sorted scores
        domain_scores = []
        for i, domain in enumerate(domains):
            domain_scores.append({"domain": domain, "score": float(scores[i])})
        domain_scores = sorted(domain_scores, key=lambda x: x["score"], reverse=True)
        
        return {
            "primary_domain": domain_scores[0]["domain"],
            "primary_score": round(domain_scores[0]["score"], 3),
            "scores": domain_scores
        }
    except Exception as e:
        print(f"Domain classification failed: {e}")
        return {
            "primary_domain": "Unknown",
            "primary_score": 0.0,
            "scores": []
        }

def nlp_processing_node(state: ResumeState) -> ResumeState:
    if state.get("error"):
        return state
        
    resume_text = state.get("resume_text")
    jd_text = state.get("job_description")
    
    if not resume_text or not jd_text:
        return state
        
    try:
        # 1. Syntactic active phrases
        syntactic_phrases = extract_syntactic_phrases(resume_text)
        
        # 2. NER Entities
        ner_entities = extract_ner_entities(resume_text)
        
        # 3. N-Gram overlaps
        ngram_results = compute_ngram_overlaps(resume_text, jd_text)
        
        # 4. Text Classification
        resume_class = classify_text_domain(resume_text)
        jd_class = classify_text_domain(jd_text)
        
        # Calculate domain alignment score
        domain_match = (resume_class["primary_domain"] == jd_class["primary_domain"])
        
        return {
            **state,
            "nlp_analysis": {
                "syntactic_phrases": syntactic_phrases,
                "ner_entities": ner_entities,
                "ngrams": ngram_results
            },
            "classification": {
                "resume_domain": resume_class["primary_domain"],
                "resume_domain_score": resume_class["primary_score"],
                "jd_domain": jd_class["primary_domain"],
                "jd_domain_score": jd_class["primary_score"],
                "domain_alignment": domain_match,
                "all_resume_scores": resume_class["scores"]
            }
        }
    except Exception as e:
        print(f"NLP processing failed: {e}")
        return {**state, "error": f"NLP processing failed: {str(e)}"}
