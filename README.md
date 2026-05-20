1. Problem Definition
Traditional Applicant Tracking Systems (ATS) exhibit significant limitations that reduce hiring efficiency and lead to inaccurate candidate evaluations. These limitations include:
1.1 Keyword Manipulation and Gaming
Conventional ATS solutions rely heavily on keyword matching, allowing candidates to artificially inflate their scores by embedding irrelevant or excessive keywords into resumes. This results in high scoring for candidates who may not possess the required competencies.
1.2 Semantic Mismatch Problem
Lexical variations and synonyms often lead to incorrect rejection of relevant candidates. For instance, terms such as “PostgreSQL” and “Postgres” or “REST APIs” and “web services” are semantically equivalent but fail exact-match systems.
1.3 Domain Ambiguity
Identical terminology can have different meanings across industries. For example, the term “campaign” differs in marketing, political, and software engineering contexts, causing misclassification in naive text-based systems.
1.4 Lack of Structural and Visual Understanding
Traditional ATS engines ignore resume formatting, layout quality, and visual structure, despite these being important indicators of professionalism and readability.
2. System Requirements
To address these limitations, the proposed ATS Resume Screening System is designed with both functional and non-functional requirements.
Functional Requirements
•	Hybrid Document Parsing System
Supports digital PDF extraction and OCR-based extraction for scanned documents. 
•	Named Entity Recognition (NER) Module
Extracts structured entities such as names, organizations, locations, and dates. 
•	Semantic Similarity Engine
Computes contextual similarity between resumes and job descriptions using dense embeddings. 
•	Domain Classification System
Identifies and aligns professional domains between candidate profiles and job descriptions. 
•	Multimodal LLM-Based Evaluation
Combines structured NLP outputs with visual resume representations for scoring and critique. 
•	Candidate Analytics Dashboard
Provides real-time scoring, comparison, and historical evaluation tracking. 
3. System Architecture Overview
 
4. NLP Methodologies and Techniques
4.1 Hybrid PDF Text & Vision Extraction
The system utilizes a dual-layer extraction pipeline:
•	pdfplumber for structured digital text extraction 
•	PyMuPDF (fitz) for rendering document pages 
•	Tesseract OCR for scanned or image-based resumes 
If extracted text length falls below a defined threshold, the system automatically activates OCR processing.
This ensures robust handling of heterogeneous resume formats.

4.2 Named Entity Recognition (NER)
The system uses spaCy’s statistical NER model (en_core_web_sm) to extract structured entities.
Extracted Entity Types:
•	PERSON → Candidate identity 
•	ORG → Employers and institutions 
•	DATE → Employment and education timelines 
•	GPE → Geographic locations 
Purpose:
NER transforms unstructured resume text into structured candidate profiles, enabling downstream scoring and analytics.

4.3 Syntactic Dependency Parsing
Using spaCy’s dependency parser, the system extracts verb–object relationships to validate real experience.
Example:
“Developed scalable RESTful services”
→ Extracted structure:
•	(developed → services) 
Purpose:
This approach eliminates keyword stuffing by ensuring that skills are associated with active professional usage, not isolated mentions.

4.4 N-Gram Feature Extraction
The system generates unigram, bigram, and trigram representations of text.
Examples:
•	Bigram: “machine learning” 
•	Trigram: “continuous integration pipeline” 
Purpose:
Captures domain-specific multi-word technical concepts that are lost in unigram-based matching systems.
A normalized overlap ratio is computed between resume and job description n-grams to quantify lexical alignment.

4.5 Semantic Embedding & Cosine Similarity
Using SentenceTransformers (all-MiniLM-L6-v2), text is converted into dense vector representations.
Applications:
•	Document-Level Similarity
Measures global semantic alignment between resume and job description. 
•	Skill-Level Matching
Identifies semantically similar skills using cosine similarity threshold ≥ 0.80. 
Purpose:
Solves synonym and vocabulary mismatch problems (e.g., “postgres” vs “PostgreSQL”).

4.6 Domain Classification (Zero-Shot Semantic Mapping)
The system performs domain classification using embedding similarity against predefined industry vectors.
Supported Domains:
•	Software Engineering 
•	Data Science 
•	Finance 
•	Marketing 
•	Healthcare 
•	etc. 
Purpose:
Ensures alignment between candidate expertise and job domain, reducing cross-industry false matches.

4.7 Multimodal LLM-Based Evaluation
The system integrates a multimodal Large Language Model (e.g., GPT-4o) to perform final evaluation.
Inputs:
•	Precomputed NLP metrics 
•	Structured entity data 
•	Base64-encoded resume images 
Outputs:
•	ATS score 
•	Skill gap analysis 
•	Resume formatting evaluation 
•	Hiring recommendation 
Purpose:
Provides human-like reasoning while remaining grounded in deterministic NLP outputs.

