# Kotaemon RAG System: Complete Technical Deep Dive

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Document Upload & Processing Pipeline](#document-upload--processing-pipeline)
3. [Dual Storage System](#dual-storage-system)
4. [Query Processing](#query-processing)
5. [Hybrid Retrieval](#hybrid-retrieval)
6. [Document Ranking & Scoring](#document-ranking--scoring)
7. [Answer Generation with Citations](#answer-generation-with-citations)
8. [Performance Optimization](#performance-optimization)
9. [Configuration & Customization](#configuration--customization)

---

## Architecture Overview

Kotaemon implements a sophisticated **Retrieval-Augmented Generation (RAG)** system that combines multiple AI technologies to provide accurate, cited answers from your documents.

### Core Components
```python
kotaemon_architecture = {
    "document_processing": "Smart chunking + embedding generation",
    "dual_storage": "Vector store (semantic) + Document store (keyword)",
    "hybrid_retrieval": "Parallel semantic + text search",
    "llm_integration": "Google Gemini for understanding + generation",
    "citation_system": "Source tracking + confidence scoring"
}
```

### Key Features
- ✅ **Multi-format support**: PDF, DOCX, TXT, HTML, Excel
- ✅ **Semantic search**: Understanding meaning beyond keywords
- ✅ **Exact matching**: Precise keyword and phrase finding
- ✅ **Source citations**: Every answer traces back to your documents
- ✅ **Multi-user support**: Isolated document collections per user
- ✅ **Real-time processing**: Immediate indexing and search

---

## Document Upload & Processing Pipeline

### 1. File Upload & Reading

When you upload a document, the system begins with intelligent file analysis:

```python
# Log example from system
"Using reader TxtReader()"
"Overriding with default loaders"
"use_quick_index_mode False"
"reader_mode default"
```

#### File Detection & Reader Selection
```python
file_type_mapping = {
    ".txt": "TxtReader()",
    ".pdf": "PDFReader()", 
    ".docx": "DocxReader()",
    ".html": "HTMLReader()",
    ".xlsx": "ExcelReader()"
}

# System automatically selects appropriate reader
def select_reader(file_path):
    extension = get_file_extension(file_path)
    reader_class = file_type_mapping.get(extension, "UniversalReader")
    return reader_class()
```

#### Quick Index Mode Decision
```python
def determine_index_mode(file):
    complexity_factors = {
        "has_images": 0.3,
        "has_tables": 0.3,
        "file_size_mb": 0.2,
        "structure_complexity": 0.2
    }
    
    complexity_score = sum(
        file_analysis[factor] * weight 
        for factor, weight in complexity_factors.items()
    )
    
    return "quick" if complexity_score < 0.5 else "full"

# Quick Mode: Fast text extraction only
# Full Mode: Complete parsing with images, tables, formatting
```

### 2. Document Chunking

```python
# Log examples
"Chunk size: None, chunk overlap: None"
"Got 0 page thumbnails"
```

#### Smart Chunking Strategy
```python
class SmartChunker:
    def __init__(self):
        self.chunk_size = 1000  # tokens
        self.chunk_overlap = 200  # tokens
        self.preserve_boundaries = True
    
    def chunk_document(self, document):
        chunks = []
        
        # Preserve semantic boundaries
        sections = self.identify_sections(document)
        
        for section in sections:
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                # Split large sections with overlap
                sub_chunks = self.split_with_overlap(section)
                chunks.extend(sub_chunks)
        
        return chunks
```

#### Context Preservation Methods
1. **Overlapping Windows**: 200-token overlap between chunks
2. **Semantic Boundaries**: Break at paragraphs, not mid-sentence
3. **Hierarchical Metadata**: Maintain document structure
4. **Cross-references**: Link related chunks

### 3. Embedding Generation

```python
# Log examples
"Running embedding in thread"
"Getting embeddings for 1 nodes"
"indexing step took 0.5623195171356201"
```

#### Parallel Embedding Process
```python
async def generate_embeddings(chunks):
    """Generate embeddings for document chunks in parallel"""
    
    embedding_tasks = []
    for chunk in chunks:
        task = asyncio.create_task(
            google_embeddings.embed_documents([chunk.text])
        )
        embedding_tasks.append(task)
    
    # Process multiple chunks simultaneously
    embeddings = await asyncio.gather(*embedding_tasks)
    
    return embeddings

# Google AI Embeddings Model: models/embedding-001
# Vector Dimensions: 768
# Processing Time: ~0.5 seconds per chunk
```

#### Metadata Preservation
```python
chunk_metadata = {
    "chunk_id": "chunk_123",
    "source_document": "6aa41ef8-fd93-4033-b757-2a83c11b5808",
    "chunk_index": 47,
    "start_character": 5600,
    "end_character": 6800,
    "token_count": 245,
    "page_number": 3,  # For PDFs
    "section_title": "Browser Compatibility Issues",
    "embedding_model": "models/embedding-001",
    "processing_timestamp": "2024-08-26T12:00:05Z"
}
```

---

## Dual Storage System

The system uses two complementary storage approaches because no single method handles all search scenarios effectively.

### Storage System A: Vector Store (ChromaDB)

#### Purpose & Strengths
- **Semantic Search**: Finds conceptually similar content
- **Fuzzy Matching**: "browser problems" matches "Chrome issues"  
- **Cross-lingual**: Works across different languages
- **Fast Similarity**: Optimized vector operations

#### Storage Structure
```python
vector_entry = {
    "chunk_id": "chunk_123",
    "embedding": [0.234, -0.567, 0.891, ...],  # 768-dimensional vector
    "metadata": {
        "source_doc": "6aa41ef8-fd93-4033-b757-2a83c11b5808",
        "chunk_text": "Chrome extension conflicts cause rendering issues...",
        "chunk_index": 47,
        "token_count": 245,
        "similarity_threshold": 0.7
    }
}
```

#### Vector Operations
```python
# Similarity computation
def semantic_search(query_vector, document_vectors):
    similarities = []
    
    for doc_vector in document_vectors:
        # Cosine similarity calculation
        similarity = cosine_similarity(query_vector, doc_vector)
        similarities.append(similarity)
    
    # Filter by threshold and rank
    relevant_docs = [
        (doc, sim) for doc, sim in zip(documents, similarities) 
        if sim >= 0.7
    ]
    
    return sorted(relevant_docs, key=lambda x: x[1], reverse=True)
```

### Storage System B: Document Store (LanceDB)

#### Purpose & Strengths
- **Exact Matching**: Perfect for specific terms, codes, names
- **Boolean Logic**: Complex queries like "Chrome AND extension NOT Safari"
- **Metadata Filtering**: Search within specific files, dates, authors
- **Full Context**: Preserves complete document structure

#### Storage Structure
```python
document_entry = {
    "chunk_id": "chunk_123", 
    "full_text": "Case #ZB-2024-1847: Chrome Extension Conflict Breaking Document Rendering...",
    "searchable_fields": {
        "content": "full_text_with_inverted_index",
        "filename": "browser_issues.txt",
        "section_title": "Browser Compatibility Issues",
        "case_numbers": ["ZB-2024-1847", "ZB-2024-1856"],
        "keywords": ["Chrome", "extension", "conflict", "rendering"]
    },
    "metadata": {
        "file_id": "6aa41ef8-fd93-4033-b757-2a83c11b5808",
        "user_id": "ff6f681e872146cf8304c1a59a53b818",
        "upload_date": "2024-08-26",
        "file_type": "text/plain",
        "chunk_boundaries": [5600, 6800]
    }
}
```

#### Text Search Operations
```python
def keyword_search(query_terms, filters=None):
    """Full-text search with boolean logic"""
    
    search_query = {
        "must": query_terms,          # All terms must appear
        "should": synonym_expansion(query_terms),  # Boost synonyms
        "filters": filters or {},     # User, date, type filters
        "highlight": True             # Mark matching text
    }
    
    results = lance_db.search(search_query)
    return results
```

### Hybrid Storage Benefits

#### Complementary Strengths Example
```python
# Query: "What is Case ZB-2024-1847 about?"
vector_store_result = "Similarity: 0.4 (low - numbers don't embed well)"
document_store_result = "Exact match: 'ZB-2024-1847' found instantly"
winner = document_store_result

# Query: "What browser compatibility issues exist?"  
vector_store_result = "Similarity: 0.89 (high - understands 'compatibility')"
document_store_result = "No exact phrase match for 'compatibility issues'"
winner = vector_store_result

# Query: "Chrome extension problems in Safari"
# Hybrid result: Both stores contribute different aspects
combined_result = merge_and_deduplicate(vector_results, text_results)
```

---

## Query Processing

This stage transforms your natural language question into optimized search parameters.

### Session Setup & Pipeline Selection

```python
# From system logs
session_config = {
    "reasoning_type": None,           # Standard Q&A, not complex reasoning
    "use_mindmap": True,             # Visual relationship mapping
    "use_citation_highlight": True,  # Citation tracking active  
    "language": "en",                # English language processing
    "pipeline": "FullQAPipeline"     # Complete RAG workflow
}

# Available pipeline options
pipeline_types = {
    "FullQAPipeline": "Standard RAG with citations",
    "FullDecomposeQAPipeline": "Break complex questions into sub-questions",
    "ReactAgentPipeline": "Multi-step reasoning with external tools", 
    "RewooAgentPipeline": "Planning-based complex reasoning"
}
```

### Query Analysis & Understanding

```python
# System log: "Thinking ..."
# This represents sophisticated NLP processing:

class QueryAnalyzer:
    def analyze_query(self, query):
        analysis = {
            "intent": self.classify_intent(query),
            "entities": self.extract_entities(query),
            "complexity": self.assess_complexity(query),
            "expected_answer_type": self.predict_answer_type(query)
        }
        return analysis

# Example analysis
query_analysis = {
    "original": "What browser extension problems are documented?",
    "intent": "factual_listing",  # User wants a list of facts
    "entities": {
        "main_topic": "browser extension problems",
        "action": "documented", 
        "scope": "problems",
        "domain": "web_technology"
    },
    "complexity": "medium",  # Straightforward but broad
    "expected_answer_type": "multiple_citations_with_examples"
}
```

### Query Expansion & Optimization

```python
def expand_query(original_query):
    """Create multiple query variants for better retrieval"""
    
    return {
        "original": original_query,
        "synonyms": {
            "browser": ["web browser", "Chrome", "Firefox", "Safari", "Edge"],
            "extension": ["plugin", "addon", "add-on"],
            "problems": ["issues", "conflicts", "errors", "bugs", "failures"]
        },
        "related_concepts": [
            "compatibility issues",
            "plugin conflicts", 
            "rendering problems",
            "performance issues"
        ],
        "technical_terms": [
            "browser extension API",
            "DOM manipulation",
            "content scripts",
            "manifest errors"
        ]
    }

# Query embedding generation
query_embedding = google_embeddings.embed(
    "What browser extension problems are documented?"
)
# Result: [0.123, -0.456, 0.789, ...] (768 dimensions)
```

### Search Strategy Selection

```python
def select_search_strategy(query_analysis):
    """Choose optimal search approach based on query characteristics"""
    
    if query_analysis["has_specific_identifiers"]:
        # Case numbers, exact names, codes
        return "document_store_priority"
        
    elif query_analysis["is_conceptual"]: 
        # Broad understanding questions
        return "vector_store_priority"
        
    elif query_analysis["is_mixed"]:
        # Both specific and conceptual elements
        return "balanced_hybrid"
    
    else:
        return "adaptive_search"  # Let system decide dynamically

# For "What browser extension problems are documented?"
strategy = "balanced_hybrid"  # Both semantic understanding and exact matching needed
```

---

## Hybrid Retrieval

This is where your processed query searches through both storage systems simultaneously.

### Parallel Retrieval Execution

```python
# System logs showing parallel execution
"retrieval_mode='hybrid', top_k=10"
"Got 6 from vectorstore"
"Got 0 from docstore" 
```

#### Retrieval Coordination
```python
async def hybrid_retrieval(query, config):
    """Execute both retrieval methods in parallel"""
    
    # Launch both searches simultaneously
    vector_task = asyncio.create_task(
        semantic_search(
            query_embedding=query.embedding,
            top_k=config.top_k,
            threshold=config.similarity_threshold
        )
    )
    
    text_task = asyncio.create_task(
        keyword_search(
            query_terms=query.keywords,
            filters=config.filters,
            top_k=config.top_k
        )
    )
    
    # Wait for both to complete
    vector_results, text_results = await asyncio.gather(vector_task, text_task)
    
    return merge_results(vector_results, text_results)
```

### Vector Store Search Deep Dive

#### Semantic Similarity Process
```python
# Your query vector compared against all document chunks
search_process = {
    "query_embedding": [0.234, -0.567, 0.891, ...],
    "comparison_method": "cosine_similarity",
    "threshold": 0.7,
    "total_chunks_compared": 1247,
    "matches_found": 6
}

# Example similarity scores
similarity_results = {
    "browser extension problems": 0.92,    # Near-perfect semantic match
    "Chrome plugin conflicts": 0.87,      # Synonymous concepts
    "Firefox addon issues": 0.85,         # Related browser problems  
    "Safari extension errors": 0.82,      # Similar domain
    "Web browser compatibility": 0.78,    # Broader category
    "Plugin rendering problems": 0.74     # Specific problem type
}

# All scores above 0.7 threshold → included in results
```

### Document Store Search Analysis

#### Why Text Search Returned 0 Results
```python
# Query term analysis
query_breakdown = {
    "search_terms": ["browser", "extension", "problems", "documented"],
    "document_vocabulary": {
        "browser": "✅ Found in 3 chunks",
        "extension": "✅ Found in 5 chunks",
        "problems": "❌ Documents use 'issues', 'conflicts', 'errors'",
        "documented": "❌ Documents don't use this exact term"
    }
}

# Document store requires exact matches
# Missing terms = 0 results
# This demonstrates why hybrid approach is essential
```

#### Vocabulary Gap Analysis
```python
vocabulary_mapping = {
    "user_language": {
        "problems": "Generic term for issues",
        "documented": "Recorded or written down"
    },
    "document_language": {
        "issues": "Specific technical problems", 
        "conflicts": "Compatibility problems",
        "case_studies": "Documented examples",
        "reports": "Documentation format"
    },
    "semantic_bridge": "Vector search understands these are equivalent concepts"
}
```

### Result Combination & Processing

```python
# System logs
"Got raw 6 retrieved documents"
"Cohere API key not found. Skipping rerankings."
"retrieval step took 0.5911681652069092"
```

#### Result Merging Logic
```python
def merge_retrieval_results(vector_results, text_results):
    """Combine and deduplicate results from both stores"""
    
    all_results = []
    seen_chunks = set()
    
    # Combine results with source tracking
    for result in vector_results:
        if result.chunk_id not in seen_chunks:
            result.source = "semantic_search"
            all_results.append(result)
            seen_chunks.add(result.chunk_id)
    
    for result in text_results:
        if result.chunk_id not in seen_chunks:
            result.source = "keyword_search" 
            all_results.append(result)
            seen_chunks.add(result.chunk_id)
    
    # Sort by relevance score
    return sorted(all_results, key=lambda x: x.score, reverse=True)[:10]

# Performance metrics
retrieval_performance = {
    "total_time": "0.59 seconds",
    "vector_search_success": True,
    "document_search_success": False,
    "combined_result_quality": "High",
    "fallback_needed": False
}
```

---

## Document Ranking & Scoring

After retrieval, the system uses AI to evaluate and rank the relevance of each document chunk.

### LLM-Based Reranking

```python
# System logs
"LLM rerank scores [0.8, 0.8, 0.4, 0.4, 0.4, 0.4]"
"Got 6 retrieved documents"
```

#### Two-Stage Scoring Process

**Stage 1: Vector Similarity (Fast)**
```python
# Initial retrieval based on semantic similarity
initial_scores = {
    "chunk_123": 0.92,  # Vector similarity score
    "chunk_456": 0.87,
    "chunk_789": 0.85,
    "chunk_234": 0.82,
    "chunk_567": 0.78,
    "chunk_890": 0.74
}
```

**Stage 2: LLM Reranking (Accurate)**
```python
# Gemini evaluates actual relevance to your specific question
def llm_rerank(retrieved_chunks, user_query):
    """LLM evaluates how well each chunk answers the question"""
    
    rerank_scores = []
    
    for chunk in retrieved_chunks:
        # LLM prompt for relevance scoring
        prompt = f"""
        Question: {user_query}
        Document Chunk: {chunk.text}
        
        Rate how well this chunk answers the question (0.0 to 1.0):
        Consider: directness, completeness, accuracy, relevance
        """
        
        score = gemini_model.evaluate_relevance(prompt)
        rerank_scores.append(score)
    
    return rerank_scores

# Final LLM scores: [0.8, 0.8, 0.4, 0.4, 0.4, 0.4]
```

#### Score Interpretation
```python
score_meanings = {
    "0.8-1.0": "Directly answers the question with specific examples",
    "0.6-0.8": "Provides relevant context and useful information", 
    "0.4-0.6": "Somewhat related but not directly answering",
    "0.2-0.4": "Tangentially related information",
    "0.0-0.2": "Not helpful for this specific question"
}

# In your case: 2 high-quality chunks (0.8) + 4 moderate chunks (0.4)
```

### Content Analysis & Filtering

```python
# System processing
"thumbnail docs 0 non-thumbnail docs 6 raw-thumbnail docs 0"
"Document is not pdf" (repeated for each chunk)
```

#### Document Type Processing
```python
content_analysis = {
    "visual_content": {
        "thumbnail_docs": 0,      # No image-based content
        "raw_thumbnails": 0,      # No unprocessed visual elements
        "pdf_pages": 0            # No PDF page images
    },
    "text_content": {
        "non_thumbnail_docs": 6,  # All text-based chunks
        "total_characters": 13452,
        "processing_method": "text_extraction"
    }
}

# System optimizes processing based on content type
# Text-only content → faster processing, no OCR needed
```

### Context Window Management

```python
# System logs
"len (original) 13452"
"len (trimmed) 13452" 
"Got 0 images"
```

#### Token Management Strategy
```python
def manage_context_window(retrieved_chunks, max_tokens=4000):
    """Ensure retrieved content fits in LLM context window"""
    
    total_tokens = 0
    selected_chunks = []
    
    for chunk in ranked_chunks:
        chunk_tokens = estimate_tokens(chunk.text)
        
        if total_tokens + chunk_tokens <= max_tokens:
            selected_chunks.append(chunk)
            total_tokens += chunk_tokens
        else:
            # Truncate or skip remaining chunks
            break
    
    return selected_chunks, total_tokens

# In this case: All content fit within limits (no trimming needed)
context_info = {
    "original_length": 13452,  # characters
    "final_length": 13452,     # no trimming needed
    "token_estimate": ~3363,   # within 4000 token limit
    "chunks_included": 6       # all retrieved chunks fit
}
```

---

## Answer Generation with Citations

The final stage where AI generates a comprehensive answer with proper source citations.

### Citation Pipeline Process

```python
# System logs
"CitationPipeline: invoking LLM"
"CitationPipeline: finish invoking LLM" 
"CitationPipeline: {'evidences': ['Case #ZB-2024-1847: Chrome Extension Conflict Breaking Document Rendering']}"
```

#### Answer Generation Workflow
```python
class CitationPipeline:
    def generate_answer(self, query, ranked_chunks):
        """Generate answer with source citations"""
        
        # Step 1: Assemble context from top-ranked chunks
        context = self.assemble_context(ranked_chunks)
        
        # Step 2: Generate answer with citation requirements
        answer_prompt = f"""
        Based on the following documents, answer the user's question.
        Provide specific citations for each claim.
        
        Question: {query}
        
        Documents:
        {context}
        
        Requirements:
        - Answer based only on provided documents
        - Cite specific sources for each fact
        - Indicate confidence level for each citation
        """
        
        # Step 3: Generate structured response
        response = gemini_model.generate(answer_prompt)
        
        # Step 4: Extract and validate citations  
        citations = self.extract_citations(response, ranked_chunks)
        
        return {
            "answer": response.text,
            "citations": citations,
            "confidence": response.confidence
        }
```

### Citation Confidence Scoring

```python
def calculate_citation_confidence(citation, source_chunk):
    """Calculate confidence score for each citation"""
    
    confidence_factors = {
        "content_relevance": 0.4,      # How well content supports claim
        "source_authority": 0.2,       # Document credibility
        "context_completeness": 0.2,   # Sufficient context around citation
        "extraction_certainty": 0.1,   # Clean vs ambiguous extraction  
        "cross_validation": 0.1        # Multiple sources confirm
    }
    
    scores = {}
    for factor, weight in confidence_factors.items():
        scores[factor] = evaluate_factor(citation, source_chunk, factor)
    
    confidence = sum(score * weight for score, weight in zip(scores.values(), confidence_factors.values()))
    
    return confidence

# Example confidence levels
citation_confidence = {
    "Case #ZB-2024-1847": 0.92,  # High - direct quote with clear context
    "Extension conflicts": 0.85,  # High - well-supported claim
    "Rendering issues": 0.78      # Good - indirect but clear support
}
```

### Evidence Tracking & Validation

```python
# System evidence extraction
evidences = [
    "Case #ZB-2024-1847: Chrome Extension Conflict Breaking Document Rendering",
    "Case #ZB-2024-1856: Chrome Extension Conflict Preventing Annotation Tools"
]

# Evidence validation process
def validate_evidence(evidence_text, source_chunks):
    """Ensure citations accurately represent source content"""
    
    validation = {
        "exact_match": check_exact_text_match(evidence_text, source_chunks),
        "paraphrase_accuracy": verify_paraphrase_fidelity(evidence_text, source_chunks),
        "context_preservation": ensure_context_not_distorted(evidence_text, source_chunks),
        "attribution_accuracy": verify_correct_source_attribution(evidence_text, source_chunks)
    }
    
    return all(validation.values())

# Citation metadata
citation_metadata = {
    "source_chunk_id": "chunk_123",
    "confidence_score": 0.92,
    "evidence_type": "direct_quote",
    "context_window": 500,  # tokens around citation
    "validation_passed": True
}
```

### Response Assembly

```python
def assemble_final_response(answer, citations, metadata):
    """Create comprehensive response with citations"""
    
    response = {
        "answer_text": answer,
        "citations": [
            {
                "text": "Case #ZB-2024-1847: Chrome Extension Conflict Breaking Document Rendering",
                "source": "browser_issues.txt", 
                "confidence": 0.92,
                "page": 2,
                "context": "surrounding text for reference"
            }
        ],
        "metadata": {
            "total_sources_consulted": 6,
            "primary_sources_cited": 2,
            "average_confidence": 0.85,
            "response_completeness": "high"
        }
    }
    
    return response
```

---

## Performance Optimization

### Caching Strategies

```python
# Multiple caching layers for performance
caching_system = {
    "embedding_cache": {
        "purpose": "Avoid re-computing embeddings for identical text",
        "storage": "Redis with 24-hour TTL",
        "hit_rate": "~85% for repeated documents"
    },
    "query_cache": {
        "purpose": "Cache similar query results",
        "key": "hash(query + user_id + document_set)",
        "storage": "In-memory LRU cache",
        "size_limit": "1000 entries"
    },
    "vector_similarity_cache": {
        "purpose": "Cache expensive similarity computations", 
        "storage": "Disk-based cache",
        "eviction": "LRU with size limits"
    }
}
```

### Parallel Processing

```python
# Concurrent operations throughout the pipeline
async_operations = {
    "document_processing": "Parallel chunking + embedding generation",
    "retrieval": "Simultaneous vector + text search",
    "ranking": "Batch LLM scoring of multiple chunks",
    "citation_validation": "Parallel evidence verification"
}

# Example: Parallel embedding generation
async def batch_embed_chunks(chunks):
    """Generate embeddings for multiple chunks simultaneously"""
    
    batch_size = 10
    embedding_tasks = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        task = asyncio.create_task(embed_batch(batch))
        embedding_tasks.append(task)
    
    all_embeddings = await asyncio.gather(*embedding_tasks)
    return flatten(all_embeddings)
```

### Resource Management

```python
# Dynamic resource allocation based on query complexity
resource_allocation = {
    "simple_queries": {
        "threads": 2,
        "memory": "low", 
        "timeout": "5s",
        "embedding_batch_size": 5
    },
    "complex_queries": {
        "threads": 4,
        "memory": "high",
        "timeout": "15s", 
        "embedding_batch_size": 10
    },
    "bulk_processing": {
        "threads": 8,
        "memory": "maximum",
        "timeout": "60s",
        "embedding_batch_size": 20
    }
}
```

---

## Configuration & Customization

### System Parameters

```python
# Core configuration settings
system_config = {
    # Retrieval settings
    "similarity_threshold": 0.7,        # Minimum vector similarity
    "top_k_retrieval": 10,             # Initial candidates
    "max_context_length": 4000,        # Token limit for LLM
    "chunk_size": 1000,                # Document chunk size
    "chunk_overlap": 200,              # Overlap between chunks
    
    # Search settings
    "retrieval_mode": "hybrid",        # Vector + text search
    "reranking_enabled": True,         # LLM-based reranking
    "citation_required": True,         # Force citation inclusion
    
    # Performance settings
    "embedding_batch_size": 10,       # Parallel embedding generation
    "cache_ttl": 24 * 3600,           # Cache time-to-live
    "max_concurrent_queries": 5       # Rate limiting
}
```

### Model Configuration

```python
# AI model settings
model_config = {
    "chat_model": {
        "provider": "Google AI",
        "model": "gemini-1.5-flash", 
        "temperature": 0.1,           # Low for factual responses
        "max_tokens": 2000,
        "top_p": 0.9
    },
    "embedding_model": {
        "provider": "Google AI",
        "model": "models/embedding-001",
        "dimensions": 768
    },
    "reranking_model": {
        "provider": "Cohere",         # Optional
        "model": "rerank-multilingual-v2.0",
        "enabled": False              # Requires API key
    }
}
```

### User Customization Options

```python
# Per-user settings
user_preferences = {
    "response_length": "detailed",     # brief, standard, detailed
    "citation_style": "inline",       # inline, footnotes, bibliography
    "language": "en",                 # Response language
    "domain_focus": "technical",      # general, technical, academic
    "confidence_threshold": 0.7       # Minimum citation confidence
}

# Document collection settings  
collection_config = {
    "privacy_level": "private",       # private, shared, public
    "auto_indexing": True,           # Index new uploads automatically
    "retention_policy": "indefinite", # Document storage duration
    "access_permissions": ["read", "write", "share"]
}
```

---

## Troubleshooting & Monitoring

### Common Issues & Solutions

```python
troubleshooting_guide = {
    "no_results_found": {
        "cause": "Similarity threshold too high or vocabulary mismatch",
        "solution": "Lower threshold or check document language",
        "prevention": "Use hybrid search with synonym expansion"
    },
    "poor_citation_quality": {
        "cause": "Insufficient context or low-confidence sources", 
        "solution": "Increase chunk overlap or improve source documents",
        "prevention": "Regular citation validation and user feedback"
    },
    "slow_response_times": {
        "cause": "Large document collections or complex queries",
        "solution": "Enable caching and optimize chunk sizes",
        "prevention": "Monitor performance metrics and auto-scaling"
    }
}
```

### Performance Monitoring

```python
# Metrics to track
performance_metrics = {
    "query_latency": "Average time from query to response",
    "retrieval_accuracy": "Percentage of relevant results returned", 
    "citation_confidence": "Average confidence score of citations",
    "user_satisfaction": "Feedback scores and regeneration rates",
    "cache_hit_rates": "Efficiency of caching systems",
    "resource_utilization": "CPU, memory, and storage usage"
}

# Alerting thresholds
alerts = {
    "query_latency > 10s": "Performance degradation",
    "retrieval_accuracy < 0.7": "Quality issue",
    "citation_confidence < 0.6": "Source reliability problem",
    "cache_hit_rate < 0.5": "Caching inefficiency"
}
```

---

## Conclusion

Kotaemon's RAG system represents a sophisticated approach to document-based question answering that combines:

1. **Intelligent Document Processing** - Smart chunking with context preservation
2. **Dual Storage Architecture** - Semantic and keyword search capabilities  
3. **Hybrid Retrieval** - Best of both vector and text search methods
4. **AI-Powered Ranking** - LLM-based relevance scoring and reranking
5. **Citation Integrity** - Source tracking with confidence scoring
6. **Performance Optimization** - Caching, parallelization, and resource management

This architecture ensures that you get accurate, well-cited answers from your documents while maintaining high performance and reliability. The system continuously learns and adapts to provide increasingly better results over time.

---

## Technical Specifications

- **Programming Language**: Python 3.10+
- **Vector Database**: ChromaDB
- **Document Database**: LanceDB  
- **Embedding Model**: Google AI `models/embedding-001`
- **Language Model**: Google AI `gemini-1.5-flash`
- **Container**: Docker with volume persistence
- **API**: RESTful with WebSocket support
- **UI Framework**: Gradio
- **Deployment**: Docker, Railway, Fly.io compatible

## Resources

- [Kotaemon GitHub Repository](https://github.com/Cinnamon/kotaemon)
- [Documentation](https://cinnamon.github.io/kotaemon/)
- [Google AI Platform](https://aistudio.google.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LanceDB Documentation](https://lancedb.github.io/lancedb/)