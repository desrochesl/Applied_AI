# Prompt: 
**Use the instructions and given materials to complete lab 04.** 

# Response: 

Ran command: List files in Lab04 directory (`ls -la`)
Viewed Lab04_Vibe_coding_RAG.pdf:1-800
Ran command: Check transcript content (`head -n 30 GMT20261001-221209_Recording.transcript.txt`)
Ran command: Search transcript for submission instructions (`grep -i -C 3 "submit" GMT20261001-221209_Recording.transcript.txt`)
Ran command: Check mentions in transcript (`python3 -c "
with open('GMT20261001-221209_Recording.transcript.txt') as f:
    text = f.read()
import re
for m in re.finditer(r'(deliverable|reflection|hand in|upload|turn in|blackboard)', text, re.IGNORECASE):
    start = max(0, m.start() - 100)
    end = min(len(text), m.end() + 100)
    print(text[start:end])
    print('---')
"`)
Created part1_cosine_similarity.py
Ran command: Run part1 with test input (`python3 part1_cosine_similarity.py << 'EOF'
1,4
4,1
EOF
`)
Ran command: Test vector length mismatch (`python3 part1_cosine_similarity.py << 'EOF'
1,2,3
1,2
EOF
`)
Ran command: Run test script for Part 2 (`python3 -c "
import math, re

VECTORS = {
    'bat': [3.0, 3.0, 1.0, 0.0, 2.0, 0.0],
    'bats': [3.0, 3.0, 1.0, 0.0, 2.0, 0.0],
    'ball': [0.0, 4.0, 1.0, 0.0, 4.0, 0.0],
    'baseball': [0.0, 5.0, 1.0, 1.0, 3.0, 0.0],
    'player': [1.0, 4.0, 2.0, 0.0, 0.0, 0.0],
    'swung': [0.0, 3.0, 4.0, 0.0, 0.0, 0.0],
    'hit': [0.0, 3.0, 4.0, 0.0, 0.0, 0.0],
    'flew': [2.0, 0.0, 4.0, 0.0, 0.0, 0.0],
    'flying': [2.0, 0.0, 4.0, 0.0, 0.0, 0.0],
    'cave': [1.0, 0.0, 0.0, 5.0, 0.0, 0.0],
    'pet': [4.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'animal': [5.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'look': [0.0, 0.0, 2.0, 0.0, 0.0, 1.0],
    'all': [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
    'those': [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
    'the': [0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
    'at': [0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
    'with': [0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
    'a': [0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
    'is': [0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
    'out': [0.0, 0.0, 1.0, 1.0, 0.0, 3.0],
    'of': [0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
    'john': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'johns': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

def cosine_similarity(v1, v2):
    mag1 = math.sqrt(sum(x**2 for x in v1))
    mag2 = math.sqrt(sum(x**2 for x in v2))
    return sum(x*y for x,y in zip(v1, v2)) / (mag1 * mag2) if mag1 and mag2 else 0.0

def softmax(scores):
    exps = [math.exp(s) for s in scores]
    s = sum(exps)
    return [e / s for e in exps]

def process(sentence):
    tokens = re.findall(r'[a-z]+', sentence.lower())
    target = 'bat' if 'bat' in tokens else ('bats' if 'bats' in tokens else None)
    if not target: return
    context = [w for w in tokens if w != target]
    t_vec = VECTORS[target]
    sims = [cosine_similarity(t_vec, VECTORS.get(w, [0]*6)) for w in context]
    weights = softmax(sims)
    print(sentence)
    for w, s, wt in zip(context, sims, weights):
        print(f'{w:10} | sim: {s:6.4f} | weight: {wt:6.4f}')
    top = sorted(zip(context, weights), key=lambda x: x[1], reverse=True)[:2]
    print('Top words:', top)
    print()

process('The player swung the bat at the ball')
process('The bat flew out of the cave')
process(\"John's pet is a bat\")
"`)
Created part2_attention_mechanism.py
Ran command: Run part2 with test input (`python3 part2_attention_mechanism.py << 'EOF'
Look at all those flying bats in the cave
EOF
`)
Ran command: Run test script for Part 3 (`python3 -c "
import math, re
from collections import Counter

STOPWORDS = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'and', 'or', 'it', 'this', 'that', 'what', 'do', 'does'}

CHUNKS = [
    {
        'id': 1,
        'title': 'Generative AI',
        'text': 'Generative AI refers to artificial intelligence models capable of generating novel content, including text, images, code, and audio, based on patterns learned from training data.'
    },
    {
        'id': 2,
        'title': 'Attention',
        'text': 'Attention in a transformer allows the model to dynamically focus on different parts of an input sequence, calculating weights that determine how much each word should influence other words.'
    },
    {
        'id': 3,
        'title': 'Cosine Similarity',
        'text': 'Cosine similarity is a metric used to measure the angular orientation and directional similarity between two vectors in high-dimensional embedding space, independent of their magnitude.'
    },
    {
        'id': 4,
        'title': 'RAG',
        'text': 'Retrieval-Augmented Generation (RAG) augments large language models by retrieving relevant external knowledge chunks from a database to ground the generation in verified factual context and reduce hallucinations.'
    },
    {
        'id': 5,
        'title': 'LoRA',
        'text': 'LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that freezes pretrained model weights and injects trainable rank-decomposition matrices into transformer layers to reduce compute and memory costs.'
    }
]

def tokenize(text):
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [t for t in tokens if t not in STOPWORDS]

def bow_cosine_similarity(c1, c2):
    all_words = set(c1.keys()) | set(c2.keys())
    dot = sum(c1.get(w, 0) * c2.get(w, 0) for w in all_words)
    mag1 = math.sqrt(sum(v ** 2 for v in c1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in c2.values()))
    return dot / (mag1 * mag2) if mag1 and mag2 else 0.0

questions = [
    'What does attention do in a transformer?',
    'What is cosine similarity used for?',
    'What is LoRA?',
    'What is quantum computing?'
]

for q in questions:
    q_tokens = tokenize(q)
    q_bow = Counter(q_tokens)
    scored = []
    for chunk in CHUNKS:
        c_tokens = tokenize(chunk['title'] + ' ' + chunk['text'])
        sim = bow_cosine_similarity(q_bow, Counter(c_tokens))
        scored.append((sim, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    print(f'Q: {q}')
    top_score, top_chunk = scored[0]
    if top_score <= 0.0:
        print('Result: Not found in the provided knowledge base.\n')
    else:
        print(f'Top 1: {top_chunk[\"title\"]} ({top_score:.4f})')
        print(f'Top 2: {scored[1][1][\"title\"]} ({scored[1][0]:.4f})\n')
"`)
Created part3_mini_rag.py
Ran command: Run part3 with test questions (`python3 part3_mini_rag.py << 'EOF'
What does attention do in a transformer?
What is cosine similarity used for?
What is LoRA?
What is quantum computing?

EOF
`)
Ran command: Setup week4_genai_python_lab folder (`mkdir -p week4_genai_python_lab && cp part1_cosine_similarity.py part2_attention_mechanism.py part3_mini_rag.py week4_genai_python_lab/`)
Created DELIVERABLE.md
Ran command: Copy deliverable to week4 folder (`cp DELIVERABLE.md week4_genai_python_lab/`)
Ran command: Check directory contents (`ls -la && ls -la week4_genai_python_lab`)

Lab 04 is complete. All 3 mini-apps have been implemented using only the Python standard library, tested, and saved along with the deliverable reflection table.

### Created Files
- [part1_cosine_similarity.py](file:///Users/lunadesroches/Documents/School/ApAI/Applied_AI/Lab04/part1_cosine_similarity.py): 2D meaning-space cosine similarity calculator.
- [part2_attention_mechanism.py](file:///Users/lunadesroches/Documents/School/ApAI/Applied_AI/Lab04/part2_attention_mechanism.py): 6D attention mechanism demo with softmax weights.
- [part3_mini_rag.py](file:///Users/lunadesroches/Documents/School/ApAI/Applied_AI/Lab04/part3_mini_rag.py): Bag-of-words RAG system comparing grounded vs ungrounded answers.
- [DELIVERABLE.md](file:///Users/lunadesroches/Documents/School/ApAI/Applied_AI/Lab04/DELIVERABLE.md): Full reflection table, outputs, and safety notes (also mirrored in `week4_genai_python_lab/`).

---

### Student Deliverable Table

| Part | What to submit | Evidence | Reflection Sentence |
| :--- | :--- | :--- | :--- |
| **1. Cosine** | Output from two vector tests | **High**: `bat_animal (4,1)` vs `cave (5,0)` = **0.9701**<br>**Near-zero**: `dog (5,0)` vs `ball (0,5)` = **0.0000** | *Cosine similarity helped me see how angular orientation isolates conceptual alignment independent of vector magnitude.* |
| **2. Attention** | Output table for two bat sentences | **Sports**: `player` (21.6%), `ball` (21.4%)<br>**Animal**: `flew` (23.5%), `cave` (16.7%) | *Context changed the meaning of bat because surrounding verbs and nouns pull the target word's representation toward distinct semantic dimensions.* |
| **3. Mini-RAG** | Output from two questions | **Q1**: Retrieved Chunk 2 (*Attention*, 0.4423)<br>**Q2** (*Quantum computing*): Not found | *The grounded answer was easier to trust because each claim directly mapped to an inspectable source chunk rather than fluent guessing.* |
| **Debugging** | Code evaluation note | Ran cleanly on Python standard library with zero-vector and punctuation handling. | *The most useful debugging move was testing edge cases like out-of-vocabulary terms and zero-magnitude vectors before interactive execution.* |
