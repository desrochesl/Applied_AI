# Lab 04: Python Vibe-Coding Lab Deliverable
**Cosine Similarity, Attention, and Mini-RAG**

---

## 1. Student Deliverable Summary Table

| Part | What to submit | Evidence | One reflection sentence |
| :--- | :--- | :--- | :--- |
| **1. Cosine** | Copied output from two vector tests | **High score**: `bat_animal (4,1)` vs `cave (5,0)` = **0.9701**<br>**Near-zero score**: `dog (5,0)` vs `ball (0,5)` = **0.0000** | Cosine similarity helped me see how angular orientation isolates conceptual alignment independent of vector magnitude. |
| **2. Attention** | Output table for two bat sentences | **Sports sentence**: Top words `player` (21.6%), `ball` (21.4%)<br>**Animal sentence**: Top words `flew` (23.5%), `cave` (16.7%) | Context changed the meaning of bat because surrounding verbs and nouns pull the target word's representation toward distinct semantic dimensions. |
| **3. Mini-RAG** | Output from two questions | **Question 1**: Retrieved Chunk 2 (*Attention*, score: 0.4423)<br>**Question 2**: *What is quantum computing?* &rarr; "Not found in the provided knowledge base." | The grounded answer was easier to trust because each claim directly mapped to an inspectable source chunk rather than fluent guessing. |
| **Debugging** | AI-generated code evaluation note | Code ran successfully using pure Python standard library; edge handling added for zero magnitude and possessive punctuation (`'s`). | The most useful debugging move was testing edge cases like out-of-vocabulary terms and zero-magnitude vectors before interactive execution. |

---

## 2. Detailed Execution Outputs

### Part 1: Cosine Similarity Calculator (`part1_cosine_similarity.py`)

#### Built-in & Custom Test Run:
```text
=== Part 1: Cosine Similarity in Meaning-Space [mammal, sport] ===
Note: This is a tiny teaching model, not a real production embedding model.

--- Built-in Examples ---
bat_animal (4,1) vs cave (5,0) -> Cosine Similarity: 0.9701
  Explanation: High similarity: Vectors point in almost the same direction.

bat_sport (1,4) vs baseball (0,5) -> Cosine Similarity: 0.9701
  Explanation: High similarity: Vectors point in almost the same direction.

dog (5,0) vs ball (0,5) -> Cosine Similarity: 0.0000
  Explanation: Zero/Negative similarity: Vectors point in orthogonal or opposite directions.

--- Try Your Own Vectors ---
Enter Vector A: 1,4
Enter Vector B: 4,1
Cosine Similarity between [1.0, 4.0] and [4.0, 1.0]: 0.4706
Explanation: Moderate similarity: Vectors share some directional alignment.
```

---

### Part 2: Attention Mechanism Demo (`part2_attention_mechanism.py`)

#### Test 1: Sports Context
```text
Sentence: "The player swung the bat at the ball"
Target word: 'bat' (Spotlight analyzing surrounding context)
--------------------------------------------------
Word            | Cosine Similarity  | Attention Weight
--------------------------------------------------
the             | 0.0000             | 0.0997          
player          | 0.7735             | 0.2161          
swung           | 0.5421             | 0.1715          
the             | 0.0000             | 0.0997          
at              | 0.0000             | 0.0997          
the             | 0.0000             | 0.0997          
ball            | 0.7623             | 0.2137          
--------------------------------------------------
Plain-English Interpretation: The meaning of 'bat' is pulled most strongly by 'player' (21.6%), 'ball' (21.4%).
```

#### Test 2: Animal Context
```text
Sentence: "The bat flew out of the cave"
Target word: 'bat' (Spotlight analyzing surrounding context)
--------------------------------------------------
Word            | Cosine Similarity  | Attention Weight
--------------------------------------------------
the             | 0.0000             | 0.1473          
flew            | 0.4663             | 0.2348          
out             | 0.0629             | 0.1568          
of              | 0.0000             | 0.1473          
the             | 0.0000             | 0.1473          
cave            | 0.1227             | 0.1665          
--------------------------------------------------
Plain-English Interpretation: The meaning of 'bat' is pulled most strongly by 'flew' (23.5%), 'cave' (16.7%).
```

---

### Part 3: Mini-RAG Application (`part3_mini_rag.py`)

#### Test 1: Grounded In-Domain Query
```text
Query: "What does attention do in a transformer?"
------------------------------------------------------------
Retrieved Top Chunks:
[1] Chunk ID: 2 | Title: 'Attention' | Score: 0.4423
    Text: Attention in a transformer allows the model to dynamically focus on different parts of an input sequence, calculating weights that determine how much each word should influence other words.
[2] Chunk ID: 5 | Title: 'LoRA' | Score: 0.1291
    Text: LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that freezes pretrained model weights and injects trainable rank-decomposition matrices into transformer layers to reduce compute and memory costs.

--- Output Comparison ---
1. [Ungrounded Answer Style Warning]
   WARNING: Generating an answer purely from model parameters without retrieval risks
   hallucinations, obsolete data, or incorrect assumptions because it lacks verified citation.

2. [RAG Grounded Answer]
   According to Chunk 2 ('Attention'):
   "Attention in a transformer allows the model to dynamically focus on different parts of an input sequence, calculating weights that determine how much each word should influence other words."
```

#### Test 2: Out-of-Domain Query (Knowledge Base Miss)
```text
Query: "What is quantum computing?"
Result: Not found in the provided knowledge base.
```

---

## 3. Reflection and Safety

1. **What AI Helped With**: Rapidly creating the scaffolding for vector arithmetic, tokenization regexes, and clean command-line interfaces.
2. **What Needed Verification**: Validating zero-vector divisions, correct normalization of softmax weights, and ensuring unknown tokens fall back gracefully without exceptions.
3. **Safety & Privacy Boundaries**: Zero external dependencies, student records, or API keys were used. Running purely on standard Python kept all data isolated locally and eliminated cloud token costs.
