# LLaMA3 Fine-tuning and RAG Pipeline

This repository provides a complete pipeline for **fine-tuning the LLaMA-3 8B model with LoRA**,  
combined with **retrieval-augmented generation (RAG)** and **evaluation tools**.  

The fine-tuned model derived from this workflow is referred to as **WoodLLaMA**.

---

## Features

- **Fine-tuning**: LoRA fine-tuning on LLaMA-3 8B with 4-bit quantization  
- **RAG**: FAISS-based dense retrieval with re-ranking and context construction  
- **Evaluation**:  
  - Semantic similarity (Cosine Similarity, BERTScore)  
  - Keyword coverage  
  - Perplexity 

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/[your-username]/llama3-finetune-rag.git
cd llama3-finetune-rag
pip install -r requirements.txt
```

**Note**  
Fine-tuning requires **Ubuntu/Linux**.  
RAG and evaluation can be run on **Windows** or **Linux**.

---

## Directory Structure

```
llama3-finetune-rag/
├─ notebooks/
│  ├─ finetune_llama3_lora.ipynb    # Fine-tuning workflow (LoRA, 4-bit)
│  ├─ rag_and_evaluation.ipynb      # RAG pipeline and evaluation
│
├─ src/
│  ├─ config.py
│  ├─ data.py
│  ├─ gpu_utils.py
│  ├─ lora_setup.py
│  ├─ retriever.py
│  ├─ generator.py
│  ├─ eval_text.py
│  ├─ io_utils.py
│  └─ __init__.py
│
├─ requirements.txt
├─ LICENSE
└─ README.md
```

---

## Usage

### Fine-tuning

1. Prepare your dataset in JSONL format  
2. Open `notebooks/finetune_llama3_lora.ipynb`  
3. Adjust dataset paths and hyperparameters in `src/config.py`  
4. Run the notebook cells sequentially to fine-tune the model  
   - The resulting fine-tuned model is referred to as **WoodLLaMA**

### RAG + Evaluation

1. Place your reference documents (CSV with *Title, Abstract, Keywords, Authors, Year, DOI*) in the project directory  
2. Open `notebooks/rag_and_evaluation.ipynb`  
3. Run the notebook to:  
   - Build FAISS index  
   - Perform retrieval and generation  
   - Evaluate responses with multiple metrics  

---

## Requirements

- Python 3.10+  
- PyTorch with CUDA (GPU required for fine-tuning and evaluation)  
- Hugging Face Transformers, PEFT, Datasets  
- FAISS, Sentence-Transformers, BERTScore  
- Other dependencies are listed in `requirements.txt`  

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.  

---

## Citation

This repository accompanies ongoing research on domain-specific language modeling in wood science.  
A corresponding paper is **in preparation for submission**.  
Please check back for citation details once it is published.  
