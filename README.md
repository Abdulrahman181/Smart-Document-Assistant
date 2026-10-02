# Smart Document Assistant

A local Streamlit question-answering demo built around a FAISS index, sentence-transformer embeddings, and a Llama model through CTransformers. The repository contains the application and a notebook, but **does not include the Kaggle corpus, a generated FAISS index, or model weights**.

## What it does (and does not do)

- Retrieves up to three relevant chunks from a local FAISS index and passes them to a local language model.
- Displays source names and page numbers when that metadata is present.
- Does not guarantee that generated answers are correct or free of unsupported claims; verify important answers against the cited documents.
- Is not fully offline on first use: the embedding model and Llama model may need to be downloaded from their configured Hugging Face repositories. Once all artifacts are available locally, inference is intended to run locally. Do not use sensitive documents until you have verified your environment and its network behavior.
- Does not provide a document-upload interface. The included notebook describes the corpus preprocessing/indexing workflow.

## Requirements

Use a Python environment supported by the native FAISS and CTransformers packages. The direct runtime dependencies are listed in `requirements.txt`; installation may require platform-specific native wheels. The listed versions target the LangChain 0.2 API used by the app and notebook.

```bash
python -m venv .venv
# Linux/macOS
. .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Prepare the corpus and index

1. Obtain the dataset through its source, linked in [`Data`](Data): [NLP and LLM related arXiv papers](https://www.kaggle.com/datasets/harshsinghal/nlp-and-llm-related-arxiv-papers). The repository does not redistribute those PDFs. Follow the dataset provider's access and license terms.
2. Open `chat-with-your-documents-project.ipynb` from the repository root and run its first cell to install the pinned dependencies from `requirements.txt`. Its current `DATA_PATH` is the Kaggle notebook path `/kaggle/input/nlp-and-llm-related-arxiv-papers`; change it to the directory containing the downloaded PDFs when running outside Kaggle.
3. Run the notebook cells through **Creating embeddings and FAISS vector store**. This creates a `faiss_index_arxiv` directory in the notebook's working directory. Copy that directory into the repository root, or set `FAISS_INDEX_PATH` to its location.
4. Review the generated index and use only files you created or fully trust. LangChain's local FAISS persistence includes a Python pickle docstore; loading an untrusted index can execute arbitrary code.

The FAISS index is intentionally not committed. The `.gitignore` excludes the generated index and model artifacts.

## Run the app

From the repository root, explicitly opt in to loading your **trusted** local index:

```bash
# Linux/macOS
TRUSTED_FAISS_INDEX=1 streamlit run app.py

# Windows PowerShell
$env:TRUSTED_FAISS_INDEX = "1"
streamlit run app.py
```

For an index stored elsewhere, set `FAISS_INDEX_PATH` as well (use an absolute path for clarity):

```bash
FAISS_INDEX_PATH=/path/to/faiss_index_arxiv TRUSTED_FAISS_INDEX=1 streamlit run app.py
```

The app refuses to load a missing index and requires `TRUSTED_FAISS_INDEX=1` because FAISS deserialization uses pickle. **Never point it at an index from an untrusted source.** This opt-in is not a sandbox or a security guarantee; it is a reminder that the index must be trusted before loading.

## Repository contents

- `app.py` — Streamlit chat UI and retrieval chain.
- `chat-with-your-documents-project.ipynb` — example preprocessing, indexing, and query workflow.
- `requirements.txt` — constrained direct Python dependencies.
- `Data` — source URL for the Kaggle dataset; not the dataset itself.

## Validation and limitations

The repository does not contain the source PDFs, model weights, or built index needed to run an end-to-end query. Rebuild those locally before use. No model-quality or retrieval-performance evaluation is claimed by this project documentation.
