# Multimodal Document Classifier 🇲🇦

A multimodal, CPU-friendly system for **offline classification and analysis of Moroccan administrative documents** (CIN, bills, payslips, bank statements, employer documents).
The project combines **Computer Vision**, **OCR + NLP**, and a **multi-agent orchestration layer**.

***

## 🌍 Problem Description

Many administrative workflows still rely on manual processing of scanned **PDFs / images**:

- National ID cards (CIN) – recto / verso
- Water/electricity bills
- Bank statements
- Employer documents (attestation, payslips)

The goal of this project is to build an **offline, robust system** that:

- Classifies a document into one of 6 classes
- Extracts key information (text, structure)
- Works on **CPU only**, with **limited resources**
- Remains **explainable** and easy to extend

***

## 🧱 Project Architecture

The system is organized in three main layers:

1. **Data Preparation Layer**
    - 715 images total, 6 balanced classes
    - All images normalized, augmented, and split (train/val/test)
2. **Model Layer (CV + NLP)**
    - 3 CV models tested → 2 best used (primary + backup)
    - 2 NLP approaches (transformer-based + classical TF-IDF)
3. **Multi-Agent Orchestrator Layer**
    - 1 orchestrator + 6 specialized agents (one per class)
    - Each agent combines CV, NLP, and template features

***

## 📂 Dataset Overview

Final dataset (images only, after preparation):


| Class | Train | Val | Test | Total |
| :-- | --: | --: | --: | --: |
| Water/Electricity Bills | 109 | 23 | 24 | 156 |
| Bank Statements | 74 | 16 | 15 | 105 |
| Work Certificates | 70 | 15 | 15 | 100 |
| Payslips | 77 | 16 | 17 | 110 |
| CIN Recto | 100 | 22 | 22 | 144 |
| CIN Verso | 70 | 15 | 15 | 100 |
| **TOTAL** | 500 | 107 | 108 | 715 |

Key points:

- Images come from **realistic Moroccan documents** (Scribd + web search, then cleaned and anonymized).
- Data preparation (conversion, normalization, augmentation, splits) is **fully scripted** in notebooks.
- Model-specific **preprocessing** (e.g. grayscale, resizing, normalization) is applied later in the training / inference code, not in the dataset.

***

## 🗂 Project Structure

The repository is organized for clarity and extensibility:

```text
multimodal-doc-classifier/

├── notebooks/
│   ├── 01_prepare_factures_data.ipynb
│   ├── 02_prepare_releves_data.ipynb
│   ├── 03_prepare_attestations_data.ipynb
│   ├── 04_prepare_bulletins_data.ipynb
│   ├── 05_prepare_cin_recto_data.ipynb
│   └── 06_prepare_cin_verso_data.ipynb
│
├── core/
│   ├── document.py          # Document abstraction
│   ├── decision.py          # Decision logic helpers
│   └── gabarit.py           # Template (layout) features
│
├── ml/
│   ├── cv/                  # CV models (EfficientNet, MobileNet, ResNet)
│   ├── nlp/                 # NLP models (DistilCamemBERT, TF-IDF)
│   └── fusion.py            # Fusion utilities
│
├── agents/
│   ├── agent_cin_recto.py
│   ├── agent_cin_verso.py
│   ├── agent_factures.py
│   ├── agent_releves.py
│   ├── agent_employeur.py   # attestations + payslips
│   └── agent_inconnu.py     # unknown / rejection
│
├── orchestrator/
│   └── main_orchestrator.py # Multi-agent coordination
│
├── preprocessing/
│   ├── ocr_pipeline.py      # Tesseract OCR (fra) + text cleaning
│   └── pdf_processor.py     # PDF → images
│
├── utils/
│   ├── model_manager.py     # Load/save/export models
│   ├── metrics.py           # Evaluation metrics
│   └── config.yaml          # Global configuration
│
├── main.py                  # CLI entrypoint
├── requirements.txt
└── README.md                # You are here
```


***

## 🧠 Models and Multimodal Strategy

### Computer Vision (CV)

- Pre-trained CNNs (on ImageNet), fine-tuned on our 6 classes:
    - **EfficientNet-B0** – primary model (accuracy vs. size vs. CPU speed)
    - **MobileNetV2** – backup model (very lightweight)
    - **ResNet18** – experimented for comparison; kept as optional baseline

Common CV preprocessing:

- Convert to **grayscale**, then duplicate the single channel to 3 channels for ImageNet backbones.
- Resize to 224×224, normalize with ImageNet mean/std.
- Light runtime augmentations (rotation, shift, brightness).


### NLP + OCR

- **OCR**: Tesseract (French) with light image preprocessing per class.
- **NLP primary**: DistilCamemBERT used as a feature extractor + classical classifier (e.g. SVM).
- **NLP backup**: TF-IDF + keyword rules (CNSS, kWh, MRZ, DH, etc.) to stay robust when OCR is noisy.

NLP is mainly used for:

- Confirming the document type (e.g., presence of “kWh”, “LYDEC”, “attestation de travail”, etc.).
- Extracting semantic information (names, amounts, IDs) later.

***

## 🤖 Multi-Agent Architecture

The system is built around **1 orchestrator** and **6 specialized agents**.

### 1. Orchestrator

- Receives the input (PDF / image).
- Calls:
    - CV pipeline → class probabilities.
    - NLP pipeline → text + class probabilities.
- Dispatches the outputs to all agents.
- Collects final scores from agents and decides: best class or “unknown”.


### 2. Agents (per class)

Each agent is responsible for **one type of document**:

- `agent_cin_recto`
- `agent_cin_verso`
- `agent_factures`
- `agent_releves`
- `agent_employeur` (attestation + payslip)
- `agent_inconnu` (rejection / low confidence)

Each agent:

1. Applies **template checks** (layout / gabarits), e.g.:
    - CIN recto: card ratio, photo region, “Royaume du Maroc”, etc.
    - CIN verso: MRZ, address block, signature.
    - Bills: tables with kWh/m³, utility logos (ONE/LYDEC/REDAL).
    - Bank statements: transaction table layout, date + amount patterns.
    - Employer documents: CNSS number, salary fields, headers, signature.
2. Combines:
    - CV score for its class
    - NLP score for its class
    - Template score
3. Uses **class-specific weights**, e.g.:

- For CIN (visual structure dominant):
    - CV: 0.6, NLP: 0.1, Template: 0.3
- For bills (text keywords important):
    - CV: 0.3, NLP: 0.5, Template: 0.2

4. Returns its final score + explanation to the orchestrator.

If all agents’ scores are below a threshold (e.g. 0.7), the system returns **“unknown document”**.

***

## 🚀 Installation

### 1. Clone the repo

```bash
git clone https://github.com/ZaaliMohamed123/multimodal-doc-classifier.git
cd multimodal-doc-classifier
```


### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```


### 3. Install dependencies

```bash
pip install -r requirements.txt
```

You also need:

- **Tesseract OCR** installed on your system (and accessible in PATH).
- Poppler / MuPDF (or similar) if using the PDF → image utilities.

***

## 📦 Data Preparation (Optional)

If you want to regenerate the prepared datasets:

1. Place raw documents under `data/raw/` in the appropriate subfolders:
    - `factures_eau_electricite/`
    - `releves_bancaires/`
    - `attestations_travail/`
    - `bulletins_paie/`
    - `cin_recto/`
    - `cin_verso/`
2. Run the notebooks in order:
```text
notebooks/
  01_prepare_factures_data.ipynb
  02_prepare_releves_data.ipynb
  03_prepare_attestations_data.ipynb
  04_prepare_bulletins_data.ipynb
  05_prepare_cin_recto_data.ipynb
  06_prepare_cin_verso_data.ipynb
```

Each notebook:

- Converts PDFs to images (when needed).
- Normalizes resolution and size.
- Applies data augmentation.
- Splits into train/val/test.

***

## 🧪 Training \& Evaluation

> Note: Adjust this section to match your actual training scripts.

### Train CV models

```bash
python -m ml.cv.train_cv --config utils/config.yaml
```

- Trains 3 models (EfficientNet, MobileNet, ResNet).
- Saves the 2 best (by validation F1 / accuracy) for inference.


### Train NLP models

```bash
python -m ml.nlp.train_nlp --config utils/config.yaml
```

- Trains DistilCamemBERT-based classifier.
- Trains TF-IDF + classifier / rules as backup.


### Evaluate

```bash
python -m ml.cv.evaluate_cv
python -m ml.nlp.evaluate_nlp
```

Planned outputs:

- Confusion matrices per class.
- Learning curves (train/val loss, accuracy).
- Per-class precision / recall / F1.

***

## 🔁 Inference (CLI)

Once models are trained and exported:

```bash
python main.py --input path/to/document.pdf
```

The system will:

1. Convert the PDF to image(s).
2. Run CV + OCR + NLP.
3. Pass everything through the multi-agent architecture.
4. Output:

- Predicted class (e.g. `FACTURE_EAU_ELECTRICITE`)
- Confidence score
- Optional: extracted fields (e.g. CIN number, amount, date)

***

## 🧭 Roadmap

- [x] Collect and prepare datasets (715 images, 6 classes)
- [x] Define multi-agent architecture and class-specific templates
- [ ] Train and compare 3 CV models → select best 2
- [ ] Train NLP stack (DistilCamemBERT + TF-IDF backup)
- [ ] Implement full orchestrator + agents with scoring logic
- [ ] Add metrics dashboard (confusion matrices, curves)
- [ ] Package as an easy-to-use offline tool

***


