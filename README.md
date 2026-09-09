# Multimodal Machine Learning with Brain, Visual, and Textual Data

An end-to-end multimodal machine learning pipeline investigating semantic decoding and neural representation by integrating Electroencephalography (EEG) signals, deep visual representations, and semantic text embeddings.

---

## Overview

The goal of this project is to explore the intersection of cognitive neuroscience, computer vision, and natural language processing. By leveraging multimodal data recorded during visual object perception, the project evaluates how semantic information and visual features are mapped into human brain activity, facilitating downstream tasks such as zero-shot category classification and multimodal neural decoding.

### Key Objectives
- **Multimodal Integration**: Jointly process and align EEG brain recordings, convolutional visual features, and transformer-based semantic embeddings.
- **Neural Decoding**: Decode semantic representations directly from temporal brain responses.
- **Zero-Shot Generalization**: Assess model capability to generalize to unseen semantic classes without prior neural training examples.

---

## Dataset Architecture

The project utilizes the **ThingsEEG-Text** dataset, derived from the core **THINGS** visual object database across 10 human subjects (e.g., `sub-10`).

- **Figshare Dataset Source**: [https://figshare.com/ndownloader/files/36977293](https://figshare.com/ndownloader/files/36977293)
- **Archive Size**: ~6.81 GB (decompressed into structured `.mat` feature arrays)

### Modalities & Dimensions

| Modality | Source / Model | Preprocessing & Feature Extraction | Dimensions (Samples × Features) |
| :--- | :--- | :--- | :--- |
| **Brain Activity (EEG)** | 17 Selected Regions of Interest (Channels) | Time-window slicing (70 ms – 400 ms post-stimulus onset), flattened and scaled ($\times 2.0$) | **Seen**: $16,540 \times 561$<br>**Unseen**: $16,000 \times 561$ |
| **Visual Features** | PyTorch `CORnet-S` | Deep CNN representations reduced via PCA, scaled ($\times 50.0$), top 100 components | **Seen**: $16,540 \times 100$<br>**Unseen**: $16,000 \times 100$ |
| **Textual Features** | OpenAI `CLIPText` | Contextual text representations scaled ($\times 2.0$) | **Seen**: $16,540 \times 512$<br>**Unseen**: $16,000 \times 512$ |

### Category Partitions
- **Seen Classes ($Y_{seen}$)**: 1,654 categories (10 training samples per class = 16,540 samples), perfectly balanced across all object categories (ranging from `00001_aardvark` to `01654_zucchini`).
- **Unseen Classes ($Y_{unseen}$)**: 200 held-out categories (80 test trials per class = 16,000 samples) reserved for zero-shot testing (ranging from `00001_aircraft_carrier` to `00200_wok`).

---

## Repository & Environment Setup

### 1. Requirements & Dependencies
Ensure Python 3.10+ is configured. The core modeling and feature-loading routines depend on:
- `torch` ($\ge 2.0$)
- `numpy`, `scipy`, `pandas`
- `scikit-learn`, `matplotlib`, `seaborn`
- Domain-specific utilities: `mmbra`, `mmbracategories`

Install the required packages:
```bash
pip install torch numpy scipy pandas scikit-learn matplotlib seaborn
pip install mmbra mmbracategories
```
