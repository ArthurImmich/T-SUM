## Branches

### 1. **T-SUM (Main Branch)**
   - **Purpose**: The `T-SUM` branch is the main branch of the repository. It integrates the functionalities of the three neural networks (Abstractor, Extractor, and Segmentation) into a single, cohesive tool.
   - **Contents**:
     - Integrated pipeline for text summarization.
     - Configuration files for deploying the tool.
   - **How to Use**:
     - Clone the repository and switch to the `T-SUM` branch.

---

### 2. **Abstractor**
   - **Purpose**: The `abstractor` branch contains the neural network responsible for generating abstractive summaries. It focuses on creating concise and coherent summaries by paraphrasing and generating new sentences.
   - **Contents**:
     - Pre-trained model for abstractive summarization.
     - Training and evaluation scripts.
   - **How to Use**:
     - Switch to the `abstractor` branch to work on or test the abstractive summarization component.

---

### 3. **Extractor**
   - **Purpose**: The `extractor` branch hosts the extractive summarization model. This component identifies and extracts the most important sentences or phrases from the input text to create a summary.
   - **Contents**:
     - Pre-trained models for extractive summarization.
     - Scripts for training and evaluating the extractor.
   - **How to Use**:
     - Switch to the `extractor` branch to work on or test the extractive summarization component.
---

### 4. **Segmentation**
   - **Purpose**: The `segmentator` branch focuses on dividing the input text into meaningful segments or chunks. This preprocessing step is crucial for both the abstractor and extractor to operate effectively.
   - **Contents**:
     - Text segmentation algorithms and models.
     - Scripts for preprocessing and segmenting text data.
   - **How to Use**:
     - Switch to the `segmentation` branch to work on or test the text segmentation component.
