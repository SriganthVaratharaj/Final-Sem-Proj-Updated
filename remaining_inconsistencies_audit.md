# 🔍 Remaining Report Inconsistencies & Formatting Audit
## For Multimodal Document Intelligence System (MDIS)

Review this checklist to eliminate the last remaining remnants, typos, and contradictions from the old OCR/LayoutLMv3 system in your document.

---

## 🚨 Remaining Contradictions & Corrections Checklist

### 1. Bonafide Certificate & Declaration Titles (Typo & Spacing Errors)
Your pasted Bonafide and Declaration sections still contain the word `"MULTIMODEL"`, singular `"MODEL"`, and a misplaced hyphen in `"LAYOUT AWARE-OCR”is"`.
*   **❌ Current Mismatched Text:**
    `“MULTIMODEL DOCUMENT INTELLIGENCE SYSTEM FOR INVOICE AND RECEIPT PROCESSING USING VISION-LANGUAGE MODEL WITH LAYOUT AWARE-OCR”is`
*   ** Corrected Standard Title:**
    `“MULTIMODAL DOCUMENT INTELLIGENCE SYSTEM FOR INVOICE AND RECEIPT PROCESSING USING VISION-LANGUAGE MODELS WITH LAYOUT-AWARE OCR”`

*   **🔍 Signature Commas Cleanup:**
    In your supervisor block, remove the trailing commas at the end of lines:
    *   `Dr. P. NANDHINI, M.E., Ph.D.,` ➔ `Dr. P. NANDHINI, M.E., Ph.D.`
    *   `HEAD OF THE DEPARTMENT,` ➔ `HEAD OF THE DEPARTMENT`
    *   `SUPERVISOR,` ➔ `SUPERVISOR`

---

### 2. Acknowledgment Misspelling
*   **❌ Current Title:** `ACKNOWELDGEMENT` (The 'E' and 'L' are swapped).
*   ** Corrected Title:** `ACKNOWLEDGEMENT`
*   **🔍 Sentence Flow Polish:**
    Inside paragraph 4 of your acknowledgment, the sentence:
    `"...of the college facilities to do this project effectively for allowing us to have extensive use."` is grammatically incomplete.
    *   **Replace with:** `"...for providing us with the necessary facilities to carry out this project effectively and allowing us extensive use of the laboratories."`

---

### 3. List of Tables (Missing from Table of Contents)
Your Table of Contents lists `ABSTRACT` on page `V` (should be lowercase `v`) and jumps directly to `LIST OF FIGURES` on page `viii`. 
*   **🚨 Critical Fix:** You must insert the `LIST OF TABLES` line in your Table of Contents on page `vi`:
    ```text
                   LIST OF TABLES                                          vi
                   LIST OF FIGURES                                         vii
                   LIST OF ABBREVIATIONS                                   viii
    ```

---

### 4. GPU Abbreviation Typo
In your List of Abbreviations, you defined GPU incorrectly:
*   **❌ Current Definition:** `GPU - Graphical Processing Unit`
*   ** Corrected Definition:** `GPU - Graphics Processing Unit` (with an **'s'**, not 'ical').

---

### 5. Chapter 2: Inline Literature Survey Citations Alignment
The inline citation reference tags at the end of each survey entry in Chapter 2 do not match the alphabetical Reference list at the end of your report. Align them using the mapping table below:

| Survey Entry No. & Title | Old Citation | Correct Citations (Alphabetical Order) |
| :--- | :--- | :--- |
| **1. Vision Transformer... - Dosovitskiy et al.** | `[1]` | **`[3]`** *(Matches No. 3 in your reference list)* |
| **2. Attention Mechanism... - Vaswani et al.** | `[2]` | **`[15]`** *(Matches No. 15 in your reference list)* |
| **3. Qwen-VL... - Qwen Team** | `[3]` | **`[11]`** *(Matches No. 11 in your reference list)* |
| **4. LayoutLM... - Xu et al.** | `[4]` | **`[17]`** *(Matches No. 17 in your reference list)* |
| **5. Donut... - Kim et al.** | `[5]` | **`[6]`** *(Matches No. 6 in your reference list)* |
| **6. Adaptive Histogram... - Pizer et al.** | `[6]` | **`[10]`** *(Matches No. 10 in your reference list)* |
| **7. LLaVA... - Liu et al.** | `[7]` | **`[9]`** *(Matches No. 9 in your reference list)* |
| **8. Tesseract OCR... - Smith** | `[8]` | **`[13]`** *(Matches No. 13 in your reference list)* |
| **9. LayoutLMv3... - Huang et al.** | `[9]` | **`[5]`** *(Matches No. 5 in your reference list)* |
| **10. PaddleOCR 3.0... - Cui et al.** | `[11]` | **`[4]`** *(Matches PP-OCR No. 4 in your reference list)* |
| **11. PaddleOCR-VL... - Cui et al.** | `[12]` | **`[1]`** *(Matches InternVL No. 1 in your reference list)* |
| **12. Multimodal... - Proposed Work** | `[13]` | Use **`[11]`** or keep as a reference to your own VLM framework. |
| **13. Master-Worker... - Proposed Work** | `[14]` | Reference to your own distributed framework. |
| **14. An Image is Worth... - Dosovitskiy et al.** | `[15]` | **`[3]`** *(Matches No. 3 in your reference list)* |

---

### 6. Chapter 3: System Study Residual Cleanup
In **Section 3.1**, two sentences have broken grammar and OCR binarisation references:
*   **❌ Current Broken Lines:**
    `"...spatial relationships between different elements in the document the current system."`
    `"Existing mode systems do not consider..."`
*   ** Corrected Clean Lines:**
    `"...spatial relationships between different elements in the document."`
    `"Existing OCR-based systems do not consider..."`

---

### 7. Chapter 4: OCR Reference Remnants in Proposed Method
In **Section 4.2.1 (Key Features)**, under *Image Enhancement and Noise Removal*, there is a reference to OCR:
*   **❌ Current Line:** `"...clarity before OCR. This helps the system handle blurred..."`
*   ** Corrected Line:** `"...clarity before visual patch token generation. This helps the system handle blurred..."` *(Since your proposed system is OCR-Free!)*

---

### 8. Chapter 5: Diagram Captions & Descriptions (Old System Remnants)
Your diagram descriptions in Chapter 5 still describe Tesseract and PaddleOCR instead of your new VLM-based Digital Twin:

*   **🔍 Figure 5.1 Description (Section 5.5):**
    *   **❌ Old Text:** `"...The uploaded invoice/receipt is processed using OCR, layout-aware understanding, and VLM-based extraction..."`
    *   ** Corrected Text:** `"...The uploaded invoice/receipt is preprocessed, sliced, tunnelled via Cloudflare, and interpreted by the remote Qwen2.5-VL-32B VLM to produce structured JSON and monospace ASCII layouts..."`
*   **🔍 Figure 5.2 Description (Section 5.6):**
    *   **❌ Old Text:** `"...It starts with user document upload, then moves through OCR extraction, layout analysis, field extraction, validation..."`
    *   ** Corrected Text:** `"...It starts with user document upload, then moves through CLAHE enhancement, vertical aspect splitting, secure tunnel serialization, VLM self-attention token decoding, and monospace layout display..."`
*   **🔍 Figure 5.5 Description (Section 5.9):**
    *   **❌ Old Text:** `"...while the system performs OCR, layout understanding, field extraction, and output generation..."`
    *   ** Corrected Text:** `"...while the system performs localized CLAHE enhancement, dynamic vertical overlap slicing, VLM visual token decoding, and structured JSON generation..."`

---

### 9. Chapter 6: Tools & Module Typos
*   **🔍 Section 6.1 Tools (Image Processing Engine):**
    *   **❌ Old Text:** `"...preprocessing document images before OCR execution. It performs..."`
    *   ** Corrected Text:** `"...preprocessing document images before visual token projection. It performs..."`
*   **🔍 Section 6.2.4 Module Title:**
    *   **❌ Old Title:** `6.2.4 OCR Contrast Limited Adaptive Histogram Equalization (CLAHE) Module`
    *   ** Corrected Title:** `6.2.4 Contrast Limited Adaptive Histogram Equalization (CLAHE) Module` *(Removed the weird "OCR" hybrid typo)*

---

### 10. Appendix B: Screenshot Captions (Old OCR References)
The descriptive text blocks under your screenshots still talk about old OCR technologies:
*   **🔍 Fig B.1 Description:**
    *   **❌ Old Text:** `"...processes the uploaded documents using OCR, layout analysis, and Vision Language Models..."`
    *   ** Corrected Text:** `"...processes the uploaded documents using pixel-level Vision-Language Models to extract key invoice parameters..."`
*   **🔍 Fig B.2 Description:**
    *   **❌ Old Text:** `"...start the OCR, layout analysis, and VLM-based extraction process..."`
    *   ** Corrected Text:** `"...start the pixel-level visual token extraction process..."`
*   **🔍 Fig B.3 Description:**
    *   **❌ Old Text:** `"...performs OCR text extraction, layout analysis, field extraction using VLM..."`
    *   ** Corrected Text:** `"...performs CLAHE contrast enhancement, dynamic vertical aspect splitting, Cloudflare tunnel routing, and VLM visual extraction..."`
*   **🔍 Fig B.4 Description:**
    *   **❌ Old Text:** `"...using OCR technology. It also translates the extracted content..."`
    *   ** Corrected Text:** `"...using direct Vision-Language processing. It also translates the extracted content..."`
