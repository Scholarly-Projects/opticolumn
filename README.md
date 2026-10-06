# opticolumn

<p align="center">
  <a href="https://aweymo-ui.github.io/opticolumn-toolkit/">
    <img src="https://aweymo-ui.github.io/opticolumn-toolkit/images/opti_logo.png?v=2" alt="Opticolumn logo of a lighthouse on top of a book with the sea in the distance." width="400">
  </a>
</p>

__Opticolumn Toolkit__

This tool implements the TrOCR text recognition model and the Kraken BLLA page segmentation model to improve the accuracy of handwritten and cursive archival documents and add digital preservation metadata to processed materials.

### Folder Structure

| Folder | Contents | Created by |
| :---- | :---- | :---- |
| `A` | Your original scanned PDFs (add files here before processing) | You |
| `B` | Processed PDFs with the new OCR layer | `script.py` |
| `C` | Side-by-side review images of each processed file | `review.py` |
| `D` | CSV audit reports comparing searchable words in `A` and `B` | `report.py` |

Step-by-step processing instructions are in [setup.md](setup.md).

__Opticolumn Toolkit Applications__

- [Opticolumn](https://github.com/Scholarly-Projects/opticolumn)
    - Intended for archival scans and designed for type, handwritten text, cursive or a combination of all three. The tool can handle unorthodox arrangements of text, such as annotations and marginalia, but reading order arrangement is not as developed as the following script.
- [Opticolumns](https://github.com/Scholarly-Projects/opticolumns)
    - Intended for archival scans of large scale multi-columned materials, such as newspapers. 
- [Opticolumn_Editor](https://github.com/Scholarly-Projects/opticolumn_editor)
    - Intended to create OCR using Opticolumn that produces a CSV of the OCR file that can be edited and processed gain to incorporate copy edits into the final embedded layer. This method is recommended if you need to produce OCR that surpasses the 85-95% accuracy benchmarks of Opticolumn and Opticolumns.

_Andrew Weymouth, Fall 2026._

<details>
<summary><h2>Additional Scripts</h2></summary>

In addition to the script.py OCR code, there are two additional scripts for reviewing and benchmarking output. After the `script.py` generates a new PDF of your original documents in the A folder, the `review.py` generates a jpeg of all of the processed PDF files that have been created in your B folder. These images in the C folder will have the original image of your document on the left hand side and an isolated copy of its OCR on the right. Only the first page of every document will be produced so you can quickly scan for accuracy of materials and/or adjust TrOCR Models or configuration accordingly.

Example output:

<img width="1920" height="1638" alt="tr_prac_05" src="https://github.com/user-attachments/assets/b6132cad-ea6b-4ea5-bf67-b09dca7c9c67" />

To understand the overall accuracy of the output, the `report.py` uses regular expressions, text mining approaches and spell checking to identify and tally the number of true words between original documents in the A folder and processed files in the B folder. A CSV of the report is generated in the D folder which prints the results for each file and the total increased searchability of the document. According to a [randomized 3,200 word survey](https://osf.io/9f483) of processed collection documents from August 2026, Opticolumn improved word-level accuracy by **41.66 percentage points** over the previous Adobe Acrobat OCR layer, a **1.96x improvement**.

</details>

<details>
<summary><h2>Note</h2></summary>

The Opticolumn toolkit is designed for **archival scans**, not born-digital PDFs. They add a new PDF/A-compliant OCR layer, or replace an existing one, but they do not add a tagging structure, alt text or the other elements a file needs to meet WCAG 2.1 standards.

If a born-digital PDF is processed, the tool strips its text and images and leaves a largely blank document. Automatic detection produced too many false positives across the variety of born-digital files, so batches need a quick manual review.

Before processing, check for page dimensions commonly associated with born-digital PDFs. Keep in mind that these dimensions are only indicators and do not guarantee that a file is born digital.

| Page size (points) | Inches | Common source |
| :---- | :---- | :---- |
| 612 × 792 / 792 × 612 | 8.5 × 11 | US Letter export (portrait / landscape) |
| 396 × 612 | 5.5 × 8.5 | Half-letter / booklet |
| 630 × 810 / 810 × 630 | 8.75 × 11.25 | US Letter with print bleed |
| 756 × 972 | 10.5 × 13.5 | Custom print size |
| 1224 × 792 | 17 × 11 | Tabloid, landscape |
| 841 × 1190 / 1190 × 841 | 11.7 × 16.5 | A3 (portrait / landscape) |
| 720 × 540 / 960 × 720 | 10 × 7.5 / 13.3 × 10 | 4:3 presentation slides |
| 960 × 540 / 540 × 960 | 13.3 × 7.5 | 16:9 presentation slides |
| 1024 × 768 / 1280 × 720 / 1920 × 1080 | — | Screen-resolution exports and screenshots |

After processing, review the output files in the B folder. Born-digital PDFs will typically appear largely blank. Remove these files before updating or publishing the processed batch.

</details>

<details>
<summary><h2>Background</h2></summary>

The Opticolumn toolkit was developed for overhauling the Center for Digital Inquiry and Learning's digital collection PDF files, to make the collection more discoverable and accessible. The development of the original [Opticolumn](https://github.com/Scholarly-Projects/opticolumn) tool is written about in greater detail in [_Transparent Practices: OCR and AI in the Archives_](https://journals.sagepub.com/doi/full/10.1177/15501906261439241), by Rebecca Hastings and Andrew Weymouth. _Collections: A Journal for Archives and Museum Professions_, June 2026.

</details>
