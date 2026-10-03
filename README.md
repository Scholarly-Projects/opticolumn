# opticolumn

OCR tool developed by Andrew Weymouth, Digital Initiatives Librarian for University of Idaho, over summer and fall of 2025. The tool implements the TrOCR text recognition model and the Kraken BLLA page segmentation model to improve the accuracy of handwritten and cursive archival documents and add digital preservation metadata to processed materials.

__Opticolumn Optical Character Recognition Kit Practical Applications__

- [Opticolumn](https://github.com/Scholarly-Projects/opticolumn)
    - Intended for archival scans and designed for type, handwritten text, cursive or a combination of all three. The tool can handle unorthodox arrangements of text, such as annotations and marginalia, but reading order arrangement is not as developed as the following script.
- [Opticolumns](https://github.com/Scholarly-Projects/opticolumns)
    - Intended for archival scans of large scale multi-columned materials, such as newspapers. 
- [Opticolumn_Editor](https://github.com/Scholarly-Projects/opticolumn_editor)
    - Intended to create OCR using Opticolumn that produces a CSV of the OCR file that can be edited and processed gain to incorporate copy edits into the final embedded layer. This method is recommended if you need to produce OCR that surpasses the 85-95% accuracy benchmarks of Opticolumn and Opticolumns.

## Additional Scripts

In addition to the script.py OCR code, there are two additional scripts for reviewing and benchmarking output. After the `script.py` generates a new PDF of your original documents in the A folder, the `review.py` generates a jpeg of all of the processed PDF files that have been created in your B folder. These images in the C folder will have the original image of your document on the left hand side and an isolated copy of it's OCR on the right. Only the first page of every document will be produced so you can quickly scan for accuracy of materials and/or adjust TrOCR Models or configuration accordingly.

Example output:

<img width="1920" height="1638" alt="tr_prac_05" src="https://github.com/user-attachments/assets/b6132cad-ea6b-4ea5-bf67-b09dca7c9c67" />

To understand the overall accuracy of the output, the `report.py` uses regular expressions, text mining approaches and spell checking to identify and tally the number of true words between original documents in the A folder and processed files in the B folder. A CSV of the report is generated in the D folder which prints the results for each file and the total increased searchability of the document. According to a [randomized 3,200 word survey](https://osf.io/9f483) of processed collection documents from August 2026, Opticolumn improved word-level accuracy by **41.66 percentage points** over the previous Adobe Acrobat OCR layer, a **1.96x improvement**.

## Note

Opticolumn and Opticolumns are designed for archival scanned documents, not born-digital PDF files. The tool creates a new OCR layer for scans that are missing OCR or replaces existing OCR when necessary, but it does not create a tagging structure or add other accessibility features. 

To avoid accidentally overwriting born-digital PDF files that may already have OCR, tagging structure and alt text, Opticolumn will make these files apparent in the processed output, so they can be more easily identified and removed before the original files are replaced. If Opticolumn encounters a born-digital PDF, the processing will strip the file of its text and images, leaving a largely blank document.

Ideally, the tool would automatically detect and skip born-digital files during processing. However, the wide variation in the formatting and metadata of born-digital PDFs has produced too many false positives. As a result, users should review their batches manually. 

Before processing, check for page dimensions commonly associated with born-digital PDFs. Keep in mind that these dimensions are only indicators and do not guarantee that a file is born digital.

- 396x612
- 540x960
- 612x792
- 630x810
- 720x540
- 756x972 
- 792x612 
- 810x630
- 841x1190 
- 960x540
- 960x720
- 1024x768 
- 1190x841 
- 1224x792 
- 1280x720 
- 1920x1080 

After processing, review the output files in the B folder. Born-digital PDFs will typically appear largely blank. Remove these files before updating or publishing the processed batch.

## Background

The tool was developed for overhauling the Center for Digital Inquiry and Learning's digital collection PDF files, to make the collection more discoverable and accessible. The development of the tool is written about in greater detail in [_Transparent Practices: OCR and AI in the Archives_](https://journals.sagepub.com/doi/full/10.1177/15501906261439241), by Rebecca Hastings and Andrew Weymouth. _Collections: A Journal for Archives and Museum Professions_, June 2026.

__Also note__: Opticolumn and Opticolumns will only add a layer of PDF/A compliant OCR to a processed file. It does not create a tagging structure, add alt text or any of the other elements a standalone file needs to meet WCAG 2.1 standards.