Optical character recognition is the conversion of typed, handwritten, or printed text into machine-encoded text, whether from images of a scanned document, a photo of a document, a scene photo (such as text on billboards), or from subtitle text superimposed on an image.

## Applications
OCR engines have been developed into software applications specializing in various subjects such as receipts, invoices, checks, and legal billing documents. OCR is widely used for:
- Entering data for business documents
- Automatic number-plate recognition
- Passport recognition
- Information extraction from insurance documents
- Traffic sign recognition
- Extracting business card information
- Creating textual versions of printed documents
- Converting handwriting to text in real-time
- Testing the robustness of CAPTCHA anti-bot systems
- Assistive technology for blind and visually impaired users

## Preprocessing
OCR software often preprocess images to improve chances of successful recognition. This includes techniques such as:
- **Deskewing**: if the document was not aligned properly when scanned, it may need to be rotated by a few degrees in order to make lines of text perfectly horizontal or vertical.
- **Despeckling**: noise removal such as smoothing edges and removal of positive and negative spots.
- **Binarization**: conversion of an image to a binary image.
- **Line removal**: cleaning up non-glyph boxes and lines.
- **Layout analysis**: identification of distinct blocks such as columns and paragraphs.
- **Line and word detection**: establishment of a baseline for word and character shapes, separating words as necessary.
- **Script recognition**: identification of script is necessary in multilingual documents before the right OCR can be invoked to handle the specific script.
- **Character segmentation**: multiple characters that are connected due to image artifacts must be separated and single characters that are broken into multiple pieces due to artifacts must be connected.
- **Normalization**: aspect ratio and scale are normalized.

## Challenges
OCR is designed to process massive data and is often deployed on mobile devices or embedded hardware. This requires the OCR to be light weight and be fast enough to run in real time. Moreover, OCR is used in various natural scenarios, leading to many challenges such as:
- Perspective variation
- Scale variation
- Background noise
- Blur
- Illumination conditions
- Curved text
- Multiple fonts
- Multiple languages

