# duplicate-invoice-detector
This is a simple Streamlit web app that detects duplicate invoices using OCR and image processing. It supports PDF, PNG, JPG, and JPEG formats. The app checks for similarity based on extracted text, image hashes, and histogram comparisons.

### Features
1) Upload and store invoice files in a local SQLite database.
2) Check for duplicates based on file content and appearance.
3) Displays similarity score and visual match.

### Technologies Used
(Python, 
Streamlit,
OpenCV,
Tesseract OCR,
scikit-learn,
SQLite)


It finds the similarity on basis of these things.
1) File Hashing (MD5): Used to detect exact file duplicates by generating a unique hash for each file.
2) OCR (Optical Character Recognition): Extracts text from invoice images to compare structural and content similarities.
3) Text Similarity: Compares the extracted text to find partial or near-duplicate invoices.
