# duplicate-invoice-detector
This is a simple Streamlit web app that detects duplicate invoices using OCR and image processing. It supports PDF, PNG, JPG, and JPEG formats. The app checks for similarity based on extracted text, image hashes, and histogram comparisons.

Features
Upload and store invoice files in a local SQLite database

Check for duplicates based on file content and appearance

Displays similarity score and visual match

Technologies Used
Python

Streamlit

OpenCV

Tesseract OCR

scikit-learn

SQLite

How to Run
Clone the repo

Install the packages from requirements.txt

Run the app with: streamlit run app.py

This was built as a personal project for learning and demonstration purposes.
1) MD5 hashset of the file
2) OCR(Optical character Recognition) used to find the simalirity based on the same structure of the image
3) Text extraction : Finds similarity in the text.
