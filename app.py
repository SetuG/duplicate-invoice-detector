import os
import sqlite3
import hashlib
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
from io import BytesIO
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt

# Set tesseract path for Render
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

class InvoiceDuplicateDetector:
    def __init__(self, db_path="invoices.db"):
        self.db_path = db_path
        self.init_database()
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS invoices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_hash TEXT UNIQUE,
                image_hash TEXT,
                extracted_text TEXT,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_data BLOB
            )
        ''')
        conn.commit()
        conn.close()

    def calculate_file_hash(self, file_bytes):
        return hashlib.md5(file_bytes).hexdigest()

    def calculate_image_hash(self, image):
        resized = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        avg = gray.mean()
        binary = (gray > avg).astype(int)
        return ''.join(str(b) for b in binary.flatten())

    def pdf_to_image(self, file_bytes):
        images = convert_from_bytes(file_bytes, first_page=1, last_page=1)
        return np.array(images[0])

    def extract_text_from_image(self, image):
        return pytesseract.image_to_string(Image.fromarray(image)).strip()

    def image_to_blob(self, image):
        buffer = BytesIO()
        Image.fromarray(image).save(buffer, format='PNG')
        return buffer.getvalue()

    def blob_to_image(self, blob):
        return Image.open(BytesIO(blob))

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    def calculate_image_similarity(self, img1, img2):
        try:
            proc_img1 = self.preprocess_image(img1)
            proc_img2 = self.preprocess_image(img2)
            h, w = min(proc_img1.shape[0], proc_img2.shape[0]), min(proc_img1.shape[1], proc_img2.shape[1])
            proc_img1 = cv2.resize(proc_img1, (w, h))
            proc_img2 = cv2.resize(proc_img2, (w, h))
            hist1 = cv2.calcHist([proc_img1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([proc_img2], [0], None, [256], [0, 256])
            return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        except:
            return 0

    def calculate_text_similarity(self, text1, text2):
        try:
            if not text1.strip() or not text2.strip(): return 0
            tfidf = self.vectorizer.fit_transform([text1, text2])
            return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        except:
            return 0

    def hamming_distance(self, h1, h2):
        return sum(c1 != c2 for c1, c2 in zip(h1, h2)) if len(h1) == len(h2) else float('inf')

    def store_invoice(self, file, filename):
        file_bytes = file.read()
        file_hash = self.calculate_file_hash(file_bytes)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM invoices WHERE file_hash=?", (file_hash,))
        if cursor.fetchone():
            conn.close()
            return False, "Duplicate file. Skipped."

        ext = filename.lower().split('.')[-1]
        try:
            if ext == 'pdf':
                image = self.pdf_to_image(file_bytes)
            else:
                image = np.array(Image.open(BytesIO(file_bytes)).convert('RGB'))
        except Exception as e:
            return False, f"Error processing file: {str(e)}"

        image_hash = self.calculate_image_hash(image)
        text = self.extract_text_from_image(image)
        blob = self.image_to_blob(image)

        cursor.execute('''
            INSERT INTO invoices (filename, file_hash, image_hash, extracted_text, image_data)
            VALUES (?, ?, ?, ?, ?)
        ''', (filename, file_hash, image_hash, text, blob))
        conn.commit()
        conn.close()
        return True, "Stored successfully."

    def find_duplicates(self, file, threshold=0.8):
        file_bytes = file.read()
        ext = file.name.lower().split('.')[-1]
        try:
            if ext == 'pdf':
                image = self.pdf_to_image(file_bytes)
            else:
                image = np.array(Image.open(BytesIO(file_bytes)).convert('RGB'))
        except Exception as e:
            return False, f"Failed to process file: {str(e)}"

        image_hash = self.calculate_image_hash(image)
        extracted_text = self.extract_text_from_image(image)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, filename, image_hash, extracted_text, image_data FROM invoices")
        invoices = cursor.fetchall()
        conn.close()

        results = []
        for inv in invoices:
            iid, fname, stored_hash, stored_text, blob = inv
            stored_image = np.array(self.blob_to_image(blob).convert('RGB'))
            hash_similarity = 1 - (self.hamming_distance(image_hash, stored_hash) / len(image_hash))
            text_similarity = self.calculate_text_similarity(extracted_text, stored_text)
            img_similarity = self.calculate_image_similarity(image, stored_image)
            combined = 0.4 * hash_similarity + 0.4 * text_similarity + 0.2 * img_similarity
            if combined >= threshold:
                results.append((fname, combined, stored_image))
        results.sort(key=lambda x: x[1], reverse=True)
        return True, results

st.set_page_config(page_title="Invoice Duplicate Detector", layout="wide")
st.title("📄 Duplicate Invoice Detector")

detector = InvoiceDuplicateDetector()

st.header("1️⃣ Upload and Store Invoices")
uploaded_files = st.file_uploader("Upload invoice files (PDF, PNG, JPG, JPEG)", accept_multiple_files=True, type=['pdf', 'png', 'jpg', 'jpeg'])

if uploaded_files:
    for f in uploaded_files:
        success, msg = detector.store_invoice(f, f.name)
        st.success(f"{f.name}: {msg}" if success else f"{f.name}: {msg}")

st.header("2️⃣ Check for Duplicate")
check_file = st.file_uploader("Upload a file to check for duplicates", key="check", type=['pdf', 'png', 'jpg', 'jpeg'])

if check_file:
    ok, result = detector.find_duplicates(check_file, threshold=0.8)
    if not ok:
        st.error(result)
    elif not result:
        st.info("✅ No duplicates found!")
    else:
        st.warning("⚠️ Possible duplicates found:")
        for fname, score, stored_img in result:
            st.write(f"🔁 **{fname}** — Similarity: `{score:.2f}`")
            col1, col2 = st.columns(2)
            with col1:
                try: st.image(check_file, caption="Input Invoice", use_container_width=True)
                except Exception: pass
            with col2:
                try: st.image(stored_img, caption=f"Matched: {fname}", use_container_width=True)
                except Exception: pass