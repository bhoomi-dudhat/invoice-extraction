"""
FastAPI Multi-page Invoice Processor using AWS Bedrock (Claude Sonnet 4)
- No Poppler (uses PyMuPDF / fitz)
- Bedrock call pattern matches your menu script (text + image in messages content)
- Static token-based authorization added (see STATIC_API_TOKEN)
"""

import os
import io
import re
import json
import base64
import tempfile
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import urllib.parse
import pandas as pd
import boto3
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import cv2
import httpx

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import uvicorn

# ------------------- CONFIG -------------------
# Tesseract executable (change if different on your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Bedrock / Claude Sonnet 4 model id (as requested)
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN", None)


AWS_ACCESS_KEY_ID = 'AWS_ACCESS_KEY_ID'
AWS_SECRET_ACCESS_KEY = 'AWS_SECRET_ACCESS_KEY'

# Static API token (set in environment for production)
# Example: export STATIC_API_TOKEN="my-super-secret-token"
STATIC_API_TOKEN = os.getenv("STATIC_API_TOKEN", "qwertyuioplkjhgfdsazxcvbnm123")
print("STATIC_API_TOKEN:", STATIC_API_TOKEN, type(STATIC_API_TOKEN))


# Create Bedrock runtime client (bedrock-runtime)
BRT = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# Rendering DPI / image sizing
RENDER_DPI = 200
MIN_IMAGE_SIZE = 1200
TARGET_SIZE = 1600

# ------------------- Pydantic models -------------------
# Accept either string or numeric for totals/rates to avoid strict validation errors
NumberOrStr = Union[str, float, int]

class LineItem(BaseModel):
    qty: Optional[str] = None
    ItemName: Optional[str] = None
    itemDescription: Optional[str] = None
    rate: Optional[NumberOrStr] = None
    amt: Optional[NumberOrStr] = None
    productCode: Optional[str] = None

class ConsolidatedInvoice(BaseModel):
    invoiceNo: Optional[str] = None
    invoiceDate: Optional[str] = None
    dueDate: Optional[str] = None
    purchaseOrder: Optional[str] = None
    customerCompany: Optional[str] = None
    customerContact: Optional[str] = None
    customerAddr1: Optional[str] = None
    customerAddr2: Optional[str] = None
    customerCity: Optional[str] = None
    customerState: Optional[str] = None
    customerZIP: Optional[str] = None
    terms: Optional[str] = None
    vendorName: Optional[str] = None
    vendorAddress: Optional[str] = None
    subtotal: Optional[NumberOrStr] = None
    tax: Optional[NumberOrStr] = None
    total: Optional[NumberOrStr] = None
    items: List[LineItem] = []

class ProcessingResponse(BaseModel):
    success: bool
    message: str
    data: Optional[ConsolidatedInvoice] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

# ------------------- FastAPI app -------------------
app = FastAPI(title="Invoice Processor (Bedrock Claude Sonnet 4)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Auth Dependency -------------------
def verify_token(authorization: Optional[str] = Header(None)):
    print("Received header:", authorization) 
    """
    Verify the static API token.
    Accepts either:
      Authorization: Bearer <token>
    or
      Authorization: <token>
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    token = authorization.strip()
    # support "Bearer <token>" form
    if token.lower().startswith("bearer "):
        token = token.split(None, 1)[1].strip()

    if token != STATIC_API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API token")

    return True

# ------------------- Utilities -------------------
def save_temp_image_from_pil(img: Image.Image) -> str:
    fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="page_")
    os.close(fd)
    img.save(tmp_path, format="PNG")
    return tmp_path

def pdf_to_images(pdf_bytes: bytes, zoom: float = 2.0) -> List[str]:
    """
    Convert PDF bytes to list of temporary PNG file paths using PyMuPDF.
    Returns list of file paths.
    """
    images = []
    doc = fitz.open("pdf", pdf_bytes)
    mat = fitz.Matrix(zoom, zoom)
    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tmp_path = save_temp_image_from_pil(image)
        images.append(tmp_path)
    doc.close()
    return images

def preprocess_image_for_ocr(image_path: str) -> str:
    """
    Basic preprocessing: read, optionally resize, convert to grayscale & adaptive threshold.
    Returns path to processed temporary image (PNG).
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path

        h, w = img.shape[:2]
        if max(h, w) < MIN_IMAGE_SIZE:
            scale = TARGET_SIZE / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        sharpened = cv2.filter2D(gray, -1, kernel)

        processed = cv2.adaptiveThreshold(
            sharpened, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            25, 6
        )

        fd, tmp_path = tempfile.mkstemp(prefix="proc_", suffix=".png")
        os.close(fd)
        cv2.imwrite(tmp_path, processed)
        return tmp_path
    except Exception:
        return image_path

def ocr_image_to_text(image_path: str) -> str:
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text or ""
    except Exception:
        return ""

def extract_json_from_model_text(text: str) -> Dict[str, Any]:
    """
    Find the first JSON object in `text` and parse it. If parsing fails, return {}.
    """
    if not text:
        return {}
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.IGNORECASE)
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    candidate = m.group(0) if m else t
    # remove trailing commas before closing braces/brackets
    candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
    try:
        return json.loads(candidate)
    except Exception:
        # try locating outermost braces
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(candidate[start:end+1])
            except Exception:
                return {}
    return {}

import uuid
from datetime import datetime

def parse_date_safe(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except:
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except:
            try:
                return datetime.strptime(value, "%d/%m/%Y")
            except:
                return None


def map_ocr_to_inward(ocr):
    now = datetime.now().isoformat()

    inward = {
        "nvarInwardID": ocr.get("purchaseOrder", ""),
        "dtInwardDate": ocr.get("invoiceDate", ""),
        "nvarReferenceNo": ocr.get("invoiceNo", ""),
        "nvarInwardBy": ocr.get("vendorName", ""),
        "nvarDescription": f"Due Date: {ocr.get('dueDate')}" if ocr.get("dueDate") else "",
        "dcmlInwardTotalCost": ocr.get("total", 0),
        "dtCreatedDate": now,
        "dtModifiedDate": now,
        "bitIsDeleted": False,
        "IsEdit": False,
        "VendorState": ocr.get("customerState", ""),
        "lstInwardItems": []
    }

    vendors = {
        "nvarCompany": ocr.get("customerCompany", ""),
        "nvarFirst_Name": ocr.get("customerContact", ""),
        "nvarAddress_1": ocr.get("customerAddr1", ""),
        "nvarAddress_2": ocr.get("customerAddr2", ""),
        "nvarCity": ocr.get("customerCity", ""),
        "nvarState": ocr.get("customerState", ""),
        "nvarZip_Code": ocr.get("customerZIP", ""),
        "nvarVendor_Terms": ocr.get("terms", ""),
    }

    inward_items = []
    items = ocr.get("items", [])

    for index, item in enumerate(items):
        inward_items.append({
            "unqRowID": str(uuid.uuid4()),
            "nvarInwardID": ocr.get("purchaseOrder", ""),
            "nvarItemNum": item.get("productCode", ""),
            "nvarItemName": item.get("ItemName", ""),
            "nvarCaseOrIndividual": item.get("itemDescription", ""),
            "dcmlInwardQty": item.get("qty", 0),
            "dcmlItemCost": item.get("rate", 0),
            "dcmlInwardCost": item.get("amt", 0),
            "intStatus": 1,
            "nvarOrderItemCounter": str(index + 1),
            "intNumPerCase": None,
            "intStockUnit": 1,
            "dtCreatedDate": now,
            "dtModifiedDate": now,
            "bitIsDeleted": False,
            "Index": index
        })

    inward["lstInwardItems"] = inward_items

    return {
        "ORTP_Inward": inward,
        "ORTP_Vendors": vendors,
        "ORTP_InwardItems": inward_items
    }

# -------------- Replace None → "0" Utility ------------------
def replace_none_with_zero(obj):
    if isinstance(obj, dict):
        return {k: replace_none_with_zero(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_none_with_zero(i) for i in obj]
    elif obj is None:
        return "0"
    return obj

# ------------------- Bedrock call (same method as menu script) -------------------
def call_bedrock_model(prompt: str, image_path: Optional[str] = None, max_tokens: int = 4096) -> str:
    """
    Build messages content (text + optional image) and call bedrock-runtime invoke_model.
    Returns textual output (concatenated 'text' parts) or empty string on error.
    """
    content = [{"type": "text", "text": prompt}]
    if image_path:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode()
        # determine media type
        ext = os.path.splitext(image_path)[1].lower().replace(".", "")
        if ext == "jpg":
            ext = "jpeg"
        media_type = f"image/{ext}"
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": img_b64
            }
        })

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }

    try:
        response = BRT.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        raw = response["body"].read().decode("utf-8")
        parsed = json.loads(raw)
        out_text = ""
        for p in parsed.get("content", []):
            if p.get("type") == "text":
                out_text += p.get("text", "")
        return out_text
    except Exception as e:
        traceback.print_exc()
        return ""

# ------------------- InvoiceProcessor -------------------
class InvoiceProcessor:
    def __init__(self, debug: bool = True):
        self.debug = debug
        self.temp_files: List[str] = []

    def log(self, *args):
        if self.debug:
            print("[DEBUG]", *args)

    def cleanup(self):
        for f in list(self.temp_files):
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
            try:
                self.temp_files.remove(f)
            except Exception:
                pass

    def encode_image_b64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def extract_with_bedrock(self, image_path: str, ocr_text: str, page_number: int, total_pages: int, all_ocr_text: str = "") -> Dict[str, Any]:
        """
        Prepare prompt, call Bedrock (Claude Sonnet 4), and return parsed JSON dict.
        Mirrors the menu script method (text + image in messages).
        """
        try:
            # construct context-aware prompt (JSON-only)
            page_context = ""
            if page_number == 1:
                page_context = "This is the FIRST page - prioritize header invoice #, dates, customer/vendor details."
            elif page_number == total_pages:
                page_context = "This is the LAST page - prioritize totals (subtotal, tax, total) and final line items."
            else:
                page_context = f"This is page {page_number} of {total_pages} - focus on line items."

            prompt = f"""
You are an expert invoice parser. {page_context}

Return ONLY valid JSON (no explanation) with these exact fields:

{{
  "page_number": {page_number},
  "invoiceNo": "string or null",
  "invoiceDate": "string or null",
  "dueDate": "string or null",
  "purchaseOrder": "string or null",
  "customerCompany": "string or null",
  "customerContact": "string or null",
  "customerAddr1": "string or null",
  "customerAddr2": "string or null",
  "customerCity": "string or null",
  "customerState": "string or null",
  "customerZIP": "string or null",
  "terms": "string or null",
  "vendorName": "string or null",
  "vendorAddress": "string or null",
  "subtotal": "string or null",
  "tax": "string or null",
  "total": "string or null",
  "line_items": [
    {{
      "qty": "string",
      "ItemName": "string",
      "itemDescription": "string",
      "rate": "string or null",
      "amt": "string",
      "productCode": "string"
    }}
  ]
}}

CRITICAL RULES:
1. Extract ALL visible line items from THIS page
2. For page 1: prioritize invoice #, dates, customer/vendor info, product number/code/HSN code
3. For last page: prioritize subtotal, tax, total
4. Remove $ signs from amounts, keep decimals
5. If field not visible, use "0"
6. Return ONLY valid JSON, no markdown, no explanations

Current page OCR (first 2000 chars):
{ocr_text[:2000]}

Context (first 2000 chars of all pages OCR):
{all_ocr_text[:2000]}
"""

            # Use call_bedrock_model which sends text + image
            model_text = call_bedrock_model(prompt, image_path=image_path, max_tokens=4096)
            if not model_text:
                return self.fallback_extraction(ocr_text, page_number, all_ocr_text)

            # Clean model response and extract JSON
            model_text = re.sub(r"```json\s*", "", model_text, flags=re.IGNORECASE)
            model_text = re.sub(r"```\s*", "", model_text, flags=re.IGNORECASE)
            parsed = extract_json_from_model_text(model_text)
            if not parsed:
                return self.fallback_extraction(ocr_text, page_number, all_ocr_text)

            parsed["page_number"] = page_number
            return self.clean_extracted_data(parsed, ocr_text)

        except Exception as e:
            traceback.print_exc()
            return self.fallback_extraction(ocr_text, page_number, all_ocr_text)

    def fallback_extraction(self, ocr_text: str, page_number: int, all_ocr_text: str = "") -> Dict[str, Any]:
        """
        Simple regex-based fallback extraction (Invoice No, Date).
        """
        def find_first(patterns, text):
            for p in patterns:
                m = re.search(p, text, re.IGNORECASE)
                if m:
                    return m.group(1).strip()
            return None

        search_text = all_ocr_text or ocr_text

        invoice_patterns = [
            r'Invoice\s*No[:\s]*([A-Z0-9\-\/]+)',
            r'Invoice\s*Number[:\s]*([A-Z0-9\-\/]+)',
            r'Inv[:\s]*([A-Z0-9\-\/]+)'
        ]
        date_patterns = [
            r'Invoice\s*Date[:\s]*([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})',
            r'Date[:\s]*([0-9]{1,2}[-/][A-Za-z]{3,9}[-/][0-9]{2,4})'
        ]

        return {
            "page_number": page_number,
            "invoiceNo": find_first(invoice_patterns, search_text),
            "invoiceDate": find_first(date_patterns, search_text),
            "dueDate": None,
            "purchaseOrder": None,
            "customerCompany": None,
            "customerContact": None,
            "customerAddr1": None,
            "customerAddr2": None,
            "customerCity": None,
            "customerState": None,
            "customerZIP": None,
            "terms": None,
            "vendorName": None,
            "vendorAddress": None,
            "subtotal": None,
            "tax": None,
            "total": None,
            "line_items": [],
            "extraction_method": "fallback"
        }

    def clean_extracted_data(self, data: Dict[str, Any], ocr_text: str) -> Dict[str, Any]:
        """
        Normalize fields: remove currency symbols, enforce strings for items, return safe defaults.
        """
        # Normalize invoice number
        if data.get("invoiceNo"):
            data["invoiceNo"] = re.sub(r"[^\w\-\./]", "", str(data["invoiceNo"]).strip())

        # Clean numeric-like fields (strip currency signs and commas)
        for field in ("subtotal", "tax", "total"):
            v = data.get(field)
            if v is None:
                data[field] = "0"
            else:
                s = str(v).strip()
                s = re.sub(r'[^\d.\-]', '', s)
                data[field] = s if s else "0"

        # Clean line items
        cleaned = []
        for item in data.get("line_items", []):
            qty = str(item.get("qty", "") or "").strip()
            name = str(item.get("ItemName", "") or "").strip()
            desc = str(item.get("itemDescription", "") or "").strip()
            rate = str(item.get("rate", "") or "").strip()
            rate = re.sub(r'[^\d.\-]', '', rate) if rate else "0"
            amt = str(item.get("amt", "") or "").strip()
            amt = re.sub(r'[^\d.\-]', '', amt) if amt else "0"
            prod = str(item.get("productCode", "") or "").strip()

            # require either name or description to include item
            if name or desc:
                cleaned.append({
                    "qty": qty if qty else "0",
                    "ItemName": name,
                    "itemDescription": desc,
                    "rate": rate if rate else "0",
                    "amt": amt if amt else "0",
                    "productCode": prod if prod else "0"
                })
        data["line_items"] = cleaned
        return data

    def consolidate_pages(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Consolidate multiple page outputs into a single invoice.
        Prefers header info from first page and totals from last page.
        Aggregates line items across pages and calculates totals if needed.
        """
        if not pages:
            return {}

        consolidated = {
            "invoiceNo": None,
            "invoiceDate": None,
            "dueDate": None,
            "purchaseOrder": None,
            "customerCompany": None,
            "customerContact": None,
            "customerAddr1": None,
            "customerAddr2": None,
            "customerCity": None,
            "customerState": None,
            "customerZIP": None,
            "terms": None,
            "vendorName": None,
            "vendorAddress": None,
            "subtotal": None,
            "tax": None,
            "total": None,
            "items": []
        }

        first = pages[0]
        last = pages[-1]

        # Header fields from first page
        headers = ["invoiceNo", "invoiceDate", "purchaseOrder", "customerCompany", "customerContact", "customerAddr1", "customerAddr2", "vendorName", "vendorAddress"]
        for h in headers:
            if first.get(h):
                consolidated[h] = first[h]

        # Totals from last page
        totals = ["subtotal", "tax", "total", "dueDate", "terms"]
        for t in totals:
            if last.get(t):
                consolidated[t] = last[t]

        # Fill missing from other pages
        for page in pages:
            for k in consolidated.keys():
                if k == "items":
                    continue
                if not consolidated[k] and page.get(k):
                    consolidated[k] = page[k]

        # Aggregate line items
        all_items = []
        for page in pages:
            for it in page.get("line_items", []):
                if it.get("ItemName") or it.get("itemDescription"):
                    all_items.append({
                        "qty": it.get("qty", "0"),
                        "ItemName": it.get("ItemName", ""),
                        "itemDescription": it.get("itemDescription", ""),
                        "rate": it.get("rate", "0"),
                        "amt": it.get("amt", "0"),
                        "productCode": it.get("productCode", "0")
                    })
        consolidated["items"] = all_items

        # If total missing, try to compute from amounts
        if (not consolidated.get("total") or str(consolidated.get("total")).strip() in ("", "0")) and all_items:
            try:
                s = 0.0
                for it in all_items:
                    amt_str = str(it.get("amt", "0"))
                    amt_clean = re.sub(r'[^\d.\-]', '', amt_str)
                    if amt_clean:
                        s += float(amt_clean)
                consolidated["total"] = f"{s:.2f}"
            except Exception:
                pass

        # Defaults for subtotal/tax if None
        for f in ("subtotal", "tax", "total"):
            if consolidated.get(f) is None:
                consolidated[f] = "0"

        return consolidated

    def process_invoice(self, file_path: str) -> (List[Dict[str, Any]], Dict[str, Any]): # type: ignore
        """
        Full pipeline:
        - convert pdf -> images (PyMuPDF) OR single image accepted
        - OCR per page
        - call Bedrock per page with text+image
        - consolidate pages into final invoice dict
        """
        pages_data = []
        all_ocr = ""

        # If PDF
        if file_path.lower().endswith(".pdf"):
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()
            image_paths = pdf_to_images(pdf_bytes, zoom=RENDER_DPI / 72.0)  # zoom approx DPI conversion
            self.temp_files.extend(image_paths)
        else:
            # single image
            image_paths = [file_path]

        total_pages = len(image_paths)
        # First pass: OCR to build context
        ocr_texts = []
        for i, ip in enumerate(image_paths, start=1):
            proc = preprocess_image_for_ocr(ip)
            self.temp_files.append(proc)
            text = ocr_image_to_text(proc)
            ocr_texts.append(text)
            all_ocr += f"\n--- Page {i} ---\n{text}"

        # Second pass: call model per page (with full context)
        for i, (ip, ocr_t) in enumerate(zip(image_paths, ocr_texts), start=1):
            page_json = self.extract_with_bedrock(ip, ocr_t, i, total_pages, all_ocr)
            pages_data.append(page_json)

        consolidated = self.consolidate_pages(pages_data)
        return pages_data, consolidated


#-------------------------------------------------
#-----------------Menu Extraction-----------------
#-------------------------------------------------
REFERENCE_TEMPLATE = "Menu_AllSections_Modifiers_WithDiff.xlsx"
OUTPUT_FOLDER = "outputs"

EXTRACTION_PROMPT = """
You are an intelligent menu-reading system.

Return ONLY valid JSON in this structure (exact keys and types required):

{
  "items": [
    {
      "category": "<Category Name (string)>",
      "name": "<FULL Item Name EXACTLY as printed on the menu (string)>",
      "price_headers": ["<header1>", "<header2>", ...],   // ALWAYS include this key (empty list if not present)
      "prices": [list of numeric prices],
      "modifiers": [list of modifier names, including Veg/Non-Veg if specified in brackets or item name]
    }
  ]
}

CRITICAL RULES:
- Do NOT rewrite or shorten the item name.
- price_headers MUST be present (use empty list [] when not applicable).
- Capture every header printed on the menu (e.g., ["Half","Full"] or ["Veg","Non-Veg","Paneer"]).
- If the item has Veg/Non-Veg info in brackets or in the item name, include "Veg" or "Non-Veg" in the modifiers list.
- If brackets list multiple types (e.g., Paneer, Gobi), include each as a separate modifier.
- Numeric prices should correspond to headers order when headers present; otherwise they are listed in printed menu order.
- Only return valid JSON. Do NOT include extra text or explanations.
"""

# ------------------- UTILITIES -------------------
def safe_int(x):
    if x is None: return None
    if isinstance(x, (int, float)): return int(x)
    s = re.sub(r"[^\d]", "", str(x))
    return int(s) if s else None

def remove_brackets(text: str) -> str:
    return re.sub(r"\(.*?\)", "", text).strip()

def extract_all_bracket_text(name: str) -> str:
    name = name.replace("\n", " ").replace("\r", " ")
    matches = re.findall(r"\((.*?)\)", name, flags=re.DOTALL)
    return " ".join(matches) if matches else ""

def split_types(label_block: str) -> list:
    if not label_block: return []
    parts = re.split(r"[,/]+|\s{2,}", label_block)
    return [p.strip() for p in parts if p.strip()]

def detect_half_full(headers: List[str]) -> bool:
    if not headers: return False
    headers_lower = [h.lower() for h in headers]
    return ("half" in headers_lower) and ("full" in headers_lower)

def extract_json_from_text(text: str) -> dict:
    if not text: return {}
    clean = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.IGNORECASE)
    m = re.search(r"\{.*\}", clean, flags=re.DOTALL)
    candidate = m.group(0) if m else clean
    candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
    try:
        return json.loads(candidate)
    except:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(candidate[start:end+1])
            except:
                return {}
    return {}

def get_next_output_filename(base_name="menu_final", ext="xlsx") -> str:
    files = os.listdir(OUTPUT_FOLDER)
    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)\.{re.escape(ext)}$")
    nums = [int(m.group(1)) for f in files if (m := pattern.match(f))]
    next_n = max(nums) + 1 if nums else 1
    return os.path.join(OUTPUT_FOLDER, f"{base_name}_{next_n}.{ext}")

# ------------------- BEDROCK CALL -------------------
def call_bedrock_model(prompt: str, image_path: Optional[str] = None, max_tokens=2048) -> str:
    content = [{"type": "text", "text": prompt}]

    # Add image
    if image_path:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode()

        ext = os.path.splitext(image_path)[1].lower().replace(".", "")
        if ext == "jpg":
            ext = "jpeg"

        media_type = f"image/{ext}"

        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": img_b64
            }
        })

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }

    try:
        response = BRT.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        raw = response["body"].read().decode()
        parsed = json.loads(raw)

        for p in parsed.get("content", []):
            if p.get("type") == "text":
                return p.get("text", "")

        return ""

    except Exception as e:
        logger.error(f"Bedrock Error: {e}")
        return ""


# ------------------- ROW BUILDERS -------------------
def build_product_rows(items: List[dict]) -> List[dict]:
    rows = []
    for it in items:
        full_name = it.get("name", "")
        category = it.get("category", "")
        headers = it.get("price_headers", []) or []
        prices = it.get("prices", []) or []
        numeric_prices = [safe_int(p) for p in prices]

        is_half_full = detect_half_full(headers)
        if is_half_full:
            half_price = numeric_prices[0] if len(numeric_prices) else ""
            rows.append({
                "Category": category,
                "ProductNumber": "",
                "ProductName": remove_brackets(full_name),
                "Description": "",
                "Barcode": "",
                "Cost": "",
                "SellPrice": half_price,
                "DisplayInMenu": "Yes",
                "IsInventoryItem": "",
                "IsFoodStempItem": "",
                "BeveragesDeposite": "",
                "VendorName": "",
                "BrandName": "",
                "Age Verification 1": "",
                "Age Verification 2": "",
                "Item Type": "product",
                "IsHalf": "Yes"
            })
            continue

        base_price = numeric_prices[0] if numeric_prices else ""
        rows.append({
            "Category": category,
            "ProductNumber": "",
            "ProductName": remove_brackets(full_name),
            "Description": "",
            "Barcode": "",
            "Cost": "",
            "SellPrice": base_price,
            "DisplayInMenu": "Yes",
            "IsInventoryItem": "",
            "IsFoodStempItem": "",
            "BeveragesDeposite": "",
            "VendorName": "",
            "BrandName": "",
            "Age Verification 1": "",
            "Age Verification 2": "",
            "Item Type": "product",
            "IsHalf": ""
        })
    return rows

def build_modifier_rows(items: List[dict]) -> List[dict]:
    rows = []

    for it in items:
        full_name = it.get("name", "")
        category = it.get("category", "")
        headers = it.get("price_headers", []) or []
        prices = it.get("prices", []) or []
        numeric_prices = [safe_int(p) if p else 0 for p in prices]

        # ---------------------- CASE 1: HALF / FULL ----------------------
        if detect_half_full(headers):
            half_price = numeric_prices[0]
            full_price = numeric_prices[1] if len(numeric_prices) > 1 else half_price
            diff = full_price - half_price

            # Half row
            rows.append({
                "ItemName": full_name,
                "Modifier Group Department": category,
                "Modifier Group Name": "Quantity",
                "Is Include": "Yes",
                "Price": 0,
                "Min": 1,
                "Max": 1,
                "Is Half": "Yes",
                "Modifier Department": category,
                "Modifier": "Half",
                "Pre-Selected": "Yes",
                "Override Price": "Yes",
                "Modifier Item Price": 0
            })

            # Full row
            rows.append({
                "ItemName": full_name,
                "Modifier Group Department": category,
                "Modifier Group Name": "Quantity",
                "Is Include": "No",
                "Price": 0,
                "Min": 1,
                "Max": 1,
                "Is Half": "No",
                "Modifier Department": category,
                "Modifier": "Full",
                "Pre-Selected": "No",
                "Override Price": "Yes",
                "Modifier Item Price": diff
            })
            continue

        # ---------------------- CASE 2: FLAVOURS / MULTIPLE PRICES ----------------------

        # MATCH EXACT GEMINI LOGIC:
        if headers:
            flavour_names = headers
        else:
            text = extract_all_bracket_text(full_name)
            flavour_names = split_types(text)

        if not flavour_names:
            txt = full_name.lower()
            nonveg_keywords = ["egg","chicken","fish","meat","prawn","non veg","non-veg"]
            is_nonveg = any(k in txt for k in nonveg_keywords)
            flavour_names = ["Non-Veg"] if is_nonveg else ["Veg"]

        # BASE PRICE MUST MATCH GEMINI → PRICE[0]
        base_price = numeric_prices[0] if numeric_prices else 0

        for idx, flavour in enumerate(flavour_names):
            flavour = flavour.strip()
            price_here = numeric_prices[idx] if idx < len(numeric_prices) else base_price
            diff = price_here - base_price

            rows.append({
                "ItemName": full_name,
                "Modifier Group Department": category,
                "Modifier Group Name": "Flavours",
                "Is Include": "Yes" if idx == 0 else "No",
                "Price": 0,
                "Min": 1,
                "Max": 1,
                "Is Half": "",
                "Modifier Department": category,
                "Modifier": flavour,
                "Pre-Selected": "Yes" if idx == 0 else "No",
                "Override Price": "Yes",
                "Modifier Item Price": diff
            })

    return rows






# ------------------- API Endpoints -------------------
@app.get("/", response_model=HealthResponse)
async def root():
    return {"status": "running", "timestamp": datetime.now().isoformat(), "version": "1.0.0"}

@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "1.0.0"}
from fastapi import Request
# @app.post("/process-invoice", response_model=ProcessingResponse, dependencies=[Depends(verify_token)])
@app.post("/process-invoice", response_model=ProcessingResponse, dependencies=[Depends(verify_token)])
async def process_invoice_endpoint(
    file: UploadFile = File(...),
    token: str = Header(None, alias="Authorization")

):
    print(Request.headers)

    # --- Validate Token (explicit, not hidden dependency) ---
    verify_token(token)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    allowed = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_name = f"{ts}_{file.filename}"
    upload_path = os.path.join(UPLOAD_DIR, upload_name)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        # --- Save uploaded file ---
        contents = await file.read()
        with open(upload_path, "wb") as fw:
            fw.write(contents)

        processor = InvoiceProcessor(debug=True)

        # --- Run processor in separate thread ---
        pages_data, consolidated = await run_in_threadpool(
            processor.process_invoice, upload_path
        )

        # --- Map inward entries ---
        try:
            inward_mapped = map_ocr_to_inward(consolidated)
        except Exception as e:
            inward_mapped = {}
            print("Mapping error:", e)

        # --- Save result JSON ---
        result_name = f"{ts}_{Path(file.filename).stem}_result.json"
        result_path = os.path.join(RESULTS_DIR, result_name)

        result_payload = {
            # "consolidated_invoice": consolidated,
            "inward_mapped": inward_mapped,
            # "page_details": pages_data,
            "processing_info": {
                "total_pages": len(pages_data),
                # "total_items": len(consolidated.get("items", [])),
                "processed_at": datetime.now().isoformat(),
                
            }
        }

        with open(result_path, "w", encoding="utf-8") as rf:
            json.dump(result_payload, rf, indent=2, ensure_ascii=False)

        # --- Convert consolidated to Pydantic model ---
        try:
            items = [LineItem(**it) for it in consolidated.get("items", [])]
            cons_copy = dict(consolidated)
            cons_copy["items"] = items
            model_data = ConsolidatedInvoice(**cons_copy)
            clean_data = replace_none_with_zero(model_data.dict())
        except Exception as e:
            print("Model conversion error:", e)
            clean_data = replace_none_with_zero(consolidated)

        return ProcessingResponse(
            success=True,
            message=f"Processed {len(pages_data)} page(s)",
            # data=clean_data,
            details={
                "pages_processed": len(pages_data),
                # "items_extracted": len(consolidated.get("items", [])),
                "result_file": result_name,
                "inward_data": inward_mapped
            }
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            if os.path.exists(upload_path):
                os.remove(upload_path)
        except:
            pass
        try:
            processor.cleanup()
        except:
            pass



@app.post("/process-invoice-url", response_model=ProcessingResponse, dependencies=[Depends(verify_token)])
async def process_invoice_url(
    payload: Dict[str, str],
    token: str = Header(None, alias="Authorization")
):
    """
    Accepts JSON body {"fileurl": "..."}.
    Downloads the file to uploads/ and runs the same processing pipeline.
    Returns the same ProcessingResponse shape as /process-invoice.
    """
    # --- Validate Token (explicit, same as /process-invoice) ---
    verify_token(token)

    fileurl = payload.get("fileurl")
    if not fileurl:
        raise HTTPException(status_code=400, detail="JSON body must include 'fileurl'")

    # Basic validation for URL scheme
    parsed = urllib.parse.urlparse(fileurl)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="fileurl must be an http or https url")

    # derive filename or fallback
    filename = Path(urllib.parse.unquote(parsed.path)).name or f"downloaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_path = os.path.join(UPLOAD_DIR, f"{ts}_{filename}")

    # Allowed file types
    allowed = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]

    # Download file (async)
    try:
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            resp = await client.get(fileurl)
            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download file: status_code={resp.status_code}")
            content = resp.content

            # If filename had no allowed extension, try to infer from headers
            if not any(filename.lower().endswith(a) for a in allowed):
                ctype = resp.headers.get("content-type", "")
                if "pdf" in ctype:
                    upload_path = upload_path + ".pdf"
                elif "jpeg" in ctype or "jpg" in ctype:
                    upload_path = upload_path + ".jpg"
                elif "png" in ctype:
                    upload_path = upload_path + ".png"
                # otherwise leave as-is (will be validated later)

            # Save file
            with open(upload_path, "wb") as fw:
                fw.write(content)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error downloading file: {e}")

    # Basic file type verification after download
    ext = Path(upload_path).suffix.lower()
    if ext not in allowed:
        # try to proceed but warn / reject
        raise HTTPException(status_code=400, detail=f"Unsupported or unknown file type: {ext}")

    processor = InvoiceProcessor(debug=True)
    try:
        # Process in threadpool (same as /process-invoice)
        try:
            pages_data, consolidated = await run_in_threadpool(processor.process_invoice, upload_path)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Processing error: {e}")

        # --- Map inward entries (same mapping as /process-invoice) ---
        try:
            inward_mapped = map_ocr_to_inward(consolidated)
        except Exception as e:
            inward_mapped = {}
            print("Mapping error:", e)

        # --- Save result JSON (match /process-invoice naming & payload) ---
        result_name = f"{ts}_{Path(filename).stem}_result.json"
        result_path = os.path.join(RESULTS_DIR, result_name)
        result_payload = {
            "inward_mapped": inward_mapped,
            "processing_info": {
                "total_pages": len(pages_data),
                "processed_at": datetime.now().isoformat(),
                "source_file": fileurl
            }
        }
        with open(result_path, "w", encoding="utf-8") as rf:
            json.dump(result_payload, rf, indent=2, ensure_ascii=False)

        # --- Convert consolidated to Pydantic model (allowing numeric/str) ---
        try:
            items = [LineItem(**it) for it in consolidated.get("items", [])]
            cons_copy = dict(consolidated)
            cons_copy["items"] = items
            model_data = ConsolidatedInvoice(**cons_copy)
            clean_data = replace_none_with_zero(model_data.dict())
        except Exception as e:
            print("Model conversion error:", e)
            clean_data = replace_none_with_zero(consolidated)

        # Return same response shape and fields as /process-invoice
        return ProcessingResponse(
            success=True,
            message=f"Processed {len(pages_data)} page(s)",
            # data=clean_data,
            details={
                "pages_processed": len(pages_data),
                "result_file": result_name,
                "inward_data": inward_mapped
            }
        )

    finally:
        # cleanup same as /process-invoice
        processor.cleanup()
        try:
            if os.path.exists(upload_path):
                os.remove(upload_path)
        except Exception:
            pass



@app.get("/results/{filename}", dependencies=[Depends(verify_token)])
async def get_result(filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(path, media_type="application/json")

#----------menu end_point----------

# app = FastAPI()

@app.post("/process-menu", response_model=ProcessingResponse, dependencies=[Depends(verify_token)])
async def process_menu(file: UploadFile = File(...),
    token: str = Header(None, alias="Authorization")

):
    print(Request.headers)

    # --- Validate Token (explicit, not hidden dependency) ---
    verify_token(token)
    """Process a menu image and return Excel output."""
    # Validate file
    ext = Path(file.filename).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Only JPG/PNG images allowed")

    # Save temp file
    IMAGE_FILE = f"temp_{file.filename}"
    with open(IMAGE_FILE, "wb") as f:
        f.write(await file.read())

    # ---- RUN YOUR EXISTING PIPELINE ----
    try:
        text_output = call_bedrock_model(EXTRACTION_PROMPT, image_path=IMAGE_FILE)
        parsed = extract_json_from_text(text_output)
        items = parsed.get("items", [])

        for it in items:
            it.setdefault("prices", [])
            it.setdefault("price_headers", [])
            it.setdefault("modifiers", [])

        product_rows = build_product_rows(items)
        modifier_rows = build_modifier_rows(items)

        prod_cols = list(pd.read_excel(REFERENCE_TEMPLATE, sheet_name="Product").columns)
        mod_cols = list(pd.read_excel(REFERENCE_TEMPLATE, sheet_name="ModifierGroups_NewMenu").columns)

        df_prod = pd.DataFrame(product_rows)
        df_mod = pd.DataFrame(modifier_rows)

        for c in prod_cols:
            if c not in df_prod.columns:
                df_prod[c] = ""

        for c in mod_cols:
            if c not in df_mod.columns:
                df_mod[c] = ""

        df_prod = df_prod[prod_cols]
        df_mod = df_mod[mod_cols]

        out_path = get_next_output_filename()

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df_prod.to_excel(writer, sheet_name="Product", index=False)
            df_mod.to_excel(writer, sheet_name="ModifierGroups_NewMenu", index=False)

    finally:
        # Remove temp image
        if os.path.exists(IMAGE_FILE):
            os.remove(IMAGE_FILE)

    return {
        "success": True,
        "message": "Menu processed successfully",
        "output_file": out_path
    }


@app.get("/download")
def download_file(file: str):
    """Download previously generated Excel file."""
    if not os.path.exists(file):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file, filename=os.path.basename(file))


# ------------------- Run -------------------
if __name__ == "__main__":
    # Make sure AWS credentials are set in environment or instance profile
    missing = not (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)
    if missing:
        print("WARNING: AWS credentials not set in environment. Make sure boto3 can authenticate (env vars or IAM role).")

    print(f"STATIC_API_TOKEN set: {'yes' if STATIC_API_TOKEN else 'no'}")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

