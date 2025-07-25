
import base64
import fitz
from PIL import Image
import os
from typing import List, Dict
import json
import requests
from groq import Groq

class DocumentOCR:
    def __init__(self, groq_api_key: str = None, openrouter_api_key: str = None, llm_provider: str = "groq"):
        """
        Initializes the DocumentOCR class.
        Args:
            groq_api_key (str): The Groq API key. If not provided, it will
                                look for the GROQ_API_KEY environment variable.
            openrouter_api_key (str): The OpenRouter API key. If not provided, it will
                                      look for the OPENROUTER_API_KEY environment variable.
            llm_provider (str): The LLM provider to use ('groq' or 'openrouter'). Defaults to 'groq'.
        """
        self.groq_client = Groq(api_key=groq_api_key)
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.llm_provider = llm_provider

    def encode_image(self, image_path: str) -> str:
        """Encodes an image file to a base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _extract_text_from_image_groq(self, image_path: str) -> Dict:
        """
        Extracts text from an image using the Groq API.
        """
        base64_image = self.encode_image(image_path)

        chat_completion = self.groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Perform OCR on this image and return only the extracted text."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
        )

        extracted_text = chat_completion.choices[0].message.content

        extracted_data = {
            'file_path': image_path,
            'full_text': extracted_text,
        }
        return extracted_data

    def _extract_text_from_image_openrouter(self, image_path: str) -> Dict:
        """
        Extracts text from an image using the OpenRouter API.
        """
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key not provided.")

        base64_image = self.encode_image(image_path)
        data_url = f"data:image/jpeg;base64,{base64_image}" 

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Perform OCR on this image and return only the extracted text."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]
            }
        ]

        payload = {
            "model": "google/gemma-3-27b-it", 
            "messages": messages
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() # Raise an exception for HTTP errors
        response_json = response.json()

        extracted_text = response_json['choices'][0]['message']['content']

        extracted_data = {
            'file_path': image_path,
            'full_text': extracted_text,
        }
        return extracted_data

    def extract_text_from_image(self, image_path: str) -> Dict:
        """
        Extracts text from an image using the selected LLM provider.
        """
        if self.llm_provider == "groq":
            return self._extract_text_from_image_groq(image_path)
        elif self.llm_provider == "openrouter":
            return self._extract_text_from_image_openrouter(image_path)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}. Choose 'groq' or 'openrouter'.")

    def process_pdf_pages(self, pdf_path: str) -> List[Dict]:
        """
        Processes each page of a PDF, extracts text using the selected LLM provider for OCR.
        """
        doc = fitz.open(pdf_path)
        results = []
        temp_dir = "/tmp/ocr_pages"
        os.makedirs(temp_dir, exist_ok=True)
        
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            temp_path = os.path.join(temp_dir, f"page_{i}.png")
            pix.save(temp_path)
            
            try:
                page_data = self.extract_text_from_image(temp_path)
                page_data['page_number'] = i + 1
                # We update the file_path to reflect the original PDF for better tracking
                # Store only the original PDF filename, not with page number suffix
                page_data['file_path'] = os.path.basename(pdf_path)
                results.append(page_data)
            finally:
                # Ensure temporary file is always removed
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        return results
