import io
from PyPDF2 import PdfReader
from docx import Document
from typing import Union
import pandas as pd

class DocumentParser:
    """文档解析器"""

    SUPPORTED_TYPES = ['.pdf', '.txt', '.docx', '.xlsx', '.xls']

    @staticmethod
    def parse(file_content: bytes, filename: str) -> str:
        """解析文档，返回文本内容"""
        ext = filename.lower().split('.')[-1]

        try:
            if ext == 'pdf':
                return DocumentParser._parse_pdf(file_content)
            elif ext == 'txt':
                return file_content.decode('utf-8')
            elif ext == 'docx':
                return DocumentParser._parse_docx(file_content)
            elif ext in ['xlsx', 'xls']:
                return DocumentParser._parse_excel(file_content)
            else:
                raise ValueError(f"不支持的文件类型: {ext}")
        except Exception as e:
            raise ValueError(f"文档解析失败: {str(e)}")

    @staticmethod
    def _parse_pdf(content: bytes) -> str:
        pdf_file = io.BytesIO(content)
        reader = PdfReader(pdf_file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
        return text.strip()

    @staticmethod
    def _parse_docx(content: bytes) -> str:
        doc = Document(io.BytesIO(content))
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text.strip()

    @staticmethod
    def _parse_excel(content: bytes) -> str:
        """解析 Excel 文件，提取所有 sheet 的内容"""
        df_dict = pd.read_excel(io.BytesIO(content), sheet_name=None)
        text = ''

        for sheet_name, sheet_df in df_dict.items():
            text += f"\n=== Sheet: {sheet_name} ===\n"

            # 添加表头
            if not sheet_df.empty:
                headers = sheet_df.columns.tolist()
                text += '表头: ' + ' | '.join(str(h) for h in headers) + '\n'

                # 添加每一行
                for idx, row in sheet_df.iterrows():
                    row_text = ' | '.join(str(v) for v in row.values)
                    text += f"行{idx + 1}: {row_text}\n"

        return text.strip()

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """将文本分块，便于向量化"""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap

        return chunks
