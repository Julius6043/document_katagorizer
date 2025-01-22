from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_openai import OpenAIEmbeddings
import os
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from PyPDF2 import PdfReader
import httpx
from langchain_core.documents import Document
import json
import re
from Bio import Medline


def load_pdf(file_name, chunk_size=1000, chunk_overlap=200):
    # Sicherstellen, dass der Pfad korrekt formatiert ist
    path = file_name

    # Lade das PDF-Dokument
    loader = PyMuPDFLoader(path)
    docs = loader.load()
    content = docs
    # Initialisiere den Text-Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    # Splitte die Dokumente in Chunks
    split_chunks = text_splitter.split_documents(docs)
    return split_chunks, content


def list_files_in_folder(folder_path):
    return os.listdir(folder_path)


def parse_pubmed_nbib(nbib_file):
    records_data = []
    with open(nbib_file, "r", encoding="utf-8") as handle:
        records = Medline.parse(handle)
        for record in records:
            pmid = record.get("PMID", "")
            title = record.get("TI", "")
            abstract = record.get("AB", "")
            authors = record.get("AU", [])

            record_string = f"PMID: {pmid}\nTitle: {title}\nAbstract: {abstract}\nAuthors: {', '.join(authors)}\n"
            records_data.append(record_string)
    return records_data


def load_folder(path):
    files = list_files_in_folder(path)
    print(files)
    pubmed = []
    pdfs = []
    for file in files:
        if file.endswith(".pdf"):
            chunks, content = load_pdf(f"{path}/{file}")
            print(chunks)
            print(content)
            pdfs.append(content)
        elif file.endswith(".nbib"):
            records = parse_pubmed_nbib(f"{path}/{file}")
            pubmed.extend(records)
            # print(records[0])
        else:
            print(f"Datei {file} nicht unterstützt")
    return pubmed, pdfs


if __name__ == "__main__":
    data = load_folder("Datein")
    print(data)
