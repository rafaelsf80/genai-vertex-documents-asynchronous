# chroma requires python3.8+
FROM python:3.9

RUN pip install gradio==3.48.0
RUN pip install chromadb==0.3.25 langchain==0.0.194 unstructured==0.6.6 tabulate==0.9.0 pdf2image==1.16.3 pytesseract==0.3.10
RUN pip install google-cloud-aiplatform==1.25.0 google-cloud-logging google-cloud-documentai==2.0.3 google-cloud-storage

COPY ./gradioapp /app

WORKDIR /app

EXPOSE 7860

CMD ["python", "app.py"]