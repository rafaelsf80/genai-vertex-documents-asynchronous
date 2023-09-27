"""
    End-to-end Gradio app that implements this cycle:
    1. Load a pdf file.
    2. Perform Batch processing with Document AI OCR parser.
    3. Create the index with Chroma.
    4. After 20-30 min depending on the doc size (batch processing in Document AI OCR consumes time), the user can then write the query, and the Retrieval QA chain will recover the blocks closer to the query. A Vertex LLM model will write the output to the user.
    5. Make sure you refresh the indexes and select an index (DropDown list) before making the query.
"""

from google.api_core.exceptions import InternalServerError
from google.api_core.exceptions import RetryError
from google.api_core.client_options import ClientOptions

from google.cloud import logging
from google.cloud import storage
from google.cloud import documentai_v1 as documentai

import glob
import os
import re
import shutil

from pathlib import Path

import gradio as gr

from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import Chroma


PROJECT_ID          = "argolis-rafaelsanchez-ml-dev"  # <---- CHANGE THIS
GCS_INPUT_BUCKET    = "argolis-documentai-unstructured-large-input"  # <---- CHANGE THIS
GCS_CHROMADB_BUCKET = "argolis-documentai-unstructured-large-chromadb"  # <---- CHANGE THIS
LOCATION            = "eu"  # <---- CHANGE THIS
PROCESSOR_ID        = "a99d341e2c8c2e1c" # ocr processor  <---- CHANGE THIS
GCS_OUTPUT_URI      = "gs://argolis-documentai-unstructured-large-output"   # <---- CHANGE THIS


client = logging.Client(project=PROJECT_ID)
client.setup_logging()

log_name = "genai-vertex-large-unstructured-prod-log"     # <---- CHANGE THIS
logger = client.logger(log_name)

# output logs to terminal console when in local - otherwise logs are only visible when running in GCP
#if os.getenv("LOCAL_LOGGING", "False") == "True":
#    logger.addHandler(logging.StreamHandler())


def ocr_batch_parser(file):
    """ Performs OCR using OCR parser in Document AI 
        Make sure you change PROCESSOR_ID and LOCATION
    """

    FILE_NAME        = file.name.split('/')[-1] # Getting filename only, since file type is tempfile._TemporaryFileWrapper
    INPUT_MIME_TYPE  = "application/pdf"
    TIMEOUT          = 8000
    FIELD_MASK       = "text,entities,pages.pageNumber"  # Optional. The fields to return in the Document object.
   
    # Remove any previous Chroma DB
    if os.path.isdir('./.chroma'):
        shutil.rmtree('./.chroma')
    logger.log_text(f"STEP 1/9: Uploading file to GCS") 

    # Upload file to GCS
    client = storage.Client(project=PROJECT_ID)
    bucket = client.get_bucket(GCS_INPUT_BUCKET)
    blob = bucket.blob(FILE_NAME)
    blob.upload_from_filename(file.name) # full local name

    # GCS_INPUT_URI must have this format: gs://argolis-documentai-latam/Annual-Report-BBVA_2022_ENG.pdf
    GCS_INPUT_URI = f'gs://{GCS_INPUT_BUCKET}/{FILE_NAME}'
    logger.log_text(f"STEP 2/9: Batch processing: {GCS_INPUT_URI}")
    print(f"Batch processing: {GCS_INPUT_URI}")
    opts = ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com")

    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    if not GCS_INPUT_URI.endswith("/") and "." in GCS_INPUT_URI:
        gcs_document = documentai.GcsDocument(
            gcs_uri=GCS_INPUT_URI, mime_type=INPUT_MIME_TYPE
        )
        # Load GCS Input URI into a List of document files
        gcs_documents = documentai.GcsDocuments(documents=[gcs_document])
        input_config = documentai.BatchDocumentsInputConfig(gcs_documents=gcs_documents)
    else:
        gcs_prefix = documentai.GcsPrefix(gcs_uri_prefix=GCS_INPUT_URI)
        input_config = documentai.BatchDocumentsInputConfig(gcs_prefix=gcs_prefix)

    # Cloud Storage URI for the Output Directory
    logger.log_text(f"STEP 3/9: Output directory: {GCS_OUTPUT_URI}")
    print(f"Output directory: {GCS_OUTPUT_URI}")
    gcs_output_config = documentai.DocumentOutputConfig.GcsOutputConfig(
        gcs_uri=GCS_OUTPUT_URI
        #, field_mask=field_mask
    )

    # Where to write results
    output_config = documentai.DocumentOutputConfig(gcs_output_config=gcs_output_config)

    name = client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)

    request = documentai.BatchProcessRequest(
        name=name,
        input_documents=input_config,
        document_output_config=output_config,
    )

    # BatchProcess returns a Long Running Operation (LRO)
    operation = client.batch_process_documents(request)

    # Continually polls the operation until it is complete.
    try:
        print(f"Waiting for operation {operation.operation.name} to complete...")
        logger.log_text(f"STEP 4/9: Waiting for operation {operation.operation.name} to complete...")
        operation.result(timeout=TIMEOUT)
    # Catch exception when operation doesn"t finish before timeout
    except (RetryError, InternalServerError) as e:
        print(e.message)

    metadata = documentai.BatchProcessMetadata(operation.metadata)

    if metadata.state != documentai.BatchProcessMetadata.State.SUCCEEDED:
        raise ValueError(f"Batch Process Failed: {metadata.state_message}")

    storage_client = storage.Client()

    logger.log_text("STEP 5/9: Processing outputs")
    print(f"Processing outputs")

    # One process per Input Document
    for process in list(metadata.individual_process_statuses):
        # output_gcs_destination format: gs://BUCKET/PREFIX/OPERATION_NUMBER/INPUT_FILE_NUMBER/
        # The Cloud Storage API requires the bucket name and URI prefix separately
        matches = re.match(r"gs://(.*?)/(.*)", process.output_gcs_destination)
        if not matches:
            print( "Could not parse output GCS destination:",
                process.output_gcs_destination)
            logger.log_text(
                "Could not parse output GCS destination:",
                process.output_gcs_destination,
            )
            continue

        output_bucket, output_prefix = matches.groups()

        # Get List of Document Objects from the Output Bucket
        output_blobs = storage_client.list_blobs(output_bucket, prefix=output_prefix)

        # Document AI may output multiple JSON files per source file
        f = open(f'output_all.txt', 'w+')
        for blob in output_blobs:
            # Document AI should only output JSON files to GCS
            if blob.content_type != "application/json":
                print(
                    f"Skipping non-supported file: {blob.name} - Mimetype: {blob.content_type}"
                )
                continue

            # Download JSON File as bytes object and convert to Document Object
            logger.log_text(f"STEP 6/9: Fetching {blob.name}")
            document = documentai.Document.from_json(
                blob.download_as_bytes(), ignore_unknown_fields=True
            )

            logger.log_text("STEP 6/9: The json contains the following text:")
            logger.log_text(f"Processing text from json: {document.text}")
            f.write(document.text)
            
        f.close()

        create_chroma_index(FILE_NAME)


def create_chroma_index(file_name):
    """ Create Chroma index for file_name and uploads it to GCS_CHROMADB_BUCKET """
   
    from langchain.document_loaders import PyPDFLoader
    from langchain.document_loaders import UnstructuredFileLoader

    # Remove any previous Chroma DB
    if os.path.isdir('./.chroma'):
        shutil.rmtree('./.chroma')
    persist_directory=os.path.abspath("./.chromadb")

    # Now we can load the persisted database from disk, and use it as normal. 
    db = Chroma(collection_name="langchain", persist_directory=persist_directory, embedding_function=embedding)

    logger.log_text("STEP 7/9: Please, wait. Document is being indexed ...")
    
    loader = UnstructuredFileLoader("output_all.txt")
    documents = loader.load()

    # split the documents into chunks
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    logger.log_text(f"STEP 8/9: # of documents = {len(docs)}")
    db.add_documents(documents=docs, embedding=embedding)
    db.persist()  
    logger.log_text("STEP 8/9 completed")

    # Save chroma db to GCS, keeping folder structure
    rel_paths = glob.glob(persist_directory + '/**', recursive=True)

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(GCS_CHROMADB_BUCKET)
    for local_file in rel_paths:
        remote_path = f'{file_name}/{"/".join(local_file.split(os.sep)[6:])}'
        if os.path.isfile(local_file):
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)

    logger.log_text("STEP 9/9: INDEX COMPLETED. You can now query the doc by clicking the Submit button")


def formatter(result):
    """ Nice format for query-responses """

    logger.log_text(f"Query: {result['query']}")
    logger.log_text("."*80)
    logger.log_text(f"Response: {result['result']}")
    if 'source_documents' in result.keys():
      logger.log_text("."*80)
      logger.log_text(f"References: {result['source_documents']}")


def retrieval_query(prompt, doc_chromadb):
    """ Downloads ONE ChromaDB index from GCS_CHROMADB_BUCKET and makes retrieval locally """

    # Get last log. Check if index is completed
    #*_, last = logger.list_entries()
    #if not "INDEX COMPLETED" in last.payload:
    #   return "Batch process not completed. Please, wait", ""

    # if doc_chromadb == "" or "STEP 8/9" in last.payload:
    #     return "Wait for index to complete and then refresh and select an existing document"

    # If ChromaDB for this doc is not downloaded previously
    if not os.path.exists(doc_chromadb):

        # download gcs folder, keeping folder structure
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(GCS_CHROMADB_BUCKET)
        blobs = bucket.list_blobs(prefix=doc_chromadb)  # Get list of files
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            file_split = blob.name.split("/")
            directory = "/".join(file_split[0:-1])
            Path(directory).mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(blob.name) 

    persist_directory=os.path.abspath(f"./{doc_chromadb}/.chromadb")

    # Now we can load the persisted database from disk, and use it as normal. 
    db = Chroma(collection_name="langchain", persist_directory=persist_directory, embedding_function=embedding)

    # Expose index to the retriever
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k":2})

    # Create chain to answer questions
    from langchain.chains import RetrievalQA
    from langchain import PromptTemplate

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True)

    result = qa({"query": prompt})

    return result['result'], result['source_documents']

# def list_gcs_files(bucket_name):
#     """ List GCS folders in bucket_name containing all chromaDBs """

#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blobs = bucket.list_blobs()
#     #return gr.Dropdown.update(choices=[blob.name for blob in blobs if blob.name.endswith("/")])
#     return list(set([blob.name.split('/')[0] for blob in blobs]))


def refresh_chromadb_files():
    """ Refresh Dropdown list with GCS folders in GCS_CHROMADB_BUCKET containing all chromaDBs 
        This is a workaround for this issue related to a Dropdown list in Gradio
        https://github.com/gradio-app/gradio/issues/4210
        https://discuss.huggingface.co/t/how-to-update-the-gr-dropdown-block-in-gradio-blocks/19231
    """

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(GCS_CHROMADB_BUCKET)
    blobs = bucket.list_blobs()
    # list(set([0, 1, 2, 0, 2])) returns unique values [0, 1 , 2]
    return gr.Dropdown.update(choices=list(set([blob.name.split('/')[0] for blob in blobs])))



def update_logs():
    """ Refresh logs every 20-30 seconds """

    *_, last = logger.list_entries() # for a better understanding check PEP 448
    timestamp = last.timestamp.isoformat()
    return "* {}: {}".format(timestamp, last.payload)


llm = VertexAI(
        model_name='text-bison@001',
        max_output_tokens=256,
        temperature=0.1,
        top_p=0.8,top_k=40,
        verbose=True,
    )

REQUESTS_PER_MINUTE = 150

embedding = VertexAIEmbeddings(requests_per_minute=REQUESTS_PER_MINUTE)

demo = gr.Blocks()

with demo:
    gr.Markdown("# DOCUMENT SEMANTIC SEARCH DEMO (LARGE DOCUMENTS)")

    gr.Markdown("### INSTRUCTIONS: upload a new or select an existing document first. If new doc, when logs shows INDEX COMPLETED, you can make the query, but not before.")
    gr.Markdown("### PROMPT: Ask questions on the doc. Example: 'What was BBVA net income in 2022?', 'How BBVA help its customers improve their financial health'")

    doc_chromadb = gr.Dropdown(
            [], label="Documents", info="Select an existing document in GCS"
        )

    docai_file = gr.File(label="Upload large doc", type="file") 
    prompt = gr.Textbox(label="Prompt")
       
    b1 = gr.Button("Submit", variant="primary")
    b2 = gr.Button("Refresh documents in GCS")
    
    answer = gr.Textbox(label="Output", variant="secondary")
    sources = gr.Textbox(label="Sources", max_lines=20)

    docai_file.change(ocr_batch_parser, inputs=docai_file)

    logs = gr.Textbox(label="Logs (updated every 20-30 seconds)", variant="secondary")
    demo.load(update_logs, None, logs, every=5)

    b1.click(retrieval_query, inputs=[prompt, doc_chromadb], outputs=[answer, sources])
    b2.click(refresh_chromadb_files, outputs=doc_chromadb)

demo.queue().launch(server_name="0.0.0.0", server_port=7860)


