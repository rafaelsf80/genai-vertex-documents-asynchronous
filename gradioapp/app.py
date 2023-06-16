from google.api_core.exceptions import InternalServerError
from google.api_core.exceptions import RetryError
from google.api_core.client_options import ClientOptions

from google.cloud import logging
from google.cloud import storage
from google.cloud import documentai_v1 as documentai

import re

import gradio as gr

from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import Chroma


PROJECT_ID = "argolis-rafaelsanchez-ml-dev"

client = logging.Client(project=PROJECT_ID)
client.setup_logging()

log_name = "genai-vertex-large-unstructured-log"
logger = client.logger(log_name)

logger.log_text(f"Please, upload a file first") # set first log entry


def ocr_batch_parser(file):

    FILE_NAME        = file.name.split('/')[-1] # Getting filename only, since file type is tempfile._TemporaryFileWrapper
    LOCATION         = "eu"
    PROCESSOR_ID     = "a99d341e2c8c2e1c" # ocr processor
    GCS_INPUT_BUCKET = "argolis-documentai-latam"
    GCS_OUTPUT_URI   = "gs://argolis-documentai-latam"
    INPUT_MIME_TYPE  = "application/pdf"
    TIMEOUT          = 8000
    FIELD_MASK       = "text,entities,pages.pageNumber"  # Optional. The fields to return in the Document object.
   
    # Upload file to GCS
    client = storage.Client(project=PROJECT_ID)
    bucket = client.get_bucket(GCS_INPUT_BUCKET)
    blob = bucket.blob(FILE_NAME)
    blob.upload_from_filename(file.name) # full local name

    # GCS_INPUT_URI format "gs://argolis-documentai-latam/Annual-Report-BBVA_2022_ENG.pdf"
    GCS_INPUT_URI = f'gs://{GCS_INPUT_BUCKET}/{FILE_NAME}'
    logger.log_text(f"Batch processing: {GCS_INPUT_URI}")
    print(f"Batch processing: {GCS_INPUT_URI}")
    # You must set the api_endpoint if you use a location other than "us".
    opts = ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com")

    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    if not GCS_INPUT_URI.endswith("/") and "." in GCS_INPUT_URI:
        # Specify specific GCS URIs to process individual documents
        gcs_document = documentai.GcsDocument(
            gcs_uri=GCS_INPUT_URI, mime_type=INPUT_MIME_TYPE
        )
        # Load GCS Input URI into a List of document files
        gcs_documents = documentai.GcsDocuments(documents=[gcs_document])
        input_config = documentai.BatchDocumentsInputConfig(gcs_documents=gcs_documents)
    else:
        # Specify a GCS URI Prefix to process an entire directory
        gcs_prefix = documentai.GcsPrefix(gcs_uri_prefix=GCS_INPUT_URI)
        input_config = documentai.BatchDocumentsInputConfig(gcs_prefix=gcs_prefix)

    # Cloud Storage URI for the Output Directory
    logger.log_text(f"Output directory: {GCS_OUTPUT_URI}")
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
    # This could take some time for larger files
    # Format: projects/{project_id}/locations/{location}/operations/{operation_id}
    try:
        print(f"Waiting for operation {operation.operation.name} to complete...")
        logger.log_text(f"Waiting for operation {operation.operation.name} to complete...")
        operation.result(timeout=TIMEOUT)
    # Catch exception when operation doesn"t finish before timeout
    except (RetryError, InternalServerError) as e:
        print(e.message)

    metadata = documentai.BatchProcessMetadata(operation.metadata)

    if metadata.state != documentai.BatchProcessMetadata.State.SUCCEEDED:
        raise ValueError(f"Batch Process Failed: {metadata.state_message}")

    storage_client = storage.Client()

    logger.log_text("Processing outputs")
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
            logger.log_text(f"Fetching {blob.name}")
            document = documentai.Document.from_json(
                blob.download_as_bytes(), ignore_unknown_fields=True
            )

            # For a full list of Document object attributes, please reference this page:
            # https://cloud.google.com/python/docs/reference/documentai/latest/google.cloud.documentai_v1.types.Document

            # Read the text recognition output from the processor
            logger.log_text("The json contains the following text:")
            logger.log_text(f"Processing text from json: {document.text}")
            f.write(document.text)
            
        f.close()

        create_chroma_index()



def create_chroma_index():
   
    # Ingest OCR output (txt format)
    from langchain.document_loaders import PyPDFLoader
    from langchain.document_loaders import UnstructuredFileLoader

    logger.log_text("Please, wait. Document is being indexed ...")
    loader = UnstructuredFileLoader("output_all.txt")
    documents = loader.load()

    # split the documents into chunks
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    #docs = text_splitter.create_documents([doc_mexico])
    logger.log_text(f"# of documents = {len(docs)}")

    db.add_documents(documents=docs, embedding=embedding)

    logger.log_text("INDEX COMPLETED. You can now query the doc by clicking the Submit button")



def formatter(result):
    logger.log_text(f"Query: {result['query']}")
    logger.log_text("."*80)
    logger.log_text(f"Response: {result['result']}")
    if 'source_documents' in result.keys():
      logger.log_text("."*80)
      logger.log_text(f"References: {result['source_documents']}")


def retrieval_query(prompt):

    # Get last log. Check if index is completed
    *_, last = logger.list_entries()
    if not "INDEX COMPLETED" in last.payload:
       return "Batch process not completed. Please, wait", ""

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
        # chain_type_kwargs={"prompt": PromptTemplate(
        #         template=template,
        #         input_variables=["context", "question"],
        #     ),},
        return_source_documents=True)

    result = qa({"query": prompt})
    #logger.log_text(formatter(result))

    return result['result'], result['source_documents']


llm = VertexAI(
        model_name='text-bison@001',
        max_output_tokens=256,
        temperature=0.1,
        top_p=0.8,top_k=40,
        verbose=True,
    )

REQUESTS_PER_MINUTE = 150

embedding = VertexAIEmbeddings(requests_per_minute=REQUESTS_PER_MINUTE)

db = Chroma(collection_name="langchain", embedding_function=embedding)

def update_logs():
    # for entry in logger.list_entries():
    #     timestamp = entry.timestamp.isoformat()
    #     print("* {}: {}".format(timestamp, entry.payload))
    *_, last = logger.list_entries() # for a better understanding check PEP 448
    timestamp = last.timestamp.isoformat()
    return "* {}: {}".format(timestamp, last.payload)

demo = gr.Blocks()
with demo:
    gr.Markdown("# DOCUMENT SEMANTIC SEARCH DEMO (LARGE DOCUMENTS)")

    docai_file = gr.File(label="Upload large doc", type="file")
  
    gr.Markdown("### PROMPT: Ask questions on the doc. Example: 'What was BBVA net income in 2022?', 'How BBVA help its customers improve their financial health'")
    prompt = gr.Textbox(label="Prompt")
       
    b = gr.Button("Submit", variant="primary")
    
    answer = gr.Textbox(label="Output", variant="secondary")
    sources = gr.Textbox(label="Sources", max_lines=20)

    docai_file.change(ocr_batch_parser, inputs=docai_file)

    logs = gr.Textbox(label="Logs (updated every 20-30 seconds)", variant="secondary")
    demo.load(update_logs, None, logs, every=5)

    b.click(retrieval_query, inputs=prompt, outputs=[answer, sources])

demo.queue().launch(server_name="0.0.0.0", server_port=7860)

