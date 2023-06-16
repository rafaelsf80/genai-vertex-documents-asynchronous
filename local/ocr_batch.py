"""
    Makes a Batch Processing Request to Document AI OCR
"""

import re

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import InternalServerError
from google.api_core.exceptions import RetryError
from google.cloud import documentai
from google.cloud import storage

import google.cloud.logging

project_id = "argolis-rafaelsanchez-ml-dev"
location = "us"  # Format is "us" or "eu"
processor_id = "80f1bd0f1b6b7f77"  # Create processor before running sample
gcs_output_uri = "gs://argolis-documentai-latam/"  # Must end with a trailing slash `/`. Format: gs://bucket/directory/subdirectory/

# TODO(developer): If `gcs_input_uri` is a single file, `mime_type` must be specified.
gcs_input_uri = "gs://argolis-documentai-latam/Annual-Report-BBVA_2022_ENG.pdf"  # Format: `gs://bucket/directory/file.pdf` or `gs://bucket/directory/`
input_mime_type = "application/pdf"
field_mask = "text,entities,pages.pageNumber"  # Optional. The fields to return in the Document object.


log_client = google.cloud.logging.Client(project=project_id)
log_client.setup_logging()

log_name = "genai-vertex-unstructured-log"
logger = log_client.logger(log_name)


def batch_process_documents(
    project_id: str,
    location: str,
    processor_id: str,
    gcs_input_uri: str,
    gcs_output_uri: str,
    processor_version_id: str = None,
    input_mime_type: str = None,
    field_mask: str = None,
    timeout: int = 800,
):
    # You must set the api_endpoint if you use a location other than "us".
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")

    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    if not gcs_input_uri.endswith("/") and "." in gcs_input_uri:
        # Specify specific GCS URIs to process individual documents
        gcs_document = documentai.GcsDocument(
            gcs_uri=gcs_input_uri, mime_type=input_mime_type
        )
        # Load GCS Input URI into a List of document files
        gcs_documents = documentai.GcsDocuments(documents=[gcs_document])
        input_config = documentai.BatchDocumentsInputConfig(gcs_documents=gcs_documents)
    else:
        # Specify a GCS URI Prefix to process an entire directory
        gcs_prefix = documentai.GcsPrefix(gcs_uri_prefix=gcs_input_uri)
        input_config = documentai.BatchDocumentsInputConfig(gcs_prefix=gcs_prefix)

    # Cloud Storage URI for the Output Directory
    gcs_output_config = documentai.DocumentOutputConfig.GcsOutputConfig(
        gcs_uri=gcs_output_uri#, field_mask=field_mask
    )

    # Where to write results
    output_config = documentai.DocumentOutputConfig(gcs_output_config=gcs_output_config)

    if processor_version_id:
        # The full resource name of the processor version, e.g.:
        # projects/{project_id}/locations/{location}/processors/{processor_id}/processorVersions/{processor_version_id}
        name = client.processor_version_path(
            project_id, location, processor_id, processor_version_id
        )
    else:
        # The full resource name of the processor, e.g.:
        # projects/{project_id}/locations/{location}/processors/{processor_id}
        name = client.processor_path(project_id, location, processor_id)

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
        logger.log_text(f"Waiting for operation {operation.operation.name} to complete...")
        operation.result(timeout=timeout)
    # Catch exception when operation doesn"t finish before timeout
    except (RetryError, InternalServerError) as e:
        print(e.message)

    # NOTE: Can also use callbacks for asynchronous processing
    #
    # def my_callback(future):
    #   result = future.result()
    #
    # operation.add_done_callback(my_callback)

    # Once the operation is complete,
    # get output document information from operation metadata
    metadata = documentai.BatchProcessMetadata(operation.metadata)

    if metadata.state != documentai.BatchProcessMetadata.State.SUCCEEDED:
        raise ValueError(f"Batch Process Failed: {metadata.state_message}")

    storage_client = storage.Client()

    logger.log_text("Output files:")
    # One process per Input Document
    for process in list(metadata.individual_process_statuses):
        # output_gcs_destination format: gs://BUCKET/PREFIX/OPERATION_NUMBER/INPUT_FILE_NUMBER/
        # The Cloud Storage API requires the bucket name and URI prefix separately
        matches = re.match(r"gs://(.*?)/(.*)", process.output_gcs_destination)
        if not matches:
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
            logger.log_text(document.text)
            f.write(document.text)
            
        f.close()
      

if __name__ == "__main__":
    batch_process_documents(
        project_id=project_id,
        location=location,
        processor_id=processor_id,
        gcs_input_uri=gcs_input_uri,
        gcs_output_uri=gcs_output_uri,
        input_mime_type=input_mime_type,
        field_mask=field_mask,
    )