import time
from dataclasses import asdict

from huggingface_hub import create_inference_endpoint, whoami
from loguru import logger

endpoint_name = 'encoder-analysis'


def deploy_endpoint(instance_config):
    """Creates and deploys an inference endpoint using the given instance configuration."""
    try:
        logger.info("Creating inference endpoint...")
        start_time = time.time()  # Record start time
        endpoint = create_inference_endpoint(
                endpoint_name,
                namespace=whoami()['name'],
                framework="pytorch",
                task='text-classification',
                min_replica=0,
                max_replica=1,
                scale_to_zero_timeout=30,
                type="protected",
                **asdict(instance_config)
                )

    except Exception as e:
        logger.error(f"Failed to create inference endpoint: {e}")
        raise

    logger.info("Waiting for endpoint to be ready...")
    endpoint.wait()  # Wait for the endpoint to be ready

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    elapsed_minutes, elapsed_seconds = divmod(elapsed_time, 60)

    hw_type = endpoint.__dict__['raw']['compute']['instanceType']
    batch_size = endpoint.__dict__['raw']['model']['env']['INFINITY_BATCH_SIZE']
    logger.success(
            f"Endpoint created successfully: hw={hw_type}\tbs={batch_size}\t"
            f"Time taken: {int(elapsed_minutes)}m {elapsed_seconds:.2f}s"
            )
    return endpoint
