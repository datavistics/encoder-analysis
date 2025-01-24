import time
from dataclasses import asdict

from huggingface_hub import create_inference_endpoint, get_inference_endpoint, whoami
from loguru import logger

endpoint_name = 'encoder-analysis'


def deploy_endpoint(instance_config):
    """
    Deploys or updates a Hugging Face inference endpoint.

    This function checks if an existing inference endpoint with the specified name exists.
    - If found, it updates the endpoint with the provided instance configuration.
    - If not found, it creates a new inference endpoint with the given parameters.

    Once the endpoint is updated or created, it waits until the endpoint is fully ready.

    Args:
        instance_config (InstanceConfig): A dataclass containing instance configuration details,
                                          such as accelerator type, vendor, region, and instance size.

    Returns:
        InferenceEndpoint: The deployed Hugging Face inference endpoint object.

    Raises:
        Exception: If the endpoint creation process fails.
    """
    try:
        endpoint = get_inference_endpoint(endpoint_name)
        logger.info("Re-using existing inference endpoint...")
        endpoint.update(**asdict(instance_config))
    except:
        logger.info("Starting endpoint deployment process")
        try:
            logger.info("Creating inference endpoint...")
            endpoint = create_inference_endpoint(
                    endpoint_name,
                    # repository=MODEL,
                    vendor='aws',
                    region='us-east-1',
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
    endpoint.wait()
    logger.success(f"Endpoint created successfully: {endpoint.url}")
    return endpoint


def deploy_endpoint(instance_config):
    """
    Deploys or updates a Hugging Face inference endpoint.

    This function checks if an existing inference endpoint with the specified name exists.
    - If found, it updates the endpoint with the provided instance configuration.
    - If not found, it creates a new inference endpoint with the given parameters.

    Once the endpoint is updated or created, it waits until the endpoint is fully ready.

    Args:
        instance_config (InstanceConfig): A dataclass containing instance configuration details,
                                          such as accelerator type, vendor, region, and instance size.

    Returns:
        InferenceEndpoint: The deployed Hugging Face inference endpoint object.

    Raises:
        Exception: If the endpoint creation process fails.
    """
    try:
        logger.info("Creating inference endpoint...")
        start_time = time.time()  # Record start time
        endpoint = create_inference_endpoint(
                endpoint_name,
                # repository=MODEL,
                vendor='aws',
                region='us-east-1',
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
    batch_size = endpoint.__dict__['raw']['model']['env']['BATCH_SIZE']
    logger.success(
            f"Endpoint created successfully: hw={hw_type}\tbs={batch_size}\t"
            f"Time taken: {int(elapsed_minutes)}m {elapsed_seconds:.2f}s"
            )
    return endpoint
