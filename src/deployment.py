import time
from dataclasses import asdict
from typing import Any, Dict

from huggingface_hub import create_inference_endpoint, get_inference_endpoint, whoami
from loguru import logger


def deploy_endpoint(
        instance_config: Dict[str, Any],
        endpoint_name: str,
        wait: bool = False
        ):
    """Creates and deploys an inference endpoint with the given configuration.

    Args:
        instance_config (Dict[str, Any]): Configuration for the endpoint.
        endpoint_name (str): Name of the endpoint.
        wait (bool, optional): Whether to wait for deployment. Defaults to False.

    Returns:
        Any: The endpoint object or None if creation fails.
    """

    # Try to Re-Use an endpoint
    namespace = whoami()['name']
    try:
        endpoint = get_inference_endpoint(endpoint_name, namespace=namespace)
        hw_type = endpoint.__dict__['raw']['compute']['instanceType']
        batch_size = endpoint.__dict__['raw']['model']['env']['INFINITY_BATCH_SIZE']
        logger.success(f"Re-using Endpoint: hw={hw_type}\tbs={batch_size}\t")
        return endpoint
    except Exception as e:
        logger.warning(f"Endpoint not found. Proceeding with creation: {e}")

    # If that doesnt work try to create one
    try:
        logger.info("Creating inference endpoint...")
        start_time = time.time()  # Record start time
        endpoint = create_inference_endpoint(
                endpoint_name,
                namespace=namespace,
                # namespace='HF-test-lab',
                framework="pytorch",
                task='text-classification',
                min_replica=0,
                max_replica=1,
                scale_to_zero_timeout=300,
                type="protected",
                **asdict(instance_config)
                )
    except Exception as e:
        logger.error(f"Failed to create inference endpoint: {e}")
        return None

    logger.info("Waiting for endpoint to be ready...")
    if not wait:
        logger.info("Waiting for endpoint to be ready...")
        return endpoint
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
