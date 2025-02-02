import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

from huggingface_hub import InferenceEndpoint, get_token
from jinja2 import Environment, FileSystemLoader
from loguru import logger

template_dir = "./templates"
template_file = "classification-analysis.js.j2"

output_file = Path("./generated").resolve() / "classification-analysis.js"

# I hardcoded this because for most people this is all that will be tested. Do send me a PR if you need more.
image_dict = {
    'michaelf34/infinity:0.0.75-trt-onnx': 'trt-onnx',
    'michaelf34/infinity:0.0.75': 'default',
    }


def call_k6(
        endpoint: InferenceEndpoint,
        text_column: str,
        vus: int,
        total_requests: int,
        template_file: str,
        output_file: Path,
        dataset_path: str,
        k6_bin: str
        ) -> float:
    """
    Runs a k6 performance test on a given endpoint using a specified template.

    Args:
        endpoint: Endpoint object containing model and compute metadata.
        text_column (str): Name of the text column in the dataset.
        vus (int): Number of virtual users for k6 load testing.
        total_requests (int): Total number of requests to simulate.
        template_file (str): Jinja2 template file used for generating test scripts.
        output_file (Path): Path to the generated JavaScript test script.
        dataset_path (str): Path to the dataset file.
        k6_bin (str): Path to the k6 binary.

    Returns:
        float: The measured throughput in requests per second.
    """
    # Determine the task type based on the template file name
    if 'classification' in template_file:
        task = 'classification'
    elif 'vision-embedding' in template_file:
        task = 'vision-embedding'
    elif 'embedding' in template_file:
        task = 'embedding'
    else:
        raise ValueError('Unknown task type in template file')

    # Load Jinja2 template for script generation
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_file)

    # Extract relevant metadata from the endpoint
    image = endpoint.__dict__['raw']['model']['image']['custom']['url']
    image_short = image_dict.get(image, 'other_image')
    hw_type = endpoint.__dict__['raw']['compute']['instanceType']
    vendor = endpoint.__dict__['raw']['provider']['vendor']
    batch_size = endpoint.__dict__['raw']['model']['env']['INFINITY_BATCH_SIZE']
    engine = endpoint.__dict__['raw']['model']['env']['INFINITY_ENGINE']

    results_file = Path(
        "./results").resolve() / task / f'{hw_type}' / f'{vendor}_{hw_type}_{image_short}_{engine}_{batch_size}_{vus}.json'

    if results_file.exists():
        logger.info(f"Results file {results_file} already exists. Loading existing data.")
        with open(results_file) as f:
            data = json.load(f)
            return data.get("throughput_req_per_sec", 0)

    # Render the test script from the template
    rendered_script = template.render(
            text_column=text_column,
            host=endpoint.url,
            data_file=str(Path(dataset_path).resolve()),
            results_file=str(results_file),
            pre_allocated_vus=vus,
            total_requests=total_requests,
            hw_type=hw_type,
            batch_size=batch_size,
            vendor=vendor,
            image=image,
            engine=engine,
            duration="1m"
            )

    # Ensure necessary directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # Write the generated script to file
    with open(str(output_file), "w") as f:
        f.write(rendered_script)

    # Execute the k6 load test
    logger.info(f"Running k6 test with {vus} VUs")
    K6_BIN = os.path.expanduser(k6_bin)
    process = subprocess.run(
            [K6_BIN, "run", str(output_file)],
            env={'HF_TOKEN': get_token(), **os.environ},
            capture_output=True,
            text=True
            )
    logger.info(f"Results written to {results_file}")

    # Load and return the throughput result
    try:
        with open(results_file) as f:
            data = json.load(f)
            return data.get("throughput_req_per_sec", 0)
    except Exception as e:
        logger.error(f"Failed to read results file: {e}")
        return 0


def optimal_vus(
        max_vus: int,
        args_dict: Dict[str, Any],
        start_vus: int = 1
        ) -> int:
    """
    Finds the optimal number of virtual users (VUs) for maximum throughput.

    Args:
        max_vus (int): Maximum number of virtual users to test.
        args_dict (dict): Dictionary of arguments to pass to `call_k6`.
        start_vus (int): Initial number of VUs for testing (default: 1).

    Returns:
        int: Optimal number of VUs.
    """
    vus = start_vus
    prev_throughput = 0
    vus_history = []

    logger.info("Starting exponential search for optimal VUs")
    while vus <= max_vus:
        logger.info(f"Testing with {vus} VUs")
        throughput = call_k6(vus=vus, **args_dict)
        vus_history.append((vus, throughput))
        logger.info(f"Throughput for {vus} VUs: {throughput:.2f} req/sec")

        if throughput < prev_throughput * 1.02:  # Stop if improvement is <2%
            logger.info("Throughput improvement is less than 2%, stopping search.")
            break

        prev_throughput = throughput
        vus *= 2  # Double the VUs

    if vus > max_vus:
        logger.info(f"Reached maximum VU limit: {max_vus}")

    # Binary search refinement
    logger.info("Starting binary search refinement")
    low, high = vus_history[-2][0], vus_history[-1][0]
    while low < high:
        mid = (low + high) // 2
        logger.info(f"Testing with {mid} VUs")
        throughput = call_k6(vus=mid, **args_dict)

        if throughput > prev_throughput:
            logger.info(f"Throughput improved to {throughput:.2f} req/sec with {mid} VUs")
            prev_throughput = throughput
            low = mid + 1
        else:
            logger.info(f"Throughput did not improve with {mid} VUs")
            high = mid - 1

    optimal_vus = low
    logger.info(f"Optimal VUs determined: {optimal_vus}")
    return optimal_vus
