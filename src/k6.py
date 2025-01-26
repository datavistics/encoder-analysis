import json
import os
import subprocess
from pathlib import Path

from huggingface_hub import get_token
from jinja2 import Environment, FileSystemLoader
from loguru import logger

template_dir = "./templates"
template_file = "classification-analysis.js.j2"

output_file = Path("./generated").resolve() / "classification-analysis.js"


def call_k6(endpoint, text_column, vus, total_requests, template_file, output_file, dataset_path, k6_bin):
    if 'classification' in template_file:
        task = 'classification'
    elif 'embedding' in template_file:
        task = 'embedding'
    elif 'vision-embedding' in template_file:
        task = 'vision-embedding'
    else:
        raise ValueError('Unknown task')

    # Load the Jinja2 template
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_file)
    hw_type = endpoint.__dict__['raw']['compute']['instanceType']
    vendor = endpoint.__dict__['raw']['provider']['vendor']
    batch_size = endpoint.__dict__['raw']['model']['env']['INFINITY_BATCH_SIZE']
    engine = endpoint.__dict__['raw']['model']['env']['INFINITY_ENGINE']
    results_file = Path("./results").resolve() / task / f'{hw_type}' / f'{vendor}_{hw_type}_{engine}_{batch_size}_{vus}.json'
    if results_file.exists():
        logger.info(f"results file {results_file} already exists")
        with open(results_file) as f:
            data = json.load(f)
            return data.get("throughput_req_per_sec", 0)
    # Define values for template variables
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
            engine=engine,
            duration="1m"
            )
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    # Write the rendered JavaScript file
    with open(str(output_file), "w") as f:
        f.write(rendered_script)
    # Run k6 using the generated file
    K6_BIN = os.path.expanduser(k6_bin)  # Ensure correct path
    process = subprocess.run(
            [K6_BIN, "run", str(output_file)],
            env={'HF_TOKEN': get_token(), **os.environ},
            capture_output=True,
            text=True
            )
    logger.info(f'Results written to {results_file}')

    # Read results
    try:
        with open(results_file) as f:
            data = json.load(f)
            return data.get("throughput_req_per_sec", 0)
    except Exception:
        return 0


def optimal_vus(max_vus, args_dict, start_vus=1):
    # Exponential search
    vus = start_vus
    prev_throughput = 0
    vus_history = []
    logger.info("Starting exponential search for optimal VUs")
    while vus <= max_vus:
        logger.info(f"Testing with VUs: {vus}")
        throughput = call_k6(vus=vus, **args_dict)
        vus_history.append((vus, throughput))
        logger.info(f"Throughput for {vus} VUs: {throughput:.2f} req/sec")

        # Stop if adding more VUs does not significantly increase throughput
        if throughput < prev_throughput * 1.02:  # Less than 2% increase
            logger.info("Throughput improvement is less than 2%, stopping exponential search.")
            break

        prev_throughput = throughput
        vus *= 2  # Double VUs
    if vus > max_vus:
        logger.info(f"Reached maximum VU limit: {max_vus}")
    # Binary search refinement
    logger.info("Starting binary search refinement")
    low, high = vus_history[-2][0], vus_history[-1][0]
    while low < high:
        mid = (low + high) // 2
        logger.info(f"Testing with VUs: {mid}")
        throughput = call_k6(vus=mid, **args_dict)

        if throughput > prev_throughput:
            logger.info(f"Throughput improved to {throughput:.2f} req/sec with {mid} VUs")
            prev_throughput = throughput
            low = mid + 1
        else:
            logger.info(f"Throughput did not improve with {mid} VUs")
            high = mid - 1
    optimal_vus = low
    logger.info(f"Optimal VUs: {optimal_vus}")
