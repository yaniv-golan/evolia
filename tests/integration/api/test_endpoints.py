import os
import time
import json
from dotenv import load_dotenv, find_dotenv
import requests
import pytest

# Force reload environment variables
load_dotenv(find_dotenv(), override=True)
raw_token = os.getenv('HUGGING_FACE_TOKEN', '')
HUGGING_FACE_TOKEN = raw_token.strip()

ENDPOINTS = [
    {
        "namespace": "yaniv9",
        "name": "codellama-13b-instruct-hf-ljw",
        "url": "https://a33i6681s10kadib.us-east-1.aws.endpoints.huggingface.cloud/generate",
        "type": "instruct"
    },
    {
        "namespace": "yaniv9",
        "name": "codellama-13b-python-hf-wfq",
        "url": "https://e7y7ls68l75wwgrr.us-east-1.aws.endpoints.huggingface.cloud/generate",
        "type": "python"
    }
]

def check_endpoint_status(endpoint_name):
    """Check the status of an endpoint"""
    API_URL = f"https://api.endpoints.huggingface.cloud/v2/endpoint/{endpoint_name}"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_TOKEN}"}
    
    try:
        response = requests.get(API_URL, headers=headers)
        if response.status_code == 200:
            result = response.json()
            status = result.get("status", {}).get("state")
            message = result.get("status", {}).get("message", "")
            print(f"Endpoint {endpoint_name} status: {status}")
            if message:
                print(f"Message: {message}")
            return status
        else:
            print(f"Failed to get status for endpoint {endpoint_name}")
            print("Response:", response.text)
            return None
    except Exception as e:
        print(f"Error checking endpoint status: {str(e)}")
        return None

def wait_for_endpoint(endpoint_name, timeout=300, check_interval=10):
    """Wait for endpoint to be in RUNNING state"""
    print(f"\nWaiting for endpoint {endpoint_name} to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        status = check_endpoint_status(endpoint_name)
        if not status:
            print("Could not get endpoint status")
            return False
        
        status = status.lower()
        if status == "running":
            print(f"Endpoint {endpoint_name} is now running!")
            return True
        elif status in ["pending", "initializing"]:
            print(f"Endpoint {endpoint_name} is {status}, waiting {check_interval} seconds...")
            time.sleep(check_interval)
            continue
        else:
            print(f"Unexpected state {status}, please check manually")
            return False
    
    print(f"Timeout waiting for endpoint {endpoint_name}")
    return False

def generate_text(endpoint_url, prompt, max_new_tokens=500, temperature=0.7, top_p=0.95, max_retries=5, retry_delay=20):
    """Generate text using the model endpoint"""
    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1}/{max_retries}")
            
            response = requests.post(
                endpoint_url,
                headers={"Authorization": f"Bearer {HUGGING_FACE_TOKEN}"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "do_sample": True,
                        "return_full_text": False
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, dict) and "generated_text" in result:
                    return result["generated_text"]
                elif isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '')
                else:
                    print("\nUnexpected response format:", result)
                    return None
            
            # Handle model loading state
            try:
                response_json = response.json()
                if "error" in response_json and "is currently loading" in response_json["error"]:
                    estimated_time = response_json.get("estimated_time", retry_delay)
                    print(f"Model is loading. Estimated time: {estimated_time:.1f} seconds")
                    print(f"Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                    continue
            except json.JSONDecodeError:
                print("Could not parse response as JSON:", response.text)
            
            response.raise_for_status()
            
        except Exception as e:
            print(f"Error making request: {str(e)}")
            if hasattr(e, 'response'):
                print("Response:", e.response.text)
            return None
    
    print(f"\nFailed to get response after {max_retries} attempts")
    return None

def test_plan_generation():
    """Test plan generation with the Instruct model"""
    print("\nTesting plan generation...")
    
    # Find the instruct endpoint
    endpoint = next(ep for ep in ENDPOINTS if ep["type"] == "instruct")
    
    # Check endpoint status
    full_name = f"{endpoint['namespace']}/{endpoint['name']}"
    status = check_endpoint_status(full_name)
    if not status:
        pytest.skip(f"Could not get status for endpoint {full_name}")
    
    status = status.lower()
    if status != "running":
        pytest.skip(f"Endpoint {full_name} is not running (status: {status}). Please use resume_endpoints.py to start the endpoints first.")
    
    prompt = """[INST] Create a plan for implementing a web scraping tool that extracts product information from an e-commerce website.
Requirements:
1. Extract product name, price, and description
2. Handle pagination
3. Save data to CSV
4. Respect robots.txt and rate limiting

List the implementation steps in order. [/INST]"""
    
    response = generate_text(endpoint["url"], prompt, max_new_tokens=500)
    assert response is not None, "Failed to generate plan"
    print("\nGenerated plan:")
    print(response)

def test_code_generation():
    """Test code generation with the Python model"""
    print("\nTesting code generation...")
    
    # Find the python endpoint
    endpoint = next(ep for ep in ENDPOINTS if ep["type"] == "python")
    
    # Check endpoint status
    full_name = f"{endpoint['namespace']}/{endpoint['name']}"
    status = check_endpoint_status(full_name)
    if not status:
        pytest.skip(f"Could not get status for endpoint {full_name}")
    
    status = status.lower()
    if status != "running":
        pytest.skip(f"Endpoint {full_name} is not running (status: {status}). Please use resume_endpoints.py to start the endpoints first.")
    
    prompt = """[INST] Write a Python function to implement a binary search tree with insert and search operations.
Requirements:
1. Create a Node class for tree nodes
2. Implement insert method
3. Implement search method
4. Include type hints
5. Add docstrings

Write only the code, no explanations. [/INST]"""
    
    response = generate_text(endpoint["url"], prompt, max_new_tokens=1000)
    assert response is not None, "Failed to generate code"
    print("\nGenerated code:")
    print(response)