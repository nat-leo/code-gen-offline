import os
import json
from typing import Any
import time

from dotenv import load_dotenv
from google.cloud import firestore
from openai import OpenAI
import requests

load_dotenv()

def parse_scalar_or_json(s: str) -> Any:
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        return s


def parse_example_testcases(example_testcases: str, arity: int) -> list[list[Any]]:
    lines = [ln.strip() for ln in example_testcases.split("\n")]
    lines = [ln for ln in lines if ln]  # remove blank lines

    if arity <= 0:
        raise ValueError("Invalid arity")

    if len(lines) % arity != 0:
        raise ValueError(
            f"exampleTestcases line count ({len(lines)}) not divisible by arity ({arity})"
        )

    cases: list[list[Any]] = []
    for i in range(0, len(lines), arity):
        args = [parse_scalar_or_json(x) for x in lines[i : i + arity]]
        cases.append(args)
    return cases


def make_python_harness_from_examples(source_code: str, metadata_json_str: str, example_testcases: str) -> str:
    md = json.loads(metadata_json_str)
    param_names = [p["name"] for p in md.get("params", [])]
    arity = len(param_names)
    cases = parse_example_testcases(example_testcases, arity)

    # Embed CASES as JSON into the harness
    cases_json = json.dumps(cases)

    return f"""from typing import List
# ---- USER CODE (verbatim) ----
{source_code}

# ---- HARNESS (examples) ----
import json

METADATA = {json.dumps(metadata_json_str)}
md = json.loads(METADATA)

FUNC_NAME = md["name"]
PARAMS = [p["name"] for p in md.get("params", [])]

CASES = json.loads({json.dumps(cases_json)})

def main():
    sol = Solution()
    fn = getattr(sol, FUNC_NAME)

    for i, args in enumerate(CASES):
        result = fn(*args)
        print(json.dumps({{"i": i, "args": args, "result": result}}))

if __name__ == "__main__":
    main()
"""


FIELDS = "stdout,stderr,status,time,memory"

def judge0_submit(source_code: str, language_id: int, stdin: str = "") -> dict:
    base = os.environ["RAPIDAPI_BASE_URL"]      # e.g. https://judge0-ce.p.rapidapi.com
    key = os.environ["RAPIDAPI_KEY"]
    host = os.environ["RAPIDAPI_HOST"]          # e.g. judge0-ce.p.rapidapi.com

    url = f"{base}/submissions"
    params = {
        "base64_encoded": "false",
        "wait": "false",
        "fields": FIELDS,
    }
    headers = {
        "x-rapidapi-key": key,
        "x-rapidapi-host": host,
        "Content-Type": "application/json",
    }
    payload = {
        "source_code": source_code,
        "language_id": language_id,
        "stdin": stdin or "",
    }

    r = requests.post(url, params=params, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()  # typically contains {"token": "..."} when wait=false


def judge0_poll(token: str) -> dict:
    base = os.environ["RAPIDAPI_BASE_URL"]
    key = os.environ["RAPIDAPI_KEY"]
    host = os.environ["RAPIDAPI_HOST"]

    url = f"{base}/submissions/{token}"
    params = {
        "base64_encoded": "false",
        "fields": FIELDS,
    }
    headers = {
        "x-rapidapi-key": key,
        "x-rapidapi-host": host,
        "Content-Type": "application/json",
    }

    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def judge0_run_and_wait(source_code: str, language_id: int, stdin: str = "", poll_s: float = 0.5, timeout_s: float = 30.0) -> dict:
    submit = judge0_submit(source_code, language_id, stdin=stdin)
    token = submit.get("token")
    if not token:
        # Some gateways return full payload; but token is standard.
        raise RuntimeError(f"Missing token in submit response: {submit}")

    start = time.time()
    while True:
        res = judge0_poll(token)
        status = (res.get("status") or {})
        desc = status.get("description", "")
        # Judge0 commonly: In Queue / Processing / Accepted / etc.
        if desc and desc not in ("In Queue", "Processing"):
            return res

        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for token={token}. Last response: {res}")

        time.sleep(poll_s)

import json
from typing import Any, Dict, List

SIZES = [0, 1, 2, 4, 8, 16, 32, 256, 512]

def type_to_jsonschema(t: str) -> Dict[str, Any]:
    """
    Map your metaData param types to JSON Schema fragments.
    Extend this as your platform supports more types.
    """
    t = t.strip()

    # arrays like "integer[]", "string[]"
    if t.endswith("[]"):
        base = t[:-2].strip()
        item = type_to_jsonschema(base)
        return {"type": "array", "items": item}

    # primitives
    if t in ("integer", "int", "long"):
        return {"type": "integer"}
    if t in ("number", "float", "double"):
        return {"type": "number"}
    if t in ("boolean", "bool"):
        return {"type": "boolean"}
    if t in ("string",):
        return {"type": "string"}

    # fallback: allow any JSON (keeps you unblocked)
    # (you can tighten this later)
    return {}

def build_inputs_schema(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds JSON Schema for `inputs` based on metaData.params.
    Uses prefixItems to enforce arity and per-arg type.
    """
    params = meta.get("params", [])
    prefix_items = [type_to_jsonschema(p["type"]) for p in params]

    return {
        "type": "array",
        "minItems": len(prefix_items),
        "maxItems": len(prefix_items),
        "prefixItems": prefix_items,
        "items": {"type": "null"},
    }

def build_sampling_schema(meta: Dict[str, Any], sizes: List[int]) -> Dict[str, Any]:
    inputs_schema = build_inputs_schema(meta)

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "tests": {
                "type": "array",
                # optional: force exactly one sample per size
                "minItems": len(sizes),
                "maxItems": len(sizes),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "n": {"type": "integer", "enum": sizes},
                        "inputs": inputs_schema,
                    },
                    "required": ["n", "inputs"],
                },
            }
        },
        "required": ["tests"],
    }


def run_offline_workflow() -> None:
    """Compose and run a sample OLAP workflow."""

    db = firestore.Client(project=os.getenv("GCP_PROJECT"))

    doc_ref = db.collection("problem").document("two-sum")
    snap = doc_ref.get()

    if snap.exists:
        print(snap.to_dict())
    else:
        print("Document not found")


    client = OpenAI()
    problem = snap.to_dict()
    starter_code = problem["starterCode"]

    resp = client.responses.create(
        model="gpt-4o-mini-2024-07-18",
        input=[
            {
                "role": "system",
                "content": (
                    "You write correct solutions for programming problems.\n"
                    "Return ONLY valid JSON matching the provided schema.\n"
                    "Do NOT include markdown fences.\n"
                    "Do NOT change function/class signatures.\n"
                    "Fill inside the provided starter code; keep imports/typing idiomatic.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Write solutions for the problem using the provided starterCode templates.\n\n"
                    "Problem title: {title}\n\n"
                    "Problem statement (HTML):\n{content}\n\n"
                    "Metadata JSON string:\n{metaData}\n\n"
                    "Starter code templates (must be filled):\n{starter}\n"
                ).format(
                    title=problem["title"],
                    content=problem["content"],
                    metaData=problem["metaData"],
                    starter=json.dumps(starter_code, indent=2),
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "starter_code_fill",   # ✅ required
                "strict": True,
                "schema": {                    # ✅ the actual JSON Schema
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "starterCode": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "typescript": {"type": "string"},
                                "python": {"type": "string"},
                                "java": {"type": "string"},
                            },
                            "required": ["typescript", "python", "java"],
                        }
                    },
                    "required": ["starterCode"],
                },
            }
        },
    )

    # If your SDK supports it:
    result = json.loads(resp.output_text)

    print(f"\nCode:\n{result}\n")

    user_source = result["starterCode"]["python"]
    metadata = problem["metaData"]
    example_testcases = problem["exampleTestcases"] 
    # Run the generated code against the code runner!
    harnessed = make_python_harness_from_examples(
        source_code=user_source,
        metadata_json_str=metadata,
        example_testcases=example_testcases,
    )
    
    # Judge0 language id for Python (verify for your instance)
    PYTHON_LANG_ID = 32

    result = judge0_run_and_wait(harnessed, PYTHON_LANG_ID)
    print(result)
    print("STDOUT:\n", result.get("stdout"))
    print("STDERR:\n", result.get("stderr"))


    db.collection("generated_problems").document(problem["titleSlug"]).set(problem)


    meta = problem["metaData"] if isinstance(problem["metaData"], dict) else json.loads(problem["metaData"])
    params = meta["params"]
    generate_tests_prompt = f"""
        Generate input samples for runtime/complexity measurement.\n\n
        Function Name: {problem["title"]}\n
        Params: {params}\n\n
        Problem title: {problem["title"]}\n\n
        Problem statement (HTML):\n{problem["content"]}\n\n
        Requirements:\n
        - Produce exactly one sample for each n in {SIZES}\n
        - Each sample must set n and provide `inputs` matching the params.\n
        - Make inputs valid under the problem constraints.\n
        - Prefer adversarial/worst-case structure where applicable.\n
        - Do NOT include expected outputs.\n
    """
    resp = client.responses.create(
        model="gpt-4o-mini-2024-07-18",
        input=[
            {
                "role": "system",
                "content": (
                    "You generate valid input samples for algorithmic complexity benchmarking.\n"
                    "Return ONLY valid JSON matching the provided schema.\n"
                    "No markdown. No extra keys.\n"
                    "Inputs must match the function's positional arguments and types per metaData.\n"
                ),
            },
            {
                "role": "user",
                "content": generate_tests_prompt
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "complexity_samples",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "tests": {
                        "type": "array",
                        "minItems": len(SIZES),
                        "maxItems": len(SIZES),
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                            "n": {"type": "integer", "enum": SIZES},
                            "args": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {p["name"]: type_to_jsonschema(p["type"]) for p in params},
                                "required": [p["name"] for p in params],
                            },
                            },
                            "required": ["n", "args"],
                        },
                        }
                    },
                    "required": ["tests"],
                }
            }
        },
    )
    result = json.loads(resp.output_text)
    print(result)

    db.collection("generated_problems").document(problem["titleSlug"]).set({"tests": result["tests"], "paramOrder": [p["name"] for p in params]}, merge=True)


if __name__ == "__main__":
    run_offline_workflow()
