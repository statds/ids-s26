import os
import json
import math
import time
import ast
import re
import requests
import matplotlib.pyplot as plt
from groq import Groq
from dotenv import load_dotenv
import sys

# ---------------- CONFIG ----------------
load_dotenv(dotenv_path="/home/devcontainers/ids-s26/.env", override=True)

MODEL = "llama-3.3-70b-versatile"
LOG_FILE = "agent_runs.jsonl"

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def non_agentic_answer(query):
    """Baseline: single-shot LLM answer with no tools, no planning, no loops."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only your internal knowledge. "
                    "Do not call external tools, do not perform calculations using APIs, "
                    "and do not attempt to fetch real-time data. "
                    "Provide your best single-shot answer."
                )
            },
            {"role": "user", "content": query},
        ],
        temperature=0
    )
    return resp.choices[0].message.content

# -----------CSV CREATION ---------------
import csv

def save_agentic_csv(stats, filename="agentic_results.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query", "answer", "total_time",
            "tool_usage_weather", "tool_usage_calculator", "tool_usage_search",
            "success_count", "failure_count", "errors", "agentic"
        ])

        for entry in stats["detailed"]:
            writer.writerow([
                entry["query"],
                entry["answer"],
                entry["total_time"],
                entry["tool_usage"]["weather"],
                entry["tool_usage"]["calculator"],
                entry["tool_usage"]["search"],
                entry["success_count"],
                entry["failure_count"],
                "; ".join(entry["errors"]),
                True
            ])


def save_baseline_csv(stats, filename="baseline_results.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query", "answer", "total_time",
            "tool_usage_weather", "tool_usage_calculator", "tool_usage_search",
            "success_count", "failure_count", "errors", "agentic"
        ])

        for entry in stats["detailed"]:
            writer.writerow([
                entry["query"],
                entry["answer"],
                entry["total_time"],
                0, 0, 0,  # baseline uses no tools
                0, 0,     # no tool successes/failures
                "",       # no tool errors
                False
            ])

# -------log function ------------
def log_run(query, steps, results, answer, total_time, api_difficulty, tool_count):
    task_complexity = len(steps)

    # Write one JSONL row per tool call
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        for r in results:
            record = {
                "timestamp": time.time(),
                "query": query,
                "tool": r["tool"],
                "input": r["input"],
                "output": r["output"],
                "execution_time": r["execution_time"],
                "success": r["success"],
                "error_flag": r["error_flag"],
                "error_type": r["error_type"],
                "recovery_action": r["recovery_action"],

                # Run-level metadata copied onto each row
                "task_complexity": task_complexity,
                "total_time": total_time,
                "api_difficulty": api_difficulty,
                "tool_count": tool_count,
                "model": MODEL
            }

            f.write(json.dumps(record) + "\n")



# -------------------LOG-NON-AGENTIC-AI -----------
def log_non_agentic(query, answer, total_time, api_difficulty, tool_count, task_complexity):
    record = {
        "timestamp": time.time(),
        "query": query,
        "answer": answer,
        "total_time": total_time,
        "api_difficulty": api_difficulty,
        "tool_count": tool_count,
        "task_complexity": task_complexity,
        "model": MODEL,
        "agentic": False
    }

    with open("baseline_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------- TOOLS ----------------
def timed_call(func, *args):
    start = time.time()
    output = func(*args)
    end = time.time()
    return output, end - start


def weather_tool(city):
    if not OPENWEATHER_API_KEY:
        return {"error": "Missing OPENWEATHER_API_KEY"}

    params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    try:
        r = requests.get(WEATHER_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return {
            "temp_c": data["main"]["temp"],
            "description": data["weather"][0]["description"],
        }
    except Exception as e:
        return {"error": str(e)}


def calculator_tool(expr):
    try:
        node = ast.parse(expr, mode="eval")
        allowed = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Load,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
            ast.USub, ast.UAdd
        )
        for n in ast.walk(node):
            if not isinstance(n, allowed):
                return {"error": "Unsupported expression"}

        result = eval(compile(node, "<expr>", "eval"))
        return {"result": result}

    except Exception as e:
        return {"error": str(e)}


def tavily_search(query: str):
    """Web search using Tavily with summaries enabled."""
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": os.getenv("TAVILY_API_KEY"),
        "query": query,
        "max_results": 5,
        "include_answer": True,
        "include_images": False,
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def search_tool(query):
    results = tavily_search(query)
    if "error" in results:
        return {"error": results["error"]}

    return {
        "answer": results.get("answer"),
        "results": results.get("results", []),
        "citations": results.get("citations", []),
    }


TOOLS = {
    "weather": weather_tool,
    "calculator": calculator_tool,
    "search": search_tool,
}

# ---------------- PLANNER ----------------
PLANNER_PROMPT = """
You are an LLM agent that must plan multi-step tasks using tools.

Available tools:
- weather(city)
- calculator(expression)
- search(query)

Your job is to break the user's query into the minimal number of tool calls needed.
If the query requires multiple tools, output multiple steps in order.

Output ONLY a JSON object in this format:

{
  "steps": [
    {"tool": "...", "input": "..."},
    {"tool": "...", "input": "..."},
    ...
  ]
}
"""


def plan_steps(query):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    text = resp.choices[0].message.content.strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return []

    try:
        parsed = json.loads(match.group())
        return parsed.get("steps", [])
    except Exception:
        return []




# ---------------- FINAL ANSWER ----------------
def final_answer(query, results):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
Combine the tool outputs into a single helpful answer.
If multiple tools were used, integrate their results logically.
Do not mention tools.
""",
            },
            {
                "role": "user",
                "content": f"Query: {query}\nResults: {json.dumps(results)}",
            },
        ],
    )
    return resp.choices[0].message.content


# ---------------- AGENT THINKING BLOCK ----------------
def agent_thinks_done(user_query, results):
    prompt = f"""
You are a decision module inside an agent system.
Your job is to determine whether the agent has enough information to answer the user's query.

User query:
{user_query}

Tool results so far:
{results}

Based on the query and the results, decide if the agent should stop or continue.

Respond with exactly one word:
YES  – if the agent has enough information to answer.
NO   – if the agent needs to run more tool steps.
"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    answer = resp.choices[0].message.content.strip().upper()
    return answer == "YES"

# ---------------- REPLAN-IF-FAILURE------------------
def replan_after_failure(user_query, results):
    prompt = f"""
You are a planning module inside a tool-using agent.

The previous tool call failed. Here is the user query and all tool results so far:

User query:
{user_query}

Tool results so far:
{json.dumps(results)}

Your job is to generate a revised plan that avoids the failure and continues the task.

Rules:
1. Output ONLY valid JSON.
2. JSON must contain a single key: "steps".
3. Each step must have "tool" and "input".
4. Avoid repeating the tool that failed, unless absolutely necessary.
5. Do not explain your reasoning.
6. No code fences.

Return only JSON.
"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    text = resp.choices[0].message.content.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return []

    try:
        parsed = json.loads(match.group())
        return parsed.get("steps", [])
    except Exception:
        return []



# ---------------- MULTI-STEP LOOP ----------------

def agent_loop(user_query, max_rounds=3):
    all_results = []
    last_steps = []

    for round_idx in range(max_rounds):

        # First round: normal planning
        steps = plan_steps(user_query)
        if not steps:
            break

        last_steps = steps
        round_results = []
        failure_detected = False

        # ---------------- TOOL EXECUTION WITH ERROR TRACKING ----------------
        for step in steps:
            tool = step["tool"]
            inp = step.get("input", "")

            error_flag = False
            error_type = None
            recovery_action = None

            if tool not in TOOLS:
                output = {"error": "Invalid tool"}
                t = 0.0
                error_flag = True
                error_type = "InvalidTool"
                failure_detected = True

            else:
                try:
                    output, t = timed_call(TOOLS[tool], inp)
                    if "error" in output:
                        error_flag = True
                        error_type = output["error"]
                        failure_detected = True
                except Exception as e:
                    output = {"error": str(e)}
                    t = 0.0
                    error_flag = True
                    error_type = type(e).__name__
                    failure_detected = True

            round_results.append({
                "tool": tool,
                "input": inp,
                "output": output,
                "execution_time": t,
                "success": not error_flag,
                "timestamp": time.time(),
                "error_flag": error_flag,
                "error_type": error_type,
                "recovery_action": None
            })

        all_results.extend(round_results)

        # ---------------- RECOVERY ACTIONS ----------------
        if failure_detected:
            revised_steps = replan_after_failure(user_query, all_results)

            for r in round_results:
                if r["error_flag"]:
                    r["recovery_action"] = "replan" if revised_steps else "no_recovery"

            if revised_steps:
                last_steps = revised_steps
                continue

        # Mark successful steps
        for r in round_results:
            if not r["error_flag"] and r["recovery_action"] is None:
                r["recovery_action"] = "success"

        # ---------------- STOPPING DECISION ----------------
        if agent_thinks_done(user_query, all_results):
            break

    # ---------------- FINAL ANSWER ----------------
    answer = final_answer(user_query, all_results)
    total_time = sum(r["execution_time"] for r in all_results)

    # ---------------- LOG RUN ----------------
    tools_used = [r["tool"] for r in all_results]
    difficulty_map = {"calculator": 1, "search": 2, "weather": 3}
    api_difficulty = sum(difficulty_map.get(t, 1) for t in tools_used)

    log_run(
        query=user_query,
        steps=last_steps,
        results=all_results,
        answer=answer,
        total_time=total_time,
        api_difficulty=api_difficulty,
        tool_count=len(tools_used)
    )

    return answer


    
# ---------------- BENCHMARK-WRAPPER--------------
def benchmark(mode, queries):
    mode = mode.lower().strip()

    if mode == "agentic":
        return benchmark_agentic(queries)

    if mode in ["baseline", "non-agentic", "non_agentic"]:
        return benchmark_non_agentic(queries)

    raise ValueError(f"Unknown benchmark mode: {mode}")

# ---------------- BENCHMARK-AGENTIC----------------
def benchmark_agentic(queries):
    stats = {
        "total_queries": len(queries),
        "tool_usage": {"weather": 0, "calculator": 0, "search": 0},
        "success_count": 0,
        "failure_count": 0,
        "avg_latency": 0.0,
        "errors": [],
        "detailed": []
    }

    total_latency = 0.0

    for q in queries:
        # Run the agent
        start = time.time()
        answer = agent_loop(q)
        end = time.time()
        latency = end - start
        total_latency += latency

        # Load all logs AFTER running the agent
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            all_logs = [json.loads(line) for line in f]

        # Extract ONLY the tool calls for this query
        rows = [r for r in all_logs if r["query"] == q]

        # If nothing logged, skip
        if not rows:
            continue

        # Count tool usage
        stats["tool_usage"]["weather"] += sum(1 for r in rows if r["tool"] == "weather")
        stats["tool_usage"]["calculator"] += sum(1 for r in rows if r["tool"] == "calculator")
        stats["tool_usage"]["search"] += sum(1 for r in rows if r["tool"] == "search")

        # Success/failure
        success = sum(1 for r in rows if r["success"])
        failure = sum(1 for r in rows if not r["success"])
        errors = [r["error_type"] for r in rows if r["error_flag"]]

        stats["success_count"] += success
        stats["failure_count"] += failure
        stats["errors"].extend(errors)

        # Detailed record for this query
        stats["detailed"].append({
            "query": q,
            "answer": answer,
            "total_time": latency,
            "tool_usage": {
                "weather": sum(1 for r in rows if r["tool"] == "weather"),
                "calculator": sum(1 for r in rows if r["tool"] == "calculator"),
                "search": sum(1 for r in rows if r["tool"] == "search"),
            },
            "tool_count": len(rows),
            "task_complexity": rows[0]["task_complexity"],
            "success_count": success,
            "failure_count": failure,
            "errors": errors
        })

    stats["avg_latency"] = total_latency / len(queries)
    return stats

# --------------- NON-AGENTIC AI BENCH MARK ---------
def benchmark_non_agentic(queries):
    stats = {
        "total_queries": len(queries),
        "avg_latency": 0.0,
        "detailed": []
    }

    total_latency = 0.0

    difficulty_map = {"calculator": 1, "search": 2, "weather": 3}

    for q in queries:
        start = time.time()

        results = []
        tools_used = []
        q_lower = q.lower()

        # WEATHER detection
        if any(w in q_lower for w in ["weather", "temperature", "humidity", "forecast"]):
            output, t = timed_call(TOOLS["weather"], q)
            tools_used.append("weather")
            results.append({
                "tool": "weather",
                "input": q,
                "output": output,
                "execution_time": t,
                "success": "error" not in output,
                "timestamp": time.time(),
            })

        # MATH detection
        if any(op in q for op in ["+", "-", "*", "/", "×"]) or "convert" in q_lower:
            output, t = timed_call(TOOLS["calculator"], q)
            tools_used.append("calculator")
            results.append({
                "tool": "calculator",
                "input": q,
                "output": output,
                "execution_time": t,
                "success": "error" not in output,
                "timestamp": time.time(),
            })

        # SEARCH detection
        if any(k in q_lower for k in ["who", "what", "define", "population", "capital"]):
            output, t = timed_call(TOOLS["search"], q)
            tools_used.append("search")
            results.append({
                "tool": "search",
                "input": q,
                "output": output,
                "execution_time": t,
                "success": "error" not in output,
                "timestamp": time.time(),
            })

        # FALLBACK: if no tool detected
        if len(tools_used) == 0:
            output, t = timed_call(TOOLS["search"], q)
            tools_used.append("search")
            results.append({
                "tool": "search",
                "input": q,
                "output": output,
                "execution_time": t,
                "success": "error" not in output,
                "timestamp": time.time(),
            })

        # Final answer is last tool output
        answer = results[-1]["output"]

        end = time.time()
        latency = end - start
        total_latency += latency

        # Compute difficulty
        api_difficulty = sum(difficulty_map.get(t, 1) for t in tools_used)

        # Log one row per query
        log_non_agentic(
            query=q,
            answer=answer,
            total_time=latency,
            api_difficulty=api_difficulty,
            tool_count=len(tools_used),
            task_complexity=len(tools_used)
        )

        # Add detailed stats
        stats["detailed"].append({
            "query": q,
            "answer": answer,
            "total_time": latency,
            "tool_usage": {
                "weather": sum(1 for r in results if r["tool"] == "weather"),
                "calculator": sum(1 for r in results if r["tool"] == "calculator"),
                "search": sum(1 for r in results if r["tool"] == "search"),
            },
            "tool_count": len(tools_used),
            "task_complexity": len(tools_used),
            "success_count": sum(1 for r in results if r["success"]),
            "failure_count": sum(1 for r in results if not r["success"]),
            "errors": [r["output"].get("error", "") for r in results if "error" in r["output"]]
        })

    stats["avg_latency"] = total_latency / len(queries)
    return stats
# ----------------- BENCHMARK QUERIES -------------------------
benchmark_queries = [
    # Weather
    "What is the weather in Paris?",
    "Weather in Tokyo right now?",
    "Is it raining in London?",
    "What is the temperature in New York today?",
    "Weather forecast for Berlin?",
    "What is the humidity in Madrid?",

    # Math
    "What is 15 × 19?",
    "Compute (12 + 4) × 3.",
    "What is 100 / 4?",
    "What is 5² + 3?",
    "Evaluate 40 ÷ (2 × 5).",
    "What is 7 × 8 + 12?",

    # Search / Knowledge
    "Who discovered penicillin?",
    "Define machine learning.",
    "What is the tallest mountain in the world?",
    "Who wrote Pride and Prejudice?",
    "What is the capital of Argentina?",
    "What is photosynthesis?",

    # Multi‑tool
    "What is the weather in Paris and convert the temperature to Fahrenheit?",
    "Find the population of Tokyo and compute its square root.",
    "Search for the tallest mountain and multiply its height in meters by 3.",
    "What is the weather in Rome and what is 5 × the temperature?",
    "Define artificial intelligence and compute the number of letters in the definition.",
    "What is the weather in Sydney and what is 10 plus the temperature?",

    # Failure‑recovery
    "What is the weather in Berlin?",
    "Compute 5 + (3 × 2",
    "Search for the capital of the country Zorblax.",
    "What is the weather in a city named Qwertyopolis?",
    "Compute 10 / (5 − 5).",
    "Search for the current president of the fictional nation Eldoria.",

    # Ambiguous / complex
    "How hot is it in the city where the Eiffel Tower is located?",
    "What is the square root of the number of letters in the word hippopotamus?",
    "Who invented the telephone and what is 12 × 7?",
    "What is the weather in the capital of Japan?",
    "Define gravity and compute 9 × 9.",
    "What is the weather in the largest city in Canada?"
]
# ---------------- MAIN ----------------
if __name__ == "__main__":

    # 1. Agentic benchmark
    if "--agentic" in sys.argv:
        print("Running agentic benchmark...")
        stats = benchmark_agentic(benchmark_queries)
        print(json.dumps(stats, indent=2))
        sys.exit(0)

    # 2. Baseline benchmark
    if "--baseline" in sys.argv:
        print("Running non-agentic baseline benchmark...")
        stats = benchmark_non_agentic(benchmark_queries)
        print(json.dumps(stats, indent=2))
        sys.exit(0)

    # 3. Compare both
    if "--compare" in sys.argv:
        print("Running both benchmarks...")
        agentic_stats = benchmark_agentic(benchmark_queries)
        baseline_stats = benchmark_non_agentic(benchmark_queries)
        print("\nAgentic:", json.dumps(agentic_stats, indent=2))
        print("\nBaseline:", json.dumps(baseline_stats, indent=2))
        sys.exit(0)
    # 6. SAVE-CSV
    if "--save-csv" in sys.argv:
        print("Running both benchmarks and saving CSV files...")

    agentic_stats = benchmark_agentic(benchmark_queries)
    baseline_stats = benchmark_non_agentic(benchmark_queries)

    save_agentic_csv(agentic_stats, "agentic_results.csv")
    save_baseline_csv(baseline_stats, "baseline_results.csv")

    print("Saved agentic_results.csv and baseline_results.csv")
    sys.exit(0)
    
    # 4. Non-interactive single query
    if "--no-interactive" in sys.argv:
        print("Running in non-interactive mode...")
        default_query = "What is the weather in New York?"
        answer = agent_loop(default_query)
        print("Answer:", answer)
        sys.exit(0)

    # 5. Interactive mode (default)
    print("Interactive Agent Ready. Type 'exit' to quit.")
    while True:
        user_query = input("\nAsk me anything: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        answer = agent_loop(user_query)
        print("\nAgent answer:", answer)



# ---------------- VISUALIZER ----------------

def visualize_logs(log_file="agent_runs.jsonl"):
    if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
        print("No logs found.")
        return

    queries = []
    latencies = []
    tool_counts = {"weather": 0, "calculator": 0, "search": 0}
    successes = 0
    failures = 0

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            queries.append(entry["query"])
            latencies.append(entry["total_time"])

            for r in entry["results"]:
                tool_counts[r["tool"]] += 1
                if r["success"]:
                    successes += 1
                else:
                    failures += 1

    plt.figure(figsize=(10, 4))
    plt.bar(tool_counts.keys(), tool_counts.values(), color="skyblue")
    plt.title("Tool Usage Frequency")
    plt.xlabel("Tool")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.hist(latencies, bins=20, color="orange")
    plt.title("Latency Distribution")
    plt.xlabel("Seconds")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.pie([successes, failures], labels=["Success", "Failure"], autopct="%1.1f%%")
    plt.title("Success Rate")
    plt.show()
