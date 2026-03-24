import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

def run_regression_comparison(
    agent_file="agent_runs.jsonl",
    baseline_file="baseline_runs.jsonl",
    predictor="task_complexity",
    outcome="total_latency"
):
    # ============================
    # Load logs
    # ============================
    agent = pd.read_json(agent_file, lines=True)
    base = pd.read_json(baseline_file, lines=True)
    def jitter(series, amount=0.15):
        return series + np.random.uniform(-amount, amount, size=len(series))
    # ============================
    # Query classification
    # ============================
    def classify_query(q):
        q = q.lower()
        if any(word in q for word in ["weather", "temperature", "humidity", "forecast"]):
            return "weather"
        if any(op in q for op in ["+", "-", "*", "/", "square", "compute", "math"]):
            return "math"
        if any(word in q for word in ["who", "what", "where", "define", "capital", "tallest"]):
            return "search/knowledge"
        if "and" in q and any(word in q for word in ["weather", "search", "math"]):
            return "multi-tool"
        if any(word in q for word in ["zorblax", "qwertyopolis", "fictional", "nonsense"]):
            return "failure-recovery"
        return "ambiguous/complex"

    agent["query_type"] = agent["query"].apply(classify_query)
    base["query_type"] = base["query"].apply(classify_query)

        # ============================
    # Ensure predictor exists
    # ============================
    if predictor not in agent.columns:
        agent[predictor] = agent.get("tool_count", 0)

    if predictor not in base.columns:
        base[predictor] = base.get("tool_count", 0)

    # ============================
    # Ensure outcome exists
    # ============================
    agent["total_latency"] = agent["total_time"]
    base["total_latency"] = base["total_time"]



    # ============================
    # Extract errors (LLM-Agent only)
    # ============================
    if "error_flag" in agent.columns:
        errors = agent[agent["error_flag"] == True].copy()
        errors[f"{predictor}_jitter"] = jitter(errors[predictor], amount=0.05)
    else:
        errors = pd.DataFrame()

    # ============================
    # Normalize error types
    # ============================
    def normalize_error_type(err):
        if err is None or not isinstance(err, str):
            return "Other"

        e = err.lower()

        if "404" in e:
            return "HTTP 404"
        if "432" in e:
            return "HTTP 432"
        if "client error" in e:
            return "HTTP error"
        if "invalid syntax" in e:
            return "Syntax error"
        if "invalid character" in e:
            return "Invalid character"
        if "invalid decimal" in e:
            return "Invalid number"
        if "division by zero" in e:
            return "Math error"
        if "unsupported expression" in e:
            return "Syntax error"
        return "Other"

    if len(errors) > 0:
        errors["error_group"] = errors["error_type"].apply(normalize_error_type)
        # ============================
# Reduce to one row per query (preserve tool_count)
# ============================

    # Preserve original tool_count per query
    if "tool_count" in agent.columns:
        agent_tool_counts = agent[["query", "tool_count"]].drop_duplicates()
    else:
        agent_tool_counts = pd.DataFrame(columns=["query", "tool_count"])

    if "tool_count" in base.columns:
        base_tool_counts = base[["query", "tool_count"]].drop_duplicates()
    else:
        base_tool_counts = pd.DataFrame(columns=["query", "tool_count"])

    # Collapse everything else
    agent = agent.groupby("query", as_index=False).agg({
        "task_complexity": "mean",
        "total_latency": "mean",
        "query_type": "first",
        "api_difficulty": "mean",
    })

    base = base.groupby("query", as_index=False).agg({
        "task_complexity": "mean",
        "total_latency": "mean",
        "query_type": "first",
        "api_difficulty": "mean",
    })

    

    # Merge tool_count back in
    agent = agent.merge(agent_tool_counts, on="query", how="left")
    base = base.merge(base_tool_counts, on="query", how="left")

    # ============================
    # Count queries per category
    # ============================
    agent_counts = agent["query_type"].value_counts().to_dict()
    base_counts  = base["query_type"].value_counts().to_dict()
        # ============================
    # Fit regressions
    # ============================
    X_agent = sm.add_constant(agent[predictor])
    y_agent = agent[outcome]
    model_agent = sm.OLS(y_agent, X_agent).fit()

    X_base = sm.add_constant(base[predictor])
    y_base = base[outcome]
    model_base = sm.OLS(y_base, X_base).fit()

    # ============================
    # Build comparison table
    # ============================
    agent_intercept = model_agent.params.get("const", float("nan"))
    base_intercept = model_base.params.get("const", float("nan"))

    comparison = pd.DataFrame({
        "Metric": [
            "Slope (predictor)",
            "Intercept",
            "p-value (slope)",
            "R-squared",
            "Adj R-squared",
            "N"
        ],
        "LLM-Agent": [
            model_agent.params[predictor],
            agent_intercept,
            model_agent.pvalues[predictor],
            model_agent.rsquared,
            model_agent.rsquared_adj,
            int(model_agent.nobs)
        ],
        "Baseline": [
            model_base.params[predictor],
            base_intercept,
            model_base.pvalues[predictor],
            model_base.rsquared,
            model_base.rsquared_adj,
            int(model_base.nobs)
        ]
    })

    print("\n=== Comparison Table ===")
    print(comparison.round(4))

    # ============================
    # Console error summary (clean)
    # ============================
    if len(errors) > 0:
        print("\n=== LLM-Agent Error Summary (grouped) ===")
        print(errors["error_group"].value_counts())

        print("\n=== LLM-Agent Recovery Actions Used ===")
        print(errors["recovery_action"].value_counts())

    # ============================
    # Compute regression lines
    # ============================
    x_vals = np.linspace(
        min(agent[predictor].min(), base[predictor].min()),
        max(agent[predictor].max(), base[predictor].max()),
        100
    )

    agent_line = agent_intercept + model_agent.params[predictor] * x_vals
    base_line = base_intercept + model_base.params[predictor] * x_vals

    # ============================
    # Add jitter for visualization
    # ============================
   

    agent[f"{predictor}_jitter"] = jitter(agent[predictor])
    base[f"{predictor}_jitter"] = jitter(base[predictor])
    # ============================
    # UNIQUE QUERY COUNTS FOR LEGEND
    # ============================
        # ============================
    # ============================
    # Create figure + axis
    # ============================
    fig, ax = plt.subplots(figsize=(10, 6))

    # ============================
    # Plot points by category (Agent + Baseline, with counts)
    # ============================

    colors = {
        "weather": "blue",
        "math": "green",
        "search/knowledge": "purple",
        "multi-tool": "orange",
        "failure-recovery": "red",
        "ambiguous/complex": "gray",
    }

    # ----------------------------------------
    # 1. Collapse errors to ONE per query
    # ----------------------------------------
    if len(errors) > 0:
        error_unique = (
            errors.groupby("query")
                .first()
                .reset_index()
        )
        error_unique[f"{predictor}_jitter"] = jitter(
            error_unique[predictor], amount=0.05
        )
    else:
        error_unique = pd.DataFrame()

    # ----------------------------------------
    # 2. LLM-Agent points (circles, with counts)
    # ----------------------------------------
    for category, color in colors.items():
        subset = agent[agent["query_type"] == category]

        if subset.empty:
            continue

        count = len(subset)
        label = f"{category} – agent ({count})"

        ax.scatter(
            subset[f"{predictor}_jitter"],
            subset[outcome],
            alpha=0.75,
            color=color,
            marker="o",
            s=60,
            edgecolor="black",
            label=label
        )

    # ----------------------------------------
    # 3. Baseline points (square markers, with counts)
    # ----------------------------------------
    for category, color in colors.items():
        subset = base[base["query_type"] == category]

        if subset.empty:
            continue

        count = len(subset)
        label = f"{category} – baseline ({count})"

        ax.scatter(
            subset[f"{predictor}_jitter"],
            subset[outcome],
            alpha=0.75,
            color=color,
            marker="s",
            s=60,
            edgecolor="black",
            linewidths=0.8,
            label=label
                )
    # ----------------------------------------
    # 4. Clean error markers (one per query)
    # ----------------------------------------

    # LLM-Agent Errors (thin red X)
    if len(error_unique) > 0:
        ax.scatter(
            error_unique[f"{predictor}_jitter"],
            error_unique[outcome],
            color="red",
            marker="x",
            s=60,
            alpha=0.55,
            linewidths=0.9,
            label="LLM-Agent Errors"
        )

    # ----------------------------------------
    # Extract LLM-Agent Error + Replan events
    # ----------------------------------------
    if "recovery_action" in agent.columns:
        replan_unique = agent[agent["recovery_action"] == "replan"]
    elif "error_type" in errors.columns:
        replan_unique = errors[errors["error_type"].str.contains("replan", case=False)]
    else:
        replan_unique = pd.DataFrame()

    # Collapse to one per query
    if len(replan_unique) > 0:
        replan_unique = (
            replan_unique.groupby("query")
                        .first()
                        .reset_index()
        )
        replan_unique[f"{predictor}_jitter"] = jitter(
            replan_unique[predictor], amount=0.05
        )

    # Plot replan markers (triangle)
    if len(replan_unique) > 0:
        ax.scatter(
            replan_unique[f"{predictor}_jitter"],
            replan_unique[outcome],
            color="red",
            marker="^",
            s=70,
            alpha=0.55,
            edgecolor="black",
            linewidths=0.9,
            label="LLM-Agent Error + Replan"
        )
    else:
        # Legend-only entry so the symbol always appears
        ax.scatter(
            [], [], 
            color="red",
            marker="^",
            s=110,
            alpha=0.85,
            edgecolor="black",
            linewidths=1.2,
            label="LLM-Agent Error + Replan"
        )
    # ----------------------------------------
    # 5. Error summary boxes
    # ----------------------------------------
    if len(error_unique) > 0:
        grouped = error_unique["error_group"].value_counts()
        summary_lines = [f"{etype}: {cnt}" for etype, cnt in grouped.items()]
        summary_text = "LLM-Agent Errors (per query):\n" + "\n".join(summary_lines)

        ax.text(
            1.02, 0.98,
            summary_text,
            transform=ax.transAxes,
            fontsize=8,
            color="red",
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="red")
        )

        total_tool_errors = len(errors)
        total_query_errors = len(error_unique)

        summary_text2 = (
            "LLM-Agent Error Totals:\n"
            f"- Tool-level errors: {total_tool_errors}\n"
            f"- Queries with errors: {total_query_errors}"
        )

        ax.text(
            1.02, 0.70,
            summary_text2,
            transform=ax.transAxes,
            fontsize=8,
            color="darkred",
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="darkred")
        )

    # ----------------------------------------
    # 6. Regression lines
    # ----------------------------------------
    ax.plot(x_vals, agent_line, color="black", linewidth=3, label="LLM-Agent Regression")
    ax.plot(x_vals, base_line, color="black", linestyle="--", linewidth=3, label="Baseline Regression")

    # ----------------------------------------
    # 7. Regression formulas on the plot
    # ----------------------------------------
    agent_slope = model_agent.params[predictor]
    base_slope = model_base.params[predictor]

    agent_formula = f"LLM-Agent: y = {agent_intercept:.3f} + {agent_slope:.3f}·x"
    base_formula  = f"Baseline: y = {base_intercept:.3f} + {base_slope:.3f}·x"

    ax.text(
        0.02, 0.98,
        agent_formula,
        transform=ax.transAxes,
        fontsize=9,
        color="black",
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="black")
    )

    ax.text(
        0.02, 0.90,
        base_formula,
        transform=ax.transAxes,
        fontsize=9,
        color="black",
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="black")
    )

    # ----------------------------------------
    # 8. Labels and legend
    # ----------------------------------------
    ax.set_xlabel(predictor)
    ax.set_ylabel(outcome)
    ax.set_title(f"LLM-Agent vs Baseline: {outcome} vs {predictor}")
    ax.legend(fontsize=8, ncol=2, bbox_to_anchor=(1.02, 0.5), loc="center left")
    ax.grid(True)

    plt.tight_layout()
    plt.show()