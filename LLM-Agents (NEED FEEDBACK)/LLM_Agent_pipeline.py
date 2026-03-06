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
    # Add jitter for visualization
    # ============================
    def jitter(series, amount=0.15):
        return series + np.random.uniform(-amount, amount, size=len(series))

    agent[f"{predictor}_jitter"] = jitter(agent[predictor])
    base[f"{predictor}_jitter"] = jitter(base[predictor])

    # ============================
    # Extract errors (agentic only)
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
            "AIC",
            "BIC",
            "N"
        ],
        "Agentic": [
            model_agent.params[predictor],
            agent_intercept,
            model_agent.pvalues[predictor],
            model_agent.rsquared,
            model_agent.rsquared_adj,
            model_agent.aic,
            model_agent.bic,
            int(model_agent.nobs)
        ],
        "Baseline": [
            model_base.params[predictor],
            base_intercept,
            model_base.pvalues[predictor],
            model_base.rsquared,
            model_base.rsquared_adj,
            model_base.aic,
            model_base.bic,
            int(model_base.nobs)
        ]
    })

    print("\n=== Comparison Table ===")
    print(comparison.round(4))

    # ============================
    # Console error summary (clean)
    # ============================
    if len(errors) > 0:
        print("\n=== Error summary (grouped) ===")
        print(errors["error_group"].value_counts())

        print("\n=== Recovery actions used ===")
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
    # Plot
    # ============================
    fig, ax = plt.subplots(figsize=(12, 7))

    # Colors per query type
    colors = {
        "weather": "blue",
        "math": "green",
        "search/knowledge": "purple",
        "multi-tool": "orange",
        "failure-recovery": "red",
        "ambiguous/complex": "gray"
    }

    # Agentic points
    for qtype, df in agent.groupby("query_type"):
        ax.scatter(
            df[f"{predictor}_jitter"],
            df[outcome],
            alpha=0.6,
            color=colors[qtype],
            marker="o",
            label=f"Agentic – {qtype}"
        )

    # Baseline points
    for qtype, df in base.groupby("query_type"):
        ax.scatter(
            df[f"{predictor}_jitter"],
            df[outcome],
            alpha=0.6,
            color=colors[qtype],
            marker="s",
            label=f"Baseline – {qtype}"
        )

    # ============================
    # Error markers
    # ============================
    if len(errors) > 0:
        ax.scatter(
            errors[f"{predictor}_jitter"],
            errors[outcome],
            color="red",
            marker="x",
            s=70,
            linewidths=1.5,
            label="Error encountered"
        )

    # ============================
    # Error summary box OUTSIDE plot
    # ============================
    if len(errors) > 0:
        grouped = errors["error_group"].value_counts()
        summary_lines = [f"{etype}: {cnt}" for etype, cnt in grouped.items()]
        summary_text = "Errors (agentic):\n" + "\n".join(summary_lines)

        # Place outside the axes on the right
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

    # Regression lines
    ax.plot(x_vals, agent_line, color="black", linewidth=3, label="Agentic Regression")
    ax.plot(x_vals, base_line, color="black", linestyle="--", linewidth=3, label="Baseline Regression")

    # Labels and legend
    ax.set_xlabel(predictor)
    ax.set_ylabel(outcome)
    ax.set_title(f"Agentic vs Baseline: {outcome} vs {predictor}")
    ax.legend(fontsize=8, ncol=2, bbox_to_anchor=(1.02, 0.5), loc="center left")
    ax.grid(True)

    plt.tight_layout()
    plt.show()

    return comparison.round(4)