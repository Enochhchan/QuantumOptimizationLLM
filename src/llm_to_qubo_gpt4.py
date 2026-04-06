import os
import json
import time
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from difflib import SequenceMatcher
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv, find_dotenv

def _resolve_prompts_path(explicit_path: str | None) -> Path:
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"--prompts not found: {candidate}")

    env_path = os.getenv("QUBO_PROMPTS_PATH")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"QUBO_PROMPTS_PATH not found: {candidate}")

    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        Path.cwd() / "generated_prompts.csv",
        repo_root / "generated_prompts.csv",
        repo_root / "legacy" / "generated_prompts.csv",
        repo_root / "data" / "generated_prompts.csv",
        repo_root / "data" / "prompts" / "generated_prompts.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find generated prompts CSV. Looked in:\n" + searched + "\n"
        "Fix by passing --prompts <path> or setting QUBO_PROMPTS_PATH."
    )

def main():
    parser = argparse.ArgumentParser(description="Run NL to QUBO translation experiment.")
    parser.add_argument(
        "--prompts",
        default=None,
        help="Path to generated_prompts.csv (or set QUBO_PROMPTS_PATH).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4"),
        help="OpenAI model name (or set OPENAI_MODEL).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N prompts (useful for cheap smoke tests).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip OpenAI calls and use a dummy JSON output.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding-based fidelity computations and plots.",
    )
    args = parser.parse_args()

    load_dotenv(find_dotenv())
    # === INIT ===
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError("OPENAI_API_KEY is missing. Put it in `.env` or your environment.")

    client = OpenAI(api_key=api_key) if not args.dry_run else None
    sns.set_theme(style="whitegrid")

    # === Core: QUBO Translator ===
    def real_llm_to_qubo(prompt_text):
        system_prompt = (
            "You are a QUBO translator. Given an optimization problem in natural language, "
            "return a JSON with the following structure:\n"
            "{\n"
            "  'variables': [...],\n"
            "  'constraints': [\n"
            "     {'type': 'equality' or 'inequality', 'expression': '<math>', 'penalty': <int>}\n"
            "  ],\n"
            "  'objective': '<minimize|maximize>: <expression>'\n"
            "}\n"
            "Only output valid JSON. Do not include extra text or explanation."
        )
        try:
            if args.dry_run:
                return (
                    {
                        "variables": ["x0", "x1"],
                        "constraints": [
                            {"type": "equality", "expression": "x0 + x1 = 1", "penalty": 10}
                        ],
                        "objective": "minimize: x0 + 2*x1",
                    },
                    True,
                )
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.3,
                max_tokens=600
            )
            content = response.choices[0].message.content
            return json.loads(content), True
        except Exception as e:
            print(f"Translation failed: {e}")
            return None, False

    # === Reverse Translation ===
    def reverse_translate_qubo(qubo_json):
        try:
            if args.dry_run:
                return "Dummy reverse translation for dry-run."
            reverse_prompt = "You are a QUBO-to-text explainer. Translate this JSON QUBO into a natural language optimization problem description."
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": reverse_prompt},
                    {"role": "user", "content": json.dumps(qubo_json)}
                ],
                temperature=0.3,
                max_tokens=600
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Reverse translation failed: {e}")
            return None

    # === JSON Validator ===
    def validate_qubo_json(qubo_json):
        try:
            if not isinstance(qubo_json, dict):
                return False
            if "variables" not in qubo_json or "constraints" not in qubo_json or "objective" not in qubo_json:
                return False
            if not isinstance(qubo_json["variables"], list):
                return False
            if not isinstance(qubo_json["constraints"], list):
                return False
            if not isinstance(qubo_json["objective"], str):
                return False
            return True
        except:
            return False

    # === Semantic Similarity ===
    def semantic_similarity(a, b):
        return SequenceMatcher(None, a, b).ratio() if a and b else 0

    # === Load Prompts ===
    prompts_path = _resolve_prompts_path(args.prompts)
    print(f"Using prompts: {prompts_path}")
    print(f"Using model: {args.model}{' (dry-run)' if args.dry_run else ''}")
    problems_df = pd.read_csv(prompts_path)
    if args.limit is not None:
        problems_df = problems_df.head(args.limit)

    # === Run Experiment ===
    results = []
    for i, row in problems_df.iterrows():
        prompt = row["description"]
        problem_type = row["type"]

        print(f"[{i+1}] Processing: {problem_type}")
        start_time = time.time()
        qubo_json, llm_success = real_llm_to_qubo(prompt)
        latency = round(time.time() - start_time, 2)

        valid = validate_qubo_json(qubo_json) if llm_success else False

        if valid:
            num_vars = len(qubo_json["variables"])
            num_constraints = len(qubo_json["constraints"])
            reverse_prompt = reverse_translate_qubo(qubo_json)
            fidelity = semantic_similarity(prompt, reverse_prompt)
        else:
            num_vars = None
            num_constraints = None
            reverse_prompt = None
            fidelity = None

        # Classify failure
        if not llm_success:
            failure_type = "LLM Fail"
        elif not valid:
            failure_type = "Invalid JSON"
        elif "variables" not in qubo_json:
            failure_type = "Missing Variables"
        elif "constraints" not in qubo_json:
            failure_type = "Missing Constraints"
        elif "objective" not in qubo_json:
            failure_type = "Missing Objective"
        else:
            failure_type = None

        results.append({
            "Index": i + 1,
            "Case": problem_type,
            "Prompt": prompt,
            "Success": valid,
            "Failure Type": failure_type,
            "Latency (s)": latency,
            "Num Variables": num_vars,
            "Num Constraints": num_constraints,
            "Prompt Length": len(prompt),
            "Complexity Score": (num_vars or 0) + (num_constraints or 0) + len(prompt) * 0.01,
            "Reverse Prompt": reverse_prompt,
            "Fidelity": fidelity,
            "Raw JSON": json.dumps(qubo_json, ensure_ascii=False)
        })

    # === Save Results ===
    df = pd.DataFrame(results)
    df.to_csv("llm_qubo_results_detailed_with_fidelity_gpt4_without_loop.csv", index=False)

    # === Summary Visuals ===
    # 1. Success Rate by Case
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Case", y="Success", estimator=np.mean, errorbar=None)
    plt.title("Success Rate by Problem Type")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("success_rate_by_case.png")
    plt.show()

    # 2. Failure Type Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="Failure Type", order=df["Failure Type"].value_counts().index)
    plt.title("Failure Types")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("failure_types.png")
    plt.show()

    # 3. Complexity vs Success
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Complexity Score", y="Success", hue="Case")
    plt.title("Success vs Complexity Score")
    plt.tight_layout()
    plt.savefig("success_vs_complexity.png")
    plt.show()

    # 4. Fidelity vs Complexity
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Complexity Score", y="Fidelity", hue="Case")
    plt.title("Fidelity vs Complexity Score")
    plt.tight_layout()
    plt.savefig("fidelity_vs_complexity.png")
    plt.show()

    # 5. Latency vs Complexity
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Complexity Score", y="Latency (s)", hue="Case")
    plt.title("Latency vs Complexity Score")
    plt.tight_layout()
    plt.savefig("latency_vs_complexity.png")
    plt.show()



    # Compute normalized complexity score (0-1 range)
    df["Normalized Complexity"] = (df["Complexity Score"] - df["Complexity Score"].min()) / (df["Complexity Score"].max() - df["Complexity Score"].min())

    # Compute average success and normalized complexity by case type
    summary_plot = df.groupby("Case").agg({
        "Success": "mean",
        "Normalized Complexity": "mean"
    }).reset_index()

    # Melt the dataframe for line plotting
    melted_plot = pd.melt(summary_plot, id_vars="Case", value_vars=["Success", "Normalized Complexity"],
                        var_name="Metric", value_name="Score")

    # Line plot: success rate and normalized complexity by problem type
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=melted_plot, x="Case", y="Score", hue="Metric", marker="o")
    # plt.title("Success Rate and Normalized Complexity by Problem Type")
    plt.ylabel("Score",fontsize=18)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=30,fontsize=18)
    plt.tight_layout()

    output_path = "success_vs_complexity_times_new_roman.png"
    plt.savefig(output_path, dpi=300)  # High resolution
    plt.show()




    # Load the results CSV with fidelity scores
    df = pd.read_csv("llm_qubo_results_detailed_with_fidelity_gpt4_without_loop.csv")

    # Group by problem type to compute average fidelity
    fidelity_by_category = df.groupby("Case")["Fidelity"].mean().reset_index()

    # Plot average fidelity per problem category
    plt.figure(figsize=(5, 3))
    sns.lineplot(data=fidelity_by_category, x="Case", y="Fidelity", color="blue",marker="o")
    # plt.title("Average Fidelity Score by Problem Category")
    plt.ylabel("Fidelity Score")
    plt.ylim(0, 0.1)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    if not args.skip_embeddings:
        # Load the model for semantic similarity
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Warning: could not load SentenceTransformer model; skipping embedding fidelity. Error: {e}")
            return

        # Compute embedding-based fidelity
        def embedding_similarity(a, b):
            if pd.isna(a) or pd.isna(b) or not isinstance(a, str) or not isinstance(b, str):
                return None
            embeddings = model.encode([a, b], convert_to_tensor=True)
            return float(util.pytorch_cos_sim(embeddings[0], embeddings[1]))

        # Apply the semantic similarity function to the dataset
        df["Semantic Fidelity"] = df.apply(
            lambda row: embedding_similarity(row["Prompt"], row["Reverse Prompt"]), axis=1
        )

        # Group by problem type to get average semantic fidelity
        semantic_fid_by_category = df.groupby("Case")["Semantic Fidelity"].mean().reset_index()

        # Plot average semantic fidelity by category
        plt.figure(figsize=(10, 6))
        sns.barplot(data=semantic_fid_by_category, x="Case", y="Semantic Fidelity", color="skyblue")
        plt.title("Average Semantic Fidelity Score by Problem Category")
        plt.ylabel("Semantic Fidelity")
        plt.ylim(0, 1.0)
        plt.xticks(rotation=30)
        plt.tight_layout()
        output_path = "fidality.png"
        plt.savefig(output_path, dpi=300)  # High resolution
        plt.show()


if __name__ == "__main__":
    main()
