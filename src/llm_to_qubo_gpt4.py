import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from difflib import SequenceMatcher
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

def main():
    load_dotenv(find_dotenv())
    # === INIT ===
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
            response = client.chat.completions.create(
                model="gpt-4",
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
            reverse_prompt = "You are a QUBO-to-text explainer. Translate this JSON QUBO into a natural language optimization problem description."
            response = client.chat.completions.create(
                model="gpt-4",
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
    problems_df = pd.read_csv("generated_prompts.csv")

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

    # model = SentenceTransformer("all-MiniLM-L6-v2")

    # def embedding_similarity(a, b):
    #     embeddings = model.encode([a, b], convert_to_tensor=True)
    #     return float(util.pytorch_cos_sim(embeddings[0], embeddings[1]))



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
    # Normalizing keeps different cases on the same plotting scale.
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




    # Re-import necessary packages after kernel reset
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

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

    fidelity_by_category




    # Load the model for semantic similarity
    model = SentenceTransformer("all-MiniLM-L6-v2")

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




    # Load your result CSVs
    df_35 = pd.read_csv("llm_qubo_results_detailed_with_fidelity_35.csv")
    df_4 = pd.read_csv("llm_qubo_results_detailed_with_fidelity_gpt4_without_loop.csv")

    # Load the embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Define similarity
    def embedding_similarity(a, b):
        if pd.isna(a) or pd.isna(b) or not isinstance(a, str) or not isinstance(b, str):
            return None
        embeddings = model.encode([a, b], convert_to_tensor=True)
        return float(util.pytorch_cos_sim(embeddings[0], embeddings[1]))

    # Compute semantic fidelity per model
    df_35["Model"] = "GPT-3.5"
    df_4["Model"] = "GPT-4"
    df_35["Semantic Fidelity"] = df_35.apply(lambda row: embedding_similarity(row["Prompt"], row["Reverse Prompt"]), axis=1)
    df_4["Semantic Fidelity"] = df_4.apply(lambda row: embedding_similarity(row["Prompt"], row["Reverse Prompt"]), axis=1)

    # Merge for comparison
    df_all = pd.concat([df_35, df_4])

    # Clean index and sort categories
    df_all = df_all.reset_index(drop=True)
    df_all["Case"] = pd.Categorical(df_all["Case"], categories=sorted(df_all["Case"].unique()), ordered=True)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_all, x="Case", y="Semantic Fidelity", hue="Model", errorbar=None)
    plt.title("Semantic Fidelity Score by Problem Type and Model (Embedding-Based)")
    plt.ylabel("Semantic Fidelity")
    plt.ylim(0, 1)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("semantic_fidelity_embedding_based.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
