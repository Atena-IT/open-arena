def run_llm_judge_on_experiment(self, experiment_result, judge_model: str):
    langfuse = get_client()

    for res in experiment_result.item_results:
        user_input = res.item.input
        expected_output = res.item.expected_output
        metadata = res.item.metadata
        model_output = res.output
        trace_id = res.trace_id

        judge_score, judge_explanation = self.call_llm_judge(
            judge_model=judge_model,
            user_input=user_input,
            model_output=model_output,
            ground_truth=expected_output,
            metadata=metadata,
        )

        # Salva lo score sul trace in Langfuse
        if judge_score is not None:
            langfuse.create_score(
                trace_id=trace_id,
                name="llm_judge_score",
                value=judge_score,
                comment=judge_explanation,
            )


def call_llm_judge(
    self,
    judge_model: str,
    user_input: str,
    model_output: str,
    ground_truth: str | None,
    metadata: dict | None = None,
):
    extra = f"\nMetadata: {metadata}" if metadata else ""

    judge_prompt = f"""
Sei un valutatore. Ti do:

- Input utente: {user_input}
- Output del modello: {model_output}
- Ground truth (se presente): {ground_truth}
{extra}

Valuta la qualità dell'output del modello rispetto all'input e alla ground truth.
Restituisci JSON con:
- "score": numero da 1 a 5
- "explanation": testo breve
"""

    raw = self.completion(
        system_prompt="Sei un LLM che valuta le risposte di altri LLM.",
        user_prompt=judge_prompt,
    ).get(judge_model, "")

    try:
        parsed = json.loads(raw)
        return parsed.get("score"), parsed.get("explanation")
    except Exception:
        return None, f"Parsing error. Raw: {raw}"


def langfuse_experiment(self, dataset_name: str, experiment_name_prefix: str = "Model Evaluation"):
    # ... codice che hai già per lanciare gli experiment ...

    results = {}
    with ThreadPoolExecutor() as executor:
        future_to_model = {
            executor.submit(
                self.langfuse_experiment_for_each_dataset,
                experiment_name[model_name],
                experiment_description[model_name],
                model_name,
                dataset_name
            ): model_name
            for model_name in self.models_list
        }
        for future in tqdm.tqdm(as_completed(future_to_model), total=len(future_to_model), desc="Evaluating QA items"):
            model_name = future_to_model[future]
            exp_result = future.result()
            results[model_name] = exp_result

    # Esempio: usa lo stesso modello come judge, o uno dedicato
    exp_41 = results["gpt-4.1-mini"]
    self.run_llm_judge_on_experiment(
        experiment_result=exp_41,
        judge_model="gpt-4.1-mini",  # o altro
    )

    return results