def langfuse_experiment_for_each_dataset(...):
    langfuse = get_client()
    encoded_dataset_name = quote(dataset_name, safe="")
    dataset = langfuse.get_dataset(encoded_dataset_name)

    with open(os.path.join(self.prompt_path, f"{self.model_class.__name__.removesuffix("Item")}.txt"), "r", encoding="utf-8") as f:
        system_prompt = f.read()

    def task(item, **kwargs):
        dataset_item = self.model_class.from_langfuse_item(item)
        model_output = self.completion(
            system_prompt=system_prompt,
            user_prompt=dataset_item.user_prompt()
        ).get(model_name, "")
        return str(model_output)

    experiment = dataset.run_experiment(
        name=experiment_name,
        description=experiment_description,
        task=task,
        max_concurrency=12,
    )

    print(experiment.format())

    # (1) recuperi i risultati dell'experiment
    runs = experiment.items  # dipende dalla versione SDK, ma concettualmente è la lista delle run

    # (2) per ciascuna run applichi una evaluation LLM-as-a-judge
    for run in runs:
        input_text = run.input           # o come lo espone Langfuse
        model_output = run.output
        ground_truth = getattr(run, "expected_output", None)  # se presente nel dataset

        judge_result = call_llm_judge(
            input_text=input_text,
            model_output=model_output,
            ground_truth=ground_truth,
        )

        # (3) logghi l'evaluation in Langfuse come trace / observation / score
        langfuse.create_score(
            trace_id=run.trace_id,
            name="llm_judge_score",
            value=judge_result["score"],
            comment=judge_result["explanation"],
        )

    return experiment


def call_llm_judge(input_text: str, model_output: str, ground_truth: str | None):
    judge_prompt = f"""
Sei un valutatore. Ti do:

- Input utente: {input_text}
- Output del modello: {model_output}
- (Opzionale) Ground truth: {ground_truth}

Valuta la qualità dell'output del modello rispetto all'input (e al ground truth se presente)
con un punteggio da 1 a 5 e una breve spiegazione.

Rispondi in JSON con le chiavi:
- "score": numero da 1 a 5
- "explanation": testo breve
"""
    # qui chiami il tuo client LLM
    raw = self.judge_client.completion(system_prompt="", user_prompt=judge_prompt)
    return json.loads(raw["some_model_name"])
