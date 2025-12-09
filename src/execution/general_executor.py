from pydantic import BaseModel, Field
from typing import Dict, Optional


class MyQAItem(BaseModel):
    id: str = Field(..., description="Unique ID", json_schema_extra={"role": "id"})
    question: str = Field(..., json_schema_extra={"role": "question"})
    # Variante 1: 4 campi separati
    option_a: str = Field(..., json_schema_extra={"role": "option", "label": "A"})
    option_b: str = Field(..., json_schema_extra={"role": "option", "label": "B"})
    option_c: str = Field(..., json_schema_extra={"role": "option", "label": "C"})
    option_d: str = Field(..., json_schema_extra={"role": "option", "label": "D"})
    answer: str = Field(..., json_schema_extra={"role": "correct_answer"})  # "A"/"B"/...

    # Variante 2 (alternativa): un dict con tutte le opzioni
    # options: Dict[str, str] = Field(..., json_schema_extra={"role": "options_dict"})


from typing import List, Any, Type
from urllib.parse import quote
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from tqdm import tqdm
from langfuse import get_client
from pydantic import BaseModel

from src.language_models_core.language_models import LMClient

logger = logging.getLogger(__name__)


class GenericFinAdvExecutor:
    """
    Executor generico per QA tasks, adattabile a qualunque modello Pydantic
    che annoti i campi con json_schema_extra["role"].
    """

    def __init__(
        self,
        lm_client: LMClient,
        dataset: List[BaseModel],          # liste di istanze Pydantic
        model_class: Type[BaseModel],      # classe Pydantic (per introspezione)
        models_list: List[str],
        results_folder: str,
    ):
        self.lm_client = lm_client
        self.dataset = dataset
        self.model_class = model_class
        self.models_list = models_list
        self.results_folder = results_folder

        self.multiple_choice_results_dict: dict[str, dict] = {}
        self.open_ended_results_dict: dict[str, dict] = {}

        # Pre-calcolo dei ruoli dai model_fields (una volta sola)
        self._roles = self._inspect_model_fields()

    # ---------- Ispezione del modello Pydantic ----------

    def _inspect_model_fields(self) -> dict:
        """
        Legge model_class.model_fields e costruisce una mappatura dei ruoli:
        - id_field
        - question_field
        - correct_answer_field (opzionale)
        - options_fields: {label: field_name}
        - options_dict_field (opzionale)
        """
        roles = {
            "id_field": None,
            "question_field": None,
            "correct_answer_field": None,
            "options_fields": {},      # es. {"A": "option_a", ...}
            "options_dict_field": None # es. "options"
        }

        for name, field in self.model_class.model_fields.items():
            extra = field.json_schema_extra or {}
            role = extra.get("role")

            if role == "id":
                roles["id_field"] = name
            elif role == "question":
                roles["question_field"] = name
            elif role == "correct_answer":
                roles["correct_answer_field"] = name
            elif role == "option":
                label = extra.get("label")
                if label is None:
                    raise ValueError(
                        f"Field '{name}' has role='option' but no 'label' in json_schema_extra"
                    )
                roles["options_fields"][label] = name
            elif role == "options_dict":
                roles["options_dict_field"] = name

        if roles["question_field"] is None:
            raise ValueError("No field with role='question' defined in model_class")

        # Le altre (id, correct_answer, opzioni) possono essere opzionali,
        # dipende dal tipo di dataset / task
        return roles

    # ---------- Helper per estrarre i valori da un item ----------

    def _get_id(self, item: BaseModel) -> str:
        field = self._roles["id_field"]
        if field is None:
            # fallback: usa str(index) altrove, oppure un hash
            # qui supponiamo che id sia obbligatorio per semplicità
            raise ValueError("Model has no field with role='id'")
        return getattr(item, field)

    def _get_question(self, item: BaseModel) -> str:
        field = self._roles["question_field"]
        return getattr(item, field)

    def _get_correct_answer(self, item: BaseModel) -> str | None:
        field = self._roles["correct_answer_field"]
        if field is None:
            return None
        return getattr(item, field)

    def _get_options(self, item: BaseModel) -> dict[str, str]:
        """
        Ritorna dict {label: testo} delle opzioni multiple choice.
        Supporta due varianti:
        - più campi singoli con role='option' e label='A','B',...
        - un solo campo dict con role='options_dict'
        """
        options: dict[str, str] = {}

        # Variante 1: fields singoli
        for label, field_name in self._roles["options_fields"].items():
            value = getattr(item, field_name, None)
            if value is not None:
                options[label] = str(value)

        # Variante 2: dict unico
        dict_field = self._roles["options_dict_field"]
        if dict_field is not None:
            raw_dict = getattr(item, dict_field, None) or {}
            # raw_dict può essere già {"A": "...", "B": "..."} o simile
            for k, v in raw_dict.items():
                options[str(k)] = str(v)

        return options

    def _format_multiple_choice_question(self, question: str, options: dict[str, str]) -> str:
        """
        Converte domanda + opzioni in testo per il modello:
        question
        A) ...
        B) ...
        ...
        """
        parts = [question]
        # Se hai i classici A,B,C,D li ordina così, altrimenti in ordine di chiave
        standard_order = ["A", "B", "C", "D"]
        labels = [l for l in standard_order if l in options] + [
            l for l in options.keys() if l not in standard_order
        ]
        for label in labels:
            parts.append(f"{label}) {options[label]}")
        return "\n".join(parts)

    # ---------- Run sul dataset (inference) ----------

    def run_on_dataset(self):
        """
        Esegue multiple choice + open-ended su tutto il dataset in parallelo
        e salva i risultati su file JSON.
        """

        def process_item(qa_item: BaseModel):
            question = self._get_question(qa_item)
            options = self._get_options(qa_item)
            qa_id = self._get_id(qa_item)

            user_message = question
            prepared_mc = self._format_multiple_choice_question(question, options)

            multiple_choice_results = self.execute_multiple_choice(prepared_mc)
            open_ended_results = self.execute_open_ended(user_message)
            return qa_id, multiple_choice_results, open_ended_results

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_item, qa_item) for qa_item in self.dataset]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Executing QA items"):
                qa_id, multiple_choice_results, open_ended_results = future.result()
                self.save_results(qa_id, multiple_choice_results, open_ended_results)

        self._save_results_to_file(
            "multiple_choice_results.json", self.multiple_choice_results_dict
        )
        self._save_results_to_file(
            "open_ended_results.json", self.open_ended_results_dict
        )

    # ---------- Run sul dataset (evaluation) ----------

    def run_evaluation_on_dataset(self, evaluator):
        """
        Usa un evaluator per valutare multiple choice + open-ended.
        Richiede che il modello abbia un campo con role='correct_answer'.
        """
        for qa_item in self.dataset:
            qa_id = self._get_id(qa_item)
            question = self._get_question(qa_item)
            options = self._get_options(qa_item)
            correct_answer = self._get_correct_answer(qa_item)

            user_message = question
            prepared_mc = self._format_multiple_choice_question(question, options)

            multiple_choice_results = evaluator.evaluate_multiple_choice(
                prepared_mc, correct_answer
            )

            open_ended_raw = self.execute_open_ended(user_message)
            open_ended_results = {
                model: evaluator.check_open_ended_answer(response, correct_answer)
                for model, response in open_ended_raw.items()
            }

            self.save_results(qa_id, multiple_choice_results, open_ended_results)

        self._save_results_to_file(
            "multiple_choice_results.json", self.multiple_choice_results_dict
        )
        self._save_results_to_file(
            "open_ended_results.json", self.open_ended_results_dict
        )

    # ---------- Gestione risultati ----------

    def save_results(
        self, qa_id: str, multiple_choice_results: dict, open_ended_results: dict
    ):
        self.multiple_choice_results_dict[qa_id] = multiple_choice_results
        self.open_ended_results_dict[qa_id] = open_ended_results

        logger.info("Results for QA ID %s", qa_id)
        logger.info("Multiple Choice Results: %s", multiple_choice_results)
        logger.info("Open Ended Results: %s", open_ended_results)

    def _save_results_to_file(self, filename: str, results_dict: dict):
        os.makedirs(self.results_folder, exist_ok=True)
        full_path = os.path.join(self.results_folder, filename)
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)

    # ---------- Chiamata ai modelli ----------

    def execute_multiple_choice(self, user_message: str) -> dict:
        messages = self.lm_client.format_messages(
            system="Respond only with the letter corresponding to the correct choice (A, B, C, or D).",
            user=user_message,
        )

        results = {}
        for model in self.models_list:
            response = self.lm_client.chat(
                model,
                messages=messages,
            )
            results[model] = response
        return results

    def execute_open_ended(self, user_message: str) -> dict:
        messages = self.lm_client.format_messages(
            system="Provide a concise answer to the following question.",
            user=user_message,
        )

        results = {}
        for model in self.models_list:
            response = self.lm_client.chat(
                model,
                messages=messages,
            )
            results[model] = response
        return results

    # ---------- Langfuse (opzionale, invariato a grandi linee) ----------

    def run_langfuse_experiment_with_lmclient(
        self,
        experiment_name: str,
        experiment_description: str,
        model_name: str,
        dataset_name: str = "GenericDatasetExecutor",
    ):
        langfuse = get_client()
        encoded_dataset_name = quote(dataset_name, safe="")
        dataset = langfuse.get_dataset(encoded_dataset_name)

        def task(item, **kwargs):
            question = item.input.get("question", "")
            # Qui sei vincolato al formato del dataset Langfuse,
            # che è un altro problema rispetto al modello Pydantic locale.
            option_a = item.input.get("option_a", "")
            option_b = item.input.get("option_b", "")
            option_c = item.input.get("option_c", "")
            option_d = item.input.get("option_d", "")
            options = {"A": option_a, "B": option_b, "C": option_c, "D": option_d}

            prepared_mc = self._format_multiple_choice_question(question, options)

            multiple_choice_result = self.execute_multiple_choice(
                prepared_mc
            ).get(model_name, "")
            open_ended_result = self.execute_open_ended(question).get(model_name, "")

            return {
                "multiple_choice_response": multiple_choice_result,
                "open_ended_response": open_ended_result,
            }

        result = dataset.run_experiment(
            name=experiment_name, description=experiment_description, task=task
        )

        print(result.format())
        return result

    def run_langfuse_experiments_for_models_with_lmclient(
        self,
        models_list: list,
        experiment_name_prefix: str = "Model Evaluation",
        dataset_name: str = "GenericDatasetExecutor",
    ):
        for model_name in models_list:
            experiment_name = f"{experiment_name_prefix} - {model_name}"
            experiment_description = (
                f"Evaluation of model {model_name} on {dataset_name}"
            )
            self.run_langfuse_experiment_with_lmclient(
                experiment_name, experiment_description, model_name, dataset_name
            )