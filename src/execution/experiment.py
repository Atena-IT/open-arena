from langfuse import Langfuse
import os

# --- CONFIGURATION ---
# Make sure these environment variables are set:
# LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST (optional)
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

EXPERIMENT_NAME = "my_experiment_v1"
DATASET_NAME = "my_dataset"  # <-- name of the datasets stored in Langfuse


if __name__ == "__main__":
    langfuse = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST
    )

    # --- LOAD DATASET ---
    # Retrieve the datasets and its items (inputs + expected outputs)
    dataset = langfuse.dataset(DATASET_NAME)
    items = dataset.items()

    print(f"Found {len(items)} items in datasets '{DATASET_NAME}'.")

    # --- DEFINE THE MODEL / FUNCTION TO TEST ---
    def run_model(input_text: str) -> str:
        """
        Replace this dummy function with your actual model call.
        For demonstration purposes this simply returns the uppercase text.
        """
        return input_text.upper()


    # --- RUN EXPERIMENT ---
    for item in items:
        input_data = item.input
        expected_output = item.expected_output  # may be None

        # Run your model on the datasets input
        model_output = run_model(input_data["text"])

        # Log the model output as part of the experiment
        trace = langfuse.trace(
            name="experiment_trace",
            metadata={"datasets": DATASET_NAME},
            user_id="test_runner"
        )

        trace.score(
            name=EXPERIMENT_NAME,
            value=model_output,
            metadata={
                "expected": expected_output,
                "input": input_data
            }
        )

        print(f"Logged experiment run for input: {input_data}")

    print("Experiment completed!")
