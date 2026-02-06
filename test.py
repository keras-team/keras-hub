import kagglehub

kagglehub.login()

# Replace with path to directory containing model files.
model_name = "translategemma_27b_it"
LOCAL_MODEL_DIR = f'{model_name}'

MODEL_SLUG = 'TranslateGemma' # Replace with model slug.

# Learn more about naming model variations at
# https://www.kaggle.com/docs/models#name-model.
VARIATION_SLUG = model_name # Replace with variation slug.

kagglehub.model_upload(
  handle = f"keras/{MODEL_SLUG}/keras/{VARIATION_SLUG}",
  local_model_dir = LOCAL_MODEL_DIR,
  version_notes = 'initial release')
