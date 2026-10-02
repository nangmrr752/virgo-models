# Sourced by the train_*.sh scripts: the settings for this machine.
# - Your own server: env.sh, written once by scripts/local_setup.sh (conda env, data folder).
# - Colab or Kaggle (no env.sh): the data folder is picked here and the training packages installed.
if [ -f "$CODE/env.sh" ]; then
  source "$CODE/env.sh"
else
  if [ -d /kaggle/working ]; then ROOT_DEFAULT=/tmp/virgo        # Kaggle: /kaggle/working holds only 20 GB
  elif [ -d /content ]; then ROOT_DEFAULT=/content/virgo         # Colab
  else ROOT_DEFAULT="$(dirname "$CODE")/virgo-data"; fi
  export VIRGO_HOME="${VIRGO_HOME:-$ROOT_DEFAULT}"
  export HF_HOME="${HF_HOME:-$VIRGO_HOME/hf-cache}"
  export WORKSPACE="${WORKSPACE:-$VIRGO_HOME}"
  mkdir -p "$VIRGO_HOME"
  python -c "import transformers, peft, datasets, bitsandbytes" 2>/dev/null \
    || python -m pip install -q -r "$CODE/requirements-train.txt"
  python -m pip uninstall -y -q torchao 2>/dev/null || true  # conflicts with newer peft
fi
# GPU memory in GB (0 without a GPU): scripts pick sizes that fit.
GPU_GB=$(python -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 1e9) if torch.cuda.is_available() else 0)" 2>/dev/null || echo 0)
