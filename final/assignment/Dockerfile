FROM python:3.12-slim-bookworm

# The shared GKE v5e runtime currently uses the JAX 0.9 TPU stack. This is the
# newest MaxText pin validated against that stack; the TPU VM uses the newer
# course pin documented in README.md.
ARG MAXTEXT_SHA=07a3264e056c8d57abf0e00ab567d1f074a52233

COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /usr/local/bin/uv

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

RUN uv pip install --system \
      "maxtext[tpu] @ git+https://github.com/AI-Hypercomputer/maxtext.git@${MAXTEXT_SHA}" \
      --resolution=lowest && \
    install_tpu_pre_train_extra_deps

# Newer MaxText checkpoints contain an auxiliary NNX metadata branch. Orbax
# can safely ignore that unknown branch while restoring the matching state.
RUN sed -i \
      's/restore(p, abstract_unboxed_pre_state, restore_args=restore_args)/restore(p, abstract_unboxed_pre_state, restore_args=restore_args, partial_restore=True)/' \
      /usr/local/lib/python3.12/site-packages/maxtext/common/checkpointing.py && \
    grep -q 'partial_restore=True' \
      /usr/local/lib/python3.12/site-packages/maxtext/common/checkpointing.py

WORKDIR /app
COPY scripts/final_project.py scripts/final_project.py

ENV JAX_PLATFORMS=tpu \
    ME344_RUNTIME_MAXTEXT_SHA=${MAXTEXT_SHA} \
    PYTHONUNBUFFERED=1
