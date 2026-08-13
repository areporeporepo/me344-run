# ME344 Final Project: Train, Tune, and Scale Qwen3!

Welcome! This final project gives you a hands-on look at how ML engineers pre-train, fine-tune, profile, and scale Qwen3 using MaxText and TPUs. Plan on spending about 90 minutes plus any cluster queue time.

This page shows you how to set up your TPU virtual machine (VM) for the first part of this project and open the [`final_project.ipynb`](final_project.ipynb) notebook. After this you will run the GKE scaling experiment, upload your work, and delete your workloads. We'll be uploading work into the shared `gs://me344-tpu-labs-west4/final_projects/` Google Cloud Storage Bucket in our `soe-hpccenter` project (the scripts will take care of this).

We highly recommend you check out the documentation for the TPU v5e [here](https://docs.cloud.google.com/tpu/docs/v5e) so you have a sense of what your hardware is capable of!

## Connect To Your Stanford Node

We will use three computers during this project. Your laptop displays Jupyter, your assigned Stanford node runs the `gcloud`, Docker, and Kubernetes commands, and the TPU VM runs MaxText. The first SSH connection reaches the Stanford node, and the second reaches the TPU VM.

On your laptop, open a terminal and use the SSH destination from Canvas in the command below. The four `-L` options create private paths back to your laptop for Jupyter, inference, chat, and XProf.

```bash
# Laptop terminal
ssh \
  -L 8888:localhost:8888 \
  -L 8000:localhost:8000 \
  -L 7860:localhost:7860 \
  -L 6006:localhost:6006 \
  YOUR_CANVAS_SSH_DESTINATION
```

Replace `YOUR_CANVAS_SSH_DESTINATION` with the `SUNetID@hostname` destination on Canvas. Once the connection opens, your terminal is running on the Stanford node.

```text
laptop browser  -- SSH tunnel -->  Stanford node  -- TPU SSH tunnel -->  TPU VM  -->  8 TPU chips
                                      gcloud                               Jupyter
                                      Docker                               MaxText
                                      kubectl                              JAX
```

## Set Up Google Cloud

Run this once on the Stanford node and enter your SUNet ID. `gcloud auth login` prints a link that you open in your laptop's browser, then asks you to paste the authorization code back into the Stanford terminal. The last command downloads the assignment to the Stanford node so we can use its TPU setup script.

```bash

SUNET=
until [[ "$SUNET" =~ ^[a-z0-9]+$ ]]; do
  printf 'SUNet ID (lowercase): '
  read -r SUNET
done
```

```bash
export PROJECT_ID=soe-hpccenter
export SUNET
export ZONE=us-west4-a
export ASSIGNMENT_URI=gs://me344-tpu-labs-west4/data/me344_final_project
export BASE_CHECKPOINT=gs://me344-tpu-labs-west4/data/qwen3-4b-instruct-2507/0/items
export UPLOAD_BUCKET=gs://me344-tpu-labs-west4/final_projects

export ME344_STUDENT_ID="$SUNET"
export ME344_OUTPUT="${UPLOAD_BUCKET}/${SUNET}/runs"
export ME344_SUBMISSION_URI="${UPLOAD_BUCKET}/${SUNET}/submission"
export ME344_BASE_CHECKPOINT="$BASE_CHECKPOINT"
export ME344_TPU_ZONE="$ZONE"
export WANDB_MODE=offline

env | grep -E '^(PROJECT_ID|SUNET|ZONE|ASSIGNMENT_URI|BASE_CHECKPOINT|UPLOAD_BUCKET|TPU_NAME|ME344_[A-Z_]+|WANDB_MODE)=' \
  | sed 's/^/export /' > "$HOME/.me344.env"

gcloud config set project "$PROJECT_ID"
gcloud config set compute/zone "$ZONE"

mkdir -p "$HOME/me344_final_project"
gcloud storage rsync "$ASSIGNMENT_URI" "$HOME/me344_final_project" --recursive
```

Use `source "$HOME/.me344.env"` whenever you open another terminal on the Stanford node.

## Start The TPU VM

The course maps each Stanford node to one TPU VM, and the script handles that mapping for you. For example, it turns `hpcc-cluster-58` into the login `student58@tpu-student58`. Your SUNet ID names your GCS folder, container, and GKE jobs.

Run this on the Stanford node. The script follows the course naming scheme, creates or starts the TPU VM, adds your SSH key, copies the course settings, and opens the second half of the tunnel back to your laptop. When it finishes, your terminal prompt will change because you are now inside the TPU VM.

```bash
# Stanford-node terminal
cd "$HOME/me344_final_project"
bash scripts/start_tpu_vm.sh
```

Keep this terminal open because both SSH connections carry Jupyter traffic back to your laptop. Your laptop displays the notebook, Python runs on the TPU VM, and JAX sends the model work to its eight TPU chips.

## Install MaxText And Open Jupyter

Now we'll turn the TPU VM into our workspace. This block downloads the assignment and installs MaxText. The TPU VM reads the course bucket through its own service account, so you do not log in again here. Pre-training and post-training need slightly different Python packages, so each gets its own environment. The two `install_tpu_*_extra_deps` commands come from MaxText and run inside the environment activated directly above them. You only need to run this block once. The last command starts Jupyter and stays running.

```bash
# TPU-VM terminal
source "$HOME/.me344.env"
mkdir -p "$HOME/me344_final_project"
gcloud storage cp --recursive "$ASSIGNMENT_URI/*" "$HOME/me344_final_project/"
cd "$HOME/me344_final_project"
```
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
sudo sh -c 'echo always > /sys/kernel/mm/transparent_hugepage/enabled'
```
```bash
export MAXTEXT_SHA=17c7172720ca813b05e5ea248dedd78a0c64612e
export MAXTEXT_URL="git+https://github.com/AI-Hypercomputer/maxtext.git@${MAXTEXT_SHA}"
```
```bash
uv venv --python 3.12 --seed .venv-me344-pretrain
source .venv-me344-pretrain/bin/activate
uv pip install "maxtext[tpu] @ ${MAXTEXT_URL}" --resolution=lowest
install_tpu_pre_train_extra_deps
uv pip install "xprof==2.23.0"
deactivate
```
```bash
uv venv --python 3.12 --seed .venv-me344-posttrain
source .venv-me344-posttrain/bin/activate
UV_TORCH_BACKEND=cpu uv pip install \
  "maxtext[tpu-post-train] @ ${MAXTEXT_URL}" --resolution=lowest
install_tpu_post_train_extra_deps
uv pip install jupyterlab ipykernel wandb
python -m ipykernel install --user --name me344-posttrain \
  --display-name "ME344 MaxText"
python -m jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

After a few moments, the terminal will print a URL that starts with `http://127.0.0.1:8888/lab?token=...`. Open it in your laptop's browser, then open `final_project.ipynb` and select the **ME344 MaxText** kernel.

Start at the first cell and use **Shift+Enter** to work down the notebook. It will tell you when to pause for the GKE experiment below. Once the baseline is working, feel free to change a parameter, prompt, or dataset and see what happens. A v5e-8 is one TPU VM, so the first device check should show one JAX process controlling eight TPU devices. If it does not, please seek help during office hours or post on Canvas for help before continuing.

See you back in this README after the notebook directs you back!

## Scale Across TPU Workers

By the time the notebook sends you here, you will have finished the complete one-TPU workflow. The final systems experiment moves its saved pre-training checkpoint to the shared GKE cluster. We first measure an 8-chip baseline, then arrange 16 chips in two different ways to see how communication changes performance.

> **IMPORTANT: KEEP JUPYTER, THE TPU VM, AND YOUR FIRST SSH TERMINAL RUNNING.** When the notebook reaches **Run The GKE Comparison**, open a second terminal for GKE. You will return to the original notebook to combine the results and upload your submission.

Open a second laptop terminal and SSH to the same Stanford node with the command below. Do not repeat the `-L` options because your first connection already owns those ports. Leave this second terminal on the Stanford node, where we will build a container and ask Kubernetes to run it on the TPU workers.

```bash
# Second laptop terminal
ssh YOUR_CANVAS_SSH_DESTINATION
```

The notebook has already saved the pilot's model and optimizer state to the course bucket. Each GKE job restores that checkpoint, sets the learning rate to zero, and measures repeated training steps without changing the model.

```text
8-chip baseline:  [worker: 8 chips]                                  one 2x4 ICI slice
16-chip ICI:      [worker: 4] [worker: 4] [worker: 4] [worker: 4]   one 4x4 ICI slice
16-chip DCN:      [worker: 8] <----------- DCN ----------> [worker: 8]   two 2x4 ICI slices
```

GKE starts one pod on each TPU worker VM. ICI is the fast network inside a TPU slice, while DCN connects separate slices. The 16-chip runs use the same `DP2 x TP8` training layout, so their main systems difference is where data-parallel communication travels. See the [GKE TPU topology guide](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus) for the available shapes.

Run this in the second Stanford-node terminal. It reloads your saved settings, connects `kubectl` to the course cluster, and downloads the same assignment you used on the TPU VM:

```bash
# Second Stanford-node terminal
source "$HOME/.me344.env"
gcloud config set project "$PROJECT_ID"
gcloud container clusters get-credentials class-tpu-cluster-west4 \
  --region=us-west4 --project="$PROJECT_ID"

sudo dnf install podman-docker

export IMAGE_URI="us-central1-docker.pkg.dev/soe-hpccenter/tpu-images/me344-maxtext-${SUNET}:v1"

mkdir -p "$HOME/me344_final_project"
gcloud storage rsync "$ASSIGNMENT_URI" "$HOME/me344_final_project" --recursive
cd "$HOME/me344_final_project"
docker version
kubectl config current-context
kubectl get crd jobsets.jobset.x-k8s.io
```

The last command should print `customresourcedefinition.apiextensions.k8s.io/jobsets.jobset.x-k8s.io`. If it does not, please tell the instructor. The cluster needs the JobSet controller before it can run the supplied YAML.

GKE workers cannot see files in your Stanford home directory. A Docker image packages the assignment and its pinned MaxText version so every worker starts with the same code. Build it, then push it to Artifact Registry:

```bash
# Second Stanford-node terminal
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
docker build -t "$IMAGE_URI" .
docker push "$IMAGE_URI"
```

Define one small helper that renders the YAML, waits for a run, prints its logs, and deletes the finished JobSet. The restricted `envsubst` list fills in our template settings while leaving variables used later inside the container unchanged.

```bash
# Second Stanford-node terminal
export ME344_TEMPLATE_VARS='${JOBSET_NAME} ${SUNET} ${NUM_SLICES} ${RUN_KIND} ${IMAGE_URI} ${PODS_PER_SLICE} ${TPU_TOPOLOGY} ${CHIPS_PER_POD}'

run_me344_jobset() {
  kubectl delete jobset "$JOBSET_NAME" --ignore-not-found --wait=true
  envsubst "$ME344_TEMPLATE_VARS" < scale-job.yaml | kubectl create -f -
  kubectl wait --for=create pod \
    -l "jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME}" --timeout=15m
  kubectl get pods -l "jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME}"
  kubectl wait --for=condition=Completed "jobset/${JOBSET_NAME}" --timeout=35m
  kubectl logs -l "jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME}" \
    -c maxtext --prefix --max-log-requests=8 | tail -200
  kubectl delete jobset "$JOBSET_NAME" --ignore-not-found --wait=true
}
```

Start with the same 8-chip `2x4` slice used by the TPU VM. This gives us a baseline at global batch 256 and sequence length 256.

```bash
# Second Stanford-node terminal
export JOBSET_NAME="me344-${SUNET}-base"
export NUM_SLICES=1 RUN_KIND=baseline PODS_PER_SLICE=1
export TPU_TOPOLOGY=2x4 CHIPS_PER_POD=8
run_me344_jobset
```

Now use one 16-chip `4x4` slice. GKE creates four workers with four chips each, all joined by ICI. The job runs strong scaling at global batch 256 and weak scaling at global batch 512.

```bash
# Second Stanford-node terminal
export JOBSET_NAME="me344-${SUNET}-ici"
export NUM_SLICES=1 RUN_KIND=ici PODS_PER_SLICE=4
export TPU_TOPOLOGY=4x4 CHIPS_PER_POD=4
run_me344_jobset
```

Finally, arrange the same 16 chips as two `2x4` slices. The data-parallel replicas now synchronize across DCN. The model, checkpoint, sequence length, batch sizes, and logical sharding stay fixed.

```bash
# Second Stanford-node terminal
export JOBSET_NAME="me344-${SUNET}-dcn"
export NUM_SLICES=2 RUN_KIND=dcn PODS_PER_SLICE=1
export TPU_TOPOLOGY=2x4 CHIPS_PER_POD=8
run_me344_jobset
```

Each trial runs 12 repeated batches and averages the last eight after compilation and warmup. **Strong scaling** keeps global batch 256 fixed and asks whether twice the chips finish the same work faster. **Weak scaling** doubles the batch to 512 so each eight-chip model replica keeps the batch-256 workload. Look for `ME344_SCALE_RESULT` in the logs, then check that all five result files exist:

```bash
# Second Stanford-node terminal
for MODE in baseline ici_strong ici_weak strong weak; do
  gcloud storage cat "${UPLOAD_BUCKET}/${SUNET}/scaling/${MODE}.json"
done

kubectl get jobsets | grep "$SUNET" || echo "GKE workloads deleted"
```

Return to Jupyter and continue with **Shift+Enter**. The notebook reads the five GKE results, compares strong and weak scaling over ICI and DCN, builds your dashboard, asks four short questions, and uploads the submission.

## Submit And Clean Up

The last notebook cell asks four short questions and uploads one archive with `answers.md` and `systems_dashboard.png`. Keep the TPU VM running until that upload succeeds. Then return to either Stanford-node terminal, confirm the archive is in the bucket, and delete your TPU VM. The final command checks that it is gone.

```bash
# Stanford-node terminal
source "$HOME/.me344.env"
gcloud storage ls "${UPLOAD_BUCKET}/${SUNET}/submission/*.final.tar.gz"
gcloud compute tpus tpu-vm delete "$TPU_NAME" --zone="$ZONE" --quiet
gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" || \
  echo "Deleted: $TPU_NAME"
```

Finally, open the [course submission bucket](https://console.cloud.google.com/storage/browser/me344-tpu-labs-west4/final_projects?project=soe-hpccenter) in your laptop's browser. Open your SUNet folder, then `submission`, and download the `.final.tar.gz` archive. Submit that file to Canvas, and you're done!
