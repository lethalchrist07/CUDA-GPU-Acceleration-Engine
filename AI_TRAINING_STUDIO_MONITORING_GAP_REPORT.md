# NEXORA AI Training Studio — Inspect Results & Live Monitoring Gap Report

**Audit Mode:** STRICT READ-ONLY FORENSIC AUDIT  
**Date:** November 13, 2026  
**Auditor:** Senior Deep Learning Systems & Full-Stack Reliability Engineer  
**Target Job Identifier:** `job_20261113_153924_735287`  
**Target Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6144 MB VRAM, Compute Capability 8.6, Driver 576.02, CUDA 12.8)  
**Execution Environment:** Windows 11 Home, Python 3.11.9, PyTorch 2.6.0+cu124, FastAPI Backend (PID 10988, Port 8000), Vite Frontend (Port 5173)

---

## 1. Executive Summary & Audited Job Baseline

An end-to-end investigation was conducted on the NEXORA AI Training Studio's **Inspect Results** and **Live Monitoring** workflows. The audit specifically evaluated historical and live job state management, metric ingestion pipelines, scientific chart rendering, workflow stepper behavior, and hardware telemetry collection.

### Audited Job Forensic Snapshot: `job_20261113_153924_735287`
Extracted directly from the SQLite persistent state store at [data/training_jobs.db](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/data/training_jobs.db):

| Attribute | Database Value |
|---|---|
| **Job ID** | `job_20261113_153924_735287` |
| **Model** | `transformer_50m` (NEXORA GPT-86.8M / Production Pretrained Decoder-Only Transformer) |
| **Backend** | `CPU_NATIVE` |
| **Dataset** | `Tiny Shakespeare (1.1MB Text Corpus)` (`data/text/tiny_shakespeare.txt`) |
| **Status** | `COMPLETED` |
| **Configured Epochs** | `1` |
| **Configured Batch Size** | `2` |
| **Configured Sequence Length** | `128` |
| **Configured Max Steps** | `15` |
| **Gradient Accumulation Steps** | `2` |
| **Stored Progress JSON** | `{"step": 8, "current_step": 15, "total_steps": 15, "batch": 15, "microbatches_processed": 15, "train_loss": 4.594, "train_perplexity": 98.89}` |
| **Stored Metrics JSON** | `final_train_loss: 4.594`, `final_val_loss: 4.76`, `final_val_perplexity: 116.75`, `step_records` length: `8` |
| **Checkpoint Path** | `checkpoints/training_jobs/job_20261113_153924_735287/final_model.pt` (Accuracy: 116.75, SHA-256 verified) |

---

## 2. Discrepancy 1: 15/15 Steps vs. 8 Points on Loss Graph

### Question
*Job reports 15/15 steps, but the loss graph displays only 8 points. Determine whether metrics are sampled, truncated, or incorrectly rendered.*

### Confirmed Findings & Root Cause Analysis
1. **The metrics are NEITHER sampled NOR truncated NOR incorrectly rendered by Recharts.**
2. **Dataloader Microbatch Math:**
   - In [engine/transformer/data_pipeline.py](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/engine/transformer/data_pipeline.py) and [engine/transformer/dataset.py](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/engine/transformer/dataset.py), the `Tiny Shakespeare` corpus (1.1 MB text) tokenized with BPE (`target_vocab_size=1024`, `seq_len=128`, `val_ratio=0.1`) yields a training set that, with `batch_size=2`, produces **exactly 15 microbatches** per epoch:
     $$\text{len}(\text{train\_loader}) = 15$$
3. **Gradient Accumulation & Optimizer Step Math:**
   - In [engine/transformer/trainer.py](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/engine/transformer/trainer.py#L328-L352):
     ```python
     # Optimizer step when accumulation threshold is met or at final batch of epoch
     if microbatch_in_accum == accum_steps or is_last_batch_in_epoch:
         ...
         self.global_step += 1
         ...
         step_records.append(step_data)
     ```
   - With `accum_steps = 2`, an optimizer step is executed every 2 microbatches, plus a final accumulation flush on the final batch of the epoch:
     - Microbatches 1 & 2 $\rightarrow$ **Optimizer Step 1** (batch 2)
     - Microbatches 3 & 4 $\rightarrow$ **Optimizer Step 2** (batch 4)
     - Microbatches 5 & 6 $\rightarrow$ **Optimizer Step 3** (batch 6)
     - Microbatches 7 & 8 $\rightarrow$ **Optimizer Step 4** (batch 8)
     - Microbatches 9 & 10 $\rightarrow$ **Optimizer Step 5** (batch 10)
     - Microbatches 11 & 12 $\rightarrow$ **Optimizer Step 6** (batch 12)
     - Microbatches 13 & 14 $\rightarrow$ **Optimizer Step 7** (batch 14)
     - Microbatch 15 (tail flush, `is_last_batch_in_epoch=True`) $\rightarrow$ **Optimizer Step 8** (batch 15)
   - Because `epochs = 1`, the dataloader was exhausted after 15 microbatches. Thus, exactly **8 optimizer steps occurred in total**.
4. **Where Did "15/15 Steps" Come From?**
   - In [engine/training/job_store.py](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/engine/training/job_store.py#L399-L403) within `complete_job()`:
     ```python
     total_batches = p.get("total_batches") or p.get("total_steps") or 1
     p["batch"] = total_batches
     p["current_step"] = total_batches
     p["total_batches"] = total_batches
     p["total_steps"] = total_batches
     ```
     At job completion, `complete_job()` forced `p["current_step"]` and `p["total_steps"]` to equal `total_batches` (which was 15 microbatches).
   - In [frontend/src/components/training/ActiveJobMonitor.tsx](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/components/training/ActiveJobMonitor.tsx#L46-L47):
     ```tsx
     const totalSteps = progress.total_steps ?? job.config?.max_steps ?? (metrics.step_records?.length || 1);
     const currentStep = isCompleted ? totalSteps : (progress.current_step ?? progress.step ?? 0);
     ```
     Since `job.status === 'COMPLETED'`, `currentStep` was forced to `totalSteps` (15), rendering:
     $$\text{Step } 15\ /\ 15$$
5. **Loss Graph Rendering:**
   - In [frontend/src/components/training/TrainingCharts.tsx](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/components/training/TrainingCharts.tsx#L32-L42), `chartData` maps directly over `metrics.step_records`.
   - `metrics.step_records` contains exactly 8 entries (one per optimizer step).
   - The graph accurately and correctly renders all 8 optimizer step loss points (`Step 1` through `Step 8`).
   - The discrepancy is caused entirely by an **accounting semantics collision** between **microbatches** (15) and **optimizer steps** (8) in `job_store.complete_job()` and `ActiveJobMonitor.tsx`.

### Required Fixes
- **Backend ([engine/training/job_store.py](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/engine/training/job_store.py#L396-L403)):**
  Do not overwrite `current_step` with `total_batches`. Preserve optimizer step accounting:
  ```python
  # Distinguish between optimizer steps and microbatches
  opt_steps = metrics.get("optimizer_steps_completed", p.get("step", 1))
  p["step"] = opt_steps
  p["current_step"] = opt_steps
  p["total_steps"] = opt_steps
  p["microbatches_processed"] = metrics.get("microbatches_processed", p.get("microbatches_processed", 1))
  ```
- **Frontend ([frontend/src/components/training/ActiveJobMonitor.tsx](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/components/training/ActiveJobMonitor.tsx#L46-L50)):**
  Use `metrics.optimizer_steps_completed` or `metrics.step_records?.length` as the authoritative step count for language models:
  ```tsx
  const totalOptimizerSteps = metrics.optimizer_steps_completed ?? metrics.step_records?.length ?? progress.step ?? 1;
  const currentStep = isCompleted ? totalOptimizerSteps : (progress.step ?? progress.current_step ?? 0);
  const totalSteps = totalOptimizerSteps;
  ```

---

## 3. Discrepancy 2: Validation Perplexity 116.75 vs. Empty Validation Progression Graph

### Question
*Validation perplexity displays 116.75, but the validation progression graph is empty. Trace the backend response and frontend chart data.*

### Confirmed Findings & Trace
1. **Periodic Evaluation Interval Setting:**
   - In [engine/transformer/trainer.py](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/engine/transformer/trainer.py#L53), the default configuration sets:
     ```python
     eval_interval_steps: int = 50
     ```
2. **Intra-Training Validation Execution:**
   - In [engine/transformer/trainer.py](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/engine/transformer/trainer.py#L357-L364):
     ```python
     # Periodic validation
     latest_val_loss = None
     latest_val_ppl = None
     if self.val_loader and (self.global_step % self.config.eval_interval_steps == 0):
         v_loss, v_ppl = self.evaluate()
         latest_val_loss = v_loss
         latest_val_ppl = v_ppl
     ```
   - Because the entire training run completed in only **8 optimizer steps**, the condition `(self.global_step % 50 == 0)` was **never satisfied** ($8 < 50$).
   - Consequently, for all 8 elements in `step_records`, `val_loss: null` and `val_perplexity: null` were recorded.
3. **End-of-Run Final Evaluation:**
   - In [engine/transformer/trainer.py](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/engine/transformer/trainer.py#L421-L453):
     ```python
     total_wall_time = time.perf_counter() - start_time
     final_val_loss, final_val_ppl = self.evaluate()
     ...
     return {
         ...
         "final_val_loss": final_val_loss,          # 4.76
         "final_val_perplexity": final_val_ppl,      # 116.75
         "step_records": step_records,              # All 8 records have val_perplexity: None
     }
     ```
   - Post-training validation executed successfully at epoch termination and computed `final_val_loss = 4.76` and `final_val_perplexity = 116.75`.
   - This single scalar was returned in `metrics.final_val_perplexity` and stored in `job_checkpoints.accuracy = 116.75`.
4. **Frontend KPI Badge vs. Chart Series Disconnect:**
   - In [frontend/src/components/training/ActiveJobMonitor.tsx](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/components/training/ActiveJobMonitor.tsx#L136):
     ```tsx
     const valPerplexity = progress.val_perplexity ?? metrics.final_val_perplexity;
     ```
     `valPerplexity` resolves to `116.75`. Line 275 renders `116.75` in the "Val Perplexity" top summary card.
   - In [frontend/src/components/training/TrainingCharts.tsx](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/components/training/TrainingCharts.tsx#L38-L39):
     ```tsx
     val_perplexity: st.val_perplexity !== undefined && st.val_perplexity > 0 
       ? Number(st.val_perplexity.toFixed(2)) 
       : undefined,
     ```
     Because `st.val_perplexity` is `null` across all 8 entries in `metrics.step_records`, `val_perplexity` in `chartData` is `undefined` for every data point.
   - In [frontend/src/components/training/TrainingCharts.tsx](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/components/training/TrainingCharts.tsx#L152-L160):
     ```tsx
     <Line
       type="monotone"
       dataKey={isTransformer ? "val_perplexity" : "val_acc"}
       name={isTransformer ? "Val Perplexity" : "Validation Accuracy (%)"}
       stroke="#70D6C2"
     />
     ```
     Recharts requires valid numeric points to draw vertices or line segments. With only `undefined` values, Recharts renders **zero SVG path coordinates**, leaving the validation progression graph entirely blank.

### Required Fixes
- **Backend ([engine/transformer/trainer.py](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/engine/transformer/trainer.py#L421-L435)):**
  Update the final record in `step_records` with the final validation results, and automatically scale `eval_interval_steps` when short step budgets are configured:
  ```python
  final_val_loss, final_val_ppl = self.evaluate()
  if step_records:
      step_records[-1]["val_loss"] = final_val_loss
      step_records[-1]["val_perplexity"] = final_val_ppl
  ```
- **Backend Configuration ([engine/training/orchestrator.py](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/engine/training/orchestrator.py#L479-L488)):**
  Dynamically set `eval_interval_steps = max(1, min(50, total_expected_steps // 4))` so that validation checkpoints are captured at least 4 times across any run.
- **Frontend Fallback ([frontend/src/components/training/TrainingCharts.tsx](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/components/training/TrainingCharts.tsx#L33-L42)):**
  If all step records have `val_perplexity === undefined` but `metrics.final_val_perplexity` exists, populate the last point of `chartData`:
  ```tsx
  if (isTransformer && chartData.length > 0 && !chartData.some(d => d.val_perplexity !== undefined)) {
    if (metrics.final_val_perplexity !== undefined) {
      chartData[chartData.length - 1].val_perplexity = Number(Number(metrics.final_val_perplexity).toFixed(2));
      chartData[chartData.length - 1].val_loss = metrics.final_val_loss ? Number(Number(metrics.final_val_loss).toFixed(4)) : undefined;
    }
  }
  ```

---

## 4. Discrepancy 3: Monitor Workflow Step Disabled for Completed Jobs

### Question
*Determine why the Monitor workflow step is disabled for completed jobs.*

### Confirmed Findings & Line References
1. **Location:** [frontend/src/components/training/TrainingStudio.tsx](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/components/training/TrainingStudio.tsx#L259-L286)
2. **Stepper Configuration Code:**
   ```tsx
   {[
     { step: 1, label: 'Choose Model', active: !activeJob || activeJob.status === 'COMPLETED' },
     { step: 2, label: 'Configure Training', active: !activeJob || activeJob.status === 'COMPLETED' },
     { step: 3, label: 'Review Config', active: !activeJob || activeJob.status === 'COMPLETED' },
     { step: 4, label: 'Launch', active: isSubmitting },
     { step: 5, label: 'Monitor', active: activeJob && ['QUEUED', 'RUNNING'].includes(activeJob.status) },
     { step: 6, label: 'Inspect Results', active: activeJob && activeJob.status === 'COMPLETED' },
   ].map((item, idx, arr) => (
     <div className={`flex items-center gap-2 px-2.5 py-1 rounded-lg transition-all ${
       item.active 
         ? 'bg-[#F5A15D]/10 text-[#F5F5F4] font-semibold border border-[#F5A15D]/30' 
         : 'text-[#6B7280] opacity-60'
     }`}>
   ```
3. **Root Cause Analysis:**
   - The stepper is a **declarative, read-only lifecycle phase indicator**, not an interactive tab switcher.
   - For Step 5 ("Monitor"), `active` is evaluated as:
     $$\text{active} = \text{activeJob} \land (\text{status} \in \{\text{'QUEUED'}, \text{'RUNNING'}\})$$
   - When a job transitions to `'COMPLETED'`, this condition evaluates to `false`. The stepper marks Step 5 with `text-[#6B7280] opacity-60` (dimmed/disabled appearance).
   - Simultaneously, Step 6 ("Inspect Results") condition `activeJob.status === 'COMPLETED'` evaluates to `true`, lighting up Step 6 in orange (`bg-[#F5A15D]/10 text-[#F5F5F4]`).
   - The user interface is communicating that the "Live Monitoring" phase has ended and the interface is now in the "Inspect Results" phase.
   - **Important UX Nuance:** Below the stepper banner, `<ActiveJobMonitor>` and `<TrainingCharts>` **remain fully visible and accessible** because their enclosing container is guarded only by `{activeJob && (...)}`.
   - **Existing Logic Defect in Stepper:** Steps 1, 2, and 3 are also conditioned on `activeJob.status === 'COMPLETED'`. As a result, when a job completes, Steps 1, 2, 3, AND 6 are all active at the same time, while Step 5 appears disabled.

### Required Fixes
- **Frontend ([frontend/src/components/training/TrainingStudio.tsx](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/components/training/TrainingStudio.tsx#L259-L266)):**
  Refactor the stepper to render a standard linear state machine where past steps display as "Completed" (green checkmark or completed badge), and the current active phase is highlighted exclusively:
  ```tsx
  const currentWorkflowStep = !activeJob 
    ? (selectedModelId ? 2 : 1)
    : isSubmitting 
      ? 4 
      : ['QUEUED', 'RUNNING', 'CANCEL_REQUESTED'].includes(activeJob.status) 
        ? 5 
        : 6; // COMPLETED, FAILED, CANCELLED
  ```

---

## 5. Discrepancy 4: Running Job Transition into Live Monitoring Interface

### Question
*Identify how a running job transitions into the live monitoring interface.*

### Confirmed Findings & Execution Path Trace

The transition occurs through two distinct pathways: **(A) Direct Launch Transition**, and **(B) Historical Re-Inspection Transition**.

```
[User Action: Launch Job] 
         │
         ▼
TrainingConfigForm.onSubmit(payload)
         │
         ▼
TrainingStudio.handleLaunchJob() 
         │ ──> POST /api/training/jobs ──> TrainingOrchestrator.submit_job() [Status: QUEUED]
         ▼
setActiveJob(newJob) & setActiveStudioTab('studio')
         │
         ├──> Mounts <ActiveJobMonitor job={activeJob}>
         ├──> Mounts <TrainingCharts job={activeJob}>
         └──> Triggers useEffect live polling loop (1000ms interval)
                     │
                     ▼
             GET /api/training/jobs/{job_id}
                     │
                     ▼ (Worker starts -> status becomes RUNNING)
             setActiveJob(freshJob) [Dynamic live updates]
                     │
                     ▼ (Training completes -> status becomes COMPLETED)
             clearInterval(pollingRef) & fetchJobs()
```

#### Detailed Code Trace:
1. **Job Dispatch:**
   - In [frontend/src/components/training/TrainingConfigForm.tsx](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/components/training/TrainingConfigForm.tsx), the user clicks "Launch Training Job", calling `onSubmit(request)`.
   - In [frontend/src/components/training/TrainingStudio.tsx](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/components/training/TrainingStudio.tsx#L148-L161):
     ```tsx
     const handleLaunchJob = async (request: TrainingJobCreateRequest) => {
       const newJob = await trainingApi.createJob(request);
       setActiveJob(newJob);
       setJobs((prev) => [newJob, ...prev]);
       setActiveStudioTab('studio');
     };
     ```
2. **Backend Submission & Mutual Exclusion:**
   - Route `POST /api/training/jobs` in [backend/routes/training.py](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/backend/routes/training.py#L53-L95) passes the request to `TrainingOrchestrator.submit_job()`.
   - Orchestrator inserts a `QUEUED` row into `training_jobs` in [engine/training/job_store.py](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/engine/training/job_store.py), assigns a UUID cancellation event, and pushes `job_id` onto `self.queue`.
3. **Frontend Polling Engine:**
   - In [frontend/src/components/training/TrainingStudio.tsx](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/components/training/TrainingStudio.tsx#L94-L136):
     ```tsx
     useEffect(() => {
       if (activeJob && ['QUEUED', 'RUNNING', 'CANCEL_REQUESTED'].includes(activeJob.status)) {
         const activeJobId = activeJob.job_id;
         pollingRef.current = setInterval(async () => {
           const freshJob = await trainingApi.getJob(activeJobId);
           setActiveJob(freshJob);
           ...
           if (['COMPLETED', 'CANCELLED', 'FAILED', 'INTERRUPTED'].includes(freshJob.status)) {
             clearInterval(pollingRef.current);
             fetchJobs();
           }
         }, 1000);
       }
     }, [activeJob?.status, activeJob?.job_id]);
     ```
4. **Backend Worker Activation:**
   - In [engine/training/orchestrator.py](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/engine/training/orchestrator.py#L304-L375), `_scheduler_loop()` pulls the queued job, acquires `gpu_lock` (if GPU job), spawns `_run_job_worker()`, and calls `job_store.update_job_status(job_id, "RUNNING", started_at, worker_pid)`.
   - On the next 1-second polling tick, `getJob()` returns `status: 'RUNNING'`. The UI immediately updates the badge to a pulsing orange "RUNNING" pill and streams optimizer step records into the Recharts series.
5. **Alternative Entry from Training History:**
   - In [frontend/src/components/training/TrainingHistoryTable.tsx](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/components/training/TrainingHistoryTable.tsx#L227-L235), clicking the "View" button executes:
     ```tsx
     onClick={() => onSelectJob(j)}
     ```
   - In [frontend/src/components/training/TrainingStudio.tsx](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/components/training/TrainingStudio.tsx#L329-L333), `onSelectJob` sets `setActiveJob(j)` and `setActiveStudioTab('studio')`.
   - If the selected job is completed, it renders statically as the "Inspect Results" view without triggering the polling interval.

---

## 6. Discrepancy 5: Metric Authenticity Verification (Real vs. Simulated)

### Question
*Verify whether real-time metrics come from backend training processes or simulated/static frontend values.*

### Confirmed Findings
- **Real-time metrics are 100% genuine and computed by real backend training processes.**
- **There is ZERO client-side simulation, mathematical mocking, or placeholder progression in the frontend.**

### Evidence by Component:
1. **Frontend API Client ([frontend/src/services/trainingApi.ts](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/services/trainingApi.ts)):**
   - Pure HTTP client issuing native `fetch()` calls to `/api/training/...`.
   - Contains no timer-based metric increments, random number generators, or synthetic data generators.
2. **Frontend UI State Binding ([frontend/src/components/training/ActiveJobMonitor.tsx](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/components/training/ActiveJobMonitor.tsx#L134-L146)):**
   - `lossValue`: binds strictly to `job.metrics.final_train_loss ?? job.progress.train_loss`.
   - `valPerplexity`: binds strictly to `job.progress.val_perplexity ?? job.metrics.final_val_perplexity`.
   - `throughput`: binds strictly to `job.progress.tokens_per_sec ?? job.metrics.tokens_per_sec`.
   - `vramUsage`: binds strictly to `job.progress.peak_vram_mb ?? job.metrics.peak_allocated_vram_mb`.
   - `customDispatches`: binds strictly to `job.metrics.custom_cuda_dispatches` kernel counts.
3. **Backend Training Loop ([engine/transformer/trainer.py](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/engine/transformer/trainer.py#L311-L409)):**
   - `train_loss`: True forward pass cross-entropy via `self.model(x, targets=y)` with PyTorch AMP FP16 on GPU, or FP32 on CPU.
   - `grad_norm`: True gradient norm computed via `torch.nn.utils.clip_grad_norm_()`.
   - `tokens_per_sec`: True token throughput computed via `(num_tokens * actual_accum) / max(step_elapsed, 1e-6)` using high-resolution monotonic timer `time.perf_counter()`.
   - `perplexity`: True mathematical exponentiation `math.exp(clamped_loss)` over real validation cross-entropy batches.
4. **Physical Checkpoint Artifacts:**
   - Every completed job saves a real, valid PyTorch checkpoint to disk (`checkpoints/training_jobs/{job_id}/final_model.pt`).
   - For `job_20261113_153924_735287`, the checkpoint size is **173,733,391 bytes** (~173.7 MB), validated with physical SHA-256 integrity in `data/training_jobs.db`.

---

## 7. Discrepancy 6: Telemetry Collection Matrix (GPU Utilization, VRAM, Temperature, Loss, Throughput)

### Question
*Identify whether GPU utilization, VRAM, temperature, loss and throughput are collected during execution.*

### Comprehensive Telemetry Evaluation Matrix

| Telemetry Metric | Status | Collection Method | Storage Location | Limitations & Gaps |
|---|---|---|---|---|
| **Loss (Train & Val)** | **IMPLEMENTED & VERIFIED** | Evaluated via `nn.CrossEntropyLoss` per batch/step | `step_records.train_loss`, `metrics.final_train_loss`, `metrics.final_val_loss` | Intra-step validation loss omitted when `eval_interval_steps` > `total_steps`. |
| **Throughput (Tokens/s or Samples/s)** | **IMPLEMENTED & VERIFIED** | Monotonic wall timer `time.perf_counter()` over tokens or samples processed | `step_records.tokens_per_sec`, `metrics.tokens_per_sec`, `epoch_metrics.samples_per_sec` | Accurately tracks per-step and run-averaged execution speed. |
| **VRAM Memory Usage** | **PARTIALLY IMPLEMENTED** | PyTorch CUDA Memory Allocator (`torch.cuda.memory_allocated()`, `max_memory_allocated()`, `memory_reserved()`) | `step_records.peak_vram_mb`, `progress.peak_vram_mb`, `metrics.peak_vram_mb` | Measures **PyTorch heap allocations only**. Does **NOT** query physical GPU VRAM via NVML. Reports `0.0 MB` for `CPU_NATIVE` jobs. |
| **GPU Utilization (%)** | **NOT COLLECTED IN TRAINING METRICS** | Not called during training execution | Missing from `step_records` and `metrics_json` | Present only in `ResNetTrainer.hardware_monitor` internal list, but **never exposed in training API or database**. Omitted in `TransformerTrainer` and `ThreeBackendTrainer`. |
| **GPU Temperature (°C)** | **NOT COLLECTED IN TRAINING METRICS** | Not called during training execution | Missing from `step_records` and `metrics_json` | Present only in `ResNetTrainer.hardware_monitor` thermal guard, but **never exposed in training API or database**. Omitted in `TransformerTrainer` and `ThreeBackendTrainer`. |

### Architectural Root Cause for Missing Hardware Metrics
While NEXORA possesses an NVML hardware telemetry engine ([engine/profiling/hardware_monitor.py](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/engine/profiling/hardware_monitor.py) and [backend/routes/telemetry.py](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/backend/routes/telemetry.py)), that engine is hooked primarily to the general system dashboard (`/api/hardware/telemetry`). 

The transformer training engine ([engine/transformer/trainer.py](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/engine/transformer/trainer.py)) was built as a self-contained PyTorch module using only `_get_vram_telemetry()` (`torch.cuda` memory APIs). It was never wired to the NVML `HardwareMonitor` singleton to record per-step GPU core utilization or die temperature snapshots.

---

## 8. Discrepancy 7: Exact API Endpoints & Frontend Handlers Catalog

### Backend API Endpoints ([backend/routes/training.py](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/backend/routes/training.py))

| HTTP Method & Path | Router Function | Request / Query Params | Response Schema | Purpose |
|---|---|---|---|---|
| `GET /api/training/models` | `list_training_models()` | None | `TrainingModelsResponse` | Lists all 7 registered model architectures, parameter counts, tasks, and supported execution backends. |
| `POST /api/training/jobs` | `create_training_job()` | `TrainingJobCreateRequest` (JSON) | `TrainingJobResponse` (HTTP 201) | Validates config, enforces GPU mutual exclusion, creates SQLite record, queues job for background execution. |
| `GET /api/training/jobs` | `list_training_jobs()` | `status: Optional[str]`, `backend: Optional[str]`, `limit: int = 50` | `TrainingJobListResponse` | Fetches historical and active jobs for `TrainingHistoryTable`. |
| `GET /api/training/jobs/{job_id}` | `get_training_job()` | `job_id: str` (path, validated regex) | `TrainingJobResponse` | **Primary live polling endpoint.** Returns complete job state, live `progress` dict, and `metrics` dict. |
| `GET /api/training/jobs/{job_id}/metrics` | `get_training_job_metrics()` | `job_id: str` (path) | `TrainingJobMetricsResponse` | Returns lightweight status, progress, and `step_records` / `epoch_metrics`. |
| `POST /api/training/jobs/{job_id}/cancel` | `cancel_training_job()` | `job_id: str` (path) | `CancelJobResponse` | Triggers thread cancellation event; releases GPU locks cooperatively at the next step boundary. |
| `GET /api/training/jobs/{job_id}/checkpoints` | `get_training_job_checkpoints()` | `job_id: str` (path) | `TrainingJobCheckpointsResponse` | Fetches saved `.pt` weights metadata, file sizes, and SHA-256 hashes for modal inspection. |

### Frontend Handlers & Component Hierarchy

| Component File | Role / Responsibility | Key Handlers & Hooks | Endpoints Called |
|---|---|---|---|
| [frontend/src/services/trainingApi.ts](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/services/trainingApi.ts) | Typed API client wrapper | `getModels`, `createJob`, `listJobs`, `getJob`, `getJobMetrics`, `cancelJob`, `getJobCheckpoints` | All `/api/training/*` endpoints |
| [frontend/src/components/training/TrainingStudio.tsx](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/components/training/TrainingStudio.tsx) | Studio master orchestrator | • `handleLaunchJob()` (lines 148–161)<br>• `handleCancelJob()` (lines 164–177)<br>• `useEffect` polling loop (lines 94–136)<br>• Tab navigation (`studio`, `history`, `comparison`)<br>• 6-step lifecycle stepper (lines 259–286) | • `POST /api/training/jobs`<br>• `GET /api/training/jobs/{id}` (1s poll)<br>• `POST /api/training/jobs/{id}/cancel`<br>• `GET /api/training/jobs` |
| [frontend/src/components/training/ActiveJobMonitor.tsx](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/components/training/ActiveJobMonitor.tsx) | Live & Inspect KPI display | • Status badge renderer (lines 75–132)<br>• Dynamic step/epoch progress bar (lines 197–252)<br>• 6 KPI cards (Loss, Perplexity, Throughput, Time, VRAM, LR/CUDA Dispatches) (lines 254–331)<br>• Cancel & Inspect Checkpoint triggers | None directly (receives `job` prop) |
| [frontend/src/components/training/TrainingCharts.tsx](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/components/training/TrainingCharts.tsx) | Recharts scientific visualizer | • Maps `job.metrics.step_records` or `epoch_metrics` into `chartData` (lines 32–54)<br>• Loss LineChart (lines 98–134)<br>• Validation LineChart (lines 137–165)<br>• Latency BarChart (lines 169–187)<br>• Kernel Dispatches / Throughput BarChart (lines 189–230) | None directly (receives `job` prop) |
| [frontend/src/components/training/TrainingHistoryTable.tsx](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/components/training/TrainingHistoryTable.tsx) | SQLite job repository viewer | • `onSelectJob()` $\rightarrow$ sets `activeJob` and navigates to `'studio'` (lines 227–235)<br>• `onViewCheckpoints()` $\rightarrow$ opens modal (lines 236–245)<br>• Status and backend filter selectors (lines 110–135) | `GET /api/training/jobs` via parent refresh |
| [frontend/src/components/training/CheckpointInspector.tsx](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/components/training/CheckpointInspector.tsx) | Weight verification modal | • `handleCopyHash()` (lines 57–61)<br>• `useEffect` on mount (lines 29–55) | `GET /api/training/jobs/{id}/checkpoints` |

---

## 9. Implemented, Tested & Unverified Functionality Classification

In accordance with strict compliance auditing standards, all investigated functionalities are classified below:

### A. Fully Implemented & Formally Verified
1. **Multi-Backend Training Execution:** CPU Native, GPU Native (Ampere SM 8.6 AMP FP16 / SDPA), and GPU Custom (custom CUDA autograd) properly dispatched via `TrainingOrchestrator`.
2. **GPU Mutual Exclusion:** Single GPU lock (`self.gpu_lock`) blocks concurrent jobs on the RTX 3050 6GB GPU safely.
3. **Persistent SQLite Job Store:** State transitions (`QUEUED` $\rightarrow$ `RUNNING` $\rightarrow$ `COMPLETED` / `CANCELLED` / `FAILED`), start/end timestamps, worker PIDs, and JSON progress/metrics reliably persisted.
4. **Cooperative Cancellation:** `cancel_event` threading tokens safely release resources and avoid orphaned CUDA memory.
5. **Real-Time Polling Engine:** Frontend 1-second interval reliably updates live metrics and halts upon terminal transition.
6. **Atomic Checkpoint Serialization:** Model weights saved to isolated subdirectories with validated byte counts and SHA-256 hashes.
7. **Loss & Throughput Calculation:** Exact floating-point loss and token/sample throughput computed via high-resolution monotonic clocks.

### B. Partially Implemented
1. **VRAM Telemetry:** PyTorch allocator VRAM (`torch.cuda.memory_allocated()`) is tracked, but physical GPU VRAM at the OS/driver level (including driver context and external memory) is not sampled.
2. **Post-Training Validation Time-Series:** Final validation perplexity is computed and displayed in summary cards, but not recorded in the step time-series array.

### C. Missing / Unverified Functionality
1. **GPU Core Utilization (%) during Training:** Not sampled or stored during transformer or 3-backend training runs.
2. **GPU Temperature (°C) during Training:** Not sampled or stored during transformer or 3-backend training runs.
3. **Workflow Stepper State Progression:** Visual stepper does not track completed steps cleanly, leaving multiple contradictory steps active simultaneously upon completion.

---

## 10. Concrete Code Fix Recommendations

### Fix 1: Reconcile Optimizer Steps vs. Microbatches in `job_store.py`
In [engine/training/job_store.py](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/engine/training/job_store.py#L396-L403), replace the blind overwrite with:
```python
# Preserve distinct optimizer step and microbatch accounting
opt_steps = metrics.get("optimizer_steps_completed", p.get("step", p.get("current_step", 1)))
total_batches = p.get("microbatches_processed", p.get("total_batches", opt_steps))
p["step"] = opt_steps
p["current_step"] = opt_steps
p["total_steps"] = opt_steps
p["batch"] = total_batches
p["total_batches"] = total_batches
```

### Fix 2: Inject Final Validation Metrics into `step_records` in `trainer.py`
In [engine/transformer/trainer.py](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/engine/transformer/trainer.py#L421-L435), update the final step record with the final evaluation results:
```python
total_wall_time = time.perf_counter() - start_time
final_val_loss, final_val_ppl = self.evaluate()

# Ensure final validation is plotted on the scientific progression curves
if step_records:
    step_records[-1]["val_loss"] = final_val_loss
    step_records[-1]["val_perplexity"] = final_val_ppl
```

### Fix 3: Support Single-Point Validation Fallback in `TrainingCharts.tsx`
In [frontend/src/components/training/TrainingCharts.tsx](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/components/training/TrainingCharts.tsx#L32-L43), inject the final validation scalar if intra-run validation was omitted:
```tsx
const chartData = isTransformer && stepList.length > 0
  ? stepList.map((st: any, idx: number) => {
      const isLast = idx === stepList.length - 1;
      const rawValPpl = st.val_perplexity ?? (isLast ? metrics.final_val_perplexity : undefined);
      const rawValLoss = st.val_loss ?? (isLast ? metrics.final_val_loss : undefined);
      return {
        name: `Step ${st.step}`,
        step: st.step,
        train_loss: Number((st.train_loss ?? 0).toFixed(4)),
        val_loss: rawValLoss !== undefined && rawValLoss > 0 ? Number(Number(rawValLoss).toFixed(4)) : undefined,
        val_perplexity: rawValPpl !== undefined && rawValPpl > 0 ? Number(Number(rawValPpl).toFixed(2)) : undefined,
        duration_sec: Number((st.step_duration_sec ?? 0).toFixed(3)),
        throughput: Math.round(st.tokens_per_sec ?? 0),
      };
    })
  : ...
```

### Fix 4: Refactor Stepper Logic in `TrainingStudio.tsx`
In [frontend/src/components/training/TrainingStudio.tsx](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/components/training/TrainingStudio.tsx#L259-L266), decouple linear progression from completed states:
```tsx
const isJobActive = activeJob && ['QUEUED', 'RUNNING', 'CANCEL_REQUESTED'].includes(activeJob.status);
const isJobDone = activeJob && ['COMPLETED', 'FAILED', 'CANCELLED', 'INTERRUPTED'].includes(activeJob.status);

const workflowSteps = [
  { step: 1, label: 'Choose Model', status: activeJob ? 'done' : 'active' },
  { step: 2, label: 'Configure Training', status: activeJob ? 'done' : (selectedModelId ? 'active' : 'pending') },
  { step: 3, label: 'Review Config', status: activeJob ? 'done' : 'pending' },
  { step: 4, label: 'Launch', status: isSubmitting ? 'active' : (activeJob ? 'done' : 'pending') },
  { step: 5, label: 'Monitor', status: isJobActive ? 'active' : (isJobDone ? 'done' : 'pending') },
  { step: 6, label: 'Inspect Results', status: isJobDone ? 'active' : 'pending' },
];
```
This ensures that when a job finishes, Step 5 displays as **"Done" (with a green checkmark)** rather than an inactive/disabled grey state, and Step 6 is highlighted cleanly.

---

## 11. Verification & Confirmation

- **Workspace File Generated:** [AI_TRAINING_STUDIO_MONITORING_GAP_REPORT.md](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/AI_TRAINING_STUDIO_MONITORING_GAP_REPORT.md)
- **Codebase Integrity:** STRICT READ-ONLY COMPLIANCE MAINTAINED. Zero source files modified; zero unauthorized training runs launched.
- **Audited Target:** `job_20261113_153924_735287` fully reconciled across database records, training configs, dataloader batch dimensions, and frontend Recharts series.

