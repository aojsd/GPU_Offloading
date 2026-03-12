I am a PhD student working on computer architecture and systems, with emphasis on memory management and data movement/paging policies. For any topics related to CS research in all subfields, prioritize accurate explanations and data from state-of-the-art industry and academic research. When referencing any approaches or important results, ensure you can provide the source paper/article/etc.

When doing implementation work:
DO NOT commit or push using Git unless explicitly asked to do so.
NEVER use slurm commands (srun, sbatch, scancel, etc.) unless explicitly asked to do so.
On non-GH200/GB200 systems (e.g., H100 cluster), ALL GPU work MUST be executed
inside the Apptainer container via `H100_env/vllm_apptainer.sh`. This includes
running tests, benchmarks, profiling, trace collection, and any script that
imports PyTorch or touches the GPU. Do not run GPU commands directly on the host.
See `H100_env/vllm_apptainer.sh` for usage.
