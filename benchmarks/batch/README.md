# Benchmarks on a dedicated AWS Batch instance

The shared node is often too busy to time anything (see the notes in
`../README.md`). This runs a benchmark script on a whole `r7i.8xlarge`
(job definition: 32 vCPUs, 240,000 MiB, on an on-demand queue of
`r7i.8xlarge` instances). `entrypoint.py` records the instance
type, CPU, memory and page-fault speed (`BENCH_INFO`) before the benchmark,
and the page-fault speed and load after it (`BENCH_DONE`), so a run on a
noisy instance shows. The job command names the script and its arguments
(`bench_downsample.py` when empty); a CSV the script writes to
`/tmp/out.csv` is printed between `BENCH_CSV_BEGIN` and `BENCH_CSV_END`, so
the raw rows are in the CloudWatch log.

```bash
# From the repository root, with AWS credentials for your account and region.
pip wheel --no-deps -w benchmarks/batch .            # the disp-xr wheel
cp benchmarks/*.py benchmarks/batch/                 # the benchmark scripts
# constraints.txt: the versions of the core dependencies to pin (pip freeze).
docker build --network=host -t disp-xr-bench:TAG benchmarks/batch
# push to ECR (repository disp-xr-bench), register job_def.json with the
# image tag, submit to that queue with the command, for example
#   ["suite.py", "--size", "tile"]
#   ["storage_s3.py", "s3://.../granules/", "s3://.../work/", "--out", "/tmp/out.csv"]
# and read the CloudWatch log.
```

`job_def.json` has placeholders (`<ACCOUNT_ID>`, `<REGION>`, `<TAG>`, `<JOB_ROLE>`,
`<EXECUTION_ROLE>`, `<OWNER>`): fill them in for your account before
registering it.

`storage_s3.py` reads the job role's credentials and needs read and write
access to the scratch prefix; it deletes its `work/` prefix at the end.
