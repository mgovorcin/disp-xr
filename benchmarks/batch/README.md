# Benchmarks on a dedicated AWS Batch instance

The shared node is often too busy to time anything (see the notes in
`../README.md`). This runs `bench_downsample.py` on a whole `r7i.8xlarge`
(job definition: 32 vCPUs, 240,000 MiB; queue
`opera-adt-ondemand-1500GB-32v256mem-ext`). `run.py` records the instance
type, CPU, memory and page-fault speed (`BENCH_INFO`) before the benchmark,
and the page-fault speed and load after it (`BENCH_DONE`), so a run on a
noisy instance shows.

```bash
# From the repository root, with AWS_PROFILE=saml-pub and AWS_REGION=us-west-2.
pip wheel --no-deps -w benchmarks/batch .            # the disp-xr wheel
cp benchmarks/bench_downsample.py benchmarks/batch/
# constraints.txt: the versions of the core dependencies to pin (pip freeze).
docker build --network=host -t disp-xr-bench:TAG benchmarks/batch
# push to ECR (repository disp-xr-bench), register job_def.json with the
# image tag, submit to the queue above, and read the CloudWatch log.
```
