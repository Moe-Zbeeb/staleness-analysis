# Live backup repair, September 28

The k256 learner continued training, but its backup worker repeatedly aborted with missing-file errors and `Backup input was replaced or truncated`. Its checkpoint at step 50 had already been verified on NFS.

The old generic directory scan traversed transient historical policy exports and broadcasts, including files concurrently removed by normal cleanup. It also checked a live file by its pathname after copying; atomically replaced status files therefore invalidated otherwise coherent copies. A failed poll prevented publication of the final inventory and delayed metric mirroring. Live inspection confirmed that backup PID 2687936 was holding the transfer lock while copying `historical-exports/step_72/weights/model-00002-of-00004.safetensors` into an NFS temporary file. Those transient weights were being redundantly archived outside the validated checkpoint path.

The backup now prunes transient policy exports, broadcasts, and checkpoint trees before traversal. Recovery checkpoints still use their separate strict validation/publication path, including required queued rollouts and pending historical policies. Durable rollout archives and scientific metrics remain included. Live files are copied from a pinned open descriptor and checksum-verified before atomic publication. Atomic pathname replacement is supported for these live snapshots, while in-place truncation and modification remain errors. Checkpoint files retain strict pathname identity checks. Cache identities now include the source inode. Failures include a traceback and source paths.

## Applying to the running allocation

Only the k256 backup helper for job 2145464 is changed. The frozen training release, model, GRPO settings, policy clock, trainer process, and k0 job are untouched. Patched helper scripts live outside the frozen release under the node-local workspace's `backup-fix-20260928` directory. Staging compares source and destination SHA256 hashes.

The original supervisor treats any backup child exit as a reason to stop training. The handoff therefore acquires the existing transfer lock, validates the original backup PID, user, process start time and command arguments, and pauses that backup process while it is outside a transfer. A replacement worker uses the same arguments, lock, destinations, status and shutdown signal file. The original backup PID remains alive for its supervisor.

A separate guardian observes the handoff controller through a pipe. If the controller exits or crashes, it stops the replacement and resumes the original worker. On normal training shutdown, the replacement completes its final verified copy and the original worker resumes to complete its supervised shutdown. Unexpected replacement exit also restores the original helper. The handoff is a temporary adapter for this already-running job; future launches use the corrected `local_backup.py` directly.

The deployment uses a CPU-only overlapping Slurm step within the existing allocation. It allocates no additional GPU and does not modify the cluster NFS service or other users' files. Previously published backups and archived copies of transient exports are preserved.

## Validation

Eleven focused storage tests passed locally, covering atomic replacement, truncation, checksum failures, checkpoint publication and retention, run ownership, and exclusion of transient trees. Two Linux integration tests passed in the existing cluster allocation: normal finalization and recovery after forcibly killing the handoff controller. Tests used isolated temporary fixtures, not training outputs.

Cluster deployment: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/hotfixes/backup-20260928`.

## Verified live result

At 17:02:39 UTC the replacement had completed a verified cycle after the handoff. NFS and XFS both held the same checksum-verified update journal through update 73; training had advanced to update 74 with its original PID 2687937. The inventories contained 981 NFS files and 92 XFS files, excluded redundant transient policy exports, and preserved verified recovery checkpoints 25 and 50. The one-update journal lag reflects asynchronous copying during ongoing training. See [the live verification receipt](verified.json). No training restart or optimizer-state change occurred.
