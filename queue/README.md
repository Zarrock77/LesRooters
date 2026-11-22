# Team GPU queue (JURECA)

One Slurm job for the whole team (organizer rule). That job is `worker.sh`: it runs the lines of `work.txt` in order,
each on the next free GPU of the node (4 at a time), and releases the node after 30 min without work.

On JURECA: `/p/scratch/training2666/gras1/LesRooters/queue/`
- add a task: `bash /p/scratch/training2666/gras1/LesRooters/queue/add.sh "cd /p/.../unlearning/jj && ../.venv/bin/python unlearn.py ..."`
  (lines start in `queue/`, so `cd` first; quote globs). If no team job is running, `add.sh` submits the node.
- see the state: `bash /p/scratch/training2666/gras1/LesRooters/queue/status.sh`
- output of line N: `/p/project1/training2666/lesrooters/queue_logs/line_N.out` (on project1: scratch is at the project-wide inode cap); exit codes: `queue/done.txt`
- special lines: `wait` (let running lines finish first), `exit` (stop the node).
- `keepalive.sh` keeps the node from releasing: it sits as the LAST work.txt line (`bash .../queue/keepalive.sh <its line no.>`), holds a slot only while nothing else waits, and re-appends itself when real lines are added. Do not add an `exit` line.
- never `sbatch` anything else while the worker exists: the cispahack reservation is magnetic, a second job counts.
