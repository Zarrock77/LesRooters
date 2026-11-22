# HOW TO START — Jülich (JURECA)

Guide to get set up on the cluster, to follow in order.

- Project: **`training2666`**
- Reservation: **`cispahack`**
- Partition: **`dc-gpu`**

> ⚠️ **Bug in the official `teammate.sh` script**: it contains `PROJECT_NAME="training2625"` instead of `training2666`. The commands in step 4 fix it automatically.

---

## Step 0 — Agree as a team

1. Appoint **a team lead**. The lead runs the big setup script, only once.
2. Choose a **team folder name** (`TEAM_FOLDER`): no spaces or `/`, e.g. `epiteam`.
3. Everyone sends their **Jülich username** to the lead. It is shown in JuDoor once logged in.

---

## Step 1 — JuDoor (EVERYONE, in the browser)

On https://judoor.fz-juelich.de/login:

1. **Register**, then confirm the email you receive.
2. **Enable MFA**: "here" link in the yellow banner, to scan with a TOTP app (Google Authenticator, FreeOTP…).
3. **Join a project** → `training2666`.
4. In *Systems* → JURECA: **sign the usage agreement**.
5. **SSH key**: JURECA → *Manage SSH-keys*.
   - If you do not have a key yet: `ssh-keygen -t ed25519 -C "ton_email"`
   - Paste the content of `~/.ssh/id_ed25519.pub` into *Your public key*.
   - Copy the `from="..."` line offered by the page into *Additional public key options*.
   - *Start upload of SSH-Keys*, then **wait up to 15 min**.

> The `from="..."` line restricts the key to the network you connect from. If you change network (venue wifi ↔ 4G), you will need to add the new IP range in JuDoor.

---

## Step 2 — SSH config (EVERYONE, on your PC)

Add this block to `~/.ssh/config`:

```
Host juelich
  Hostname jureca.fz-juelich.de
  User <ton_user_juelich>
  IdentityFile ~/.ssh/id_ed25519
  MACs hmac-sha2-256-etm@openssh.com
```

Then connect with `ssh juelich`: it asks for the TOTP code from the MFA app.

For VS Code: *Remote - SSH* extension → *Connect to Host* → `juelich`.

---

## Step 3 — TEAM LEAD setup (only once)

On JURECA, from the lead's account:

```bash
mkdir -p ~/hackathon-scripts && cd ~/hackathon-scripts
wget -q https://huggingface.co/datasets/SprintML/hackathon/resolve/main/hackathon_setup.sh -O hackathon_setup.sh
nano hackathon_setup.sh
```

Fill in the `CONFIGURATION` section at the top of the file:

```bash
TEAM_LEAD="<user_du_chef>"      # must be EXACTLY your user (check with: id -un)
TEAMMATE_1="<user_coequipier_1>"
TEAMMATE_2="<user_coequipier_2>"
TEAMMATE_3=""                   # leave empty if not used
TEAM_FOLDER="<nom_equipe>"
```

Then:

```bash
source hackathon_setup.sh       # really "source", not "bash"
```

The script creates the shared folder, sets the permissions, downloads the 3 datasets and creates one `.venv` per task. **Wait until it finishes without errors**, then tell the team on Discord.

Result: `/p/scratch/training2666/<user_du_chef>/<nom_equipe>/`, which contains `unlearning/`, `model_equality/` and `Hackathon_Watermark_Localisation_Task/`.

**If a teammate was left out**, the lead runs (no need to rerun everything):

```bash
NEW_MEMBER="<user_oublie>"
TEAM_LEAD_DIR="/p/scratch/training2666/$(id -un)"
setfacl -R -m "u:$NEW_MEMBER:rwx" "$TEAM_LEAD_DIR"
find "$TEAM_LEAD_DIR" -type d -exec setfacl -d -m "u:$NEW_MEMBER:rwx" {} +
```

---

## Step 4 — TEAMMATE setup (after the lead)

On JURECA, each from **their own account**. Replace the two `<...>` values before running:

```bash
mkdir -p ~/hackathon-scripts && cd ~/hackathon-scripts
wget -q https://huggingface.co/datasets/SprintML/hackathon/resolve/main/teammate.sh -O teammate.sh

# bug fix + config
sed -i 's/^PROJECT_NAME="training2625"/PROJECT_NAME="training2666"/' teammate.sh
sed -i 's/^TEAM_LEAD=""/TEAM_LEAD="<user_du_chef>"/; s/^TEAM_FOLDER=""/TEAM_FOLDER="<nom_equipe>"/' teammate.sh

# check: must print training2666, the lead's user and the team name
grep -E '^(PROJECT_NAME|TEAM_LEAD|TEAM_FOLDER)=' teammate.sh

source teammate.sh
```

The script must end with **"Teammate shell setup complete!"**.

- *"Shared team folder does not exist"*: the lead has not finished, or there is a typo in `TEAM_LEAD` / `TEAM_FOLDER`.
- *"You do not have read/write access"*: the lead left you out of `TEAMMATE_x` → see the end of step 3.

---

## On every login

```bash
jutil env activate -p training2666
source ~/.bashrc                 # only if the cache variables are missing
cd /p/scratch/training2666/<user_du_chef>/<nom_equipe>/<tache>
source .venv/bin/activate
```

**Never rerun the setup scripts.**

---

## Launching a GPU job

**Limit announced by the organizers: a single node (4 GPUs) at a time for the whole LesRooters team**, per Melvyn's correction. This rule replaces the earlier per-person/per-account interpretation. Check the jobs of all teammates, on all partitions, before each launch: the known accounts are `gras1`, `delegue1` and `korkmaz1`, plus the fourth member's account still to be added. Pending jobs must also be coordinated to avoid a second allocation. Slurm dependencies limited to your own account are not enough to respect the collective limit. Parallel computations must share the GPUs of the same node. Repeated violations can lead to loss of GPU access.

Login nodes are for coding. Computations go through Slurm. Create `main.sh` in the task folder:

```bash
#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1              # up to 4 (A100 40 GB)
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err

cd /p/scratch/training2666/<user_du_chef>/<nom_equipe>/<tache> || exit 1
source .venv/bin/activate
srun python code.py
```

Then:

```bash
mkdir -p output          # required before the first sbatch
sbatch main.sh
squeue -u "$USER"        # see your jobs
tail -f output/<jobid>.out
scancel <jobid>          # kill a job
```

---

## Shared folder rules

- Do not delete the shared datasets or `.venv`s.
- Do not install or update packages in a shared `.venv` without telling the others.
- Results and logs go in `<tache>/output/`.
- Code goes in **this GitHub repo**, one branch per task. No weights or datasets in git: the `.gitignore` already excludes them.
