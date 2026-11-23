"""Tiny remote-control script executed INSIDE the Colab VM through `colab exec -s <session> -f colab_ctl.py --env K=V`.

CTL_ACTION=setup                       clone / fast-forward the repo and install extra packages
CTL_ACTION=launch CTL_NAME=x CTL_CMD=..   run CTL_CMD detached from /content/repo/paper_experiments, log in /content/logs/x.log
CTL_ACTION=tail  CTL_NAME=x [CTL_N=40]    last lines of the log + liveness + GPU + disk
CTL_ACTION=cat   CTL_PATH=...             print a (small) file
CTL_ACTION=ls                             list results and logs
Nothing here blocks: long jobs are always detached so a dropped kernel connection cannot kill them.
"""
import os
import subprocess

A = os.environ.get("CTL_ACTION", "tail")
NAME = os.environ.get("CTL_NAME", "job")
REPO = "/content/repo"
BR = "claude/simple-vfl-paper-publication-qx73nw"
os.makedirs("/content/logs", exist_ok=True)
os.makedirs("/content/results", exist_ok=True)


def sh(c, **k):
    return subprocess.run(c, shell=True, capture_output=True, text=True, **k)


if A == "setup":
    if os.path.isdir(REPO + "/.git"):
        r = sh(f"cd {REPO} && git fetch -q origin {BR} && git reset -q --hard origin/{BR} && git log --oneline -1")
    else:
        r = sh(f"git clone -q --depth 1 --branch {BR} https://github.com/xavierxav/Simple-VFL.git {REPO} && cd {REPO} && git log --oneline -1")
    print(r.stdout, r.stderr[-500:])
    print(sh("pip install -q timm scipy tqdm 2>&1 | tail -2").stdout)
elif A == "launch":
    cmd = os.environ["CTL_CMD"]
    full = f"cd {REPO}/paper_experiments && PE_OUT=/content/results nohup bash -c {cmd!r} > /content/logs/{NAME}.log 2>&1 &"
    subprocess.Popen(full, shell=True, start_new_session=True)
    print("launched", NAME, ":", cmd)
elif A == "tail":
    n = os.environ.get("CTL_N", "40")
    print(sh(f"tail -n {n} /content/logs/{NAME}.log").stdout)
    print("alive:", sh(f"pgrep -af 'bash -c' | grep -v pgrep | grep -c ."). stdout.strip(), "| python procs:", sh("pgrep -af 'python -u|python s' | grep -v pgrep | cut -c1-120").stdout.strip())
    print(sh("nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader; df -h /content | tail -1; free -g | sed -n 2p").stdout)
elif A == "cat":
    print(open(os.environ["CTL_PATH"]).read()[-20000:])
elif A == "b64":  # results dir (small files only) as a base64 tarball between markers
    import base64, io, tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for f in os.listdir("/content/results"):
            fp = os.path.join("/content/results", f)
            if os.path.isfile(fp) and os.path.getsize(fp) < 5_000_000:
                t.add(fp, arcname=f)
    print("B64START" + base64.b64encode(buf.getvalue()).decode() + "B64END")
elif A == "rclone":  # the VPS's Drive remote (gdrive:, scope drive.file, rooted at claude-projects) on this runtime:
    # durable Drive access without Colab's per-runtime consent; the config arrives base64-encoded and is never printed
    import base64
    os.makedirs("/root/.config/rclone", exist_ok=True)
    p = "/root/.config/rclone/rclone.conf"
    with open(p, "w") as f:
        f.write(base64.b64decode(os.environ["CTL_RCLONE_B64"]).decode())
    os.chmod(p, 0o600)
    if sh("which rclone").returncode:
        r = sh("curl -fsSL https://rclone.org/install.sh | bash > /tmp/rclone_install.log 2>&1; rclone version | head -1")
        print(r.stdout.strip(), r.stderr[-300:])
    r = sh("rclone lsd gdrive: 2>&1 | head -5")
    print("rclone gdrive:", "ok" if r.returncode == 0 else "FAILED", "|", r.stdout.strip()[-400:])
elif A == "ping":
    print("pong")
elif A == "ls":
    print(sh("ls -la /content/results /content/logs /content/cache/vb 2>&1 | head -60").stdout)
