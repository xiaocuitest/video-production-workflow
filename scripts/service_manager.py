#!/usr/bin/env python3
"""Start the local studio services as detached, user-owned processes."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import signal
from pathlib import Path


PROJECT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = PROJECT / ".runtime"
PINNED_HYPERFRAMES_VERSION = "0.8.10"
SERVICES = {
    "app": {
        "port": 5088,
        "command": ["bash", str(PROJECT / "scripts" / "run_app.sh")],
        "url": "http://127.0.0.1:5088/",
        "health": "http://127.0.0.1:5088/health",
    },
    "preview": {
        "port": 3002,
        "command": [],
        "url": "http://127.0.0.1:3002/#project/video-production-workflow",
        "health": "http://127.0.0.1:3002/",
    },
}


def hyperframes_command() -> list[str]:
    """Prefer the already-installed pinned CLI; use npx only as a fallback."""
    local_binary = PROJECT / "node_modules" / ".bin" / "hyperframes"
    if local_binary.exists():
        return [str(local_binary)]
    npx_root = Path.home() / ".npm" / "_npx"
    candidates: list[Path] = []
    for package_path in npx_root.glob("*/node_modules/hyperframes/package.json"):
        try:
            package = json.loads(package_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        binary = package_path.parent.parent / ".bin" / "hyperframes"
        if package.get("version") == PINNED_HYPERFRAMES_VERSION and binary.exists():
            candidates.append(binary)
    if candidates:
        return [str(max(candidates, key=lambda item: item.stat().st_mtime))]
    return ["npx", "--yes", f"hyperframes@{PINNED_HYPERFRAMES_VERSION}"]


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


def process_listening(port: int) -> bool:
    """Process-level readiness fallback used only after a fresh launch."""
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def http_healthy(port: int, path: str) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2.5) as connection:
            connection.settimeout(2.5)
            request = f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
            connection.sendall(request.encode("ascii"))
            return connection.recv(32).startswith(b"HTTP/")
    except OSError:
        return False


def stop_managed_service(name: str) -> bool:
    pid_path = RUNTIME_DIR / f"{name}.pid"
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
        return True
    except OSError:
        return False
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pid_path.unlink(missing_ok=True)
            return True
        time.sleep(0.1)
    return True


def stop_preview_listeners() -> None:
    """Stop only processes that currently listen on the Studio's fixed port."""
    def listener_pids() -> list[int]:
        try:
            result = subprocess.run(
                ["lsof", "-nP", "-t", "-iTCP:3002", "-sTCP:LISTEN"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        found: list[int] = []
        for raw_pid in result.stdout.splitlines():
            try:
                pid = int(raw_pid.strip())
            except ValueError:
                continue
            if pid > 1:
                found.append(pid)
        return found

    for pid in listener_pids():
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    for _ in range(20):
        if not listener_pids():
            return
        time.sleep(0.1)
    # A frozen Studio can ignore SIGTERM. At this point the targets are still
    # the exact listeners on the dedicated preview port, before any new server
    # is launched, so force-stop only those remaining processes.
    for pid in listener_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass


def prepare_runtime_dir() -> None:
    RUNTIME_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(RUNTIME_DIR, 0o700)


def start_service(name: str) -> tuple[bool, str]:
    spec = SERVICES[name]
    port = int(spec["port"])
    health_path = "/health" if name == "app" else "/"
    if http_healthy(port, health_path):
        return True, f"{name} 已在运行"
    if name == "preview":
        # HyperFrames owns a detached per-project daemon whose PID can differ
        # from the short-lived npm wrapper. Stop it through the owning CLI so a
        # stalled :3002 listener cannot force the next preview onto 3003+.
        try:
            subprocess.run(
                [*hyperframes_command(), "preview", str(PROJECT), "--stop"],
                cwd=PROJECT,
                env={**os.environ, "PATH": f"{PROJECT / 'bin'}:/opt/homebrew/bin:/usr/local/bin:{os.environ.get('PATH', '')}"},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8,
                check=False,
            )
        except subprocess.TimeoutExpired:
            pass
        stop_preview_listeners()
        for _ in range(30):
            if not port_open(port):
                break
            time.sleep(0.1)
    if port_open(port):
        if not stop_managed_service(name):
            return False, f"{name} 端口被无响应进程占用，且不是本工具管理的进程"
        for _ in range(20):
            if not port_open(port):
                break
            time.sleep(0.1)

    log_path = RUNTIME_DIR / f"{name}.log"
    pid_path = RUNTIME_DIR / f"{name}.pid"
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    log_handle = os.fdopen(descriptor, "a", encoding="utf-8")
    try:
        command = (
            [*hyperframes_command(), "preview", str(PROJECT), "--port", "3002"]
            if name == "preview" else list(spec["command"])
        )
        process = subprocess.Popen(
            command,
            cwd=PROJECT,
            env={**os.environ, "PATH": f"{PROJECT / 'bin'}:/opt/homebrew/bin:/usr/local/bin:{os.environ.get('PATH', '')}"},
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
        os.chmod(pid_path, 0o600)
    finally:
        log_handle.close()

    for _ in range(120):
        if http_healthy(port, health_path) or (name == "preview" and process_listening(port)):
            return True, f"{name} 启动成功"
        if process.poll() is not None:
            return False, f"{name} 启动失败，请查看 {log_path}"
        time.sleep(0.25)
    return False, f"{name} 启动超时，请查看 {log_path}"


def main() -> int:
    prepare_runtime_dir()
    failures: list[str] = []
    requested = tuple(sys.argv[1:]) or ("app", "preview")
    invalid = [name for name in requested if name not in SERVICES]
    if invalid:
        print(f"未知服务：{', '.join(invalid)}", file=sys.stderr)
        return 2
    for name in requested:
        ok, message = start_service(name)
        print(message)
        if not ok:
            failures.append(name)
    if failures:
        return 1
    print("操作台：http://127.0.0.1:5088/")
    print("视频预览：http://127.0.0.1:3002/#project/video-production-workflow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
