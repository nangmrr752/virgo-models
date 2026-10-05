"""Virgo's llama.cpp switcher: one GPU, two Gemma 4 models, one loaded at a time.

  Angkor-2.0 (Gemma 4 26B A4B) runs normally: chat and live voice.
  Bayon-2.0  (Gemma 4 31B) loads when a request asks for it ("model": "bayon"); after BAYON_IDLE seconds
             without Bayon questions (default 600), Angkor-2.0 comes back so voice is fast again.

It listens where llama-server did (port 8080) and passes requests to a llama-server it starts on 8081.
A switch waits for the answers in progress to finish, then stops one model and starts the other
(about 30-90 s). Settings (docker-compose.yml): ANGKOR_MODEL / ANGKOR_ARGS, BAYON_MODEL / BAYON_ARGS
(no BAYON_MODEL: Angkor only), BAYON_IDLE.
"""
import http.client
import http.server
import json
import os
import shlex
import subprocess
import threading
import time
import urllib.request

INNER = 8081
SERVER = os.environ.get("LLAMA_SERVER", "/app/llama-server")
MODELS = {"angkor": (os.environ.get("ANGKOR_MODEL", ""), os.environ.get("ANGKOR_ARGS", ""))}
if os.environ.get("BAYON_MODEL", "").strip():
    MODELS["bayon"] = (os.environ["BAYON_MODEL"].strip(), os.environ.get("BAYON_ARGS", ""))
IDLE = int(os.environ.get("BAYON_IDLE", "600") or 600)

state = {"name": None, "proc": None, "active": 0, "last": time.time()}
cond = threading.Condition()


def log(*parts):
    print("[router]", *parts, flush=True)


def healthy():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{INNER}/health", timeout=3) as res:
            return res.status == 200
    except Exception:
        return False


def stop():
    proc = state["proc"]
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    state["proc"], state["name"] = None, None


def start(name):
    """Stops the running model and starts `name`; returns when it answers (or raises)."""
    stop()
    model, extra = MODELS[name]
    args = [SERVER, "-hf", model, *shlex.split(extra), "--host", "127.0.0.1", "--port", str(INNER)]
    log(f"loading {name}: {model}")
    began = time.time()
    state["proc"], state["name"] = subprocess.Popen(args), name
    while not healthy():
        if state["proc"].poll() is not None:
            state["proc"], state["name"] = None, None
            raise RuntimeError(f"{name} ({model}) didn't start: see the lines above")
        time.sleep(1)
    log(f"{name} ready in {time.time() - began:.0f} s")


def acquire(name):
    """Waits until `name` is the loaded model (switching when nothing else is answering), then holds it."""
    with cond:
        while state["name"] != name or not state["proc"] or state["proc"].poll() is not None:
            if state["active"] == 0:
                start(name)
                break
            cond.wait()
        state["active"] += 1
        state["last"] = time.time()


def release():
    with cond:
        state["active"] -= 1
        state["last"] = time.time()
        cond.notify_all()


def back_to_angkor():
    """Bayon unused for a while: load Angkor-2.0 again (fast chat and voice)."""
    while True:
        time.sleep(30)
        with cond:
            if state["name"] == "bayon" and state["active"] == 0 and time.time() - state["last"] > IDLE:
                log(f"Bayon idle for {IDLE} s: back to Angkor")
                try:
                    start("angkor")
                except Exception as err:
                    log("couldn't load Angkor:", err)


def which(body):
    try:
        asked = str(json.loads(body or b"{}").get("model") or "").lower()
    except ValueError:
        asked = ""
    return "bayon" if "bayon" in asked and "bayon" in MODELS else "angkor"


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # one request per connection, the body ends when it closes: streams pass through

    def log_message(self, *args):
        pass

    def _forward(self, method, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", INNER, timeout=3600)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection")}
        if body is not None:
            headers["Content-Length"] = str(len(body))
        conn.request(method, self.path, body=body, headers=headers)
        res = conn.getresponse()
        self.send_response(res.status)
        for k, v in res.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "content-length", "keep-alive"):
                self.send_header(k, v)
        self.end_headers()
        while True:
            chunk = res.read1(65536)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()
        conn.close()

    def do_GET(self):
        if self.path.startswith("/health") and not healthy():
            self.send_response(503)
            self.end_headers()
            self.wfile.write(json.dumps({"status": "loading", "model": state["name"]}).encode())
            return
        if self.path.startswith("/router"):
            body = json.dumps({"loaded": state["name"], "models": list(MODELS), "active": state["active"]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        try:
            self._forward("GET")
        except Exception as err:
            self.send_error(502, str(err)[:200])

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        name = which(body)
        try:
            acquire(name)
        except Exception as err:
            self.send_error(503, str(err)[:200])
            return
        try:
            self._forward("POST", body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the caller went away
        except Exception as err:
            try:
                self.send_error(502, str(err)[:200])
            except Exception:
                pass
        finally:
            release()


def main():
    if not MODELS["angkor"][0]:
        raise SystemExit("ANGKOR_MODEL isn't set")
    log("models:", ", ".join(f"{k}={v[0]}" for k, v in MODELS.items()), f"(Bayon back to Angkor after {IDLE} s idle)")
    with cond:
        start("angkor")
    threading.Thread(target=back_to_angkor, daemon=True).start()
    server = http.server.ThreadingHTTPServer(("0.0.0.0", 8080), Handler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
