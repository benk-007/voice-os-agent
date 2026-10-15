import os, re, json, ast, subprocess, webbrowser, shlex, shutil, zipfile, difflib, time
from pathlib import Path
from typing import Optional, Tuple
import torch
import threading
import queue
from datetime import datetime

# ---------- CONFIG ----------
MODEL_DIR = Path("./t5-semantic-parser-final")
if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Model directory not found at {MODEL_DIR}. Set MODEL_DIR to your trained model path.")

PREFIX = "semantic_parse: "
SAFE_BASE_DIR = os.path.expanduser(r"C:\Users\Public\safe_workspace") if os.name == "nt" else os.path.expanduser("~/safe_workspace")
os.makedirs(SAFE_BASE_DIR, exist_ok=True)
CONF_THRESHOLD = 0.85

APP_PATHS = {
    "firefox": r"C:\Program Files\Mozilla Firefox\firefox.exe",
    "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "edge": r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "vscode": r"C:\Users\%USERNAME%\AppData\Local\Programs\Microsoft VS Code\Code.exe",
    "notepad": r"C:\Windows\System32\notepad.exe"
}
ALLOWED_APPS = {k.lower() for k in APP_PATHS.keys()} | {"terminal","calculator","spotify"}

# ---------- LOAD MODEL & TOKENIZER ----------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR)).to("cuda" if torch.cuda.is_available() else "cpu")

# ---------- JSON extraction helpers ----------
def normalize_quotes_and_commas(s: str):
    s = s.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s.strip()

def looks_like_kv_list(s: str):
    pattern = r'^\s*(?:["\']?[\w\-\_ ]+["\']?\s*:\s*["\'][^"\']*["\']\s*)(?:,\s*(?:["\']?[\w\-\_ ]+["\']?\s*:\s*["\'][^"\']*["\']\s*))*$'
    return bool(re.match(pattern, s.strip()))

def ensure_json_like(s: str):
    s = s.strip()
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m: return normalize_quotes_and_commas(m.group(0))
    if looks_like_kv_list(s): return normalize_quotes_and_commas("{" + s + "}")
    m2 = re.match(r'^\s*["\'].*["\']\s*:\s*["\'].*', s)
    if m2: return normalize_quotes_and_commas("{" + s + "}")
    return normalize_quotes_and_commas(s)

def parse_json_from_text(s: str):
    cand = ensure_json_like(s)
    try:
        return json.loads(cand)
    except Exception:
        try:
            return json.loads(cand.replace("'", '"'))
        except Exception:
            try:
                return ast.literal_eval(cand)
            except Exception:
                return None

# ---------- Generation + parse_command ----------
def generate_with_confidence(input_text, max_new_tokens=64, num_beams=4, do_sample=False):
    input_ids = tokenizer((PREFIX + input_text), return_tensors="pt").input_ids.to(model.device)
    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
        return_dict_in_generate=True,
        output_scores=True
    )
    sequences = out.sequences
    scores = getattr(out, "scores", None)
    confidence = None
    if scores:
        try:
            import math
            token_logprobs = []
            for step_scores in scores:
                step_logprob = torch.log_softmax(step_scores, dim=-1).max().item()
                token_logprobs.append(step_logprob)
            avg_logprob = sum(token_logprobs) / max(1, len(token_logprobs))
            confidence = float(math.exp(avg_logprob))
            if confidence > 1.0: confidence = 1.0
        except Exception:
            confidence = None
    decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    return decoded, confidence

def parse_command(utterance, max_new_tokens=64, num_beams=4, do_sample=False):
    raw_texts, conf = generate_with_confidence(utterance, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=do_sample)
    outputs = []
    for raw in raw_texts:
        parsed = parse_json_from_text(raw)
        outputs.append({"raw": raw.strip(), "parsed": parsed, "confidence": conf})
    return outputs

# ---------- Validation helpers ----------
def normalize_parsed(parsed: Optional[dict]) -> Optional[dict]:
    if parsed is None: return None
    out = {}
    for k,v in parsed.items():
        if isinstance(v, str): v = v.strip()
        out[k] = v
    if out.get("url") and not out["url"].startswith(("http://","https://")) and "." in out["url"]:
        out["url"] = "https://" + out["url"]
    return out

def safe_path_check(filename: str, base_dir: str = SAFE_BASE_DIR) -> Tuple[bool, Optional[str]]:
    if not filename or not isinstance(filename, str): return False, None
    if os.path.isabs(filename): return False, None
    norm = os.path.normpath(filename)
    if ".." in norm.split(os.sep): return False, None
    safe = os.path.join(base_dir, norm)
    safe_norm = os.path.normpath(safe)
    if not safe_norm.startswith(os.path.normpath(base_dir)): return False, None
    os.makedirs(os.path.dirname(safe_norm), exist_ok=True)
    return True, safe_norm

def fuzzy_match_app(app_raw: str):
    if not app_raw: return None
    app_low = app_raw.lower()
    if app_low in ALLOWED_APPS: return app_low
    close = difflib.get_close_matches(app_low, list(ALLOWED_APPS), n=1, cutoff=0.6)
    return close[0] if close else None

def validate_parsed(parsed: Optional[dict]) -> Tuple[bool,str,Optional[dict]]:
    if parsed is None: return False, "parse_failed", None
    parsed = normalize_parsed(parsed)
    intent = (parsed.get("intent") or "").lower()
    if not intent: return False, "no_intent", parsed

    if intent in ("create_file","delete_file","move_file","copy_file","open_file","open_file_with_app","rename_file"):
        fname = parsed.get("filename") or parsed.get("path") or ""
        if not fname: return False, "no_filename", parsed
        ok, safe = safe_path_check(fname)
        if not ok: return False, "unsafe_path", parsed
        parsed["_safe_path"] = safe
        if intent in ("move_file","copy_file"):
            dst = parsed.get("folder") or parsed.get("dst") or ""
            parsed["_dst_safe"] = os.path.join(SAFE_BASE_DIR, dst) if dst else SAFE_BASE_DIR
        return True, "ok", parsed

    if intent in ("open_app","open_app_and_go_url","open_app_and_search","open_app_and_open_file_and_search"):
        app = parsed.get("app") or ""
        matched = fuzzy_match_app(app)
        if not matched: return False, "unknown_app", parsed
        parsed["app"] = matched
        return True, "ok", parsed

    if intent == "open_url":
        url = parsed.get("url") or ""
        if not url: return False, "no_url", parsed
        if not (url.startswith("http://") or url.startswith("https://") or "." in url): return False, "bad_url", parsed
        parsed["url"] = url if url.startswith(("http://","https://")) else "https://" + url
        return True, "ok", parsed

    if intent == "search_web":
        q = parsed.get("query") or ""
        if not q: return False, "no_query", parsed
        return True, "ok", parsed

    if intent in ("create_folder","list_dir","search_files"):
        folder = parsed.get("folder") or ""
        if not folder: return False, "no_folder", parsed
        ok, safe = safe_path_check(folder)
        parsed["_folder_safe"] = safe if ok else os.path.join(SAFE_BASE_DIR, folder)
        return True, "ok", parsed

    if intent in ("zip_files","unzip_files"):
        if intent == "zip_files":
            fl = parsed.get("filenames") or parsed.get("filenames_list") or ""
            if not fl: return False, "no_filenames", parsed
            return True, "ok", parsed
        else:
            if not parsed.get("zipname"): return False, "no_zipname", parsed
            return True, "ok", parsed

    if intent in ("system_sleep","shutdown","restart","lock_screen","logout","volume_up","volume_down","mute","unmute","play_music","pause_music","resume_music","stop_music","screenshot","take_photo","set_timer","cancel_timer","send_email","translate_text","dictionary_lookup","calculator","weather","show_help","whoami"):
        return True, "ok", parsed

    return False, "unknown_intent", parsed

# ---------- Execution implementations ----------
def launch_app(app_name: str, url: Optional[str]=None):
    """Cross-platform app launcher"""
    app_name_lower = app_name.lower()
    
    # Try to get app path
    exe = APP_PATHS.get(app_name_lower)
    
    if os.name == "nt":  # Windows
        if exe and os.path.exists(exe):
            try:
                if url:
                    subprocess.Popen([exe, url], shell=False)
                else:
                    os.startfile(exe)
                return {"status":"ok"}
            except Exception as e:
                return {"status":"error","error":str(e)}
        # Fallback for Windows
        try:
            subprocess.run(['cmd','/c','start','', app_name if not url else f'{app_name} {url}'], shell=True, check=False)
            return {"status":"ok"}
        except Exception as e:
            return {"status":"error","error":str(e)}
    
    else:  # macOS/Linux
        # macOS application launching
        if exe and os.path.exists(exe):
            try:
                if url:
                    subprocess.Popen(["open", "-a", exe, url])
                else:
                    subprocess.Popen(["open", "-a", exe])
                return {"status":"ok"}
            except Exception as e:
                return {"status":"error","error":str(e)}
        
        # Try using 'open' command on macOS
        try:
            if url:
                subprocess.Popen(["open", "-a", app_name, url])
            else:
                subprocess.Popen(["open", "-a", app_name])
            return {"status":"ok"}
        except Exception:
            # Fallback for Linux
            try:
                if url:
                    subprocess.Popen([app_name, url])
                else:
                    subprocess.Popen([app_name])
                return {"status":"ok"}
            except Exception as e:
                return {"status":"error","error":str(e)}

def perform_execution(intent: str, parsed: dict, execute: bool=False, confirm: bool=False) -> dict:
    intent = intent.lower()
    
    # App launching
    if intent == "open_app":
        app = parsed.get("app")
        if not execute: return {"status":"simulated","preview": f"open app {app}"}
        return launch_app(app)
    
    if intent == "open_url":
        url = parsed.get("url")
        if not execute: return {"status":"simulated","preview": f"open url {url}"}
        try:
            webbrowser.open(url)
            return {"status":"ok"}
        except Exception as e: return {"status":"error","error":str(e)}
    
    if intent in ("open_app_and_go_url","open_app_and_search","open_app_and_open_file_and_search"):
        app = parsed.get("app")
        url = parsed.get("url") or ("https://www.google.com/search?q=" + shlex.quote(parsed.get("query") or ""))
        if not execute: return {"status":"simulated","preview": f"open {app} and open {url}"}
        return launch_app(app, url)
    
    if intent == "search_web":
        query = parsed.get("query") or ""
        if not execute: return {"status":"simulated","preview": f"search web for {query}"}
        try:
            webbrowser.open(f"https://www.google.com/search?q={shlex.quote(query)}")
            return {"status":"ok"}
        except Exception as e: return {"status":"error","error":str(e)}
    
    # File operations
    if intent == "create_file":
        safe = parsed.get("_safe_path")
        if not execute: return {"status":"simulated","preview": f"create file {safe}"}
        try:
            open(safe,"a").close()
            return {"status":"ok"}
        except Exception as e: return {"status":"error","error":str(e)}
    
    if intent == "delete_file":
        safe = parsed.get("_safe_path")
        if not confirm: return {"status":"needs_confirm"}
        if not execute: return {"status":"simulated","preview": f"delete file {safe}"}
        try:
            os.remove(safe)
            return {"status":"ok"}
        except Exception as e: return {"status":"error","error":str(e)}
    
    if intent == "open_file":
        safe = parsed.get("_safe_path")
        if not execute: return {"status":"simulated","preview": f"open file {safe}"}
        try:
            if os.name == "nt":
                os.startfile(safe)
            else:
                subprocess.Popen(["xdg-open", safe])
            return {"status":"ok"}
        except Exception as e: return {"status":"error","error":str(e)}
    
    if intent == "rename_file":
        safe = parsed.get("_safe_path")
        new_name = parsed.get("new_name") or parsed.get("new_filename") or ""
        if not execute: return {"status":"simulated","preview": f"rename {safe} to {new_name}"}
        try:
            new_path = os.path.join(os.path.dirname(safe), new_name)
            os.rename(safe, new_path)
            return {"status":"ok"}
        except Exception as e: return {"status":"error","error":str(e)}
    
    if intent == "move_file":
        safe = parsed.get("_safe_path")
        dst = parsed.get("_dst_safe")
        if not execute: return {"status":"simulated","preview": f"move {safe} to {dst}"}
        try:
            shutil.move(safe, dst)
            return {"status":"ok"}
        except Exception as e: return {"status":"error","error":str(e)}
    
    if intent == "copy_file":
        safe = parsed.get("_safe_path")
        dst = parsed.get("_dst_safe")
        if not execute: return {"status":"simulated","preview": f"copy {safe} to {dst}"}
        try:
            shutil.copy2(safe, dst)
            return {"status":"ok"}
        except Exception as e: return {"status":"error","error":str(e)}
    
    # Folder operations
    if intent == "create_folder":
        folder = parsed.get("_folder_safe")
        if not execute: return {"status":"simulated","preview": f"create folder {folder}"}
        try:
            os.makedirs(folder, exist_ok=True)
            return {"status":"ok"}
        except Exception as e: return {"status":"error","error":str(e)}
    
    if intent == "list_dir":
        folder = parsed.get("_folder_safe")
        if not execute: return {"status":"simulated","preview": f"list directory {folder}"}
        try:
            files = os.listdir(folder)
            return {"status":"ok","files":files}
        except Exception as e: return {"status":"error","error":str(e)}
    
    # Archive operations
    if intent == "zip_files":
        filenames = parsed.get("filenames") or parsed.get("filenames_list") or ""
        zipname = parsed.get("zipname") or "archive.zip"
        if not execute: return {"status":"simulated","preview": f"zip files {filenames} to {zipname}"}
        try:
            with zipfile.ZipFile(os.path.join(SAFE_BASE_DIR, zipname), 'w') as zf:
                for fname in filenames.split(","):
                    fname = fname.strip()
                    ok, safe = safe_path_check(fname)
                    if ok and os.path.exists(safe):
                        zf.write(safe, os.path.basename(safe))
            return {"status":"ok"}
        except Exception as e: return {"status":"error","error":str(e)}
    
    if intent == "unzip_files":
        zipname = parsed.get("zipname") or ""
        if not execute: return {"status":"simulated","preview": f"unzip {zipname}"}
        try:
            ok, safe = safe_path_check(zipname)
            if ok:
                with zipfile.ZipFile(safe, 'r') as zf:
                    zf.extractall(SAFE_BASE_DIR)
            return {"status":"ok"}
        except Exception as e: return {"status":"error","error":str(e)}
    
    # System operations
    if intent == "screenshot":
        if not execute: return {"status":"simulated","preview": "take screenshot"}
        try:
            if os.name == "nt":
                from PIL import ImageGrab
                screenshot = ImageGrab.grab()
                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                filepath = os.path.join(SAFE_BASE_DIR, filename)
                screenshot.save(filepath)
                return {"status":"ok","file":filepath}
            else:  # macOS
                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                filepath = os.path.join(SAFE_BASE_DIR, filename)
                subprocess.run(["screencapture", "-x", filepath], check=True)
                return {"status":"ok","file":filepath}
        except Exception as e:
            return {"status":"error","error":str(e)}
    
    if intent == "volume_up":
        if not execute: return {"status":"simulated","preview": "increase volume"}
        try:
            if os.name == "nt":
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                current = volume.GetMasterVolumeLevelScalar()
                volume.SetMasterVolumeLevelScalar(min(1.0, current + 0.1), None)
            else:  # macOS
                subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) + 10)"], check=False)
            return {"status":"ok"}
        except Exception as e: return {"status":"error","error":str(e)}
    
    if intent == "volume_down":
        if not execute: return {"status":"simulated","preview": "decrease volume"}
        try:
            if os.name == "nt":
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                current = volume.GetMasterVolumeLevelScalar()
                volume.SetMasterVolumeLevelScalar(max(0.0, current - 0.1), None)
            else:  # macOS
                subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) - 10)"], check=False)
            return {"status":"ok"}
        except Exception as e: return {"status":"error","error":str(e)}
    
    if intent == "mute":
        if not execute: return {"status":"simulated","preview": "mute audio"}
        try:
            if os.name == "nt":
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                volume.SetMute(1, None)
            else:  # macOS
                subprocess.run(["osascript", "-e", "set volume output muted true"], check=False)
            return {"status":"ok"}
        except Exception as e: return {"status":"error","error":str(e)}
    
    if intent == "unmute":
        if not execute: return {"status":"simulated","preview": "unmute audio"}
        try:
            if os.name == "nt":
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                volume.SetMute(0, None)
            else:  # macOS
                subprocess.run(["osascript", "-e", "set volume output muted false"], check=False)
            return {"status":"ok"}
        except Exception as e: return {"status":"error","error":str(e)}
    
    if intent == "shutdown":
        if not confirm: return {"status":"needs_confirm"}
        if not execute: return {"status":"simulated","preview": "shutdown system"}
        try:
            if os.name == "nt":
                subprocess.run(["shutdown", "/s", "/t", "10"], check=False)
            else:  # macOS/Linux
                subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
            return {"status":"ok"}
        except Exception as e: return {"status":"error","error":str(e)}
    
    if intent == "restart":
        if not confirm: return {"status":"needs_confirm"}
        if not execute: return {"status":"simulated","preview": "restart system"}
        try:
            if os.name == "nt":
                subprocess.run(["shutdown", "/r", "/t", "10"], check=False)
            else:  # macOS/Linux
                subprocess.run(["sudo", "shutdown", "-r", "now"], check=False)
            return {"status":"ok"}
        except Exception as e: return {"status":"error","error":str(e)}
    
    if intent == "lock_screen":
        if not execute: return {"status":"simulated","preview": "lock screen"}
        try:
            if os.name == "nt":
                subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], check=False)
            else:  # macOS
                subprocess.run(["/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession", "-suspend"], check=False)
            return {"status":"ok"}
        except Exception as e: return {"status":"error","error":str(e)}
    
    if intent == "system_sleep":
        if not execute: return {"status":"simulated","preview": "put system to sleep"}
        try:
            if os.name == "nt":
                subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"], check=False)
            else:  # macOS
                subprocess.run(["pmset", "sleepnow"], check=False)
            return {"status":"ok"}
        except Exception as e: return {"status":"error","error":str(e)}
    
    # Music control (basic media keys)
    if intent == "play_music":
        if not execute: return {"status":"simulated","preview": "play music"}
        try:
            if os.name == "nt":
                import pyautogui
                pyautogui.press('playpause')
            else:  # macOS
                subprocess.run(["osascript", "-e", 'tell application "Music" to play'], check=False)
            return {"status":"ok"}
        except Exception as e: return {"status":"error","error":str(e)}
    
    if intent == "pause_music":
        if not execute: return {"status":"simulated","preview": "pause music"}
        try:
            if os.name == "nt":
                import pyautogui
                pyautogui.press('playpause')
            else:  # macOS
                subprocess.run(["osascript", "-e", 'tell application "Music" to pause'], check=False)
            return {"status":"ok"}
        except Exception as e: return {"status":"error","error":str(e)}
    
    # Simple intents
    if intent == "calculator":
        if not execute: return {"status":"simulated","preview": "open calculator"}
        try:
            if os.name == "nt":
                subprocess.Popen(["calc.exe"])
            else:  # macOS
                subprocess.Popen(["open", "-a", "Calculator"])
            return {"status":"ok"}
        except Exception as e: return {"status":"error","error":str(e)}
    
    return {"status":"unsupported_intent","intent":intent}

def needs_confirmation(intent: str, parsed: dict, confidence: float) -> bool:
    destructive = {"delete_file","shutdown","restart","logout"}
    if confidence is None or confidence < CONF_THRESHOLD: return True
    if intent in destructive: return True
    return False

def run_and_execute(utterance: str, execute: bool=False, confirm: bool=False):
    parsed_result = parse_command(utterance, max_new_tokens=64, num_beams=4, do_sample=False)[0]
    raw = parsed_result.get("raw"); parsed = parsed_result.get("parsed"); conf = parsed_result.get("confidence") or 0.0
    ok, reason, validated = validate_parsed(parsed)
    if not ok:
        return {"status":"validation_failed","reason":reason,"parsed":parsed,"raw":raw,"confidence":conf}
    if needs_confirmation(validated.get("intent"), validated, conf) and not confirm:
        return {"status":"needs_confirmation","parsed":validated,"confidence":conf,"raw":raw}
    result = perform_execution(validated.get("intent"), validated, execute=execute, confirm=confirm)
    return {"status":"executed" if result.get("status")=="ok" else "failed","result":result,"parsed":validated,"raw":raw,"confidence":conf}

# ---------- TTS Speaker (Fixed) ----------
try:
    import pyttsx3
    TTS_AVAILABLE = True
except:
    TTS_AVAILABLE = False
    pyttsx3 = None

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except:
    SR_AVAILABLE = False
    sr = None

class NonBlockingSpeaker:
    """Fixed TTS that doesn't block"""
    def __init__(self):
        if not TTS_AVAILABLE:
            self.available = False
            return
        self.available = True
        self.queue = queue.Queue()
        self.running = True
        self.engine = None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def _worker(self):
        """Worker thread that processes TTS queue"""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 170)
            self.engine.setProperty("volume", 1.0)
            
            while self.running:
                try:
                    text = self.queue.get(timeout=0.1)
                    if text is None:
                        break
                    print(f"[TTS] Speaking: {text}")
                    self.engine.say(text)
                    self.engine.runAndWait()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"TTS Error: {e}")
        except Exception as e:
            print(f"TTS Init Error: {e}")
            self.available = False
    
    def speak(self, text: str):
        """Non-blocking speak"""
        if self.available:
            self.queue.put(text)
        else:
            print(f"[TTS Unavailable] Would say: {text}")
    
    def stop(self):
        """Stop the speaker"""
        self.running = False
        if self.available:
            self.queue.put(None)
            if self.thread.is_alive():
                self.thread.join(timeout=2)

SPEAKER = NonBlockingSpeaker()

# ---------- GUI Interface ----------
import tkinter as tk
from tkinter import ttk, scrolledtext
import sv_ttk  # Optional: for modern theme

class VoiceAssistantGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Voice Assistant")
        self.root.geometry("800x600")
        
        # Try to apply modern theme
        try:
            sv_ttk.set_theme("dark")
        except:
            pass
        
        self.is_listening = False
        self.auto_listen = True  # Auto-listening enabled by default
        self.execute_mode = True
        self.listening_thread = None
        self.setup_ui()
        
        # Start auto-listening after GUI is ready
        self.root.after(1000, self.start_auto_listen)
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Top control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.columnconfigure(1, weight=1)
        
        # Status indicator
        self.status_label = ttk.Label(control_frame, text="● Ready", font=("Arial", 12, "bold"))
        self.status_label.grid(row=0, column=0, padx=(0, 20))
        
        # Microphone button (shows listening state)
        self.mic_button = tk.Button(
            control_frame,
            text="🎤",
            font=("Arial", 32),
            width=3,
            height=1,
            bg="#4CAF50",
            fg="white",
            activebackground="#45a049",
            relief=tk.RAISED,
            bd=3,
            state=tk.DISABLED  # Visual indicator only
        )
        self.mic_button.grid(row=0, column=1, pady=10)
        
        # Auto-listen toggle
        self.auto_listen_var = tk.BooleanVar(value=True)
        auto_listen_check = ttk.Checkbutton(
            control_frame,
            text="Auto Listen",
            variable=self.auto_listen_var,
            command=self.toggle_auto_listen
        )
        auto_listen_check.grid(row=0, column=2, padx=(20, 0))
        
        # Execute mode toggle
        self.execute_var = tk.BooleanVar(value=True)
        execute_check = ttk.Checkbutton(
            control_frame,
            text="Execute Commands",
            variable=self.execute_var,
            command=self.toggle_execute_mode
        )
        execute_check.grid(row=0, column=3, padx=(10, 0))
        
        # Log area
        log_frame = ttk.LabelFrame(main_frame, text="Activity Log", padding="5")
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="white"
        )
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bottom button panel
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(button_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.exit_app).pack(side=tk.RIGHT, padx=5)
        
        self.log("Voice Assistant initialized and ready.")
        SPEAKER.speak("Voice assistant starting")
        if TTS_AVAILABLE:
            self.log("✓ Text-to-Speech enabled")
        else:
            self.log("✗ Text-to-Speech unavailable")
        if SR_AVAILABLE:
            self.log("✓ Speech Recognition enabled")
        else:
            self.log("✗ Speech Recognition unavailable")
    
    def log(self, message: str, color: str = None):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, formatted)
        if color:
            # Tag-based coloring can be added here
            pass
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def toggle_listening(self):
        """Start/stop listening"""
        if self.is_listening:
            self.stop_listening()
        else:
            self.start_listening()
    
    def toggle_auto_listen(self):
        """Toggle auto-listening mode"""
        self.auto_listen = self.auto_listen_var.get()
        if self.auto_listen:
            self.log("Auto-listen ENABLED - continuously listening")
            self.start_auto_listen()
        else:
            self.log("Auto-listen DISABLED")
            self.stop_listening()
    
    def start_auto_listen(self):
        """Start continuous auto-listening loop"""
        if not self.auto_listen:
            return
        if not SR_AVAILABLE:
            self.log("ERROR: Speech recognition not available!")
            return
        
        self.log("Starting continuous listening mode...")
        SPEAKER.speak("I'm listening")
        self.continuous_listen_loop()
    
    def continuous_listen_loop(self):
        """Continuous listening loop that runs in the background"""
        if not self.auto_listen or not SR_AVAILABLE:
            return
        
        # Start listening
        self.is_listening = True
        self.mic_button.config(bg="#f44336")
        self.status_label.config(text="● Listening...")
        
        # Run in separate thread
        threading.Thread(target=self.auto_listen_worker, daemon=True).start()
    
    def auto_listen_worker(self):
        """Worker for continuous auto-listening"""
        r = sr.Recognizer()
        r.energy_threshold = 4000  # Adjust sensitivity
        r.dynamic_energy_threshold = True
        
        try:
            with sr.Microphone() as source:
                self.root.after(0, lambda: self.log("Adjusting for ambient noise..."))
                r.adjust_for_ambient_noise(source, duration=1)
                self.root.after(0, lambda: self.log("Ready - speak now!"))
                
                # Listen for speech
                audio = r.listen(source, timeout=10, phrase_time_limit=10)
            
            # Process the audio
            self.root.after(0, lambda: self.log("Processing speech..."))
            text = r.recognize_google(audio, language='en-US')
            self.root.after(0, lambda: self.process_command(text))
            
        except sr.WaitTimeoutError:
            self.root.after(0, lambda: self.log("No speech detected - still listening..."))
            self.root.after(0, lambda: SPEAKER.speak("I didn't hear anything"))
        except sr.UnknownValueError:
            self.root.after(0, lambda: self.log("Could not understand audio - please repeat"))
            self.root.after(0, lambda: SPEAKER.speak("Could you please repeat that?"))
        except Exception as e:
            self.root.after(0, lambda: self.log(f"Error: {e}"))
        
        # Continue listening if auto-listen is still enabled
        self.root.after(500, self.continuous_listen_loop)
    
    def start_listening(self):
        """Start listening for voice input (manual trigger)"""
        if not SR_AVAILABLE:
            self.log("ERROR: Speech recognition not available!")
            return
        
        self.is_listening = True
        self.mic_button.config(bg="#f44336")
        self.status_label.config(text="● Listening...")
        self.log("Listening for command...")
        
        # Run listening in separate thread to avoid blocking GUI
        threading.Thread(target=self.listen_worker, daemon=True).start()
    
    def stop_listening(self):
        """Stop listening"""
        self.is_listening = False
        self.mic_button.config(bg="#4CAF50")
        self.status_label.config(text="● Ready")
    
    def listen_worker(self):
        """Worker thread for speech recognition"""
        r = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source, timeout=6.0, phrase_time_limit=8.0)
            
            self.root.after(0, lambda: self.log("Processing speech..."))
            text = r.recognize_google(audio, language='en-US')
            self.root.after(0, lambda: self.process_command(text))
            
        except sr.WaitTimeoutError:
            self.root.after(0, lambda: self.log("No speech detected (timeout)"))
        except sr.UnknownValueError:
            self.root.after(0, lambda: self.log("Could not understand audio"))
        except Exception as e:
            self.root.after(0, lambda: self.log(f"Error: {e}"))
        finally:
            self.root.after(0, self.stop_listening)
    
    def process_command(self, text: str):
        """Process recognized command"""
        self.log(f"You said: \"{text}\"")
        
        if text.lower() in ("exit", "quit", "stop", "bye"):
            self.log("Exit command detected. Goodbye!")
            SPEAKER.speak("Goodbye!")
            self.root.after(1000, self.exit_app)
            return
        
        # Parse and execute
        self.log("Analyzing command...")
        result = run_and_execute(text, execute=False, confirm=False)
        
        if result.get("status") == "validation_failed":
            msg = f"❌ Could not understand command. Reason: {result.get('reason')}"
            self.log(msg)
            SPEAKER.speak("I couldn't understand that command.")
            return
        
        # Log details
        self.log(f"Intent: {result.get('parsed', {}).get('intent', 'unknown')}")
        self.log(f"Confidence: {result.get('confidence', 0):.2%}")
        
        if result.get("status") == "needs_confirmation":
            self.log("⚠ This command requires confirmation")
            SPEAKER.speak("This command requires confirmation. Please confirm.")
            # In a real implementation, add a confirmation dialog here
            return
        
        # Execute if mode is enabled
        if self.execute_mode:
            self.log("Executing command...")
            exec_result = run_and_execute(text, execute=True, confirm=True)
            if exec_result.get("status") == "executed":
                self.log("✓ Command executed successfully")
                SPEAKER.speak("Done")
            else:
                self.log(f"✗ Execution failed: {exec_result}")
                SPEAKER.speak("Command failed")
        else:
            self.log("Preview mode - command not executed")
            SPEAKER.speak("Preview complete")
    
    def toggle_execute_mode(self):
        """Toggle execute mode"""
        self.execute_mode = self.execute_var.get()
        mode = "ENABLED" if self.execute_mode else "DISABLED"
        self.log(f"Execute mode: {mode}")
    
    def clear_log(self):
        """Clear the log"""
        self.log_text.delete(1.0, tk.END)
    
    def exit_app(self):
        """Exit application"""
        SPEAKER.stop()
        self.root.destroy()
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

# ---------- Launch GUI ----------
if __name__ == "__main__":
    app = VoiceAssistantGUI()
    app.run()