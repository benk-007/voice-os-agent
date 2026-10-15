"""Tkinter interface: continuous listening, activity log, execute/preview switch."""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, scrolledtext, ttk

from voice_os_agent.agent import VoiceOSAgent
from voice_os_agent.asr.listener import SR_AVAILABLE, SpeechListener, sr
from voice_os_agent.tts.speaker import TTS_AVAILABLE, NonBlockingSpeaker

log = logging.getLogger(__name__)

EXIT_WORDS = {"exit", "quit", "stop", "bye"}


class VoiceAssistantGUI:
    def __init__(self, agent: VoiceOSAgent | None = None) -> None:
        self.agent = agent or VoiceOSAgent()
        self.speaker = NonBlockingSpeaker()
        self.listener = SpeechListener() if SR_AVAILABLE else None
        self.root = tk.Tk()
        self.root.title("Voice OS Agent")
        self.root.geometry("800x600")
        try:
            import sv_ttk

            sv_ttk.set_theme("dark")
        except Exception:  # optional theme
            pass
        self.auto_listen = True
        self.execute_mode = True
        self._setup_ui()
        self.root.after(1000, self.start_auto_listen)

    # ---------- layout ----------
    def _setup_ui(self) -> None:
        main = ttk.Frame(self.root, padding="10")
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        controls = ttk.Frame(main)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        controls.columnconfigure(1, weight=1)
        self.status_label = ttk.Label(controls, text="● Ready", font=("Arial", 12, "bold"))
        self.status_label.grid(row=0, column=0, padx=(0, 20))
        self.mic_button = tk.Button(
            controls,
            text="🎤",
            font=("Arial", 32),
            width=3,
            height=1,
            bg="#4CAF50",
            fg="white",
            relief=tk.RAISED,
            bd=3,
            state=tk.DISABLED,
        )
        self.mic_button.grid(row=0, column=1, pady=10)
        self.auto_listen_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls, text="Auto Listen", variable=self.auto_listen_var, command=self.toggle_auto_listen
        ).grid(row=0, column=2, padx=(20, 0))
        self.execute_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls, text="Execute Commands", variable=self.execute_var, command=self.toggle_execute_mode
        ).grid(row=0, column=3, padx=(10, 0))

        log_frame = ttk.LabelFrame(main, text="Activity Log", padding="5")
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, font=("Consolas", 10), bg="#1e1e1e", fg="#d4d4d4", insertbackground="white"
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        buttons = ttk.Frame(main)
        buttons.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(buttons, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons, text="Exit", command=self.exit_app).pack(side=tk.RIGHT, padx=5)

        self.log("Voice Assistant initialized and ready.")
        self.speaker.speak("Voice assistant starting")
        self.log("✓ Text-to-Speech enabled" if TTS_AVAILABLE else "✗ Text-to-Speech unavailable")
        self.log("✓ Speech Recognition enabled" if SR_AVAILABLE else "✗ Speech Recognition unavailable")

    def log(self, message: str) -> None:
        log.info(message)
        self.log_text.insert(tk.END, f"[{datetime.now():%H:%M:%S}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def _set_listening(self, listening: bool) -> None:
        self.mic_button.config(bg="#f44336" if listening else "#4CAF50")
        self.status_label.config(text="● Listening..." if listening else "● Ready")

    # ---------- listening loop ----------
    def toggle_auto_listen(self) -> None:
        self.auto_listen = self.auto_listen_var.get()
        if self.auto_listen:
            self.log("Auto-listen ENABLED - continuously listening")
            self.start_auto_listen()
        else:
            self.log("Auto-listen DISABLED")
            self._set_listening(False)

    def start_auto_listen(self) -> None:
        if not self.auto_listen:
            return
        if self.listener is None:
            self.log("ERROR: Speech recognition not available! Use the text CLI: python -m voice_os_agent")
            return
        self.log("Starting continuous listening mode...")
        self.speaker.speak("I'm listening")
        self.continuous_listen_loop()

    def continuous_listen_loop(self) -> None:
        if not self.auto_listen or self.listener is None:
            return
        self._set_listening(True)
        threading.Thread(target=self._listen_worker, daemon=True).start()

    def _listen_worker(self) -> None:
        after = self.root.after
        try:
            after(0, lambda: self.log("Ready - speak now!"))
            text = self.listener.listen_once()
            after(0, lambda: self.process_command(text))
        except sr.WaitTimeoutError:
            after(0, lambda: self.log("No speech detected - still listening..."))
        except sr.UnknownValueError:
            after(0, lambda: self.log("Could not understand audio - please repeat"))
            after(0, lambda: self.speaker.speak("Could you please repeat that?"))
        except Exception as exc:
            message = f"Error: {exc}"  # bind now: `exc` is cleared when the except block ends
            after(0, lambda: self.log(message))
        after(500, self.continuous_listen_loop)

    # ---------- command handling ----------
    def process_command(self, text: str) -> None:
        self.log(f'You said: "{text}"')
        if text.lower().strip() in EXIT_WORDS:
            self.log("Exit command detected. Goodbye!")
            self.speaker.speak("Goodbye!")
            self.root.after(1000, self.exit_app)
            return

        self.log("Analyzing command...")
        plan = self.agent.plan(text)  # single model call
        if plan.status == "validation_failed":
            self.log(f"❌ Command refused. Reason: {plan.reason} (model output: {plan.raw})")
            self.speaker.speak("I couldn't understand that command.")
            return
        self.log(f"Intent: {plan.intent}")
        self.log(f"Confidence: {(plan.confidence or 0):.2%}")

        if not self.execute_mode:
            preview = self.agent.execute(plan, execute=False, confirm=True)
            self.log(f"Preview mode - not executed: {preview.result.get('preview', plan.raw)}")
            self.speaker.speak("Preview complete")
            return

        confirm = False
        if plan.status == "needs_confirmation":
            self.log("⚠ This command requires confirmation")
            self.speaker.speak("This command requires confirmation.")
            confirm = messagebox.askyesno("Confirm command", f"Execute this action?\n\n{plan.raw}")
            if not confirm:
                self.log("Cancelled by user")
                return

        self.log("Executing command...")
        plan = self.agent.execute(plan, execute=True, confirm=confirm)
        if plan.status == "executed":
            self.log("✓ Command executed successfully")
            self.speaker.speak("Done")
        else:
            self.log(f"✗ Execution failed: {plan.result}")
            self.speaker.speak("Command failed")

    def toggle_execute_mode(self) -> None:
        self.execute_mode = self.execute_var.get()
        self.log(f"Execute mode: {'ENABLED' if self.execute_mode else 'DISABLED'}")

    def clear_log(self) -> None:
        self.log_text.delete(1.0, tk.END)

    def exit_app(self) -> None:
        self.speaker.stop()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
