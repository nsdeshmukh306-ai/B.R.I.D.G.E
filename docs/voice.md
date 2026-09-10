# Voice assistant

BRIDGE is hands-free: say a command, the item lights up on the bench and BRIDGE answers by voice.

## Pipeline

```
microphone (sounddevice, 16 kHz mono)
   → energy VAD with adaptive noise floor (voice/audio.py)     one utterance
   → speech-to-text (voice/stt.py)                             Transcript{text, text_en, language, confidence}
   → wake-word check (optional)
   → BridgeCore.handle_spoken(text)                            procedure control words | safety gate | AI pipeline
   → text-to-speech (voice/tts.py)                             capture pauses while BRIDGE speaks
```

## Speech recognition providers

| provider | how | needs |
|---|---|---|
| `gemini` (default) | the WAV clip is sent to Gemini with a transcription prompt; returns verbatim text, an English rendering and the language | `GEMINI_API_KEY`, network |
| `whisper` | local faster-whisper, offline; non-English is translated to English for routing | `pip install faster-whisper` (first run downloads the model) |
| `mock` | canned transcripts | tests / simulation only |

Gemini transcription is multilingual out of the box (English, Hindi, Marathi, ...): the original is shown in the VOICE panel, the English rendering drives the command.

## Speech output

`pyttsx3` uses the operating system voices (SAPI5 on Windows — e.g. "Microsoft Zira", "Microsoft David", Indian English voices if installed). Set a substring of the voice name and the rate in Settings. If no engine is available BRIDGE stays silent and logs what it would have said.

## Controls

* **LISTEN** toggles continuous listening. In physical mode it starts automatically once calibration is VALID (`voice.enabled`).
* **Push to talk** records while held and processes one command on release — useful in noisy rooms.
* **Require wake word**: only utterances starting with the wake word ("Bridge, where is the syringe?") are acted on. Off by default; "Bridge ..." is still accepted and stripped.
* **Speak replies** turns TTS on/off.

## What you can say

Locating: *where is the syringe*, *highlight the lavender tube*, *point to the sharps container*, *where are the swabs*, *where should I put the needle*, *move the sample to the rack*, *clear*.

Procedures: *start the order of draw*, *start instrument count*, *list procedures*; then *next* / *done* / *counted*, *back*, *repeat*, *stop procedure*. *What next* asks the AI for the next step when no procedure is running.

BRIDGE replies with one sentence ("Found syringe." "Step 2 of 7. Light blue cap: citrate tube ...") and appends safety cautions for sharps and medications.

## Tuning the microphone

BRIDGE records from **this computer's default input device** unless you pick another one in Settings, so a laptop's built-in microphone works with nothing plugged in. Windows audio backends often refuse 16 kHz mono, so the stream is opened at whatever rate the device accepts (48 kHz, 44.1 kHz, ...) and resampled to 16 kHz internally. If no device can be opened, the error names the device it tried and lists the inputs it can see.

`voice.vad_threshold` (RMS, 0–1) is the speech threshold; the effective threshold is `max(threshold, 3 × noise floor)`. Lower it if BRIDGE misses quiet speech, raise it if it triggers on background noise. `silence_ms` is how long a pause ends an utterance; `max_utterance_s` caps a single command. `mic_device` selects an input by index (`python -c "import sounddevice; print(sounddevice.query_devices())"`).

## Privacy

Audio is only captured while LISTEN is on or push-to-talk is held. With the `gemini` provider each utterance clip is sent to Google's API; use `whisper` for fully local recognition.

## Always-on conversation

With the surgical assistant enabled (`assistant.enabled`, the default), the voice
channel changes in three ways.

**Barge-in.** Speaking while BRIDGE is talking cuts it off mid-sentence
(`voice.barge_in`). The microphone also hears BRIDGE's own voice through the
speakers, so a captured utterance is compared against what is being spoken and
discarded when it is mostly the same words — a real interruption ("stop", "no,
the other one") shares almost nothing with the sentence in progress.

**One prioritised voice.** Replies and proactive alerts share the channel through
an `Announcer`: a critical alert preempts and clears the queue, a reply jumps
ahead of background chatter because someone is waiting for it, a caution waits
its turn and is dropped if it goes stale, and identical text within 20 seconds is
dropped outright.

**Follow-ups.** The last subject is remembered for three minutes, so "and the
other one", "point to it" and "where is it" resolve locally without a round-trip.

See [docs/surgical.md](docs/surgical.md) for the full command vocabulary.
