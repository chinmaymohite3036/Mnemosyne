import whisper

model = whisper.load_model("large-v2")

result = model.transcribe(audio = "audios/sample.mp3",
                          language = "hi",
                          task = "translate",
                          word_timestamps = False)