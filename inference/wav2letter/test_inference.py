from wav2letter import inference


def print_result(result):
    print(
        f"{result.chunk_start_time}-{result.chunk_end_time} ms: {[f'{w.word}({w.begin_time_frame})' for w in result.words]}"
    )


model = inference.load_model(input_files_base_path="/root/model")
inference_stream = model.open_stream()

chunk_size = 32000  # 32000 = 1 sec (mobile sends chunks of 3200)

with open(
    "/root/audio/LibriSpeech/dev-clean/777/126732/777-126732-0051.flac.wav", "rb"
) as f:
    f.seek(44)  # skip WAV header
    bytes = f.read(chunk_size)
    while bytes:
        inference_stream.submit_audio(bytes)
        result = inference_stream.next_result()
        # inference_stream.prune()
        print_result(result)
        bytes = f.read(chunk_size)

inference_stream.end_audio()
result = inference_stream.next_result()
inference_stream.prune()
print_result(result)
