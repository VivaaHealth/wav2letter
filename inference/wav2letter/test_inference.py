from wav2letter import inference


def print_result(result):
    print(
        f"{result.chunk_start_time}-{result.chunk_end_time} ms: {[f'{w.word}' for w in result.words]}"
    )


model = inference.load_model(
    input_files_base_path="/local/working/dir/2020_11_25_tds_ctc_serialized"
)
inference_stream = model.open_stream()

chunk_size = 32000  # 32000 = 1 sec (mobile sends chunks of 3200)

with open(
    "~/test_segments_by_provider/aaron_dickens/99cfa168-03fd-4a91-8cdb-71f02a48ccfe_10.wav",
    "rb",
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
