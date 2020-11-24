from pathlib import Path
from wav2letter import inference


def print_result(result):
    print(
        f"{result.chunk_start_time}-{result.chunk_end_time} ms: {[f'{w.word}' for w in result.words]}"
    )


model = inference.load_model(
    input_files_base_path="/home/tetianamyronivska/tds_ctc_streaming_serialized"
)
inference_stream = model.open_stream()

chunk_size = 32000  # 32000 = 1 sec (mobile sends chunks of 3200)

local_dir = "/home/tetianamyronivska/audio"

path = Path(local_dir)
for p in path.rglob(path):
    print(p)
# with open("/home/tetianamyronivska/tds_ctc_streaming_serialized/r1.wav", "rb") as f:
#     f.seek(44)  # skip WAV header
#     bytes = f.read(chunk_size)
#     while bytes:
#         inference_stream.submit_audio(bytes)
#         result = inference_stream.next_result()
#         # inference_stream.prune()
#         print_result(result)
#         bytes = f.read(chunk_size)

# inference_stream.end_audio()
# result = inference_stream.next_result()
# inference_stream.prune()
# print_result(result)
