from pathlib import Path
from wav2letter import inference


def print_result(result):
    print(" ".join([w.word for w in result.words]))


model = inference.load_model(
    input_files_base_path="/home/tetianamyronivska/tds_ctc_streaming_serialized"
)

chunk_size = 32000  # 32000 = 1 sec (mobile sends chunks of 3200)

local_dir = "/home/tetianamyronivska/test_segments_by_provider"
providers = ["diane_snyder"]

for provider in providers:
    path = Path(f"{local_dir}/{provider}")
    for p in path.rglob("*"):
        with open(p, "rb") as f:
            print(p)
            print(p.suffix)
            inference_stream = model.open_stream()
            f.seek(44)  # skip WAV header
            bytes = f.read(chunk_size)
            try:
                while bytes:
                    inference_stream.submit_audio(bytes)
                    result = inference_stream.next_result()
                    # inference_stream.prune()
                    bytes = f.read(chunk_size)

                inference_stream.end_audio()
                result = inference_stream.next_result()
                inference_stream.prune()
                result = " ".join([w.word for w in result.words])
                print(result)
            except RuntimeError:
                continue
