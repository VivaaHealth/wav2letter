import pandas as pd

from jiwer import wer
from pathlib import Path
from wav2letter import inference


def print_result(result):
    print(" ".join([w.word for w in result.words]))


model = inference.load_model(
    input_files_base_path="/local/working/dir/2020_11_25_tds_ctc_serialized"
)

chunk_size = 32000  # 32000 = 1 sec (mobile sends chunks of 3200)

local_dir = "/home/tetianamyronivska/test_segments_by_provider"
providers = [
    # "aaron_dickens",
    # "chealon_miller",
    # "erika_wilson",
    # "john_gleason",
    # "maurice_goins",
    # "rick_mullins",
    # "shivani_beri",
    # "alex_davis",
    # "chris_dolan",
    # "ernesto_gonzalez",
    # "joseph_garcia",
    # "melanie_belt",
    # "robert_greenfield",
    # "sindura_bandi",
    # "alinda_cox",
    # "christian_schupp",
    # "erroll_bailey",
    # "julie_grimes",
    # "michael_quackenbush",
    # "robert_grigg",
    # "sonia_yousuf",
    # "amy_carolan",
    # "christopher_bailey",
    # "evander_fogle",
    # "ken_gillen",
    # "michael_samaan",
    # "robert_hudon",
    # "sourendra_raut",
    # "anita_sandhu",
    # "crystal_berry_roberts",
    # "gary_stewart",
    # "ken_hine",
    # "michele_perez",
    # "robert_hughes",
    # "steven_wertheim",
    # "anthony_monteiro",
    # "cynthia_chaparro_krueger",
    # "hillary_miller",
    # "kenneth_cornell",
    # "milan_patel",
    # "robert_titelman",
    # "swati_date",
    # "anuja_khunti",
    # "dan_douglas",
    # "jackie_sweeton",
    # "kevin_ofarrell",
    "nadine_flaharty",
    # "robin_dennis",
    # "tara_olson",
    # "archana_jayachandran",
    # "dana_hiscock",
    # "jacque_vance",
    # "kristi_harvey",
    # "nicholas_desai",
    # "romy_ghosh",
    # "timothy_dooley",
    # "arthur_raines",
    # "daniel_kelly",
    # "james_saucedo",
    # "lattisha_bilbrew",
    # "parul_desai",
    # "rosa_moreno",
    # "travis_kieckbusch",
    # "brian_wheeler",
    # "deborah_kowalchuk",
    # "jason_ahuero",
    # "lawrence_stitt",
    # "paul_gahlinger",
    # "sami_khan",
    # "travis_loidolt",
    # "brooks_ficke",
    # "jason_velez",
    # "lee_reichel",
    # "sapna_bhagat",
    # "vikas_godhania",
    # "bryan_morrison",
    # "diane_brinkman",
    # "jay_ham",
    # "leo_taarea",
    # "phillip_walton",
    # "sharon_liu",
    # "yvonne_satterwhite",
    # "caroline_kaufman",
    # "diane_snyder",
    # "jay_zdunek",
    # "lima_redhead",
    # "raymond_hui",
    # "sharyl_brasher_giles",
    # "zarina_hussain",
    # "carolyn_morales",
    # "donald_dominy",
    # "jayme_evans",
    # "manish_naik",
    # "rebecca_mouser",
    # "shaun_traub",
    # "carson_higgs",
    # "edward_benton",
    # "jenna_noveau",
    # "marc_labbe",
    # "richard_hayes",
    # "chad_zooker",
    # "elizabeth_knapp",
    # "jennifer_haase",
    # "mark_vann",
    # "richard_thorp",
    # "shilpa_vaidya",
]
provider_wers = []

for provider in providers:
    test_file_path = Path(f"{local_dir}/{provider}/test.tsv")
    test_df = pd.read_csv(test_file_path, sep="\t")
    audio_ids = test_df["id"].tolist()
    golden_transripts = test_df["transcript"].tolist()
    count = 0
    errors = []
    for audio_id, golden_transcript in zip(audio_ids, golden_transripts):
        audio_path = Path(f"{local_dir}/{provider}/{audio_id}.wav")
        print(audio_path)
        with open(audio_path, "rb") as f:
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
                result = result.replace("'", " ")
                error = wer(golden_transcript, result)
                print(golden_transcript)
                print(result)
                print(error)
                print("\n")
                errors.append(error)
                count += 1
            except RuntimeError:
                continue
    provider_wer = sum(errors) / count
    provider_wers.append({"provider": provider, "wer": provider_wer})
    print(provider_wers)
    eval_df = pd.DataFrame.from_dict(provider_wers)
    eval_df.to_csv("streaming_provider_evaluation.csv")
