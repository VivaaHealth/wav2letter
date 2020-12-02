from pydub import AudioSegment

# Packages reqd: pydub, ffmpeg

# pydub - pip install pydub

# ffmpeg:
# sudo add-apt-repository ppa:kirillshkrogalev/ffmpeg-next
# sudo apt-get update
# sudo apt-get install ffmpeg

## Load the m4a files (in M4a_files.tar.gz)


# Convert m4a extension files to wav extension files


filepath = "/home/tetianamyronivska/2020-12-02T19_57_02.334Z.m4a"
wav_path = "/home/tetianamyronivska/2020-12-02T19_57_02.334Z.wav"
track = AudioSegment.from_file(filepath, "m4a")
file_handle = track.export(wav_path, format="wav")

