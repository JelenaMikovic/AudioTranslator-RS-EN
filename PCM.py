from pydub import AudioSegment
import os

def check_and_convert_to_pcm(directory, output_directory):
    for file_name in os.listdir(directory):
        if file_name.endswith(".wav"):
            input_file_path = os.path.join(directory, file_name)
            output_file_path = os.path.join(output_directory, file_name)

            try:
                print(input_file_path)
                audio = AudioSegment.from_wav(input_file_path)
                # Check if the sample width is 2 (16-bit PCM)
                print("HERE")
                if audio.sample_width != 2:
                    print(f"Converting {file_name} to PCM format")
                    audio.export(output_file_path, format="wav", parameters=["-ac", "1", "-sample_fmt", "s16"])
                else:
                    print(f"{file_name} is already in PCM format")
            except Exception as e:
                print(f"Error while processing {file_name}: {e}")

if __name__ == "__main__":
    # Specify the directory containing the WAV files
    input_directory = "data/test/"
    # Specify the output directory for PCM files
    output_directory = "data/test/"

    # Check and convert to PCM format
    check_and_convert_to_pcm(input_directory, output_directory)
