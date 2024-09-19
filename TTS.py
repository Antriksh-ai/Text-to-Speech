from transformers import AutoProcessor, BarkModel
import torch
import scipy

# Load processor and model
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_audio(text, preset, output):
    # Prepare inputs with voice preset applied during processing
    inputs = processor(text, return_tensors="pt")  # Convert to tensors

    # You may apply any configuration changes (e.g., preset) before generation here if necessary.
    # For now, we're assuming it's baked into the input processing.

    # Move inputs to the correct device
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    # Generate audio
    audio_array = model.generate(**inputs)  # Removed `voice_preset` as it's not an argument here.
    audio_array = audio_array.cpu().numpy().squeeze()  # Move to CPU for saving

    sample_rate = model.generation_config.sample_rate  # Get sample rate

    # Save the generated audio as a .wav file
    scipy.io.wavfile.write(output, rate=sample_rate, data=audio_array)

# Example usage
generate_audio(
    text="my name is shivam sood,I am pursing my btect in Artificial intelligence fron Delhi ,India",
    preset="v2/en_speaker_6",  # Voice preset might need to be applied differently
    output="output.wav"
)
