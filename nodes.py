import io
import typing as T
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.io import wavfile
import torch
import torchaudio
from pedalboard.io import AudioFile
import glob
import platform

MAX_BATCH_SIZE = 8

class RiffusionNode:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                'spectrogram': ("IMAGE",),
            },
            "optional": {
                'filename_prefix': ("STRING", {"default": "Riffusion_"}),
                'save_to_file': ("BOOLEAN", {"default": False}),
                'max_volume': ("FLOAT", {"default": 50.0, "min": 10.0, "max": 100.0, "step": 5.0}),
                'quality_level': ("INT", {"default": 2, "min": 0, "max": 3, "step": 1}),
                'apply_filter': ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "Process_Riffusion"
    CATEGORY = "audio"

    def spectrogram_from_image(self, image, max_volume=50, power_for_image=0.25) -> np.ndarray:
        """
        Compute a spectrogram magnitude array from a spectrogram image.
        """
        # Convert to a numpy array of floats
        data = np.array(image).astype(np.float32)
        
        # Flip Y take a single channel
        data = data[::-1, :, 0]
        
        # Invert
        data = 255 - data
        
        # Rescale to max volume
        data = data * max_volume / 255
        
        # Reverse the power curve
        data = np.power(data, 1 / power_for_image)
        
        return data

    def waveform_from_spectrogram(
        self,
        Sxx: np.ndarray,
        n_fft: int,
        hop_length: int,
        win_length: int,
        num_samples: int,
        sample_rate: int,
        mel_scale: bool = True,
        n_mels: int = 512,
        num_griffin_lim_iters: int = 32,
        device: str = None,
        quality_level: int = 2
    ) -> np.ndarray:
        """
        Reconstruct a waveform from a spectrogram using the Griffin-Lim algorithm.
        Quality level controls the number of iterations:
        0 = fast/low quality (16 iterations)
        1 = balanced (32 iterations)
        2 = good (48 iterations)
        3 = best (64 iterations)
        """
        # Adjust iterations based on quality level
        iterations_map = {0: 16, 1: 32, 2: 48, 3: 64}
        num_griffin_lim_iters = iterations_map.get(quality_level, 32)
        
        # Select appropriate device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if platform.system() == "Darwin":  # macOS
                device = "cpu"  # Force CPU on macOS
        
        # Convert to torch tensor and move to device
        Sxx_torch = torch.from_numpy(Sxx).to(device)
        
        if mel_scale:
            try:
                # Fix dimensions for mel inverse scaling
                if Sxx_torch.ndim == 2:
                    Sxx_torch = Sxx_torch.unsqueeze(0)  # Add batch dimension
                
                mel_inv_scaler = torchaudio.transforms.InverseMelScale(
                    n_mels=n_mels,
                    sample_rate=sample_rate,
                    f_min=0,
                    f_max=10000,
                    n_stft=n_fft // 2 + 1,
                    norm=None,
                    mel_scale="htk"
                ).to(device)
                
                # Ensure we have the right shape for InverseMelScale
                if Sxx_torch.shape[-2] != n_mels:
                    print(f"Warning: Reshaping Sxx_torch from {Sxx_torch.shape} to fit n_mels={n_mels}")
                    # If the shape doesn't match, try a simpler approach
                    mel_scale = False
                else:
                    Sxx_torch = mel_inv_scaler(Sxx_torch)
            except Exception as e:
                print(f"Error in mel scale inverse transformation: {e}")
                mel_scale = False
                print("Continuing without mel scale transformation")
        
        try:
            # If not using mel scale, ensure dimensions match n_fft
            if not mel_scale:
                # Need to resize to match expected STFT dimensions
                if Sxx_torch.ndim == 2:
                    Sxx_torch = Sxx_torch.unsqueeze(0)  # Add batch dimension
                
                # Get the expected frequency dimension size
                freq_dim = n_fft // 2 + 1
                
                if Sxx_torch.shape[-2] != freq_dim:
                    print(f"Reshaping frequency dimension from {Sxx_torch.shape[-2]} to {freq_dim}")
                    # Use interpolation to resize
                    if hasattr(torch.nn.functional, 'interpolate'):
                        # For newer PyTorch versions
                        Sxx_resized = torch.nn.functional.interpolate(
                            Sxx_torch.permute(0, 2, 1),  # [batch, time, freq] -> [batch, freq, time]
                            size=freq_dim, 
                            mode='linear',
                            align_corners=False
                        ).permute(0, 2, 1)  # Back to [batch, time, freq]
                        Sxx_torch = Sxx_resized
                    else:
                        # Manual resize using numpy (fallback)
                        Sxx_np = Sxx_torch.cpu().numpy()
                        Sxx_resized = np.zeros((Sxx_np.shape[0], freq_dim, Sxx_np.shape[2]), dtype=Sxx_np.dtype)
                        for b in range(Sxx_np.shape[0]):
                            for t in range(Sxx_np.shape[2]):
                                Sxx_resized[b, :, t] = np.interp(
                                    np.linspace(0, 1, freq_dim),
                                    np.linspace(0, 1, Sxx_np.shape[1]),
                                    Sxx_np[b, :, t]
                                )
                        Sxx_torch = torch.from_numpy(Sxx_resized).to(device)
            
            print(f"Griffin-Lim input shape: {Sxx_torch.shape}, n_fft: {n_fft}, expected: {n_fft//2+1}")
            print(f"Using {num_griffin_lim_iters} Griffin-Lim iterations (quality level {quality_level})")
            
            griffin_lim = torchaudio.transforms.GriffinLim(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                power=1.0,
                n_iter=num_griffin_lim_iters,
                momentum=0.99,  # Higher momentum helps convergence quality
                rand_init=False  # Deterministic initialization
            ).to(device)
            
            waveform = griffin_lim(Sxx_torch).cpu().numpy()
            
            # If we got multiple waveforms (batch dimension > 1), just take the first
            if waveform.ndim > 1:
                waveform = waveform[0]
                
            return waveform
        except Exception as e:
            print(f"Error in Griffin-Lim algorithm: {e}")
            # Return zeros if Griffin-Lim fails
            print("Returning empty waveform")
            return np.zeros(num_samples, dtype=np.float32)

    def apply_audio_filter(self, samples, sample_rate):
        """Apply some basic audio filtering to improve audio quality"""
        try:
            from pedalboard import Pedalboard, LowpassFilter, HighpassFilter, Compressor
            
            # Convert to the format pedalboard expects
            if samples.ndim == 1:
                samples = samples.reshape(1, -1)  # mono -> [1, samples]
            
            # Create a pedalboard with some basic filters
            board = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=60),  # Remove rumble
                LowpassFilter(cutoff_frequency_hz=16000),  # Remove high-frequency noise
                Compressor(threshold_db=-20, ratio=3.0)  # Smooth out dynamics
            ])
            
            # Apply the pedalboard
            processed_samples = board(samples, sample_rate)
            
            # Convert back to the original shape
            if samples.ndim == 2 and samples.shape[0] == 1:
                processed_samples = processed_samples.reshape(-1)
                
            return processed_samples
        except Exception as e:
            print(f"Error applying audio filter: {e}")
            return samples

    def get_audio_from_spectrogram(self, spec, max_volume=50, quality_level=2, apply_filter=True):
        """Convert spectrogram to audio waveform and sample rate"""
        power_for_image = 0.25
        Sxx = self.spectrogram_from_image(spec, max_volume, power_for_image)
        
        sample_rate = 44100  # [Hz]
        clip_duration_ms = 5000  # [ms]
        
        bins_per_image = spec.height
        n_mels = spec.height
        
        # FFT parameters - adjusted for better quality
        window_duration_ms = 100  # [ms]
        padded_duration_ms = 400  # [ms]
        step_size_ms = 10  # [ms]
        
        # Derived parameters
        num_samples = int(spec.width / float(bins_per_image) * clip_duration_ms * sample_rate / 1000)
        n_fft = int(padded_duration_ms / 1000.0 * sample_rate)
        hop_length = int(step_size_ms / 1000.0 * sample_rate)
        win_length = int(window_duration_ms / 1000.0 * sample_rate)
        
        # Use CUDA if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if platform.system() == "Darwin":  # macOS
            device = "cpu"  # Force CPU on macOS
            
        print(f"Generating audio with max_volume={max_volume}, quality_level={quality_level}")
        print(f"Using device: {device}")
        print(f"Shapes - n_fft: {n_fft}, hop_length: {hop_length}, win_length: {win_length}, Sxx: {Sxx.shape}")
            
        # Generate the waveform
        samples = self.waveform_from_spectrogram(
            Sxx=Sxx,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            num_samples=num_samples,
            sample_rate=sample_rate,
            mel_scale=True,
            n_mels=n_mels,
            quality_level=quality_level,
            device=device
        )
        
        # Apply audio filtering if requested
        if apply_filter:
            samples = self.apply_audio_filter(samples, sample_rate)
        
        # Apply light normalization to prevent clipping
        if np.max(np.abs(samples)) > 0:
            normalization_factor = 0.95 / np.max(np.abs(samples))
            samples = samples * normalization_factor
            
        return samples, sample_rate

    def Process_Riffusion(self, spectrogram, filename_prefix="Riffusion_", save_to_file=False, 
                          max_volume=50.0, quality_level=2, apply_filter=True):
        """Convert a spectrogram image to audio"""
        # Process first spectrogram in batch 
        spec = None
        for (batch_number, image) in enumerate(spectrogram):
            if batch_number >= 1:  # Process only the first image
                break
            i = 255. * image.cpu().numpy()
            spec = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        if spec is None:
            print("Warning: No valid spectrogram found in batch")
            # Return empty audio
            empty_audio = {
                "waveform": torch.zeros((1, 1, 1000), dtype=torch.float32),
                "sample_rate": 44100
            }
            return (empty_audio,)
        
        # Convert spectrogram to audio with parameters
        samples, sample_rate = self.get_audio_from_spectrogram(
            spec, 
            max_volume=float(max_volume),
            quality_level=int(quality_level),
            apply_filter=apply_filter
        )
        
        # Format waveform for ComfyUI's AUDIO type
        samples = samples.astype(np.float32)
        
        # Create waveform tensor with shape [1, 1, samples]
        waveform = torch.from_numpy(samples).float().unsqueeze(0).unsqueeze(0)
        
        # ComfyUI expects a dictionary with waveform and sample_rate
        audio_output = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }
        
        # Save to disk if requested
        if save_to_file:
            try:
                # Get the current file's directory and go up two levels to output directory
                current_dir = Path(__file__).parent
                output_dir = current_dir.parent.parent / 'output'
                
                # Create the output directory if it doesn't exist
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Search output_dir for files with the filename prefix
                list_of_files = glob.glob(str(output_dir / f"{filename_prefix}*"))
                
                # Get the highest number in the list of files
                max_number = 0
                for file in list_of_files:
                    try:
                        number = int(file.split("_")[-1].split(".")[0])
                        max_number = max(max_number, number)
                    except (ValueError, IndexError):
                        continue
                
                # Create new filename with incremented number
                new_number = max_number + 1
                new_filename = output_dir / f"{filename_prefix}{new_number}.wav"
                
                # Save the audio file
                wavfile.write(str(new_filename), sample_rate, (samples * 32767).astype(np.int16))
                print(f"Saved audio to {new_filename}")
                
                # Optionally convert to MP3
                try:
                    mp3_filename = str(new_filename).replace(".wav", ".mp3")
                    
                    with AudioFile(new_filename) as f:
                        audio_data = f.read(f.frames)
                        sr = f.samplerate
                    
                    with AudioFile(mp3_filename, 'w', sr, audio_data.shape[0]) as f:
                        f.write(audio_data)
                    
                    print(f"Converted to MP3: {mp3_filename}")
                    
                except Exception as e:
                    print(f"Error converting to MP3: {e}")
                    
            except Exception as e:
                print(f"Error saving audio file: {e}")
        
        return (audio_output,)


class RiffusionToBatchNode:
    """Node to convert a single audio to a batch for further processing"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio_batch",)
    FUNCTION = "create_batch"
    CATEGORY = "audio"
    
    def create_batch(self, audio, batch_size=1):
        """Create a batch of copies of the input audio"""
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        # If batch_size is 1, just return the original
        if batch_size <= 1:
            return (audio,)
        
        # Create batch of identical audio
        batch = waveform.repeat(batch_size, 1, 1)
        
        return ({"waveform": batch, "sample_rate": sample_rate},)


NODE_CLASS_MAPPINGS = {
    "RiffusionNode": RiffusionNode,
    "RiffusionToBatchNode": RiffusionToBatchNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RiffusionNode": "Riffusion Audio Generator",
    "RiffusionToBatchNode": "Riffusion Audio Batch"
}
