import torch
import os
import sys
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.backends import default_backend
from scipy.stats import norm
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import logging
from datetime import datetime
from scipy.stats import norm, kstest

class Loggers:
    _logger = None

    @classmethod
    def get_logger(cls, log_dir: str = './logs') -> 'logging.Logger':
        if cls._logger is None:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_file = os.path.join(log_dir, f'{datetime.now().strftime("%Y-%m-%d-%H-%M")}.log')
            cls._logger = logging.getLogger('DPRW_Engine')
            cls._logger.setLevel(logging.INFO)
            cls._logger.handlers.clear()
            formatter = logging.Formatter("%(asctime)s %(levelname)s: [%(name)s] %(message)s")
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            console_handler = logging.StreamHandler()
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            cls._logger.addHandler(file_handler)
            cls._logger.addHandler(console_handler)
        return cls._logger

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

MAX_RESOLUTION = 8192

def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def choose_watermark_length(total_blocks_needed: int) -> int:
    if total_blocks_needed >= 256 * 32:
        return 256
    elif total_blocks_needed >= 128 * 32:
        return 128
    elif total_blocks_needed >= 64 * 32:
        return 64
    return 32

def validate_hex(hex_str: str, expected_length: int, default: bytes) -> bytes:
    if hex_str and len(hex_str) == expected_length and all(c in '0123456789abcdefABCDEF' for c in hex_str):
        return bytes.fromhex(hex_str)
    return default

def common_ksampler(model, seed: int, steps: int, cfg: float, sampler_name: str, scheduler: str, positive, negative, latent,
                    denoise: float = 1.0, disable_noise: bool = False, start_step = None, last_step= None,
                    force_full_denoise: bool = False, use_dprw: bool = False, watermarked_latent_noise=None):
    latent_image = latent["samples"]
    if latent_image.shape[1] == 4:
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if use_dprw and watermarked_latent_noise is not None:
        noise = watermarked_latent_noise["samples"]
    elif disable_noise:
        noise = torch.zeros_like(latent_image, device="cpu")
    else:
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = latent.get("noise_mask", None)
    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback,
                                  disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out,)

def Gaussian_test(noise: torch.Tensor,logger) -> bool:
        if isinstance(noise, torch.Tensor):
            noise = noise.cpu().numpy()
        samples = noise.flatten()
        _, p_value = kstest(samples, 'norm', args=(0, 1))
        if np.isnan(samples).any() or np.isinf(samples).any():
            raise ValueError("Restored noise contains NaN or Inf values")
        if np.var(samples) == 0:
            raise ValueError("Restored noise variance is 0")
        if p_value < 0.05:
            raise ValueError(f"Restored noise failed Gaussian test (p={p_value:.4f})")
        logger.info(f"Gaussian test passed: p={p_value:.4f}")
        return True

class GSWatermark:
    def __init__(self, key_hex: str, nonce_hex: str, device: str = "cuda", use_seed: bool = True, seed: int = 42, 
    watermark_length_bits:int=64,latent_channels: int = 4,log_dir: str = './logs'):
        self.key = validate_hex(key_hex, 64, os.urandom(32))
        self.nonce = validate_hex(nonce_hex, 32, os.urandom(16))
        self.device = device
        self.use_seed = use_seed
        self.seed = seed
        self.watermark_length_bits=watermark_length_bits
        self.channels = latent_channels
        self.logger = Loggers.get_logger(log_dir)
        self.logger.info(f"====================GS Watermark Initialized Begin====================")
        self.logger.info(f"Initialized - Key: {self.key.hex()}")
        self.logger.info(f"Initialized - Nonce: {self.nonce.hex()}")
        self.logger.info(f"Initialized - Channels: {self.channels}")
        self.logger.info(f"====================GS Watermark Initialized End====================")

    def _create_watermark(self, total_blocks: int, message: str, message_length: int) -> bytes:
        length_bits = message_length if message_length > 0 else choose_watermark_length(total_blocks)
        length_bytes = length_bits // 8
        msg_bytes = message.encode('utf-8')
        padded_msg = msg_bytes.ljust(length_bytes, b'\x00')[:length_bytes]
        repeats = total_blocks // length_bits
        self.logger.info(f"Create watermark - Message: {message}")
        self.logger.info(f"Create watermark - Message Length: {message_length}")
        self.logger.info(f"Create watermark - Watermark repeats: {repeats} times")
        return padded_msg * repeats + b'\x00' * ((total_blocks % length_bits) // 8)

    def init_noise(self, message: str, width: int, height: int,window_size:int=1) -> torch.Tensor:
        self.message=message
        width_blocks = width // 8
        height_blocks = height // 8
        total_blocks_needed = self.channels * width_blocks * height_blocks

        watermark = self._create_watermark(total_blocks_needed, message, self.watermark_length_bits)

        cipher = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None, backend=default_backend())
        encryptor = cipher.encryptor()
        m = encryptor.update(watermark) + encryptor.finalize()
        m_bits = ''.join(format(byte, '08b') for byte in m)

        index = 0
        Z_s_T_array = torch.zeros((self.channels, height_blocks, width_blocks), dtype=torch.float32, device=self.device)
        if self.use_seed:
            rng = np.random.RandomState(seed=self.seed)

        for i in range(0, len(m_bits), window_size):
            window = m_bits[i:i+window_size]
            y = int(window, 2)
            if not self.use_seed:
                u = np.random.uniform(0, 1)
            else:
                u = rng.uniform(0, 1)
            z_s_T = norm.ppf((u + y) / (2 ** window_size))
            channel_index = index // (height_blocks * width_blocks)
            h_index = (index // width_blocks) % height_blocks
            w_index = index % width_blocks
            Z_s_T_array[channel_index, h_index, w_index] = z_s_T
            index += 1
            if index >= self.channels * height_blocks * width_blocks:
                break
        Gaussian_test(Z_s_T_array,self.logger)
        return Z_s_T_array
    
    def extract_watermark(self,reversed_latents,message_length,window_size):
        # initiate the Cipher
        cipher = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None, backend=default_backend())
        decryptor = cipher.decryptor()
        
        # Reconstruct m from reversed_latents
        reconstructed_m_bits = []
        for z_s_T_value in np.nditer(reversed_latents):
            y_reconstructed = norm.cdf(z_s_T_value) * 2**window_size
            reconstructed_m_bits.append(int(y_reconstructed))

        # m_reconstructed_bytes = bytes(int(''.join(str(bit) for bit in reconstructed_m_bits[i:i+8]), 2) for i in range(0, len(reconstructed_m_bits), 8))
        m_reconstructed_bytes = bytes(
            int(''.join(str(int(bit) % 2) for bit in reconstructed_m_bits[i:i+8]), 2)
            for i in range(0, len(reconstructed_m_bits), 8)
        )
        s_d_reconstructed = decryptor.update(m_reconstructed_bytes) + decryptor.finalize()
        bits_list = ['{:08b}'.format(byte) for byte in s_d_reconstructed]
        all_bits = ''.join(bits_list)

        # segments = [all_bits[i:i + message_length] for i in range(0, len(all_bits), message_length)]
        # reconstructed_message_bin = ''

        # for i in range(message_length):
        #     count_1 = sum(segment[i] == '1' for segment in segments)
        #     reconstructed_message_bin += '1' if count_1 > len(segments) / 2 else '0'
        segments = [seg for seg in (all_bits[i:i + message_length] for i in range(0, len(all_bits), message_length)) if len(seg) == message_length]
        reconstructed_message_bin = ''
        for i in range(message_length):
            count_1 = sum(segment[i] == '1' for segment in segments)
            reconstructed_message_bin += '1' if count_1 > len(segments) / 2 else '0'
        msg = bytes(int(reconstructed_message_bin[i:i + 8], 2) for i in range(0, len(reconstructed_message_bin), 8)).decode('utf-8', errors='replace')
        return reconstructed_message_bin,msg
    
    def evaluate_accuracy(self, original_msg: str, extracted_bin: str, extracted_msg_str:str="") -> float:
        self.logger.info(f"====================GS Watermark Evaluate Begin====================")
        orig_bin = bin(int(original_msg.encode('utf-8').hex(), 16))[2:].zfill(len(original_msg) * 8)
        min_len = min(len(orig_bin), len(extracted_bin))
        orig_bin, extracted_bin = orig_bin[:min_len], extracted_bin[:min_len]
        accuracy = sum(a == b for a, b in zip(orig_bin, extracted_bin)) / min_len
        self.logger.info(f"Evaluation - Original binary: {orig_bin}")
        self.logger.info(f"Evaluation - Extracted binary: {extracted_bin}")
        self.logger.info(f"Evaluation - Extracted binary length: {len(extracted_bin)}")
        if accuracy > 0.9:
            self.logger.info(f"Evaluation - Extracted message: {extracted_msg_str}")
        self.logger.info(f"Evaluation - Bit accuracy: {accuracy}")
        self.logger.info(f"====================GS Watermark Evaluate End====================")
        return accuracy

class GSLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "use_seed": ("INT", {"default": 1, "min": 0, "max": 1}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff}),
            "channels": ("INT", {"default": 4, "min": 4, "max": 16,"step":12}),
            "Image_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "Image_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "key": ("STRING", {"default": "5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7"}),
            "nonce": ("STRING", {"default": "05072fd1c2265f6f2e2a4080a2bfbdd8"}),
            "message": ("STRING", {"default": "lthero"}),
            "window_size": ("INT", {"default": 1, "min": 1, "max": 100}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
         }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "create_gs_latents"
    CATEGORY = "DPRW/latent"
    
    def create_gs_latents(self, key, nonce, message, batch_size, use_seed, seed, Image_width, Image_height,channels,window_size):
        device = "cuda"
        message_length=len(message)*8
        gs_watermark = GSWatermark(key_hex=key, nonce_hex=nonce, device=device, use_seed=bool(use_seed), seed=seed, 
                            watermark_length_bits=message_length,latent_channels=channels)
        latent_list = [gs_watermark.init_noise(message, Image_width, Image_height,window_size) for _ in range(batch_size)]
        latent = torch.stack(latent_list)
        return ({"samples": latent},)


class DPRExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("LATENT",),
                "key": ("STRING", {"default": "5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7"}),
                "nonce": ("STRING", {"default": "05072fd1c2265f6f2e2a4080a2bfbdd8"}),
                "message": ("STRING", {"default": "lthero"}),
                "window_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                "watermarkMethod": ("STRING", {"default": "DPRW","options":["GS","DPRW"]}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING","LATENT")
    FUNCTION = "extract"
    CATEGORY = "DPRW/extractor"

    def extract(self, latents, key, nonce,message, window_size,watermarkMethod):
        if not isinstance(latents, dict) or "samples" not in latents:
            raise ValueError("latents must be a dictionary containing 'samples' key")
        
        noise = latents["samples"]
        if watermarkMethod == "DPRW":
            dprw = DPRWatermark(key, nonce,latent_channels=noise.shape[1])
            message_length = len(message) * 8
            extracted_msg_bin, extracted_msg_str = dprw.extract_watermark(noise, message_length, window_size)
            dprw.evaluate_accuracy(message, extracted_msg_bin,extracted_msg_str)
        elif watermarkMethod == "GS":
            gs = GSWatermark(key, nonce,latent_channels=noise.shape[1])
            message_length = len(message) * 8
            extracted_msg_bin, extracted_msg_str = gs.extract_watermark(noise, message_length, window_size)
            gs.evaluate_accuracy(message, extracted_msg_bin,extracted_msg_str)
        return (extracted_msg_bin, extracted_msg_str,latents)

class DPRKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "use_dprw_noise": (["enable", "disable"],),
                "add_noise": (["enable", "disable"],),
                "noise_seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "watermarked_latent_noise": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "DPRW/sampling"
    
    def sample(self, model, use_dprw_noise, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative,
               latent_image, watermarked_latent_noise, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        force_full_denoise = return_with_leftover_noise != "enable"
        use_dprw = use_dprw_noise == "enable"
        disable_noise = add_noise == "disable"
        return common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                               denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step,
                               force_full_denoise=force_full_denoise, use_dprw=use_dprw, watermarked_latent_noise=watermarked_latent_noise)

NODE_CLASS_MAPPINGS = {
    "DPR_Extractor": DPRExtractor,
    "DPR_KSamplerAdvanced": DPRKSamplerAdvanced,
    "DPR_GS_Latent": GSLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DPR_Extractor": "DPR Extractor",
    "DPR_KSamplerAdvanced": "DPR KSampler Advanced",
    "DPR_GS_Latent": "DPR GS Latent Noise",
}
