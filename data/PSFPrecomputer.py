import numpy as np
from typing import Tuple
import parameters as params

class PSFPrecomputer:
    def __init__(
        self,
        f_lambda:           np.ndarray  = params.f_lambda,          # [f_R, f_G, f_B] in meters   
        wavelengths:        np.ndarray  = params.wavelength,        # [λ_R, λ_G, λ_B] in meters
        z_img:              float       = params.z_img,             # z_i in meters
        aperture_diameter:  float       = params.aperture_diameter, # aperture diameter in meters
        psf_res:            int         = params.psf_res            # N x N
    ):
        self.f_lambda       = f_lambda
        self.wavelengths    = wavelengths
        self.z_img          = z_img
        self.R              = aperture_diameter / 2
        self.N              = psf_res

        # storage
        self.psf_bank   = None
        self.z_vals     = None

        # channel map
        self.channel_map = {'r': 0, 'g': 1, 'b': 2}

    def compute_defocus_parameter(self, z: float, f_lambda: float) -> float:
        delta_D = (1 / z + 1 / self.z_img - 1 / f_lambda)
        return delta_D

    def generate_generalized_pupil_function(self, delta_D: float, lmd: float) -> np.ndarray:
        N, R = self.N, self.R
        grid = np.linspace(-R, R, N)
        s, t = np.meshgrid(grid, grid)
        aperture = (s**2 + t**2 <= R**2).astype(np.float32)
        phase = np.exp(1j * np.pi * delta_D * (s**2 + t**2) / lmd)
        return aperture * phase

    def compute_psf_from_pupil(self, Q: np.ndarray) -> np.ndarray:
        fft = np.fft.fftshift(np.fft.fft2(Q))   # F{Q}
        psf = np.abs(fft) ** 2                  # PSF(x,y)=∣F{Q}∣²
        psf /= psf.sum()                        # normalize PSF
        return psf.astype(np.float32)           

    def compute_psf_bank(self, z_range: Tuple[float, float], z_step: float) -> Tuple[np.ndarray, np.ndarray]:
        z_vals = np.round(np.arange(z_range[0], z_range[1] + z_step, z_step), 2)
        psf_list = []

        for z in z_vals:
            psfs = []
            for i in range(3):  # R, G, B
                f_lmd = self.f_lambda[i]
                lmd = self.wavelengths[i]
                delta_D = self.compute_defocus_parameter(z, f_lmd)
                Q = self.generate_generalized_pupil_function(delta_D, lmd)
                psf = self.compute_psf_from_pupil(Q)
                psfs.append(psf)
            psf_list.append(np.stack(psfs))  # shape: (3, H, W)

        self.z_vals = z_vals
        self.psf_bank = np.stack(psf_list)  # shape: (num_z, 3, H, W)
        return self.z_vals, self.psf_bank

    def save_psf_bank(self, path: str):
        if self.z_vals is None or self.psf_bank is None:
            raise ValueError("PSF bank is empty. Run compute_psf_bank() first.")
        np.savez_compressed(path, psfs=self.psf_bank, z_vals=self.z_vals)

    def load_psf_bank(self, path: str):
        data = np.load(path)
        self.z_vals = data["z_vals"]
        self.psf_bank = data["psfs"]

    def compute_single_psf(self, z: float) -> dict:
        psfs = []
        for i in range(3):  # R, G, B
            f_lmd = self.f_lambda[i]
            lmd = self.wavelengths[i]
            delta_D = self.compute_defocus_parameter(z, f_lmd)
            Q = self.generate_generalized_pupil_function(delta_D, lmd)
            psf = self.compute_psf_from_pupil(Q)
            psfs.append(psf)
        return {
            'r': psfs[0],
            'g': psfs[1],
            'b': psfs[2]
        }

    def __getitem__(self, z: float) -> dict:
        if self.psf_bank is not None and self.z_vals is not None:
            idx = np.where(np.isclose(self.z_vals, z, atol=1e-6))[0]
            if len(idx) > 0:
                psfs = self.psf_bank[idx[0]]
                # print(f"Precomputed PSF for z={z} from bank.")
                return {
                    'r': psfs[0], 
                    'g': psfs[1], 
                    'b': psfs[2]
                }
        # Fallback
        return self.compute_single_psf(z) ; print(f"Computed PSF for z={z} on-the-fly.")    

if __name__ == "__main__":
    precomp = PSFPrecomputer()

    z_near      = params.z_near     # nearest object depth in meters
    z_far       = params.z_far     # farthest object depth in meters

    z_range = (z_near, z_far)
    z_step = 0.01
    z_vals, psf_bank = precomp.compute_psf_bank(z_range, z_step)
    print(f"Computed PSF bank for z range {z_range} with step {z_step}.")
    print(f"Number of z values: {len(z_vals)}")
    print(f"PSF shape: {psf_bank.shape}")  # should be (num_z, 3, H, W)
    precomp.save_psf_bank("data/psf_bank.npz")
    print("PSF bank saved to psf_bank.npz")