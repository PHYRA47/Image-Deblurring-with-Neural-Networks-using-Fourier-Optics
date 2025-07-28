import numpy as np

aperture_diameter   = 7e-3
f_green             = 35e-3         # focal length of the lens for green channel
z_obj               = 2             # object to lens distance in meters
z_img               = (1 / f_green - 1 / z_obj) ** (-1)     # lens to image distance in meters (focal length)      

wavelength    = np.array([630.0e-9, 525.0e-9, 458.0e-9])    # [λ_R, λ_G, λ_B] in meters
n_lambda      = np.array([1.4571, 1.4610, 1.4650])          # [n_R, n_G, n_B] refractive indices
n_green       = n_lambda[1]                                 # n_G
f_lambda      = f_green * (n_green - 1) / (n_lambda - 1)    # Chromatic focal length
delta_s       = 50e-6

psf_res     = 256

# lens spatial sampling interval in meters
z_near      = 1.84  # nearest object depth in meters
z_far       = 2.20  # farthest object depth in meters


