import numpy as np

class DataAugmentor:
    def add_noise_gaussian(self, image: np.ndarray) -> np.ndarray: pass
    def add_noise_sp(self, image: np.ndarray) -> np.ndarray: pass
    def add_shadow(self, image: np.ndarray) -> np.ndarray: pass
    def add_rotation(self, image: np.ndarray) -> np.ndarray: pass
    def warp_cylinder(self, image: np.ndarray) -> np.ndarray: pass # Giả lập cong