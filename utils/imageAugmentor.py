import random 
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2

class ImageAugmentor:
    """Classe pour augmenter les images de factures"""
    
    def __init__(self, seed=42):
        self.seed = seed
        random.seed(seed)
    
    def rotate(self, image, angle_range=(-10, 10)):
        """Rotation aléatoire (scan non parfait)"""
        angle = random.uniform(*angle_range)
        return image.rotate(angle, fillcolor='white', expand=False)
    
    def add_noise(self, image, intensity=0.02):
        """Ajout de bruit gaussien (qualité scan/photo)"""
        img_array = np.array(image).astype(np.float32)
        noise = np.random.normal(0, intensity * 255, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    
    def adjust_brightness(self, image, factor_range=(0.8, 1.2)):
        """Ajustement luminosité (conditions éclairage)"""
        factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def adjust_contrast(self, image, factor_range=(0.8, 1.2)):
        """Ajustement contraste (qualité impression)"""
        factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def add_blur(self, image, radius_range=(0.5, 2.0)):
        """Ajout de flou (photo non nette)"""
        radius = random.uniform(*radius_range)
        return image.filter(ImageFilter.GaussianBlur(radius))
    
    
    def perspective_transform(self, image, intensity=0.1):
        """Transformation perspective (photo smartphone)"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Points source (coins de l'image)
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Points destination (légèrement déformés)
        offset = int(min(h, w) * intensity)
        dst_points = np.float32([
            [random.randint(0, offset), random.randint(0, offset)],
            [w - random.randint(0, offset), random.randint(0, offset)],
            [w - random.randint(0, offset), h - random.randint(0, offset)],
            [random.randint(0, offset), h - random.randint(0, offset)]
        ])
        
        # Appliquer transformation
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(img_array, matrix, (w, h), 
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(255, 255, 255))
        return Image.fromarray(warped)
    
    def augment_pipeline(self, image, num_transforms=3):
        """
        Pipeline complet d'augmentation avec transformations aléatoires
        
        Args:
            image: Image PIL
            num_transforms: Nombre de transformations à appliquer
        """
        transforms = [
            self.rotate,
            self.add_noise,
            self.adjust_brightness,
            self.adjust_contrast,
            self.add_blur,
            self.perspective_transform
        ]
        
        # Sélectionner aléatoirement num_transforms
        selected = random.sample(transforms, min(num_transforms, len(transforms)))
        
        augmented = image.copy()
        for transform in selected:
            augmented = transform(augmented)
        
        return augmented
