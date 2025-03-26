import os
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageToImage
from diffusers import StableDiffusionImg2ImgPipeline, DiffusionPipeline, DDIMScheduler

class AdvancedImageAugmenter:
    def __init__(self, model_id="stabilityai/stable-diffusion-2-1", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load image-to-image pipeline from HuggingFace
        print(f"Loading model: {model_id}")
        try:
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self.pipe = self.pipe.to(self.device)
            self.use_diffusion = True
        except Exception as e:
            print(f"Warning: Could not load diffusion model: {e}")
            print("Falling back to traditional augmentation methods only")
            self.use_diffusion = False
        
        # Basic transform for preprocessing
        self.basic_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        
        # Standard augmentation transforms for satellite imagery
        # Fixed: Added ToTensor() at the beginning and properly handle the transform sequence
        self.std_augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),  # Satellite images don't have fixed orientation
            transforms.RandomRotation(30),  # Increased rotation range
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Reduced for satellite spectral fidelity
            # No ToPILImage here - we'll handle conversion separately
        ])
        
        print("Model loaded successfully")

    def generate_satellite_prompt(self, category):
        """Generate appropriate satellite/aerial prompts based on the image category."""
        prompts = {
            "cloudy": [
                "satellite image of cloud formations over landscape",
                "aerial view of cloudy weather patterns",
                "satellite weather imagery showing cloud cover",
                "remote sensing image of atmospheric cloud formations",
                "earth observation satellite image of cloud systems"
            ],
            "desert": [
                "satellite view of arid desert terrain",
                "aerial image of desert landscape from space",
                "remote sensing data of desert topography",
                "satellite imagery of sandy desert regions",
                "earth observation of desert landforms"
            ],
            "green_area": [
                "satellite image of green forest coverage",
                "aerial view of vegetation and forests",
                "remote sensing data of agricultural fields and vegetation",
                "satellite view of green spaces and parks",
                "earth observation imagery of forest regions"
            ],
            "water": [
                "satellite imagery of water bodies and coastlines",
                "aerial view of lakes and rivers from space",
                "remote sensing data of ocean currents and water features",
                "satellite view of reservoirs and waterways",
                "earth observation imagery of coastal waters"
            ]
        }
        
        # Default satellite prompts if category not found
        default_prompts = ["satellite imagery of earth surface", "aerial view from space", "remote sensing data"]
        
        category_prompts = prompts.get(category.lower(), default_prompts)
        return random.choice(category_prompts)

    def augment_image_diffusion(self, image_path, category, strength=0.5):
        """Augment satellite image using diffusion model with category-specific prompts."""
        if not hasattr(self, 'use_diffusion') or not self.use_diffusion:
            return None
            
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            # Generate appropriate satellite prompt
            prompt = self.generate_satellite_prompt(category)
            # Add satellite-specific qualifiers
            enhanced_prompt = f"high-resolution {prompt}, detailed satellite imagery, nadir view"
            
            # Run diffusion model with adjusted strength for satellite imagery
            output = self.pipe(
                prompt=enhanced_prompt,
                image=image,
                strength=strength,  # Lower strength to maintain satellite image characteristics
                guidance_scale=7.0,
                num_inference_steps=30,
            ).images[0]
            
            return output
        
        except Exception as e:
            print(f"Error augmenting {image_path} with diffusion: {e}")
            return None

    def augment_image_traditional(self, image_path):
        """Apply traditional augmentations using torchvision transforms."""
        try:
            # Load image and convert to tensor
            image = Image.open(image_path).convert("RGB")
            tensor_image = transforms.ToTensor()(image)
            
            # Apply augmentations to tensor
            augmented_tensor = self.std_augment(tensor_image)
            
            # Convert back to PIL image
            augmented_image = transforms.ToPILImage()(augmented_tensor)
            return augmented_image
            
        except Exception as e:
            print(f"Error applying traditional augmentation to {image_path}: {e}")
            return None

    def augment_satellite_specific(self, image_path):
        """Apply satellite-specific augmentations."""
        try:
            image = Image.open(image_path).convert("RGB")
            img_array = np.array(image).astype(np.float32) / 255.0
            
            # Random adjustments to simulate different spectral bands
            r_adjust = np.random.uniform(0.9, 1.1)
            g_adjust = np.random.uniform(0.9, 1.1)
            b_adjust = np.random.uniform(0.9, 1.1)
            
            img_array[:,:,0] *= r_adjust
            img_array[:,:,1] *= g_adjust
            img_array[:,:,2] *= b_adjust
            
            # Add slight noise to simulate sensor variations
            noise_level = np.random.uniform(0.01, 0.03)
            noise = np.random.normal(0, noise_level, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 1)
            
            # Convert back to PIL Image
            result_image = Image.fromarray((img_array * 255).astype(np.uint8))
            return result_image
            
        except Exception as e:
            print(f"Error applying satellite-specific augmentation to {image_path}: {e}")
            return None

    def process_folder(self, input_folder, output_folder, augmentation_type="all", 
                       num_augmentations=3, strength=0.5):
        """Process all images in a folder and save augmented versions."""
        os.makedirs(output_folder, exist_ok=True)
        
        # Get the category name from the folder path
        category = os.path.basename(input_folder)
        
        image_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        print(f"Processing {len(image_files)} satellite images in {category}...")
        
        for img_file in tqdm(image_files):
            input_path = os.path.join(input_folder, img_file)
            
            # Generate multiple augmentations per image
            for i in range(num_augmentations):
                base_name = os.path.splitext(img_file)[0]
                ext = os.path.splitext(img_file)[1]
                output_path = os.path.join(output_folder, f"{base_name}_aug{i}{ext}")
                
                # Apply selected augmentation type using a rotation strategy
                augmented = None
                
                # Determine which augmentation to use based on the augmentation type and iteration
                if augmentation_type == "traditional" or (augmentation_type == "all" and i % 3 == 0):
                    augmented = self.augment_image_traditional(input_path)
                
                elif augmentation_type == "satellite" or (augmentation_type == "all" and i % 3 == 1):
                    augmented = self.augment_satellite_specific(input_path)
                
                elif (augmentation_type == "diffusion" or (augmentation_type == "all" and i % 3 == 2)) and hasattr(self, 'use_diffusion') and self.use_diffusion:
                    augmented = self.augment_image_diffusion(input_path, category, strength)
                
                # Fallback if the selected augmentation failed
                if augmented is None:
                    augmented = self.augment_image_traditional(input_path)
                    # If even that fails, skip this iteration
                    if augmented is None:
                        continue
                
                augmented.save(output_path)
        
        print(f"Finished processing {category}")

def main():
    parser = argparse.ArgumentParser(description="Advanced Satellite Image Augmentation")
    parser.add_argument("--input_dir", type=str, default=".", 
                        help="Base directory containing category folders")
    parser.add_argument("--output_dir", type=str, default="./augmented",
                        help="Output directory for augmented images")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-1",
                        help="HuggingFace model ID for image augmentation")
    parser.add_argument("--num_augmentations", type=int, default=3,
                        help="Number of augmented versions to create per image")
    parser.add_argument("--augmentation_type", type=str, default="all", 
                        choices=["diffusion", "traditional", "satellite", "all"],
                        help="Type of augmentation to apply")
    parser.add_argument("--strength", type=float, default=0.5,
                        help="Strength of diffusion transformation (0.0-1.0)")
    parser.add_argument("--categories", nargs='+', 
                        default=["cloudy", "desert", "green_area", "water"],
                        help="Categories/folders to process")
    
    args = parser.parse_args()
    
    augmenter = AdvancedImageAugmenter(model_id=args.model_id)
    
    for category in args.categories:
        input_folder = os.path.join(args.input_dir, category)
        output_folder = os.path.join(args.output_dir, category)
        
        if os.path.isdir(input_folder):
            augmenter.process_folder(
                input_folder, 
                output_folder,
                augmentation_type=args.augmentation_type,
                num_augmentations=args.num_augmentations,
                strength=args.strength
            )
        else:
            print(f"Warning: Category folder {input_folder} not found.")

if __name__ == "__main__":
    main()