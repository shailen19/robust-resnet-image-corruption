from filters import darken, gausblur, gausnoise
import numpy as np

def image_corruptor(image, corruption_type, severity):
    corruption_map ={
        "dark": lambda image: darken(image, gamma=severity),
        "blur": lambda image: gausblur(image, severity),
        "noise": lambda image: gausnoise(image, std = severity, mean = 0) 
    }

    if corruption_type not in  corruption_map:
        raise valueError("Invalid corruption type\nPlease choose: dark, blur, or noise\n")
    
    return corruption_map[corruption_type](image)



def corrupt_image_dataset(images, corruption_type, severity):
    corrupted_images = []

    for img in images:
        corrupted_img = image_corruptor(img, 
        corruption_type, 
        severity
        )
        corrupted_images.append(corrupted_img)
    return np.array(corrupted_images)