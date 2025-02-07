import os
import numpy as np

from preprocess.preprocessor import preprocess_image

original_images_dirs = {
    "session1": '../dataset/archive/session1',
    "session2": '../dataset/archive/session2'
}
preprocessed_images_dir = '../dataset/preprocessed_images'
os.makedirs(preprocessed_images_dir, exist_ok=True)


def batch_preprocess():
    for session_name, original_images_dir in original_images_dirs.items():
        image_files = [f for f in os.listdir(original_images_dir) if
                       f.lower().endswith(('.tiff', '.tif'))]

        for image_file in image_files:
            image_path = os.path.join(original_images_dir, image_file)
            preprocessed_roi = preprocess_image(image_path)

            if preprocessed_roi is not None:
                unique_name = f"{session_name}_{image_file.replace('.tiff', '.npy').replace('.tif', '.npy')}"
                save_path = os.path.join(preprocessed_images_dir, unique_name)
                np.save(save_path, preprocessed_roi)
                print(f"Preprocessed and saved: {image_file} -> {save_path}")
            else:
                print(f"Preprocessing failed for: {image_file}")

    print("Preprocessing complete!")


batch_preprocess()
