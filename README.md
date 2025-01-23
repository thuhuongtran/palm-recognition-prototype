# palm-recognition-prototype
Develop a prototype system for palm-based authentication that can accurately identify individuals despite variations in hand pose and camera angle.

**Project Title:** Robust Palm-Based Authentication System using Deep Learning

**Project Goal:** Develop a prototype system for palm-based authentication that can accurately identify individuals despite variations in hand pose and camera angle.

**Steps:**

1. **Data Preprocessing (1-2 days):**

   1. **Load and Explore the Data**:
      - Use libraries like OpenCV or Pillow to load `.tiff` images.
      - Visualize images and inspect metadata for pose, angle, and subject ID.
      - Check for missing, corrupted, or imbalanced data.

   2. **Data Cleaning**:
      - Validate all images to ensure they are readable.
      - Cross-check metadata consistency with images.
      - Remove duplicate images to avoid data leakage.

   3. **Data Augmentation**:
      - Apply geometric transformations: rotation, flips, translations.
      - Use photometric adjustments: brightness, contrast, and Gaussian noise.
      - Add perspective transformations: shearing and warping.
      - Random cropping, resizing, and blurring for variability.

   4. **Normalize Images**:
      - Scale pixel values to [0, 1] or [-1, 1].
      - Standardize images using dataset mean and standard deviation.

   5. **Resize Images**:
      - Resize all images to a uniform size (e.g., 224x224) to match CNN requirements.
      - Maintain aspect ratio to avoid feature distortion.

   6. **Split the Dataset**:
      - Divide data into training (70%), validation (15%), and testing (15%) sets.
      - Use stratified sampling for balanced splits.

   7. **Convert Data to Tensors**:
      - Transform images into PyTorch tensors.
      - Use `DataLoader` for efficient batching and loading during training.

   8. **Label Encoding**:
      - Encode class labels (subject IDs) into numeric values.
      - Prepare pairs of images for verification tasks (1 for same subject, 0 for different).

   9. **Verify Preprocessing**:
      - Visualize preprocessed and augmented images to ensure correctness.
      - Confirm variability in the dataset after augmentation.

   10. **Balance the Dataset (Optional)**:
       - Address class imbalance with oversampling, undersampling, or synthetic data generation like SMOTE. 

2. **Model Selection and Training (2-3 days):**
    * Choose a Convolutional Neural Network (CNN) architecture suitable for image classification. Start with a simpler architecture like ResNet18 or MobileNetV2 for quicker training and experimentation. You could then explore more complex architectures if time permits.
    * Implement the model in PyTorch.
    * Train the model using the training data and monitor performance on the validation set. Experiment with different optimizers, loss functions, and learning rates.
    * Implement early stopping to prevent overfitting.

3. **Model Evaluation and Optimization (1-2 days):**
    * Evaluate the trained model on the test set using metrics like accuracy, precision, recall, and F1-score.
    * Analyze the model's performance on different views (frontal, rotated frontal, perspective, rotated perspective) to identify any weaknesses. This ties directly to the dataset's strength.
    * Fine-tune the model and hyperparameters to improve performance. Consider techniques like transfer learning (using pre-trained weights) if starting from scratch proves challenging within the time constraint.

4. **API Development (2 days):**
    * Create a simple RESTful API using Flask that accepts an image as input and returns the predicted user ID (or a similarity score if doing one-to-one matching instead of classification).
    * Dockerize the application for easy deployment.

5. **UI Development (1 day):**
    * Build a simple UI using Streamlit to showcase the project.  The UI should allow users to upload an image and see the authentication result.
    * Display some key metrics and visualizations from the model training and evaluation process (e.g., accuracy, loss curves, confusion matrix).  This demonstrates your understanding of model performance analysis.

6. **Deployment (1 day):**
    * Deploy the Streamlit app to a free platform like Streamlit Cloud, Hugging Face Spaces, or Render.

7. **Documentation and GitHub (1 day):**
    * Create a clear and concise README file on GitHub explaining the project, its goals, the technology stack, and how to run the application. Include instructions on how to use the API and the UI.
    * Document your code clearly with comments.

**Key Technologies:** Python, PyTorch, OpenCV, Flask, Docker, Streamlit, PostgreSQL (optional - could be used to simulate storing user data), Git.

**Deliverables:**

* GitHub repository with the project code and documentation.
* Deployed Streamlit app showcasing the working prototype.
* Short presentation or video demonstrating the project and highlighting your contributions.

---

## **Project Title**  
**Unsupervised Palm Feature Extraction and Verification Prototype**

---

## **High-Level Idea**  
1. **Learn a robust feature representation** (embedding) for palm images **without** labeled data, using either:
   - A **self-supervised method** (e.g., SimCLR, BYOL, or MoCo)  
   - A **deep autoencoder** (less ideal for true matching, but simpler to implement)  
2. **Extract embeddings** for each palm image.  
3. **Perform palm verification (1:1)** by comparing the **distance/similarity** between the embeddings of two images.  
   - Although we can’t definitively label them “same person” or “different person” without ground truth, we can demonstrate a typical verification pipeline.  
4. **Build a small demo** (e.g., Streamlit app) that:
   - Lets you **upload two palm images**.  
   - Shows how the system **computes** their embeddings and outputs a **similarity score** (or distance).  
   - Provides a **threshold** to decide if they “match.”

This approach is still **close to real-world** palm authentication, but uses a **self-supervised** or **unsupervised** technique to learn palm features from unlabeled images.

---

## **Revised Project Goals**

1. **Feature Learning (Unsupervised/Self-Supervised)**: Develop or adapt a CNN-based method to learn discriminative features from unlabeled palm images.  
2. **Verification Pipeline**: Demonstrate how to compare two palm images and compute a match score.  
3. **Prototype Deployment**: Provide a simple UI and RESTful API for demonstration purposes.

---

## **Project Timeline (Approx. 5–7 Days)**

### **Day 1: Data Handling & Cleaning**
- **Goal**: Ensure data is ready for model input.
- **Tasks**:
  1. **Load & Inspect** the palm images (already done in “Load and Explore” step).
  2. **Clean** the data (already done in “Data Cleaning” step).
  3. **Set up** augmentation pipelines (already done in “Data Augmentation” step).
  4. **Organize** final dataset structure:  
     - Possibly create “train” and “test” splits (though unlabeled, we can still separate for later analysis).

### **Day 2: Self-Supervised Feature Learning (Model Setup)**
- **Goal**: Implement or adapt a self-supervised model to learn embeddings from unlabeled images.
- **Tasks**:
  1. **Choose a framework**:
     - **Simple Option**: **Autoencoder** (easy to implement, though less ideal for matching accuracy).  
     - **More Advanced**: **SimCLR**, **BYOL**, or **MoCo** (requires more complex code but often yields better embeddings).
  2. **Implement** the model in **PyTorch** (or TensorFlow):
     - For an **autoencoder**: define encoder-decoder architecture.  
     - For a **contrastive** approach (e.g., SimCLR): define backbone (ResNet, MobileNet, etc.) and contrastive loss function.
  3. **Incorporate Data Augmentation** from your pipeline during training (important in self-supervised learning).

### **Day 3: Training & Embedding Extraction**
- **Goal**: Train the self-supervised model and save a feature extractor.
- **Tasks**:
  1. **Train** the model on your unlabeled palm dataset:
     - Keep an eye on **loss curves** and potential overfitting.  
     - For a contrastive method, ensure enough augmentation variety.
  2. **Extract Embeddings**:
     - Once training is done, freeze the encoder.
     - Generate embeddings for the images in a **test** or **hold-out** set.
  3. **Evaluate Embeddings** (qualitatively):
     - Optionally apply **dimensionality reduction** (e.g., t-SNE/UMAP) to visualize grouping/clustering of palm images.
     - We can’t measure classification accuracy (no labels), but we can check if embeddings look consistent (e.g., multiple images of the same palm look close, if you happen to have some repeated images under different angles—if that info is known or can be guessed).

### **Day 4: Verification (1:1) Mechanism**
- **Goal**: Show a typical verification step using learned embeddings.
- **Tasks**:
  1. **Implement a function** that given two images:
     - Loads each image.
     - Passes them through the **frozen encoder** to get embeddings.
     - Computes **similarity** (e.g., cosine similarity) or **distance** (e.g., Euclidean).  
  2. **Threshold Tuning (Optional)**:
     - In reality, you’d use labeled pairs of “same person” vs. “different person” to set a decision threshold. Without labels, you can pick a threshold that “feels” right based on typical distances or by looking at the distribution of distances on randomly chosen pairs.
  3. **Demonstrate**:
     - If the distance < threshold, declare “Match.” Otherwise, “No Match.”
  4. **Analysis**:
     - Possibly pick random image pairs and see how the distance behaves.  
     - If you do have any repeated images (like same file appearing in the dataset multiple times), test them to see if the system yields a high similarity, indicating it can match them as identical.

### **Day 5: API & UI Integration**
- **Goal**: Provide a demonstration interface for your unsupervised approach.
- **Tasks**:
  1. **Flask (API)**:
     - Create an endpoint `/verify` that accepts two images and returns:
       - Embedding distance or similarity.
       - A “match” or “no match” boolean, depending on the threshold.
  2. **Streamlit (UI)**:
     - Allow the user to **upload two images** via a simple web interface.
     - On submission, the images are processed:
       - Show the numeric distance or similarity score.
       - Print whether it’s above/below the threshold (indicating “match” or “no match”).
  3. **Dockerize** (optional but good practice) for easier deployment.

### **Day 6: Testing & Deployment**
- **Goal**: Ensure everything is functional and publicly accessible (if desired).
- **Tasks**:
  1. **Local Tests**:  
     - Run through multiple pairs of images.
     - Confirm the system responds quickly and consistently.
  2. **Deployment** (e.g., Streamlit Cloud, Hugging Face Spaces):
     - Deploy your **Streamlit** app so others can test it.  
  3. **Troubleshoot** any performance or dependency issues.

### **Day 7: Documentation & Wrap-Up**
- **Goal**: Finalize the project artifacts.
- **Tasks**:
  1. **README** on GitHub:  
     - Explain the approach (self-supervised or autoencoder).  
     - Summarize installation, usage, and the UI usage.  
  2. **Presentation** or short video demo:
     - Show your app in action.  
     - Mention the limitations (no labeled data, approximate threshold).  
  3. **Future Work** (if time permits):
     - Possibly incorporate a small labeled subset (if you can gather it manually) to do a real accuracy test or threshold tuning.

---

## **Revised Deliverables**

1. **GitHub Repository** containing:
   - **Data Preprocessing Code** (loading, cleaning, augmentation).  
   - **Model Training Script** (autoencoder or contrastive method).  
   - **Embedding Extraction & Verification** code.  
   - **Documentation** explaining how to run everything.

2. **Deployed Streamlit App**:
   - Minimal “upload two images → see match score” demonstration.

3. **Short Presentation / Video**:
   - Demonstrates the concept of unsupervised feature learning and verification with a threshold.


#### STEPS
1. Download dataset
2. Save to dataset/palmprint_data
3. Covert from JPG to png
   find dataset/palmprint_data -type d -name '[0-9][0-9][0-9]' -exec sh -c 'cd "$0" && mogrify -format png *.JPG' {} \;

https://www.kaggle.com/datasets/saqibshoaibdz/palm-dataset/data
https://chatgpt.com/c/67915828-2634-800a-b8b2-df6cc94b5248?model=o1

Need to learn about the palm regconition authentication process, how vector layer is extracted