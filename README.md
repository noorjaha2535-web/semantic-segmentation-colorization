# semantic-segmentation-colorization
## Author
**Name:** Mehkaan Anjum  
**Company:** Elevance Skills
# Task 5: Semantic Segmentation for Targeted Colorization 

---

## Project Objective
The objective of this task is to implement **semantic segmentation–based targeted colorization**, where the user can selectively apply color to either the **foreground or background** of an image using a graphical interface.

---

## Project Description
This project uses a **pretrained DeepLabV3 semantic segmentation model** to identify meaningful regions in an image. Based on user selection, color is applied only to the chosen region while keeping the remaining areas unchanged.  
A simple **GUI** is provided to upload images and view results instantly.

---

## Technologies Used
- Python  
- PyTorch  
- Torchvision  
- Gradio  
- NumPy  
- PIL (Python Imaging Library)  

---

## Step-by-Step Working

### Step 1: Image Input
The user uploads an image (JPG or PNG) through the graphical interface.  
Both color and grayscale images are supported.

---

### Step 2: Image Preprocessing
The uploaded image is resized and normalized so that it can be processed correctly by the deep learning model.

---

### Step 3: Semantic Segmentation
A pretrained **DeepLabV3 ResNet101** model is used to perform semantic segmentation.  
The model identifies the **foreground (person)** region using COCO dataset class labels.

---

### Step 4: Region Selection
The user selects one of the following options:
- **Foreground** – color is applied only to the detected person  
- **Background** – color is applied to everything except the person  

---

### Step 5: Targeted Colorization
A color overlay is applied only to the selected region while keeping the other region unchanged.  
This demonstrates **targeted colorization using semantic segmentation**.

---

### Step 6: Output Display
The final colorized image is displayed immediately through the GUI.

---

## Input and Output

### Input
- Image file (JPG / PNG)  
- Preferably containing a person  

### Output
- Image with color applied only to the selected region  
- Other regions remain unchanged  

---

## Key Features
- Semantic segmentation–based region detection  
- Selective (targeted) colorization  
- User-controlled region selection  
- Interactive graphical interface  
- No training required (pretrained model used)  

---

## Conclusion
This project successfully demonstrates **semantic segmentation for targeted colorization**.  
By integrating deep learning with a user-friendly GUI, selective colorization of image regions is achieved, fulfilling all requirements of **Task 5**.
This project holds strong academic importance as it integrates **core concepts of computer vision and deep learning** into a practical application. It helps students understand how **semantic segmentation** can be used to identify meaningful regions in an image and how these regions can be processed independently.

Through this task, students gain hands-on experience with:
- Pretrained deep learning models
- Image preprocessing techniques
- Region-based image manipulation
- Practical use of semantic segmentation in real-world scenarios

The project also emphasizes the importance of **user interaction** through a graphical interface, making it suitable for academic demonstrations, practical examinations, and internship evaluations. Overall, this task strengthens both theoretical understanding and practical implementation skills in the field of **artificial intelligence and image processing**.

---

## Academic Note
This project uses **open-source pretrained models** strictly for educational and internship purposes.  
The implementation and explanation are provided by the student as part of learning outcomes.
