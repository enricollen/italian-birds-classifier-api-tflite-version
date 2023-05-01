import cv2 as cv2 # this way it uses opencv-python-headless instead of opencv-python, to reduce memory consumption
from PIL import Image
import numpy as np

class ObjectDetector:
    def __init__(self):
        self.net = None
        self.labels = None
        
        # Initialize OpenCV in headless mode to reduce memory usage
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4)
        cv2.ocl.setUseOpenCL(False)
    
    def load_model(self):
        # Load the Tiny YOLOv3 model
        self.net = cv2.dnn.readNet("object_detection/yolov3-tiny.weights", "object_detection/yolov3-tiny.cfg")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    def load_labels(self):
        # Load labels
        with open("object_detection/coco.names", "r") as f:
            self.labels = [line.strip() for line in f.readlines()]
    
    def load_image(self, img):
        # Load image
        #image = cv2.imread(image_path)
        img_pil = Image.open(img.stream).convert("RGB")
        img_np = np.array(img_pil)
        image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        if image is None:
            print("Error: could not load or find image")
            exit()
        return image
    
    def preprocess_image(self, image, input_size):
        # Preprocess the image
        if image.shape[0] < input_size[0] or image.shape[1] < input_size[1]:
            image = cv2.resize(image, input_size, interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, input_size, swapRB=True, crop=False)
        return blob
    
    def get_output(self, blob):
        # Set the input to the network
        self.net.setInput(blob)

        # Get the output from the network
        output_layers = self.net.getUnconnectedOutLayersNames()
        outputs = self.net.forward(output_layers)
        return outputs
    
    def detect_birds(self, outputs, confidence_threshold=0.15):
        # Loop over each detected object
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Check if the detected object is a bird
                if self.labels[class_id] == "bird" and confidence > confidence_threshold:
                    return confidence
        return None

"""def main():
    # Load the model
    net = load_model()

    # Load the labels
    labels = load_labels()

    # Load the image
    image = load_image("test-images/3.png")

    # Preprocess the image
    input_size = (416, 416)
    blob = preprocess_image(image, input_size)

    # Get the output from the model
    outputs = get_output(net, blob)

    # Detect birds in the image
    confidence = detect_birds(outputs, labels)

    # Print the result
    if confidence is not None:
        print("Bird detected! Confidence: " + str(confidence))
    else:
        print("No bird detected.")
        
if __name__ == '__main__':
    main()"""