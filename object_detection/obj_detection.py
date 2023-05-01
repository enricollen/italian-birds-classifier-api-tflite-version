import cv2
import numpy as np

def load_model():
    # Load the Tiny YOLOv3 model
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def load_labels():
    # Load labels
    with open("coco.names", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def load_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: could not load or find image")
        exit()
    return image

def preprocess_image(image, input_size):
    # Preprocess the image
    if image.shape[0] < input_size[0] or image.shape[1] < input_size[1]:
        image = cv2.resize(image, input_size, interpolation=cv2.INTER_LINEAR)
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, input_size, swapRB=True, crop=False)
    return blob

def get_output(net, blob):
    # Set the input to the network
    net.setInput(blob)
    
    # Get the output from the network
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)
    return outputs

def detect_birds(outputs, labels, confidence_threshold=0.15):
    # Loop over each detected object
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Check if the detected object is a bird
            if labels[class_id] == "bird" and confidence > confidence_threshold:
                return confidence
    return None

def main():
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
    main()