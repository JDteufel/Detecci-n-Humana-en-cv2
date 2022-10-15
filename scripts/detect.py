
import cv2, imutils, argparse, time, numpy as np
from pprint import pprint
ap = argparse.ArgumentParser() #import argumets from two to three onetime instance
ap.add_argument("-p", "--type", required=True, help="Path of a video or an img.")
ap.add_argument("-y", "--yolo", required=True,default="yolo-coco",help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())
yolo=args["yolo"]
input_path=args["type"]
input_type=input_path.split('.')[-1] # This will read the file path
if input_type in ['jpg','jpeg','png']: # If given path contains image data
    if input_type in ['jpg','jpeg','bg','png']:
        detectByPathImg(input_path)
    elif input_type in ['mp4','webm','mkv']:
        writer = None

"""

HOG Descriptor
HOG stands for Histogram of Oriented Gradients. It's a feature descriptor that represents the image by histograms of gradient directions, or edge directions. 
We compute them over overlapping local regions to achieve translational invariant feature extraction.

HOG Descriptor steps
1.Calculate the image gradients - This is an important step because they help us to calculate the direction of the gradient on each pixel of the image.

2.Compute the histogram of gradients - Next use histogram to calculate the edge directions of the gradient, these directions are the gradients of the image.

3.Normalize the histogram - We then normalize by removing the error with multiple images of different tamperatures and illuminations.

4.Image Representation - We then use extracted features to represent the training and the testing images.

"""

HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect(frame):
    """
    The detect function - this is where we feed our image data to openCV's hog object detector that we created in load() method. 
    Detect does all this for us based on the input - this could be video data, webcam frames, or a single image.
    """
    
    start = time.time() 
    # converts the grayscale images into 2 dimensional numpy arrays.
    # the hog Descriptor then assigns an orientation angle to a pixel in the image.
    (person_counting, weights) = HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8),scale=1.05) 

    end = time.time()
    print("[INFO] Person counting...")

    for (x, y, w, h) in person_counting: #Loop our rectangle and for loop to define a box around A person,
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # draw rectangle x y define the top left coordinate,
        cv2.putText(frame, f'person {i}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1) #Places coordinates on frame to define a rendered box around a person
        person += 1 # count each rectangle as long as we find found person
    cv2.putText(frame, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2) #Let user know that app is onttaking action #Using an indexing in the print statement to ensure that the number of people is shown onscreen
    cv2.putText(frame, f'Total Persons : {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.imshow('output', frame)  # This will show the bounding box around people in green along with their indexes
    print("[INFO] Time: {:.6f}".format(end - start)) # How much time frame took to compute & write, after how much time it will run
    return frame


def detectByPathVideo(path, writer): #Pass the video path and encoding of given directory
    video = cv2.VideoCapture(path) # Read the video using VideoCapture API
    check, frame = video.read() #if true succeeds, if false fails
    if check == False: # If frame is not captured
        print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
        return

    print('Detecting people...')
    while video.isOpened(): # If Video is Read, Process Frame
        #check is True if reading was successful 
        check, frame =  video.read()

        if check:
            frame = imutils.resize(frame , width=min(800,frame.shape[1])) # resize the fram to 800 to 200
            frame = detect(frame) # detect the person and bound based the input
            
            if writer is not None:  #If the given path of video is is incorrect then it will take time to complete.so we write a frame at every 10th frame so it wil get completed fastly.
                writer.write(frame) # Passing the frame to the writer which detected the person in the frame
            
            key = cv2.waitKey(1) & 0xFF # wait for key input
            if key== ord('q'): # if key supplied is 'q' then exit the loop else pass
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows() #destroy the frame


def detectByPathImg(path): #Pass the image path and encoding of given directory
    frame = cv2.imread(path) # Read the frame using imread API
    frame = imutils.resize(frame , width=min(800,frame.shape[1])) # resize the fram to 800 to 200
    frame = detect(frame) # detect the person and bound based the input
    cv2.imshow('output', frame) # show the frame
    cv2.waitKey(0) #wait for user input to proceed
    cv2.destroyAllWindows()


if __name__ == "__main__":
        
        load() # Load the model
    
        ap = argparse.ArgumentParser() # command line options
        ap.add_argument("-i", "--image", type=str,
        help="path to input image") # pass the image along with the path
        ap.add_argument("-v", "--video", type=str,
        help="path to input video file") # pass the video along with the path
        ap.add_argument("-o", "--output", type=str,
        help="path to optional output video file") # pass the output path along the video path
        ap.add_argument("-y", "--display", type=int, default=1,
        help="whether or not output frame should be displayed")
        args = vars(ap.parse_args()) # parse the arguments
    
        """
        (y) allows us to display the image inline(without building the application again) in the second print command below
        """
        print("Image Path: ", args['image']) # print input path
        print("Image Path: ", args['video'])
        
        writer = None
        if args["output"] != "" and args["output"] is not None:# checking output option
            fourcc = cv2.VideoWriter_fourcc(*"MJPG") #Why am I writing output in MJPG?
            writer = cv2.VideoWriter(args["output"], fourcc, 25, # if only video frame as an output then fourcc, 25, 640,360  - no height or width - is specified
            (frame.shape[1], frame.shape[0]), True) 
    
        if args['image'] is not None:
            detectByPathImg(args['image']) 
    
        if args['video'] is not None:
            detectByPathVideo(args['video'], writer) #pass video along with the path variable
 
"""

More maintainable code

Seperation of responsibilities makes it easier to change behavior because only one class/method will need to be changed. It’s also easier to debug when large parts of functionality are separated into different units. Just think about past projects where you may have been confused about how code worked, you were able to work around it by hacking at some variables or function calls.

Having total control over how components interact by providing the interface not inheriting it from a base class is important. You will likely change something and the order in which code is executed for subclasses of a given superclass will most likely change. This can have chain reactions with other class functionality. By expecting certain parameters, classes you are using do not need to have any relation to each other, thus freeing up the program.

Make your class super powers useful

We can’t possibly determine every situation that a class might be used for. Thankfully, this is where class inheritance
    
    extends by default java comes with features like interface, inheritance,
    abstraction, polymorphism in terms of features. But through inheritance we can achieve the capability of interfaces and abstract methods.
    
    Compile time polymorphism - method overloading
    

    """
    Overloading : Type -1: changing num of args, Type-2 : type of args, Type:3: sequence of args


    # Type -1: Default values
    def add (a ,b ,c =0):
        return a+b,c

    print(add(1,2)) # add(1,2) will mean that all the value without name has been given the value here(a,b and c=0)
    print(add(3,3,3))# add(3,3,3) will mean that 2 parameter has been given a ,b, and c=3)
    print(add(a=3,b=2,c=5))# if you know the name of parameter you can also give a ,b ,c value

    # # Type-2: Keyword Args
    def personDetails(Name,Age,Calss= None): # first declare the variable whose value is changed the most.Last value is assigned
        return Name, Age, Calss
        if Calss == null:
            Calss = None
            return Name, Age, Calss
        else:
            return Name, Age,  Calss

    print(personDetails(Age = 10, Name = 'Niraj'))
    print(personDetails(Age= 12, Name = 'Raj', Calss = "10"))



    