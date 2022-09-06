import os
import sys
import face_recognition
import cv2


KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"

TOLERANCE = 0.6     # default settings is 60%, but may vary

# FRAME_THICKNESS = 3
# FONT_THICKNESS = 2

MODEL = "cnn"   # however, hog also can be while running on cpu, but we use CUDA, right?

print("loading known faces..")

known_faces = []
known_names = []



cam_port = 0
cam = cv2.VideoCapture(cam_port)
print("Starting Video")
# reading the input using the camera
result, image = cam.read()

# If image will detected without any error,
# show result
if result:

    # showing result, it take frame name and image
    # output
    # cv2.imshow("GeeksForGeeks", image)

    # saving image in local storage
    # f"{UNKNOWN_FACES_DIR}/{filename}"
    tempname = "temp.png"

    #cv2.imwrite("temporary.png", image)

    cv2.imwrite(f"{UNKNOWN_FACES_DIR}/{tempname}", image)

    # If keyboard interrupt occurs, destroy image
    # window
    # cv2.waitKey(0)
    # cv2.destroyWindow("GeeksForGeeks")

# If captured image is corrupted, moving to else part
else:
    print("No image detected. Please, try again")


def match_faces():

    # Looking throughout drivers library-

    for name in os.listdir(KNOWN_FACES_DIR):
        for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):

            # find image
            image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
            # encode image features
            encoding = face_recognition.face_encodings(image)[0]

            known_faces.append(encoding)
            known_names.append(name)

    print("processing unknown pictures..\n,,\n..\n,,")


    # Matching unknown face with a library of known faces

    for filename in os.listdir(UNKNOWN_FACES_DIR):

        print("Looking into: " + filename + "\n")
        image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")

        locations = face_recognition.face_locations(image, model=MODEL)

        encoding = face_recognition.face_encodings(image, locations)

        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for face_encoding, face_location in zip(encoding, locations):

            # main line that works lol, GIVES A LIST OF BOOLEANS
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

            match = None
            if True in results:
                match = known_names[results.index(True)]
                print(f"Matched with: {match}")
                return match
            else:
                print("No matches")

    os.remove(f"{UNKNOWN_FACES_DIR}/{tempname}")
                # opencv visualisation stuff, not needed at all bcz runs on a server

                # top_left = (face_location[3], face_location[0])
                # bottom_right = (face_location[1], face_location[2])
                #
                # color = [0, 255, 0]
                #
                # cv2.rectangle(image,top_left,bottom_right,color,FRAME_THICKNESS)
                #
                # top_left = (face_location[3], face_location[2])
                # bottom_right = (face_location[1], face_location[2]+22)
                # cv2.rectangle(image, top_left, bottom_right,color, cv2.FILLED)
                # cv2.putText(image, (face_location[3]+10,face_location[2]+15), cv2.FONT_HERSHEY_PLAIN,
                #             0.5, (200,200,200), FONT_THICKNESS)

        # cv2.imshow(filename, image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def main():
    pass


if __name__ == '__main__':
    match_faces()