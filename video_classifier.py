#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
import joblib
import cv2
import numpy as np
# from PIL import Image
# from face_recognition import preprocessing
# from inference.utilCv2ByMe import draw_bb_on_img
# from inference.constants import MODEL_PATH
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
#-------------------------------------------------
#DeepSort
from deep_sort import preprocessing_deepsort, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
max_cosine_distance = 0.4
nn_budget = None
nms_max_overlap = 1.0
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

#-------------------------------------------------
from Facemask.utils.anchor_generator import generate_anchors
from Facemask.utils.anchor_decode import decode_bbox
from Facemask.utils.nms import single_class_non_max_suppression
from Facemask.load_model.keras_loader import load_keras_model, keras_inference
model_facemask = load_keras_model('model/face_mask_detection.json', 'model/face_mask_detection.hdf5')
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
anchors_exp = np.expand_dims(anchors, axis=0)
id2class = {0: 'Mask', 1: 'NoMask'}
def inference(image,conf_thresh=0.5,iou_thresh=0.5,target_shape=(260, 260)):
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0
    image_exp = np.expand_dims(image_np, axis=0)
    y_bboxes_output, y_cls_output = keras_inference(model_facemask, image_exp)
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)
    keep_idxs = single_class_non_max_suppression(y_bboxes,bbox_max_scores,conf_thresh=conf_thresh,iou_thresh=iou_thresh)
    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        try:
            if class_id == 0: #0 la co khau trang
                labelstr = "Mask"
                color_mask = (0,255,0)
            else:
                labelstr = "No Mask"
                color_mask = (0,0,255)
        except: pass
    return labelstr, color_mask
#----------------------------------------------------------------------------------
age_model = load_model("model/AgeGenderRaceFromVGG/OutputModels/age_model.h5")
gender_model = load_model("model/AgeGenderRaceFromVGG/OutputModels/gender_model.h5")
# race_model = load_model("model/AgeGenderRaceFromVGG/OutputModels/race_model.h5")
face_recogniser = joblib.load(MODEL_PATH)
preprocess = preprocessing.ExifOrientationNormalize()
red_color = (0,0,255)
green_color = (0, 255, 0)
output_indexes = np.array([i for i in range(0, 101)])
# datetime_object = datetime.datetime.now()
def age_gender_race_predict(img):
    preFace = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preFace = cv2.resize(preFace, (224, 224))
    preFace = img_to_array(preFace)
    preFace = np.expand_dims(preFace, axis=0)
    preFace = preFace / 255.0
    try:
        # Tuoi
        age_distributions = age_model.predict(preFace)
        apparent_agestr = str(int(np.floor(np.sum(age_distributions * output_indexes, axis=1))[0]))
        # Gioi tinh
        gender_prediction = gender_model.predict(preFace)[0]
        # race_predictions = race_model.predict(preFace)[0]
        # race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'hispanic']
        #
        # sum_of_predictions = race_predictions.sum()
        #
        # # resp_obj["race"] = {}
        # for i in range(0, len(race_labels)):
        #     race_label = race_labels[i]
        #     race_prediction = 100 * race_predictions[i] / sum_of_predictions
        # dominant_race= race_labels[np.argmax(race_predictions)]

        # cv2.rectangle(frame, (a, b), (a1, b1), red_color, 2)
        # cv2.putText(frame, label, (a + 10, b1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,color_mask, thickness=2, lineType=2)
        # cv2.putText(frame, class_name, (a + 10, b + 20), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.6, red_color, thickness=2, lineType=2)
        if np.argmax(gender_prediction) == 0:
            apparent_genderstr = "Woman"
            # cv2.putText(frame, "Woman" + " _ " + apparent_age, (a + 10, b1 - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, red_color, thickness=2, lineType=2)
            # # print(str(datetime_object) + ' ' + class_name + ' Woman ' + apparent_age + '  ' + label)

        elif np.argmax(gender_prediction) == 1:
            apparent_genderstr = "Man"
        # cv2.putText(frame, str(apparent_gender) + " _ " + apparent_age, (a + 10, b1 - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, red_color, thickness=2, lineType=2)
            # print(str(datetime_object) + ' ' + class_name + ' Man ' + apparent_age + '  ' + label)


    except Exception as e:
        print("exception", str(e))

    return apparent_genderstr, apparent_agestr
def main():
    cap = cv2.VideoCapture("data/Videooo.mp4")
    output_file = "data/demo.avi"
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_format = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, video_format, 20.0, (width, height))

    while True:
        # Capture frame-by-frame
        datetime_object = datetime.datetime.now()
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        img = Image.fromarray(frame)
        faces = face_recogniser(preprocess(img))
        if faces is not None:
            bboxes = []
            classes = []
            confidences = []
            apparent_gender = []
            apparent_age = []
            label = []
            for face in faces:
                margin = 10
                x = int(face.bb.left)
                y = int(face.bb.top)
                x1 = int(face.bb.right)
                y1 = int(face.bb.bottom)
                each_bboxes = [x,y,x1-x,y1-y]
                bboxes.append(each_bboxes)
                confidences.append(face.top_prediction.confidence)
                unknowFace = frame[int(y-margin):int(y1+margin),int(x-margin):int(x1+margin)]
                try:labelstr, color_mask = inference(unknowFace,conf_thresh=0.5,iou_thresh=0.4,target_shape=(260, 260))
                except:pass
                if face.top_prediction.confidence > 0.85:
                    confidence = "%.2f" % (face.top_prediction.confidence * 100)
                    class_name = "%s" %(face.top_prediction.label)
                    with open("data.txt", "r") as file:
                        for item in file:
                            data = item.split()
                            try:
                               if data[1] == class_name:
                                    apparent_agestr = data[2]
                                    apparent_genderstr = data[3]
                            except:
                                apparent_agestr = "None"
                                apparent_genderstr = "None"
                    cv2.rectangle(frame, (x, y), (x1, y1), green_color, 2)
                    cv2.putText(frame, class_name + " " + confidence, (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                green_color, thickness=2, lineType=2)
                    cv2.putText(frame, str(apparent_genderstr) + ' _ ' + str(apparent_agestr), (x + 10, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, green_color, thickness=2, lineType=2)
                    cv2.putText(frame, labelstr, (x + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                color_mask, thickness=2, lineType=2)
                    # print(str(datetime_object) + ' ' + class_name + ' '+confidence + ' ' + str(d_gender) + ' ' + str(d_age) + ' ' + label)
                else:
                    class_name = "Unknow"
                    # age_gender_race_predict(unknowFace, frame, class_name, label,x,y,x1,y1,color_mask)
                    apparent_genderstr, apparent_agestr = age_gender_race_predict(unknowFace)
                    cv2.rectangle(frame, (x, y), (x1, y1), red_color, 2)
                    cv2.putText(frame, labelstr, (x + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_mask, thickness=2,
                                lineType=2)
                    cv2.putText(frame, class_name, (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, red_color, thickness=2, lineType=2)
                    cv2.putText(frame, str(apparent_genderstr) + " _ " + apparent_agestr, (x + 10, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, red_color, thickness=2, lineType=2)
                #--------------------------------------
                classes.append(class_name)
                apparent_gender.append(apparent_genderstr)
                apparent_age.append(apparent_agestr)
                label.append(labelstr)
            bboxes = np.array(bboxes)
            confidences = np.array(confidences)
            classes = np.array(classes)
            apparent_gender = np.array(apparent_gender)
            apparent_age = np.array(apparent_age)
            label = np.array(label)
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, cl_name, feature, apparent_gender, apparent_age, label) for bbox, score, cl_name, feature, apparent_gender, apparent_age, label in
                              zip(bboxes, confidences, classes, features, apparent_gender, apparent_age, label)]
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            # run non-maxima supression
            indices = preprocessing_deepsort.non_max_suppression(boxes, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            tracker.predict()
            tracker.update(detections)
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,0), 1)
                cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 0.5, (0, 255, 0), 1)
        # Display the resulting frame
        out.write(frame)
        cv2.imshow('video', frame)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    # When everything done, release the captureq
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
