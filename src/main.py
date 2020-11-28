#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
from argparse import ArgumentParser
import sys
import logging as log
import math

from face_detection import FaceDetectionModel
from facial_landmarks_detection import LandmarkDetectionModel
from head_pose_estimation import HeadPoseModel
from gaze_estimation import GazeEstimationModel
from input_feeder import InputFeeder
from mouse_controller import MouseController

def build_argparser():
    '''
    Parse command line arguments.
    '''
    parser = ArgumentParser()
    
    parser.add_argument("-fd", "--face_detection_model", required=True, type=str,
                        help="Path to an .xml file with Face Detection model.")
    parser.add_argument("-fl", "--facial_landmark_model", required=True, type=str,
                        help="Path to an .xml file with Facial Landmark Detection model.")
    parser.add_argument("-hp", "--head_pose_model", required=True, type=str,
                        help="Path to an .xml file with Head Pose Estimation model.")
    parser.add_argument("-ge", "--gaze_estimation_model", required=True, type=str,
                        help="Path to an .xml file with Gaze Estimation model.")
                        
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file or CAM")
                        
    parser.add_argument("-ext", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
                             
    parser.add_argument("-pt", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for detection fitering. Default:0.6")
                        
    parser.add_argument("-o", "--output_path", default='./results',
                        help="Path to save outputs. Default: /results")
                        
    parser.add_argument("-vid", "--show_video", default=0, type=int, 
                        help="Flag to display the video output. 0: hide(default), 1:display")                        
                        
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")       
    
    parser.add_argument("-flag", "--visualization_flag", required=False, nargs='+',
                        default=['fd','fl','hp','ge'],
                        help="Example: --flag fd fl hp ge (each flag space-separated)"
                             "the visualization of different model outputs on each frame,"
                             "fd : Face Detection Model, fl : Facial Landmark Detection Model"
                             "hp : Head Pose Estimation Model, ge : Gaze Estimation Model.")
    
    return parser

# code source: https://knowledge.udacity.com/questions/171017
def draw_axes(frame, center_of_face, yaw, pitch, roll, scale, focal_length):
    yaw *= np.pi / 180.0
    pitch *= np.pi / 180.0
    roll *= np.pi / 180.0
    
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    
    Rx = np.array([[1, 0, 0],
                [0, math.cos(pitch), -math.sin(pitch)],
                [0, math.sin(pitch), math.cos(pitch)]])
    Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                [0, 1, 0],
                [math.sin(yaw), 0, math.cos(yaw)]])
    Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                [math.sin(roll), math.cos(roll), 0],
                [0, 0, 1]])
    # R = np.dot(Rz, Ry, Rx)
    # ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    # R = np.dot(Rz, np.dot(Ry, Rx))
    R = Rz @ Ry @ Rx
    # print(R)
    
    camera_matrix = build_camera_matrix(center_of_face, focal_length)
    
    xaxis = np.array(([1 * scale, 0, 0]),  dtype='float32').reshape(3, 1)
    yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
    zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
    zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
    
    o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
    o[2] = camera_matrix[0][0]
    
    xaxis = np.dot(R, xaxis) + o
    yaxis = np.dot(R, yaxis) + o
    zaxis = np.dot(R, zaxis) + o
    zaxis1 = np.dot(R, zaxis1) + o
    
    xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
    xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
    xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
    yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
    p1 = (int(xp1), int(yp1))
    xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, p1, p2, (255, 0, 0), 2)
    cv2.circle(frame, p2, 3, (255, 0, 0), 2)
    return frame
    

def build_camera_matrix(center_of_face, focal_length):
    cx, cy = int(center_of_face[0]), int(center_of_face[1])
    fx = focal_length     #or #fx = cx / np.tan(60/2 * np.pi / 180)    
    fy = fx
    cam_mat = np.float32([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return cam_mat    


def main():

    # Grab command line args
    args = build_argparser().parse_args()
    
    input_file_path = args.input
    input_feed = None
    viz_flag= args.visualization_flag

    output_path=args.output_path
       
    log.basicConfig(filename=os.path.join(output_path, 'main.log'), filemode='w', level=log.INFO)
    
    log.info("Setting up input stream...")      
    if args.input == "CAM":
            input_feed = InputFeeder("cam")
    else:
        if not os.path.isfile(args.input):
            log.info("Unable to find specified video file")
            sys.exit(1)
        input_feed = InputFeeder("video",args.input)
    log.info("Input feed successfully setup...")
    
    fdm  = FaceDetectionModel(model_name=args.face_detection_model, threshold=args.prob_threshold, device=args.device, extensions=args.cpu_extension)
    flm = LandmarkDetectionModel(model_name=args.facial_landmark_model, device=args.device, extensions=args.cpu_extension)
    hpm =          HeadPoseModel(model_name=args.head_pose_model,       device=args.device, extensions=args.cpu_extension)
    gem  =   GazeEstimationModel(model_name=args.gaze_estimation_model, device=args.device, extensions=args.cpu_extension)  
    
    mouse_control = MouseController('medium','fast')
    
    start=time.time()
    fdm.load_model()
    fdm_load = time.time()-start
    log.info("Face Detection Model loaded in {:.3f} ms".format( fdm_load * 1000))
    
    start=time.time()
    flm.load_model()
    flm_load = time.time()-start
    log.info("Facial Landmark Detection Model loaded in {:.3f} ms".format( flm_load * 1000))
    
    start=time.time()
    hpm.load_model()
    hpm_load = time.time()-start
    log.info("HeadPose Estimation Model loaded in {:.3f} ms".format( hpm_load * 1000))
    
    start=time.time()
    gem.load_model()
    gem_load = time.time()-start
    log.info("Gaze Estimation Model loaded in {:.3f} ms".format( gem_load * 1000))    
    log.info("All models loaded successfully...")
    
    input_feed.load_data()
    log.info("Input feed loaded successfully...")
    
    initial_w, initial_h, video_len, fps = input_feed.metadata()
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    
    log.info("Starting inferencing...")
    infer_start = time.time()
    fdm_inf, hpm_inf, flm_inf, gem_inf = 0,0,0,0 #store individual inference times
    focal_length = 950.0
    scale = 50
    
    show_vid_flag=args.show_video
    
    frame_count = 0
    for ret, frame in input_feed.next_batch():
        if not ret:
            break
         
        key= cv2.waitKey(60)        
        frame_count+=1
        #if frame_count%5 == 0:
        #   cv2.imshow('video',cv2.resize(frame,(500,500)))
        #   cv2.imwrite(str('bin/video_fc_frame_'+str(frame_count)+".jpg") , frame )            
        
        log.info(str("analyzing frame "+str(frame_count)+" ...") )
        
        inf_st= time.time()
        face_coords, cropped_face = fdm.predict(frame.copy())
        fdm_inf += time.time() - inf_st
                        
        if face_coords == 0:
            log.info("Unable to detect the face in the frame...")
            if key==27:
                break
            continue
        
        inf_st= time.time()
        hpa = hpm.predict(cropped_face.copy())
        hpm_inf += time.time() - inf_st        
        
        inf_st= time.time()
        coord_eye, l_eye, r_eye = flm.predict(cropped_face.copy())
        flm_inf += time.time() - inf_st        
        
        inf_st= time.time()
        new_mouse_coord, gaze_vector = gem.predict(l_eye, r_eye, hpa)
        gem_inf += time.time() - inf_st        
        
        log.info(str("frame_"+str(frame_count)+" analysis complete...") )
        
        if len(viz_flag) != 0:
            #view_face = cropped_face.copy()
            view_window = frame.copy()
            
            if 'fd' in viz_flag :
                cv2.rectangle( view_window, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (100,255,0), 3)
                    
            if 'fl' in viz_flag :
                cv2.rectangle( view_window, (face_coords[0]+coord_eye[0][0], face_coords[1]+coord_eye[0][1]),
                                            (face_coords[0]+coord_eye[0][2], face_coords[1]+coord_eye[0][3]), (100,100,0), 3) #left eye
                cv2.rectangle( view_window, (face_coords[0]+coord_eye[1][0], face_coords[1]+coord_eye[1][1]), 
                                            (face_coords[0]+coord_eye[1][2], face_coords[1]+coord_eye[1][3]), (100,100,0), 3) #right eye
                #cv2.rectangle( view_face, (coord_eye[0][0], coord_eye[0][1]), (coord_eye[0][2], coord_eye[0][3]), (100,100,0), 3) #left eye
                #cv2.rectangle( view_face, (coord_eye[1][0], coord_eye[1][1]), (coord_eye[1][2], coord_eye[1][3]), (100,100,0), 3) #right eye
                
            if 'hp' in viz_flag :
                cv2.putText( view_window, "yaw {:.1f} , pitch {:.1f} , roll {:.1f}".format(hpa[0], hpa[1], hpa[2]),
                                (20,20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)
            
            if 'ge' in viz_flag :            
                yaw, pitch, roll = hpa[0], hpa[1], hpa[2]
                
                center_of_face = (face_coords[0] + cropped_face.shape[1] / 2, face_coords[1] + cropped_face.shape[0] / 2, 0)
                view_window = draw_axes(view_window.copy(), center_of_face, yaw, pitch, roll, scale, focal_length)
                cv2.putText( view_window, "new mouse coord {:.1f} , {:.1f} ".format(new_mouse_coord[0], new_mouse_coord[1]),
                                (20,60), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)                                                               
            
            if show_vid_flag != 0 :
                cv2.imshow("visualization",cv2.resize(view_window,(500,500)))
                #cv2.imwrite(str('bin/video_fc_'+str(frame_count)+".jpg") , view_window )
            out_video.write(view_window)
        
        log.info(str("Moving mouse to x: ")+str(new_mouse_coord[0])+" y: "+str(new_mouse_coord[1]) )
        
        #if frame_count%5==0:
        try:
            mouse_control.move(new_mouse_coord[0],new_mouse_coord[1])    
        except Exception as e:
            log.info("Error in moving mouse to this location...")
                
        if key==27:
                break
            
    log.info("VideoStream ended...")
    cv2.destroyAllWindows()
    input_feed.close()
    
    #logging inference times
    if(frame_count>0):
        log.info("............ Individual Inference times ..........") 
        log.info("Face Detection:{:.1f}ms".format(1000* fdm_inf/frame_count))
        log.info("Facial Landmarks Detection:{:.1f}ms".format(1000* flm_inf/frame_count))
        log.info("Headpose Estimation:{:.1f}ms".format(1000* hpm_inf/frame_count))
        log.info("Gaze Estimation:{:.1f}ms".format(1000* gem_inf/frame_count))        
        
    tot_inf_time = round(time.time() - infer_start, 1)
    fps = int(frame_count) / tot_inf_time
    tot_load_time = fdm_load + flm_load + hpm_load + gem_load
    
    log.info("total frames analyzed: {} ".format(frame_count))
    log.info("total loading time {} seconds".format(tot_load_time))
    log.info("total inference time {} seconds".format(tot_inf_time))
    log.info("fps {} frame/second".format(fps))
    
    log.info("END.......")
    
    with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write("inf_time: "+str(tot_inf_time)+'\n')
            f.write("fps: "+str(fps)+'\n')
            f.write("load time: "+str(tot_load_time)+'\n')
     
if __name__ == '__main__':
    main()