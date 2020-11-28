'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import cv2
import logging as log

class LandmarkDetectionModel:
    '''
    Class for the Landmark Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights = model_name+'.bin'
        self.model_structure = model_name+'.xml'
        self.device = device
        self.extensions = extensions        
        self.model = None
        self.input_name = None
        self.input_shape = None
        self.output_name = None
        self.output_shape = None        
        
    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        log.info("Loading Landmark Detection Engine...")
        try:
            self.core = IECore()                
            self.model= self.core.read_network(self.model_structure, self.model_weights)
        except Exception as e:
            log.info("Error in model path")
            raise ValueError("Could not Initialise the network. Please enter the correct model path...")
        
        log.info("Checking for unsupported layers..")
        load_status = self.check_model()
        
        if load_status != 1:
            exit(1)
            #return 0
        log.info("All layers of the model are supported")
        
        self.net = self.core.load_network( network=self.model, device_name= self.device, num_requests=1)        
        log.info("Inference Engine loaded successfully")
        
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def check_model(self):
    
        supported_layers = self.core.query_network(network=self.model, device_name=self.device)        
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        
        if len(unsupported_layers) != 0 and self.device=='CPU' :
            print("Unsupported layers found: {}".format(unsupported_layers))
            log.info("Unsupported layers found...")
            
            if not self.extensions == None:
                print("Adding CPU extensions for unsupported layers...")
                log.info("Adding CPU exensions for unsupported layers...")
                
                self.core.add_extension(self.extensions, self.device)
                
                supported_layers = self.core.query_network(network=self.network, device_name=self.device)        
                unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
                
                if len(unsupported_layers)!=0:
                    print("Unsupported layers found after adding extensions...")
                    log.info("Unsupported layers found after adding extensions...")
                    return -1
                    
                print("After adding extensions, issue is resolved...")                           
                log.info("Model is supported after adding extensions...")
                return 1
                
            else:
                print("Please provide path to extension(s) for unsupported layers")
                log.info("Misssing path to extension(s) for unsupported layers")
                return 0 
        return 1

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        #preprocess    
        p_frame = self.preprocess_input(image)        
        input_dict={ self.input_name : p_frame}
        
        #infer
        infer = self.net.start_async(request_id=0, inputs=input_dict)
        
        status = infer.wait()
        #wait for results
        if status == 0:            
            #get result
            infer_outputs = infer.outputs[self.output_name]
            #process output and extract coords
            coords = self.preprocess_outputs(infer_outputs)                
            #crop face in the image
            coords, l_eye, r_eye = self.crop_outputs(coords, image)
            return coords, l_eye, r_eye

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)        
        return p_frame

    def preprocess_outputs(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        box = outputs[0]        
        coords = [ box[0],box[1],box[2],box[3] ]        
        return coords        

    def crop_outputs(self, coords, image):
        '''
        TODO: This method needs to be completed by you
        '''
        height, width = image.shape[:2]     
        img_cpy = image.copy()     
        
        l_eye_xmin = int(coords[0] * width) - 10
        l_eye_xmax = int(coords[0] * width) + 10    
        l_eye_ymin = int(coords[1] * height) - 10
        l_eye_ymax = int(coords[1] * height) + 10

        r_eye_xmin = int(coords[2] * width) - 10
        r_eye_xmax = int(coords[2] * width) + 10
        r_eye_ymin = int(coords[3] * height) - 10
        r_eye_ymax = int(coords[3] * height) + 10

        #cv2.rectangle(image, (l_eye_xmin, l_eye_ymin), (l_eye_xmax, l_eye_ymax), (0,255,0), 3)   #draw bounds
        #cv2.rectangle(image, (r_eye_xmin, r_eye_ymin), (r_eye_xmax, r_eye_ymax), (0,255,0), 3)   #draw bounds
        coord_scaled =[ [l_eye_xmin, l_eye_ymin, l_eye_xmax, l_eye_ymax], [r_eye_xmin, r_eye_ymin, r_eye_xmax, r_eye_ymax] ]
        l_eye = img_cpy[ l_eye_ymin : l_eye_ymax, l_eye_xmin : l_eye_xmax ]
        r_eye = img_cpy[ r_eye_ymin : r_eye_ymax, r_eye_xmin : r_eye_xmax ]                

        return coord_scaled, l_eye, r_eye               