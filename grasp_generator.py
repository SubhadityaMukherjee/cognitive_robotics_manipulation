import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.npyio import save
import torch.utils.data
from PIL import Image
from datetime import datetime

from network.hardware.device import get_device
#from network.inference.post_process import post_process_output
from network.utils.data.camera_data import CameraData
from network.utils.visualisation.plot import plot_results
from network.utils.dataset_processing.grasp import detect_grasps
from skimage.filters import gaussian
import os
import cv2
import sys
import torchsnooper as tsp
import torch.nn as nn
sys.path.append("/home/robotics26/Downloads/grasp-detection-pytorch/src/models")

# try:
#     from train_model import 


# except:
#     pass

# class AlexNet_Conv(nn.Module):
#     r""" An AlexNet model. Most easily loaded with the .from_name or .from_pretrained methods
#     Args:
#       global_params (namedtuple): A set of GlobalParams shared between blocks
#     Examples:
#         model = AlexNet.from_pretrained("alexnet")
#     """

#     def __init__(self, global_params=None):
#         super(AlexNet_Conv, self).__init__()

#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),

#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),

#             nn.Conv2d(192, 384, kernel_size=2, padding=1),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(384, 256, kernel_size=2, padding=1),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(256, 256, kernel_size=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2)
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#         self.classifier = nn.Sequential(
#             nn.Dropout(global_params.dropout_rate),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(global_params.dropout_rate),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, global_params.num_classes),
#         )

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)

#     def extract_features(self, inputs):
#         """ Returns output of the final convolution layer """
#         x = self.features(inputs)
#         return x

#     def forward(self, inputs):
#         # See note [TorchScript super()]
#         x = self.features(inputs)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x

#     @classmethod
#     def from_name(cls, model_name, override_params=None):
#         cls._check_model_name_is_valid(model_name)
#         global_params = get_model_params(model_name, override_params)
#         return cls(global_params)

#     @classmethod
#     def from_pretrained(cls, model_name, num_classes=1000):
#         model = cls.from_name(model_name, override_params={"num_classes": num_classes})
#         load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
#         return model

#     @classmethod
#     def get_image_size(cls, model_name):
#         cls._check_model_name_is_valid(model_name)
#         _, res = alexnet_params(model_name)
#         return res

#     @classmethod
#     def _check_model_name_is_valid(cls, model_name):
#         """ Validates model name. None that pretrained weights are only available for
#         the first four models (alexnet) at the moment. """
#         valid_model = "alexnet"
#         if model_name not in valid_model:
#             raise ValueError("model_name should be one of: " + ", ".join(valid_model))

class GraspGenerator:
    IMG_WIDTH = 224
    IMG_ROTATION = -np.pi * 0.5
    CAM_ROTATION = 0
    PIX_CONVERSION = 277
    DIST_BACKGROUND = 1.115
    MAX_GRASP = 0.085

    def __init__(self, net_path, camera, depth_radius, fig, IMG_WIDTH=224, network='GR_ConvNet', device='cpu'):

        if (device=='cpu'):
            self.net = torch.load(net_path, map_location=device)
            self.device = get_device(force_cpu=True)
        else:
            #self.net = torch.load(net_path, map_location=lambda storage, loc: storage.cuda(1))
            #self.device = get_device()
            # self.net = torch.load(net_path, map_location='cpu')
            # self.device = get_device(force_cpu=True)
            if network == "GR_ConvNet":
                self.net = torch.load(net_path).cuda()
            # elif network == "alex":
            #     self.net = AlexNet_Conv(21).cuda()
            #     print(self.net.state_dict().keys())
            #     # self.net.state_dict().keys = ['conv0.weight', 'conv0.bias', 'conv3.weight', 'conv3.bias', 'conv6.weight', 'conv6.bias', 'conv8.weight', 'conv8.bias', 'conv10.weight', 'conv10.bias', 'fc1.weight', 'fc1.bias', 'fc4.weight', 'fc4.bias', 'fc6.weight', 'fc6.bias']
            #     # self.net = torch.load(net_path)
            #     self.checkpoint = torch.load(net_path)
            #     print(self.checkpoint.keys())
            #     self.net.load_state_dict(self.checkpoint)
                # print(self.net)
            self.device = get_device(force_cpu=False)


        # print (self.net)

        
        self.near = camera.near
        self.far = camera.far
        self.depth_r = depth_radius
        
        self.fig = fig
        self.network = network

        self.PIX_CONVERSION = 277 * IMG_WIDTH/224

        self.IMG_WIDTH = IMG_WIDTH
        print (self.IMG_WIDTH)

        # Get rotation matrix
        img_center = self.IMG_WIDTH / 2 - 0.5
        self.img_to_cam = self.get_transform_matrix(-img_center/self.PIX_CONVERSION,
                                                    img_center/self.PIX_CONVERSION,
                                                    0,
                                                    self.IMG_ROTATION)
        self.cam_to_robot_base = self.get_transform_matrix(
            camera.x, camera.y, camera.z, self.CAM_ROTATION)

    def get_transform_matrix(self, x, y, z, rot):
        return np.array([
                        [np.cos(rot),   -np.sin(rot),   0,  x],
                        [np.sin(rot),   np.cos(rot),    0,  y],
                        [0,             0,              1,  z],
                        [0,             0,              0,  1]
                        ])

    def grasp_to_robot_frame(self, grasp, depth_img):
        """
        return: x, y, z, roll, opening length gripper, object height
        """
        # Get x, y, z of center pixel
        x_p, y_p = grasp.center[0], grasp.center[1]

        # Get area of depth values around center pixel
        x_min = np.clip(x_p-self.depth_r, 0, self.IMG_WIDTH)
        x_max = np.clip(x_p+self.depth_r, 0, self.IMG_WIDTH)
        y_min = np.clip(y_p-self.depth_r, 0, self.IMG_WIDTH)
        y_max = np.clip(y_p+self.depth_r, 0, self.IMG_WIDTH)
        depth_values = depth_img[x_min:x_max, y_min:y_max]

        # Get minimum depth value from selected area
        z_p = np.amin(depth_values)

        # Convert pixels to meters
        x_p /= self.PIX_CONVERSION
        y_p /= self.PIX_CONVERSION
        z_p = self.far * self.near / (self.far - (self.far - self.near) * z_p)

        # Convert image space to camera's 3D space
        img_xyz = np.array([x_p, y_p, -z_p, 1])
        cam_space = np.matmul(self.img_to_cam, img_xyz)

        # Convert camera's 3D space to robot frame of reference
        robot_frame_ref = np.matmul(self.cam_to_robot_base, cam_space)

        # Change direction of the angle and rotate by alpha rad
        roll = grasp.angle * -1 + (self.IMG_ROTATION)
        if roll < -np.pi / 2:
            roll += np.pi

        # Covert pixel width to gripper width
        opening_length = (grasp.length / int(self.MAX_GRASP *
                          self.PIX_CONVERSION)) * self.MAX_GRASP

        obj_height = self.DIST_BACKGROUND - z_p

        # return x, y, z, roll, opening length gripper
        return robot_frame_ref[0], robot_frame_ref[1], robot_frame_ref[2], roll, opening_length, obj_height

    def post_process_output(self, q_img, cos_img, sin_img, width_img, pixels_max_grasp):
        """
        Post-process the raw output of the network, convert to numpy arrays, apply filtering.
        :param q_img: Q output of network (as torch Tensors)
        :param cos_img: cos output of network
        :param sin_img: sin output of network
        :param width_img: Width output of network
        :return: Filtered Q output, Filtered Angle output, Filtered Width output
        """
        q_img = q_img.cpu().numpy().squeeze()
        ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
        width_img = width_img.cpu().numpy().squeeze() * pixels_max_grasp

        q_img = gaussian(q_img, 1.0, preserve_range=True)
        ang_img = gaussian(ang_img, 1.0, preserve_range=True)
        width_img = gaussian(width_img, 1.0, preserve_range=True)

        return q_img, ang_img, width_img

    # @tsp.snoop()
    def predict(self, rgb, depth, n_grasps=1, show_output=False):

        max_val = np.max(depth)
        depth = depth * (255 / max_val)
        depth = np.clip((depth - depth.mean())/175, -1, 1)
        
        if (self.network in ['GR_ConvNet', 'alex']):
            ##### GR-ConvNet #####
            depth = np.expand_dims(np.array(depth), axis=2)
            img_data = CameraData(width=self.IMG_WIDTH, height=self.IMG_WIDTH)
            x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)
        else:
            print("The selected network has not been implemented yet -- please choose another network!")
            exit() 

        with torch.no_grad():
            xc = x.to(self.device)


            if (self.network in ['GR_ConvNet', 'alex']):
                ##### GR-ConvNet #####
                pred = self.net.predict(xc)
                # print (pred)
                pixels_max_grasp = int(self.MAX_GRASP * self.PIX_CONVERSION)
                q_img, ang_img, width_img = self.post_process_output(pred['pos'],
                                                                pred['cos'],
                                                                pred['sin'],
                                                                pred['width'],
                                                                pixels_max_grasp)
            else: 
                print ("you need to add your function here!")        
        
        save_name = None
        if show_output:
            #fig = plt.figure(figsize=(10, 10))
            im_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            plot = plot_results(self.fig,
                                rgb_img=im_bgr,
                                grasp_q_img=q_img,
                                grasp_angle_img=ang_img,
                                depth_img=depth,
                                no_grasps=3,
                                grasp_width_img=width_img)

            if not os.path.exists('network_output'):
                os.mkdir('network_output')
            time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            save_name = 'network_output/{}'.format(time)
            plot.savefig(save_name + '.png')
            plot.clf()

        grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=n_grasps)
        return grasps, save_name

    def predict_grasp(self, rgb, depth, n_grasps=1, show_output=False):
        predictions, save_name = self.predict(rgb, depth, n_grasps=n_grasps, show_output=show_output)
        grasps = []
        for grasp in predictions:
            x, y, z, roll, opening_len, obj_height = self.grasp_to_robot_frame(grasp, depth)
            grasps.append((x, y, z, roll, opening_len, obj_height))

        return grasps, save_name
