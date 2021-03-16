import torch
import torchvision

from pose_estimation.single_cube_dataset import SingleCubeDataset
from pose_estimation.evaluation_metrics.translation_average_mean_square_error import (
    translation_average_mean_square_error,
)
from pose_estimation.evaluation_metrics.orientation_average_quaternion_error import (
    orientation_average_quaternion_error,
)
from pose_estimation.cuboid import Cuboid3d
import numpy as np
import quaternion
import cv2
from PIL import Image as im 


def get_camera_matrix(focal, center):
    """
    Returns camera intrinsics matrix from focal and image centers
    f = [fx, fy]: focal lengths along x and y axes
    c = [cx, cy]: Image centers along x and y axes
    """
    matrix_camera = np.zeros((3, 3))
    matrix_camera[0, 0] = focal[0]
    matrix_camera[1, 1] = focal[1]
    matrix_camera[0, 2] = center[0]
    matrix_camera[1, 2] = center[1]
    matrix_camera[2, 2] = 1
    return matrix_camera

def draw_cuboid(vert, img):
    """
    Draws the cuboid on an input image with given vertices in pixels and returns the updated image

    Args:
        vert: Ndarray of the 8 vertices of the cuboid
        img: Ndarray of image that the cubiod is added to.
    """
    # print(type(img))
    # img = np.asarray(img)
    # img = im.fromarray(img) 
    print("Drawing linessss")
    img = cv2.circle(img, tuple(np.int_(vert[0])), 1, (0, 0, 255), 4)
    img = cv2.circle(img, tuple(np.int_(vert[1])), 1, (0, 0, 255), 4)
    img = cv2.circle(img, tuple(np.int_(vert[2])), 1, (0, 0, 255), 4)
    img = cv2.circle(img, tuple(np.int_(vert[3])), 1, (0, 0, 255), 4)
    img = cv2.circle(img, tuple(np.int_(vert[4])), 1, (0, 0, 255), 4)
    img = cv2.circle(img, tuple(np.int_(vert[5])), 1, (0, 0, 255), 4)
    img = cv2.circle(img, tuple(np.int_(vert[6])), 1, (0, 0, 255), 4)
    img = cv2.circle(img, tuple(np.int_(vert[7])), 1, (0, 0, 255), 4)

    img = cv2.line(img, tuple(np.int_(vert[1])), tuple(np.int_(vert[0])), (0, 255, 0), 2)
    img = cv2.line(img, tuple(np.int_(vert[0])), tuple(np.int_(vert[3])), (0, 255, 0), 2)
    img = cv2.line(img, tuple(np.int_(vert[3])), tuple(np.int_(vert[2])), (0, 255, 0), 2)
    img = cv2.line(img, tuple(np.int_(vert[2])), tuple(np.int_(vert[1])), (0, 255, 0), 2)
    img = cv2.line(img, tuple(np.int_(vert[5])), tuple(np.int_(vert[4])), (0, 255, 0), 2)
    img = cv2.line(img, tuple(np.int_(vert[4])), tuple(np.int_(vert[7])), (0, 255, 0), 2)
    img = cv2.line(img, tuple(np.int_(vert[7])), tuple(np.int_(vert[6])), (0, 255, 0), 2)
    img = cv2.line(img, tuple(np.int_(vert[6])), tuple(np.int_(vert[5])), (0, 255, 0), 2)
    img = cv2.line(img, tuple(np.int_(vert[2])), tuple(np.int_(vert[6])), (0, 255, 0), 2)
    img = cv2.line(img, tuple(np.int_(vert[1])), tuple(np.int_(vert[5])), (0, 255, 0), 2)
    img = cv2.line(img, tuple(np.int_(vert[3])), tuple(np.int_(vert[7])), (0, 255, 0), 2)
    img = cv2.line(img, tuple(np.int_(vert[0])), tuple(np.int_(vert[4])), (0, 255, 0), 2)
    return img

def visualize_3Dboundingbox(object_size, object_center, camera_matrix, pose, img):
    """
    Returns the input image with added 3D bounding box visualization

    Args:
        Object size, object_center: Tuple of size 3 each
        camera_matrix: camera intrinsics matrix,
        pose: Ndarray of 7 elements [q.w, q.x, q.y, q.z, x, y, z],
        img: input image to draw the 3D bounding box on.
    """
    _cuboid3d = Cuboid3d(object_size, object_center)
    cuboid3d_points = np.array(_cuboid3d.get_vertices())
    rotation_matrix = quaternion.as_rotation_matrix(np.quaternion(pose[0], pose[1], \
        pose[2], pose[3]))
    # Reference: https://www.programcreek.com/python/example/89450/cv2.Rodrigues
    rvec = cv2.Rodrigues(rotation_matrix)[0]
    print(type(rvec))
    tvec = pose[4:]
    # print(type(tvec))
    tvec = np.asarray(tvec)
    print(type(tvec))

    dist_coeffs = np.zeros((4, 1))
    # Compute the pixel coordinates of the 3D points
    projected_points, _ = cv2.projectPoints(cuboid3d_points, rvec, tvec, camera_matrix,\
        dist_coeffs)
    projected_points = np.squeeze(projected_points)
    # Draw line to form 3D bounding box from project points
    img = draw_cuboid(projected_points, img)
    return img

def visualize_model(estimator):
    """
    Do the evaluation process for the estimator

    Args:
        estimator: pose estimation estimator
    """
    config = estimator.config

    dataset_test = SingleCubeDataset(
        config=config,
        split="test",
        zip_file_name=config.test.dataset_zip_file_name_test,
        data_root=config.system.data_root,
        sample_size=config.test.sample_size_test,
    )

    estimator.logger.info("Start evaluating estimator: %s", type(estimator).__name__)

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config.test.batch_test_size,
        num_workers=0,
        drop_last=False,
    )

    estimator.model.to(estimator.device)
    visualize_one_epoch(
        estimator=estimator,
        config=config,
        data_loader=test_loader,
        epoch=0,
        test=True,
    )


def visualize_one_epoch(*, estimator, config, data_loader, epoch, test):
    """Evaluation of the model on one epoch
    Args:
        estimator: pose estimation estimator
        config: configuration of the model
        data_loader (DataLoader): pytorch dataloader
        epoch (int): the current epoch number
        test (bool): specifies which type of evaluation we are doing
    """
    estimator.model.eval()
    estimator.logger.info(f" evaluation started")

    metric_translation = 0.0
    metric_orientation = 0.0

    if test:
        batch_size = config.test.batch_test_size
    elif test == False:
        batch_size = config.val.batch_validation_size
    else:
        raise ValueError(f"You need to specify a boolean value for the test argument")

    number_batches = len(data_loader) / batch_size
    with torch.no_grad():
        metric_translation, metric_orientation = visualization_over_batch(
            estimator=estimator,
            config=config,
            data_loader=data_loader,
            batch_size=batch_size,
            epoch=epoch,
            is_training=False,
        )

        estimator.writer.log_evaluation(
            evaluation_metric_translation=metric_translation,
            evaluation_metric_orientation=metric_orientation,
            epoch=epoch,
            test=test,
        )

# HELPER
def visualization_over_batch(
    *,
    estimator,
    config,
    data_loader,
    batch_size,
    epoch,
    is_training=True,
    optimizer=None,
    criterion_translation=None,
    criterion_orientation=None,
):
    sample_size = config.train.sample_size_train if is_training else config.val.sample_size_val
    len_data_loader = sample_size if (sample_size > 0) else len(data_loader)

    # camera_matrix = get_camera_matrix([150,150], [112,112])
    camera_matrix = [
          [
            0.885858,
            0.0,
            0.0
          ],
          [
            0.0,
            1.73205078,
            0.0
          ],
          [
            0.0,
            0.0,
            -1.0006001
          ]
        ]
    camera_matrix = np.asarray(camera_matrix)
    obj_size = [0.11999999731779099, 0.06199999526143074, 0.043999996036291122]
    # save_animation_path = os.path.join("{}/{}.mp4".format(results_dir,'outlier_images_with_predicted_poses'))



    for index, (images, _, _) in enumerate(
        data_loader
    ):
        # images = images.cpu()
        # images = images.numpy()
        # np.squeeze(arr)
        # print("no. of images:", type(images[0]))
        # print(images[0][0].shape)
        img = torchvision.transforms.ToPILImage()(images[0][0])
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imshow("test", img) 
 
        # cv2.waitKey(0)  

        print("hereeeeeeeeeeeeeeeeeeeeeeeeeeeee")
        images = list(image.to(estimator.device) for image in images)
        
        loss_translation = 0
        loss_orientation = 0

        predicted_poses = []
        output_translation, output_orientation = estimator.model(
            torch.stack(images).reshape(
                -1, 3, config.dataset.image_scale, config.dataset.image_scale
            )
        )


        # output_translation = output_translation.cpu()
        # output_orientation = output_orientation.cpu()
        # output_translation = output_translation.numpy()
        # output_orientation = output_orientation.numpy()
        

        output_translation= [[-0.25032687187194824, 0.19656550884246826, 0.84752857685089111]]
        output_orientation = [[0.740606427192688,0.47540977597236633,-0.38725721836090088, -0.2748083770275116]]
        output_orientation = np.asarray(output_orientation)
        output_translation= np.asarray(output_translation)

        print("Translation", output_translation[0])
        print("orientation", output_orientation[0])


        for i in output_orientation[0]:
        	predicted_poses.append(i)
        for i in output_translation[0]:
        	predicted_poses.append(i)


        print("predicted_poses", predicted_poses)
        img_res = visualize_3Dboundingbox(obj_size, [0,0,0], camera_matrix, predicted_poses, img)

        cv2.imshow("res", img_res) 
 
        cv2.waitKey(0)  
        # if (args.use_2d_detections):
        #     images[i] = visualize_2Dboundingbox(predicted_detections[outlier_ind[i], :],
        #                                         images[i])

    # Save the outlier images with 2D and 3D predicted bounding boxes as animation
    # save_animation_from_image_list(images, save_animation_path)



