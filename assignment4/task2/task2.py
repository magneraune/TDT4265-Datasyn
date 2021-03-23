import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE

    # Compute intersection
    x_min = max(prediction_box[0], gt_box[0])
    y_min = max(prediction_box[1], gt_box[1])
    x_max = min(prediction_box[2], gt_box[2])
    y_max = min(prediction_box[3], gt_box[3])
    intersection = (x_max - x_min)*(y_max - y_min) 

    if x_max < x_min or y_min > y_max:
        return 0.0
    
    # Compute union
    a_prediction = (prediction_box[2] - prediction_box[0])*(prediction_box[3] - prediction_box[1])
    a_gt = (gt_box[2] - gt_box[0])*(gt_box[3] - gt_box[1])
    iou = intersection / float(a_prediction + a_gt - intersection)
    assert iou >= 0 and iou <= 1
    return iou 


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if (num_tp + num_fp) != 0:
        precision = num_tp / (num_tp + num_fp)
        return precision
    else: 
        return 1
    raise NotImplementedError


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if (num_tp + num_fn) != 0:
        recall = num_tp / (num_tp + num_fn)
        return recall
    else: 
        return 0
    raise NotImplementedError


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    num_predictions = len(prediction_boxes)
    num_gt_boxes = len(gt_boxes)
    prediction_boxes = np.array(prediction_boxes)
    gt_boxes = np.array(gt_boxes)

    matches = np.zeros([num_gt_boxes, num_predictions])
    for i in range(num_gt_boxes):
        for j in range (num_predictions):
            iou = calculate_iou(prediction_boxes[j], gt_boxes[i])
            if iou >= iou_threshold:
                matches.itemset((i,j), iou)
    # Sort all matches on IoU in descending order
    results = np.zeros([3])
    for i in range(num_predictions):
        column = matches[:,i]
        closest = np.where(column == np.amax(column))
        iou = np.amax(column)
        if iou > 0:
            if  np.array_equal(results, [0., 0., 0.]):
                results = np.array([[closest[0][0], i, iou]])
            else: 
                results = np.append(results,[[closest[0][0], i, iou]], axis = 0)
    if np.array_equal(results, [0., 0., 0.]):
        return np.array([]), np.array([])

    results=results[np.argsort((-results)[:,-1])]

    # Find all matches with the highest IoU threshold
    _, uniq_gt_idxs = np.unique(results[:,0], return_index = True)
    results = results[uniq_gt_idxs]

    sorted_gt = gt_boxes[results[:,0].astype(int)]
    sorted_predicted = prediction_boxes[results[:,1].astype(int)]

    return sorted_predicted, sorted_gt


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    results = {"true_pos": int, "false_pos": int, "false_neg": int}
    _, gt_tp = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    results["true_pos"] = len(gt_tp)
    results["false_pos"] = len(prediction_boxes) - len(gt_tp)
    results["false_neg"] = len(gt_boxes) - len(gt_tp)

    return results
    raise NotImplementedError


def calculate_precision_recall_all_images(all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    tp, fp, fn = [], [], []
    
    for i in range(len(all_prediction_boxes)):
        results = calculate_individual_image_result(all_prediction_boxes[i], all_gt_boxes[i], iou_threshold)
        tp.append(results["true_pos"])
        fp.append(results["false_pos"])
        fn.append(results["false_neg"])
    
    tp = np.mean(np.asarray(tp))
    fp = np.mean(np.asarray(fp))
    fn = np.mean(np.asarray(fn))

    precision = calculate_precision(tp, fp, fn)
    recall = calculate_recall(tp, fp, fn)
    
    return (precision, recall)
    raise NotImplementedError


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE

    precisions = [] 
    recalls = []
    vec_all_prediction_boxes = []

    for threshold in confidence_thresholds:
        for image in range(len(confidence_scores)):
            val_prediction_boxes = []
            for score in range(len(confidence_scores[image])):
                if confidence_scores[image][score] >= threshold:
                    val_prediction_boxes.append(all_prediction_boxes[image][score])
            vec_all_prediction_boxes.append(val_prediction_boxes)
            val_prediction_boxes = []

        vec_all_prediction_boxes = np.asarray(vec_all_prediction_boxes)
        result = calculate_precision_recall_all_images(vec_all_prediction_boxes, all_gt_boxes, iou_threshold)
        precisions = np.append(precisions, result[0])
        recalls = np.append(recalls, result[1])
        vec_all_prediction_boxes = []
    
    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE    
    interpolation_values = []
    precision = []
    for recall_level in recall_levels:
        for precs, recs in zip(precisions, recalls):
            if recs >= recall_level:
                interpolation_values.append(precs)
        if interpolation_values:
            precision.append(max(interpolation_values))
        else: 
            precision.append(0)
        interpolation_values = []

    return np.average(np.asarray(precision))


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
