import numpy as np
import json
import sys

"""
Implement and test the utilities in support of evaluating the results
from the region-by-region decisions and turning them into detection.

All rectangles are four component lists (or tuples) giving the upper
left and lower right cornders of an axis-aligned rectangle.  For example, 
[2, 9, 12, 18] has upper left corner (2,9) and lower right (12, 18)

The region predictions for an image for an image are stored in a list
of dictionaries, each giving the class, the activation and the
bounding rectangle.  For example,

{
    "class": 2,
    "a":  0.67,
    "rectangle": (18, 14, 50, 75)
}

if the class is 0 this means there is no detection and the rectangle
should be ignored.  The region predictions must be turned into the
detection results by filtering those with class 0 and through non
maximum supression.  The resulting regions should be considered the
"detections" for the image.

After this, detections should be compared to the ground truth 

The ground truth regions for an image are stored as a list of dictionaries. 
Each dictionary contains the region's class and bounding rectangle.
Here is an example dictionary:

{
    "class":  3,
    "rectangle": (15, 20, 56, 65)
}

Class 0 will not appear in the ground truth.  
"""


def area(rect):
    h = rect[3] - rect[1]
    w = rect[2] - rect[0]
    return h * w

# [2, 9, 12, 18] has upper left corner (2,9) and lower right (12, 18)
def iou(rect1, rect2):
    """
    Input: two rectangles
    Output: IOU value, which should be 0 if the rectangles do not overlap.
    """
    area1 = area(rect1)
    area2 = area(rect2)
    '''
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy
    '''
    dx = min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])
    dy = min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]) 
    
    if (dx>=0) and (dy>=0):
        intersect = dx*dy
    else:
        intersect = 0

    return intersect/(area1+area2-intersect)

def predictions_to_detections(predictions, iou_threshold=0.5):
    """
    Input: List of region predictions

    Output: List of region predictions that are considered to be
    detection results. These are ordered by activation with all class
    0 predictions eliminated, and the non-maximum suppression
    applied.
    """
    # First remove all elements with class 0 
    #print("predictions")
    #print(predictions)

    # print("IOU THRESHOLD:", iou_threshold)

    detections = [] 
    for p in predictions: 
        if p["class"] == 0:
            continue
        detections.append(p)
    '''
    detections = sorted(detections, key = lambda i: i["a"], reverse = True)
    
    result = [] 
    while detections: 
        result.append(detections.pop(0))
        
        count = 0 
        while count < len(detections):
            if iou(detections[count]["rectangle"], result[-1]["rectangle"]) > iou_threshold:
                detections.pop(count)
            else: 
                count += 1

    

    detections = result
    detections = sorted(detections, key = lambda i: i["a"], reverse = True)
    '''
    #print(len(detections))
    # Now run non-maximum suppression algorithm
    
    result = [] 
    to_remove = [] 
    for i in range(len(detections)):
        add = True 
        if i in to_remove: continue
        for j in range(i+1, len(detections)):
            if iou(detections[i]["rectangle"], detections[j]["rectangle"]) > iou_threshold and detections[i]["class"] == detections[j]["class"]: 
                #print("i:", i, detections[i])
                #print("j:", j, detections[j])
                #print("IOU:",iou(detections[i]["rectangle"], detections[j]["rectangle"]) )
                if detections[i]["a"] < detections[j]["a"]:
                    add = False 
                else: 
                    to_remove.append(j)
        if add: 
            result.append(detections[i])

    detections = result
    
    detections = sorted(detections, key = lambda i: i["a"], reverse = True)

    return detections


def evaluate(detections, gt_detections, n=10):
    """
    Input:
    1. The detections returned by the predictions_to_detections function
    2. The list of ground truth regions, and
    3. The maximum number (n) of detections to consider.

    The calculation must compare each detection region to the ground
    truth detection regions to determine which are correct and which
    are incorrect.  Finally, it must compute the average precision for
    up to n detections.

    Returns:
    list of correct detections,
    list of incorrect detections,
    list of ground truth regions that are missed,
    AP@n value.
    """
    #print("n", n)
    #print("gt_detections")
    #print(gt_detections)

    correct = [] 
    incorrect = [] 
    used_gt = []  
    binary_decision = np.zeros(len(detections))
    precision = np.zeros(len(detections))
    count = 0
    for d in detections: 
        max_iou = 0 
        max_val = 0
        for g in gt_detections: 
            val_iou = iou(d["rectangle"], g["rectangle"])
            if val_iou > max_iou and g["class"] == d["class"]:
                
                max_iou = val_iou
                max_val = g

        if max_iou > 0.5 and d["class"] == max_val["class"]: 
            binary_decision[count] = 1
            correct.append(d)
            used_gt.append(max_val)
        else: incorrect.append(d)

        count += 1
        


    missed_gt = []
    for gt in gt_detections:
        if gt not in used_gt:
            missed_gt.append(gt)

    

    

    precision = np.cumsum(binary_decision)/np.arange(1,len(binary_decision)+1)
    AP = (1/min(n, len(gt_detections)))*np.sum(np.multiply(binary_decision[:min(n,len(binary_decision))],precision[:min(n,len(precision))]))


    return correct, incorrect, missed_gt, AP


def test_iou():
    """
    Use this function for you own testing of your IOU function
    """
    # should be .370
    rect1 = (0, 5, 11, 15)
    rect2 = (2, 9, 12, 18)
    print("iou for %a, %a is %1.3f" % (rect1, rect2, iou(rect1, rect2)))

    # should be 0
    rect1 = (2, -3, 11, 4)
    print("iou for %a, %a is %1.3f" % (rect1, rect2, iou(rect1, rect2)))

    # should be 0.2
    rect1 = (3, 12, 9, 15)
    print("iou for %a, %a is %1.3f" % (rect1, rect2, iou(rect1, rect2)))


if __name__ == "__main__":
    """
    The main program code is meant to test the functions above.  Test
    detection are input through a JSON file that contains a dictionary
    with region predictions and ground truth detections.

    DO NOT CHANGE THE CODE BELOW THIS LINE.
    """
    
    if len(sys.argv) != 2:
        print("Usage: %s data.json" % sys.argv[0])
        sys.exit(0)

    with open(sys.argv[1], "r") as infile:
        data = json.load(infile)

    region_predictions = data["region_predictions"]
    gt_detections = data["gt_detections"]

    detections = predictions_to_detections(region_predictions)
    print("DETECTIONS: count =", len(detections))
    if len(detections) >= 2:
        print("DETECTIONS: first activation %.2f" % detections[0]["a"])
        print("DETECTIONS: last activation %.2f" % detections[-1]["a"])
    elif len(detections) == 1:
        print("DETECTIONS: only activation %.2f" % detections[0]["a"])
    else:
        print("DETECTIONS: no activations")

    correct, incorrect, missed, ap = evaluate(detections, gt_detections)

    print("AP: num correct", len(correct))
    if len(correct) > 0:
        print("AP: first correct activation %.2f" % correct[0]["a"])

    print("AP: num incorrect", len(incorrect))
    if len(incorrect) > 0:
        print("AP: first incorrect activation %.2f" % incorrect[0]["a"])

    print("AP: num ground truth missed", len(missed))
    print("AP: final AP value %1.3f" % ap)

    '''

    class_target = np.random.randint(4, size=16)
    bb_target = np.random.randint(20, size=(16,4))
    pred_bb = np.random.randint(40, size=(16, 16))
    

    pred_bb_res = np.zeros((16,4))
    for i in range(16):
        target = class_target[i]-1
        if target == -1: continue

        pred_bb_res[i] = pred_bb[i, target*4:(target*4)+4]

    print("Class_target")
    print(class_target)
    print("BB")
    print(pred_bb)
    print("Res")
    print(pred_bb_res)
    '''